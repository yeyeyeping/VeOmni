"""Immutable-by-contract, microbatch-owned MiniMax sparse attention metadata.

Prepare after final packing/SP padding, before entering the model. Keep the
object alive through backward and checkpoint recomputation; never overwrite its
storage or pool it across live microbatches.
"""

import torch


class SparseAttentionMetadata:
    """Opaque to FSDP tensor-container traversal; owns host and device buffers."""

    __slots__ = (
        "cu_host",
        "cu",
        "cu_i64",
        "lengths",
        "block_counts",
        "cu_blocks",
        "positions",
        "gbsa_present",
        "gbsa_rows",
        "gbsa_slots",
        "start_loc",
        "atten_mask",
        "max_seqlen",
        "backward_max_seqlen",
        "total_tokens",
        "total_rows",
        "max_kv",
        "pinned",
        "storage",
        "ready",
        "producer_stream",
    )

    def __init__(
        self,
        cu_host,
        cu,
        cu_i64,
        lengths,
        block_counts,
        cu_blocks,
        positions,
        gbsa_present,
        gbsa_rows,
        gbsa_slots,
        start_loc,
        atten_mask,
        max_seqlen,
        backward_max_seqlen,
        total_tokens,
        total_rows,
        max_kv,
        pinned,
        storage,
        ready,
        producer_stream,
    ):
        object.__setattr__(self, "cu_host", cu_host)
        object.__setattr__(self, "cu", cu)
        object.__setattr__(self, "cu_i64", cu_i64)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "block_counts", block_counts)
        object.__setattr__(self, "cu_blocks", cu_blocks)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "gbsa_present", gbsa_present)
        object.__setattr__(self, "gbsa_rows", gbsa_rows)
        object.__setattr__(self, "gbsa_slots", gbsa_slots)
        object.__setattr__(self, "start_loc", start_loc)
        object.__setattr__(self, "atten_mask", atten_mask)
        object.__setattr__(self, "max_seqlen", max_seqlen)
        object.__setattr__(self, "backward_max_seqlen", backward_max_seqlen)
        object.__setattr__(self, "total_tokens", total_tokens)
        object.__setattr__(self, "total_rows", total_rows)
        object.__setattr__(self, "max_kv", max_kv)
        object.__setattr__(self, "pinned", pinned)
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "ready", ready)
        object.__setattr__(self, "producer_stream", producer_stream)

    def __setattr__(self, name, value):
        raise AttributeError("Sparse attention metadata is immutable")

    def wait(self, device):
        """Enqueue only a stream dependency, never a host synchronization."""
        if self.cu.device != device:
            raise ValueError(f"Sparse metadata device {self.cu.device} != consumer {device}")
        if device.type == "npu":
            stream = torch.npu.current_stream(device)
            if stream != self.producer_stream:
                stream.wait_event(self.ready)
                for tensor in (
                    self.storage,
                    self.positions,
                    self.gbsa_present,
                    self.gbsa_rows,
                    self.gbsa_slots,
                    self.atten_mask,
                ):
                    tensor.record_stream(stream)


def prepare_sparse_attention_metadata(cu_seqlens, max_seqlen, device):
    """Snapshot host boundaries and upload one packed pinned buffer per batch.

    CPU-only support is for adapter unit tests. Device prefix tensors must be
    materialized on the host by the input preparer, never implicitly here.
    Empty segments are legal; decreasing boundaries and overflow are rejected.
    """
    device = torch.device(device)
    if cu_seqlens.device.type != "cpu":
        raise ValueError("Prepare sparse metadata from CPU boundaries before model.forward")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be a 1-D CPU int32/int64 prefix tensor")
    if not isinstance(max_seqlen, int):
        raise TypeError("max_seqlen must be a host Python int")
    with torch.profiler.record_function("msa_metadata_prepare"):
        host = cu_seqlens.to(dtype=torch.int64).clone()
        lengths = host.diff()
        if int(host[0]) != 0 or bool((lengths < 0).any()) or int(host[-1]) > torch.iinfo(torch.int32).max:
            raise ValueError("Invalid sparse prefix boundaries")
        if max_seqlen < int(lengths.max()):
            raise ValueError("max_seqlen is smaller than an actual segment")
        total = int(host[-1])
        backward_max = min(max_seqlen, total)
        counts = (lengths + 127) // 128
        cu_blocks = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
        # int64 first ensures alignment; all following fields use int32 views.
        fields = [
            host,
            host.int(),
            lengths.int(),
            counts.int(),
            cu_blocks.int(),
            torch.zeros_like(lengths, dtype=torch.int32),
        ]
        sizes = [t.numel() * t.element_size() for t in fields]
        pinned = torch.empty(sum(sizes), dtype=torch.uint8, device="cpu", pin_memory=device.type == "npu")
        offset = 0
        for tensor, size in zip(fields, sizes):
            pinned[offset : offset + size].copy_(tensor.view(torch.uint8))
            offset += size
        with torch.profiler.record_function("msa_metadata_pinned_h2d"):
            storage = pinned.to(device=device, non_blocking=True)
        views = []
        offset = 0
        for tensor, size in zip(fields, sizes):
            views.append(storage[offset : offset + size].view(tensor.dtype))
            offset += size
        cu64, cu, used, block_counts, blocks_cu, start_loc = views
        positions = torch.arange(total, device=device) - torch.repeat_interleave(
            cu64[:-1], used.long(), output_size=total
        )
        blocks = (backward_max + 127) // 128
        present = torch.arange(blocks, device=device)[:, None] < block_counts[None, :]
        rows = (present.flatten().long().cumsum(0) - 1).clamp(min=0).view(blocks, lengths.numel()).T
        slots = torch.arange(backward_max, device=device)
        mask = torch.zeros((2048, 2048), dtype=torch.int8, device=device)
        stream = torch.npu.current_stream(device) if device.type == "npu" else None
        ready = torch.npu.Event() if device.type == "npu" else None
        if ready is not None:
            ready.record(stream)
        return SparseAttentionMetadata(
            host,
            cu,
            cu64,
            used,
            block_counts,
            blocks_cu,
            positions,
            present,
            rows,
            slots,
            start_loc,
            mask,
            max_seqlen,
            backward_max,
            total,
            int(cu_blocks[-1]),
            int(counts.max()),
            pinned,
            storage,
            ready,
            stream,
        )
