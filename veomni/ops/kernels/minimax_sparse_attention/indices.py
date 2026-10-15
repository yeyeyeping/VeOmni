# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packed sequence metadata and sparse adjacency conversions for MiniMax MSA.

These helpers use ordinary PyTorch operations so their indexing contracts can
be tested without CANN. Q/K/V are never padded. CSR rows follow Split-KV's
block-major, batch-interleaved order; GBSAG uses sample-local query ids.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PackedMetadata:
    cu_seqlens: torch.Tensor
    lengths: torch.Tensor
    token_batch: torch.Tensor
    positions: torch.Tensor
    row_map: torch.Tensor
    row_batch: torch.Tensor
    row_block: torch.Tensor
    max_seqlen: int
    block_size: int
    batch_size: int
    num_blocks: int


def prepare_metadata(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int,
    device: torch.device,
    block_size: int = 128,
) -> PackedMetadata:
    """Validate packed boundaries and materialize int32 device metadata.

    VeOmni normally provides CPU boundaries on NPU. Device boundaries are
    accepted but require a host copy for validation and allocation sizes.
    Zero-length segments and the synthetic SP tail sequence are preserved.
    """
    if type(max_seqlen) is not int:
        raise TypeError("MiniMax MSA max_seqlen must be a Python int.")
    if block_size <= 0 or max_seqlen < 0 or total_tokens > torch.iinfo(torch.int32).max:
        raise ValueError("MiniMax MSA requires valid int32 sizes and a positive block_size.")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("MiniMax MSA cu_seqlens must be an integer prefix sum [B+1].")
    host_cu = cu_seqlens.detach().to(device="cpu", dtype=torch.int64)
    boundaries = host_cu.tolist()
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    if boundaries[0] != 0 or boundaries[-1] != total_tokens or any(n < 0 or n > max_seqlen for n in lengths):
        raise ValueError("MiniMax MSA cu_seqlens must span [0,T] with lengths in [0,max_seqlen].")

    batch_size = len(lengths)
    num_blocks = (max_seqlen + block_size - 1) // block_size
    rows = torch.tensor([(n + block_size - 1) // block_size for n in lengths], dtype=torch.long, device="cpu")
    # Flattening this J-by-B mask enumerates only existing blocks, interleaving
    # batches at each local block index, including ragged/empty sequences.
    alive = torch.arange(num_blocks, device="cpu")[:, None] < rows[None, :]
    packed_rows = (alive.flatten().long().cumsum(0) - 1).view(num_blocks, batch_size)
    row_map = packed_rows.masked_fill(~alive, -1).T.contiguous().to(device)
    row_batch = torch.arange(batch_size, device="cpu").expand(num_blocks, -1)[alive].to(device)
    row_block = torch.arange(num_blocks, device="cpu")[:, None].expand(-1, batch_size)[alive].to(device)
    cu = host_cu.to(device=device, dtype=torch.int32)
    device_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    token_batch = torch.repeat_interleave(
        torch.arange(batch_size, device=device), device_lengths.long(), output_size=total_tokens
    )
    positions = torch.arange(total_tokens, device=device) - cu.long()[token_batch]
    return PackedMetadata(
        cu,
        device_lengths,
        token_batch,
        positions,
        row_map,
        row_batch,
        row_block,
        max_seqlen,
        block_size,
        batch_size,
        num_blocks,
    )


def select_blocks(
    scores: torch.Tensor,
    metadata: PackedMetadata,
    topk_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Turn [G,T,J_aligned] raw scores into the public [T,G,K] selection."""
    if topk_blocks <= 0 or local_blocks < 0:
        raise ValueError("MiniMax MSA requires topk_blocks > 0 and local_blocks >= 0.")
    total_tokens = metadata.positions.numel()
    if scores.ndim != 3 or scores.shape[1] != total_tokens or scores.shape[-1] < metadata.num_blocks:
        raise ValueError("MiniMax MSA scores must be [G,T,J_aligned] and cover max_seqlen.")
    if total_tokens == 0:
        return torch.empty((0, scores.shape[0], topk_blocks), dtype=torch.int32, device=scores.device)
    scores = scores[..., : metadata.num_blocks]
    query_block = metadata.positions // metadata.block_size
    block_id = torch.arange(metadata.num_blocks, device=scores.device)
    scores = scores.masked_fill(block_id[None, None, :] > query_block[None, :, None], -float("inf"))
    if local_blocks:
        local_idx = (query_block[:, None] - torch.arange(local_blocks, device=scores.device)[None, :]).clamp(min=0)
        scores.scatter_(-1, local_idx[None].expand(scores.shape[0], -1, -1), float("inf"))
    width = min(topk_blocks, metadata.num_blocks)
    values, indices = scores.topk(width, dim=-1)
    indices = indices.masked_fill(values == -float("inf"), -1)
    indices = F.pad(indices, (0, topk_blocks - width), value=-1)
    return indices.permute(1, 0, 2).to(torch.int32).contiguous()


def validate_selection(block_indices: torch.Tensor, metadata: PackedMetadata, num_groups: int) -> None:
    """Reject patterns that would give different forward/backward semantics."""
    if (
        block_indices.ndim != 3
        or block_indices.shape[:2] != (metadata.positions.numel(), num_groups)
        or block_indices.shape[-1] <= 0
        or block_indices.dtype != torch.int32
        or block_indices.device != metadata.cu_seqlens.device
    ):
        raise ValueError("MiniMax MSA block_indices must be int32 [T,G,K] on the QKV device.")
    if block_indices.shape[0] * block_indices.shape[-1] > torch.iinfo(torch.int32).max:
        raise ValueError("MiniMax MSA CSR capacity T*K exceeds int32.")
    ordered = block_indices.sort(dim=-1).values
    duplicate = (ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)
    query_block = metadata.positions // metadata.block_size
    checks = torch.stack(
        (
            (block_indices >= -1).all(),
            (block_indices <= query_block[:, None, None]).all(),
            (block_indices >= 0).any(dim=-1).all(),
            (~duplicate).all(),
        )
    )
    # One explicit synchronization protects the opaque kernels from invalid
    # ids, repeated blocks and fully masked rows.
    if not bool(checks.all()):
        raise ValueError(
            "MiniMax MSA requires unique causal block ids, at least one per token/group, with -1 padding."
        )


def _csr_edges(
    block_indices: torch.Tensor, metadata: PackedMetadata
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort edges by packed KV row then global Q id, retaining top-k slots."""
    tokens, groups, topk = block_indices.shape
    capacity = tokens * topk
    rows = metadata.row_batch.numel()
    selected = block_indices.permute(1, 0, 2).reshape(groups, capacity).long()
    query_ids = torch.arange(tokens, device=selected.device).repeat_interleave(topk)
    slots = torch.arange(topk, device=selected.device).repeat(tokens)
    if capacity == 0:
        empty = torch.empty((groups, 0), dtype=torch.int32, device=selected.device)
        return torch.zeros((groups, rows + 1), dtype=torch.int32, device=selected.device), empty, empty, empty.long()
    batch = metadata.token_batch[query_ids]
    row_ids = metadata.row_map[batch[None, :], selected.clamp(min=0)]
    row_ids = row_ids.masked_fill(selected < 0, rows)
    counts = torch.zeros((groups, rows + 1), dtype=torch.int32, device=selected.device)
    counts.scatter_add_(1, row_ids, torch.ones_like(row_ids, dtype=torch.int32))
    row_ptr = F.pad(counts[:, :rows].cumsum(dim=1, dtype=torch.int32), (1, 0))
    # Composite keys give ascending Q order without relying on stable sort.
    order = (row_ids * (tokens + 1) + query_ids[None, :]).argsort(dim=-1)
    sorted_rows = row_ids.gather(1, order)
    q_indices = query_ids.expand(groups, -1).gather(1, order).masked_fill(sorted_rows == rows, -1).to(torch.int32)
    slot_indices = slots.expand(groups, -1).gather(1, order).masked_fill(sorted_rows == rows, -1).to(torch.int32)
    return row_ptr, q_indices, slot_indices, sorted_rows


def build_k2q_csr(
    block_indices: torch.Tensor, metadata: PackedMetadata
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Split-KV CSR from public [T,G,K] ids; input must be validated."""
    row_ptr, q_indices, slot_indices, _ = _csr_edges(block_indices, metadata)
    return row_ptr, q_indices, slot_indices


def build_gbsag_indices(block_indices: torch.Tensor, metadata: PackedMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sorted [B,G,J,max_seqlen] local Q lists and [B,G,J] counts."""
    row_ptr, q_indices, _, sorted_rows = _csr_edges(block_indices, metadata)
    groups, capacity = q_indices.shape
    batch, blocks, width = metadata.batch_size, metadata.num_blocks, metadata.max_seqlen
    counts = torch.zeros((batch, groups, blocks), dtype=torch.int32, device=block_indices.device)
    if capacity == 0:
        return counts.new_full((batch, groups, blocks, width), -1), counts

    rows = metadata.row_batch.numel()
    safe_rows = sorted_rows.clamp(max=rows - 1)
    row_batch = metadata.row_batch[safe_rows]
    row_block = metadata.row_block[safe_rows]
    rank = torch.arange(capacity, device=block_indices.device)[None, :] - row_ptr.long().gather(1, sorted_rows)
    group = torch.arange(groups, device=block_indices.device)[:, None]
    destinations = ((row_batch * groups + group) * blocks + row_block) * width + rank
    total_size = batch * groups * blocks * width
    destinations = destinations.masked_fill(sorted_rows == rows, total_size)
    local_q = q_indices - metadata.cu_seqlens[row_batch]
    local_q = local_q.masked_fill(sorted_rows == rows, -1)
    # A throwaway final element receives all invalid capacity slots.
    flat = counts.new_full((total_size + 1,), -1)
    flat.scatter_(0, destinations.flatten(), local_q.flatten())
    count_destinations = (metadata.row_batch[None, :] * groups + group) * blocks + metadata.row_block[None, :]
    counts.view(-1).scatter_(0, count_destinations.flatten(), row_ptr.diff(dim=1).flatten())
    return flat[:-1].view(batch, groups, blocks, width), counts
