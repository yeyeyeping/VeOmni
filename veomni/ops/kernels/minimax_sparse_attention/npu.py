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

"""Packed MiniMax M3 CANN indexer and split-KV/GBSA training adapters."""

import importlib
import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from ....utils import logging
from ....utils.import_utils import is_torch_npu_available
from .eager import minimax_sparse_attention_eager


logger = logging.get_logger(__name__)
_BLOCK_SIZE = 128
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@lru_cache(maxsize=1)
def _cann_ops():
    if not is_torch_npu_available():
        raise RuntimeError("MiniMax CANN attention requires torch_npu and an Ascend NPU.")
    try:
        return importlib.import_module("cann_ops_transformer.ops")
    except ImportError as exc:
        raise ImportError(
            "Install the MiniMax operators and their matching cann_ops_transformer.ops Torch extensions "
            "to use minimax_sparse_attention_implementation='npu_split_kv_gbsa'."
        ) from exc


def _cann_op(name):
    op = getattr(_cann_ops(), name, None)
    if not callable(op):
        raise ImportError(f"The installed cann_ops_transformer.ops must export callable {name!r}.")
    return op


@dataclass(frozen=True)
class _PackedLayout:
    prefix: tuple[int, ...]
    lengths: tuple[int, ...]
    block_counts: tuple[int, ...]
    max_seqlen: int

    @property
    def max_blocks(self):
        return (self.max_seqlen + _BLOCK_SIZE - 1) // _BLOCK_SIZE


def _packed_layout(cu_seqlens, max_seqlen, total_tokens):
    if not isinstance(max_seqlen, int):
        raise TypeError("MiniMax CANN max_seqlen must be a Python int.")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("MiniMax CANN cu_seqlens must be a 1D int32/int64 prefix sum with at least two entries.")
    # CANN needs a Python total_rows for its CSR allocation. Reuse the given
    # pack boundaries; never reconstruct them from device position_ids. Host
    # metadata avoids this copy; device metadata incurs a small D2H sync.
    prefix = tuple(cu_seqlens.detach().cpu().tolist())
    lengths = tuple(end - begin for begin, end in zip(prefix, prefix[1:]))
    if prefix[0] != 0 or prefix[-1] != total_tokens or min(lengths) < 0:
        raise ValueError(
            "MiniMax CANN cu_seqlens must start at 0, be nondecreasing, and end at the packed token count."
        )
    if max_seqlen < max(lengths) or max_seqlen < 0:
        raise ValueError("MiniMax CANN max_seqlen must cover every packed sequence.")
    if total_tokens > torch.iinfo(torch.int32).max:
        raise ValueError("MiniMax CANN packed token count exceeds int32 capacity.")
    return _PackedLayout(prefix, lengths, tuple((length + 127) // 128 for length in lengths), max(lengths))


def _check_npu_tensors(*tensors):
    if any(tensor.device.type != "npu" for tensor in tensors):
        raise ValueError("MiniMax CANN inputs must be NPU tensors.")
    if any(tensor.device != tensors[0].device for tensor in tensors):
        raise ValueError("MiniMax CANN inputs must be on the same NPU.")


def _token_positions(layout, device):
    lengths = torch.tensor(layout.lengths, dtype=torch.long, device=device)
    starts = torch.tensor(layout.prefix[:-1], dtype=torch.long, device=device)
    return torch.arange(layout.prefix[-1], device=device) - torch.repeat_interleave(
        starts, lengths, output_size=layout.prefix[-1]
    )


@lru_cache(maxsize=8)
def _mask_template(device):
    # Current CANN validates this tensor but computes causal bounds internally.
    return torch.zeros((2048, 2048), dtype=torch.int8, device=device)


def _select_from_scores(scores, layout, topk_blocks, local_blocks):
    """Map CANN block scores to VeOmni's fixed-width [T,Hkv,K] contract."""
    blocks = layout.max_blocks
    expected_stride = (blocks + 15) // 16 * 16
    if scores.ndim != 3 or scores.shape[1:] != (layout.prefix[-1], expected_stride) or scores.dtype != torch.float32:
        raise RuntimeError("msa_index_score returned an unexpected score shape or dtype.")
    scores = scores[..., :blocks].clone()
    # CANN writes FP32 minimum, whereas eager marks invalid entries with -inf.
    scores.masked_fill_(scores == torch.finfo(torch.float32).min, -torch.inf)
    positions = _token_positions(layout, scores.device)
    scores.masked_fill_(
        torch.arange(blocks, device=scores.device)[None, None, :] > positions[None, :, None] // 128, -torch.inf
    )
    if local_blocks:
        local = torch.arange(min(local_blocks, blocks), device=scores.device)
        local_idx = (positions[:, None] // 128 - local[None, :]).clamp(min=0)
        scores.scatter_(-1, local_idx[None].expand(scores.shape[0], -1, -1), torch.inf)
    count = min(topk_blocks, blocks)
    values, indices = scores.topk(count, dim=-1)
    indices.masked_fill_(torch.isneginf(values), -1)
    indices = F.pad(indices, (0, topk_blocks - count), value=-1)
    return indices.permute(1, 0, 2).to(torch.int32).contiguous()


@torch.no_grad()
def minimax_sparse_indexer_npu(index_q, index_k, cu_seqlens, max_seqlen, *, block_size, topk_blocks, local_blocks):
    _check_npu_tensors(index_q, index_k)
    if block_size != _BLOCK_SIZE:
        raise ValueError("MiniMax CANN indexer requires index_block_size=128.")
    if topk_blocks <= 0 or local_blocks < 0:
        raise ValueError("MiniMax CANN requires positive topk_blocks and nonnegative local_blocks.")
    if index_q.ndim != 3 or index_k.shape != (index_q.shape[0], 1, index_q.shape[-1]) or index_q.shape[1] <= 0:
        raise ValueError("MiniMax CANN indexer expects Q [T,H,D] and a single index key head [T,1,D].")
    if index_q.dtype not in _SUPPORTED_DTYPES or index_k.dtype != index_q.dtype:
        raise ValueError("MiniMax CANN index Q/K must have matching BF16 or FP16 dtype.")
    if not 0 < index_q.shape[-1] <= 128 or index_q.shape[-1] % 16:
        raise ValueError("MiniMax CANN index head dimension must be a positive multiple of 16, at most 128.")
    layout = _packed_layout(cu_seqlens, max_seqlen, index_q.shape[0])
    if index_q.shape[0] == 0:
        return torch.full((0, index_q.shape[1], topk_blocks), -1, dtype=torch.int32, device=index_q.device)
    cu = torch.tensor(layout.prefix, dtype=torch.int32, device=index_q.device)
    scores = _cann_op("msa_index_score")(
        query=index_q.contiguous(),
        key=index_k.contiguous(),
        start_loc=torch.zeros(len(layout.lengths), dtype=torch.int32, device=index_q.device),
        block_table=None,
        scale=None,
        atten_mask=_mask_template(index_q.device),
        actual_seq_qlen=cu,
        actual_seq_klen=cu,
        layout_key="TND",
        sparse_mode=3,
        init_blocks=0,
        local_blocks=0,
    )
    if scores.shape[0] != index_q.shape[1] or scores.device != index_q.device:
        raise RuntimeError("msa_index_score must preserve the local index query heads and device.")
    return _select_from_scores(scores, layout, topk_blocks, local_blocks)


def _canonical_indices(indices, layout):
    """Discard invalid/future blocks, deduplicate, and left-pack selection slots."""
    positions = _token_positions(layout, indices.device)
    valid = (indices >= 0) & (indices <= positions[:, None, None] // 128)
    sentinel = layout.max_blocks
    ordered = indices.masked_fill(~valid, sentinel).sort(dim=-1).values
    previous = F.pad(ordered[..., :-1], (1, 0), value=-1)
    ordered = ordered.masked_fill(ordered == previous, sentinel).sort(dim=-1).values
    return ordered.masked_fill(ordered == sentinel, -1).contiguous()


def _backward_indices(indices, layout):
    """Build sorted batch-local KV-to-Q lists without data-dependent allocation."""
    _, heads, _ = indices.shape
    blocks = layout.max_blocks
    idx = torch.full(
        (len(layout.lengths), heads, blocks, layout.max_seqlen), -1, dtype=torch.int32, device=indices.device
    )
    counts = torch.zeros((len(layout.lengths), heads, blocks), dtype=torch.int32, device=indices.device)
    for b, length in enumerate(layout.lengths):
        if length == 0:
            continue
        selected = indices[layout.prefix[b] : layout.prefix[b + 1]].permute(1, 2, 0).long()
        valid = (selected >= 0) & (selected < layout.block_counts[b])
        # An extra block receives invalid slots. Scatter into a boolean map
        # also deduplicates edges, matching the eager attention keep mask.
        keep = torch.zeros((heads, blocks + 1, length), dtype=torch.bool, device=indices.device)
        keep.scatter_(1, selected.masked_fill(~valid, blocks), True)
        keep = keep[:, :blocks]
        counts[b] = keep.sum(-1, dtype=torch.int32)
        queries = torch.arange(length, dtype=torch.int32, device=indices.device).expand(heads, blocks, -1)
        queries = queries.masked_fill(~keep, length).sort(dim=-1).values
        idx[b, :, :, :length] = queries.masked_fill(queries == length, -1)
    return idx, counts


class _SplitKVGBSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_indices, layout, scale):
        q, k, v = (tensor.contiguous() for tensor in (q, k, v))
        indices = _canonical_indices(block_indices, layout)
        device = q.device
        cu_i32 = torch.tensor(layout.prefix, dtype=torch.int32, device=device)
        lengths = torch.tensor(layout.lengths, dtype=torch.int32, device=device)
        cu_blocks = [0]
        for count in layout.block_counts:
            cu_blocks.append(cu_blocks[-1] + count)
        row_ptr, q_idx, slot_idx = _cann_op("k2q_csr")(
            indices.permute(1, 0, 2).contiguous(),
            cu_i32,
            torch.tensor(cu_blocks, dtype=torch.int32, device=device),
            order_method=1,
            total_rows=cu_blocks[-1],
            max_kv=max(layout.block_counts),
            use_simt=False,
            q_global_offset=True,
        )
        out, lse = _cann_op("npu_minimax_sparse_attention_split_kv")(
            query=q,
            key=k,
            value=v,
            k2q_row_ptr=row_ptr,
            k2q_q_indices=q_idx,
            k2q_slot_indices=slot_idx,
            actual_seq_lengths=lengths,
            actual_seq_lengths_kv=lengths,
            num_key_value_heads=k.shape[1],
            scale_value=scale,
            block_size=_BLOCK_SIZE,
            top_k=indices.shape[-1],
            block_table=None,
            inner_precise=0,
            softmax_lse_flag=True,
            input_layout="TND",
        )
        if (
            out.shape != q.shape
            or out.dtype != q.dtype
            or lse.shape != (*q.shape[:2], 1)
            or lse.dtype != torch.float32
        ):
            raise RuntimeError("split-KV must return Q-shaped output and FP32 global LSE [T,Hq,1].")
        ctx.save_for_backward(q, k, v, out, lse, indices)
        ctx.layout = layout
        ctx.scale = scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, out, lse, indices = ctx.saved_tensors
        layout = ctx.layout
        sparse_idx, sparse_count = _backward_indices(indices, layout)
        cu_i64 = torch.tensor(layout.prefix, dtype=torch.int64, device=q.device)
        attrs = dict(
            cu_seqlens_q=cu_i64,
            cu_seqlens_kv=cu_i64,
            seqused_q=None,
            seqused_kv=None,
            layout_q="TND",
            layout_kv="TND",
            layout_sparse_pattern=1,
            mask_mode=1,
            softmax_precision=0,
            win_left=-1,
            win_right=-1,
            residual_block_mode=0,
            is_consistent_topk=False,
        )
        metadata = _cann_op("generic_block_sparse_attention_grad_metadata")(
            sparse_idx,
            sparse_count,
            q.shape[1],
            k.shape[1],
            q.shape[2],
            [1, _BLOCK_SIZE],
            max_seqlen_q=layout.max_seqlen,
            max_seqlen_kv=layout.max_seqlen,
            **attrs,
        )
        dq, dk, dv = _cann_op("generic_block_sparse_attention_grad")(
            q,
            k,
            v,
            dout.contiguous(),
            out,
            lse,
            sparse_idx,
            sparse_count,
            [1, _BLOCK_SIZE],
            metadata=metadata,
            attn_mask=None,
            softmax_scale=ctx.scale,
            **attrs,
        )
        return dq, dk, dv, None, None, None


def minimax_sparse_attention_npu(q, k, v, block_indices, cu_seqlens, max_seqlen, *, block_size, scale, dropout_p=0.0):
    if block_indices is None or dropout_p != 0.0:
        if dropout_p != 0.0:
            logger.warning_once("MiniMax split-KV has no dropout support; using eager attention for nonzero dropout.")
        return minimax_sparse_attention_eager(
            q, k, v, block_indices, cu_seqlens, max_seqlen, block_size=block_size, scale=scale, dropout_p=dropout_p
        )
    _check_npu_tensors(q, k, v, block_indices)
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or k.shape[0] != q.shape[0] or k.shape[-1] != q.shape[-1]:
        raise ValueError("MiniMax CANN attention expects Q [T,Hq,D] and matching K/V [T,Hkv,D].")
    if block_size != _BLOCK_SIZE or q.shape[-1] != 128:
        raise ValueError("MiniMax split-KV/GBSA requires block_size=128 and attention head dimension=128.")
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("MiniMax split-KV/GBSA Q/K/V must have matching BF16 or FP16 dtype.")
    if (
        not 0 < k.shape[1] <= 128
        or not 0 < q.shape[1] <= 128
        or q.shape[1] % k.shape[1]
        or q.shape[1] // k.shape[1] > 16
    ):
        raise ValueError("MiniMax split-KV/GBSA requires at most 128 heads and an integral GQA group size in [1,16].")
    if (
        block_indices.ndim != 3
        or block_indices.shape[:2] != k.shape[:2]
        or block_indices.shape[-1] <= 0
        or block_indices.dtype != torch.int32
    ):
        raise ValueError("MiniMax block_indices must be int32 [T,Hkv,K] with positive K.")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("MiniMax CANN attention requires a positive finite softmax scale.")
    layout = _packed_layout(cu_seqlens, max_seqlen, q.shape[0])
    if q.shape[0] == 0:
        return q + (k.sum() + v.sum()) * 0
    if len(layout.lengths) * q.shape[1] * layout.max_blocks > 1048576:
        raise ValueError("MiniMax GBSA metadata exceeds its B * Hq * J task limit (1048576).")
    if q.shape[0] * block_indices.shape[-1] > torch.iinfo(torch.int32).max:
        raise ValueError("MiniMax CSR edge capacity exceeds int32 offsets.")
    return _SplitKVGBSA.apply(q, k, v, block_indices, layout, float(scale))
