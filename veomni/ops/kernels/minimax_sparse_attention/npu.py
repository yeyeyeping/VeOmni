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

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from ....utils import logging
from .eager import minimax_sparse_attention_eager


logger = logging.get_logger(__name__)
_BLOCK_SIZE = 128


def _select_from_scores(scores, cu_seqlens, max_seqlen, topk_blocks, local_blocks):
    """Return unique causal KV block IDs in [T,Hkv,K], with trailing -1 slots."""
    # Match eager's TopK width, including when max_seqlen is an upper bound.
    blocks = (max_seqlen + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    scores = F.pad(scores, (0, blocks - scores.shape[-1]), value=-torch.inf)
    scores.masked_fill_(scores == torch.finfo(torch.float32).min, -torch.inf)
    total = scores.shape[1]
    positions = torch.arange(total, device=scores.device) - torch.repeat_interleave(
        cu_seqlens[:-1], cu_seqlens.diff().long(), output_size=total
    )
    scores.masked_fill_(
        torch.arange(blocks, device=scores.device)[None, None, :] > positions[None, :, None] // _BLOCK_SIZE,
        -torch.inf,
    )
    if local_blocks:
        local = torch.arange(min(local_blocks, blocks), device=scores.device)
        local_idx = (positions[:, None] // _BLOCK_SIZE - local[None, :]).clamp(min=0)
        scores.scatter_(-1, local_idx[None].expand(scores.shape[0], -1, -1), torch.inf)
    count = min(topk_blocks, blocks)
    values, indices = scores.topk(count, dim=-1)
    indices.masked_fill_(torch.isneginf(values), -1)
    indices = F.pad(indices, (0, topk_blocks - count), value=-1)
    return indices.permute(1, 0, 2).to(torch.int32).contiguous()


@torch.no_grad()
def minimax_sparse_indexer_npu(index_q, index_k, cu_seqlens, max_seqlen, *, block_size, topk_blocks, local_blocks):
    if block_size != _BLOCK_SIZE:
        raise ValueError("MiniMax CANN indexer requires index_block_size=128.")
    if topk_blocks <= 0 or local_blocks < 0:
        raise ValueError("MiniMax CANN requires positive topk_blocks and nonnegative local_blocks.")
    if index_q.shape[0] == 0:
        return torch.full((0, index_q.shape[1], topk_blocks), -1, dtype=torch.int32, device=index_q.device)

    from cann_ops_transformer import ops

    cu = cu_seqlens.to(device=index_q.device, dtype=torch.int32)
    scores = ops.msa_index_score(
        query=index_q.contiguous(),
        key=index_k.contiguous(),
        start_loc=torch.zeros(cu.numel() - 1, dtype=torch.int32, device=index_q.device),
        block_table=None,
        scale=None,
        # The current kernel validates the template and computes causal bounds internally.
        atten_mask=torch.zeros((2048, 2048), dtype=torch.int8, device=index_q.device),
        actual_seq_qlen=cu,
        actual_seq_klen=cu,
        layout_key="TND",
        sparse_mode=3,
        init_blocks=0,
        local_blocks=0,
    )
    return _select_from_scores(scores, cu, max_seqlen, topk_blocks, local_blocks)


def _csr_to_gbsa(row_ptr, q_indices, cu_seqlens, max_seqlen):
    """Expand Q-sorted interleaved CSR into batch-local BNKQ lists on device."""
    batch = cu_seqlens.numel() - 1
    heads = row_ptr.shape[0]
    blocks = (max_seqlen + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    block_counts = (cu_seqlens.diff() + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    # Split-KV interleaves batches within each KV block, skipping absent blocks.
    present = torch.arange(blocks, device=row_ptr.device)[:, None] < block_counts[None, :]
    rows = (present.flatten().long().cumsum(0) - 1).clamp(min=0).view(blocks, batch).T
    idx = torch.empty((batch, heads, blocks, max_seqlen), dtype=torch.int32, device=row_ptr.device)
    counts = torch.empty((batch, heads, blocks), dtype=torch.int32, device=row_ptr.device)
    slots = torch.arange(max_seqlen, device=row_ptr.device)
    # Bound temporary gather storage to one batch; output capacity is fixed by GBSA.
    for b in range(batch):
        starts = row_ptr[:, rows[b]]
        counts[b] = (row_ptr[:, rows[b] + 1] - starts).masked_fill(~present[:, b], 0)
        offsets = (starts[:, :, None].long() + slots).clamp(max=q_indices.shape[1] - 1)
        queries = q_indices.gather(1, offsets.flatten(1)).view(heads, blocks, max_seqlen) - cu_seqlens[b]
        valid = slots < counts[b, :, :, None]
        # Native k2q_csr's MC path already writes ascending Q IDs within each row.
        idx[b] = queries.masked_fill(~valid, -1)
    return idx, counts


class _SplitKVGBSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_indices, cu_seqlens, max_seqlen, scale):
        from cann_ops_transformer import ops

        q, k, v = (tensor.contiguous() for tensor in (q, k, v))
        cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
        lengths = cu.diff()
        block_counts = (cu_seqlens.diff() + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        cu_blocks = F.pad(block_counts.cumsum(0, dtype=torch.int32), (1, 0))
        # Use known host/shape metadata; otherwise the native wrapper infers sizes.
        total_rows = max_kv = None
        if cu_seqlens.device.type == "cpu":
            total_rows, max_kv = int(cu_blocks[-1]), int(block_counts.max())
        elif cu_seqlens.numel() == 2:
            total_rows = max_kv = (q.shape[0] + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        # The indexer owns uniqueness, causality and -1 padding of selections.
        row_ptr, q_idx, slot_idx = ops.k2q_csr(
            block_indices.permute(1, 0, 2).contiguous(),
            cu,
            cu_blocks.to(device=q.device),
            order_method=1,
            total_rows=total_rows,
            max_kv=max_kv,
            use_simt=False,
            q_global_offset=True,
        )
        out, lse = ops.npu_minimax_sparse_attention_split_kv(
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
            top_k=block_indices.shape[-1],
            block_table=None,
            inner_precise=0,
            softmax_lse_flag=True,
            input_layout="TND",
        )
        ctx.save_for_backward(q, k, v, out, lse, row_ptr, q_idx, cu)
        # GBSA bounds TND's KV-block dimension by the total packed K tokens.
        ctx.max_seqlen = min(max_seqlen, q.shape[0])
        ctx.scale = scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        from cann_ops_transformer import ops

        q, k, v, out, lse, row_ptr, q_idx, cu = ctx.saved_tensors
        sparse_idx, sparse_count = _csr_to_gbsa(row_ptr, q_idx, cu, ctx.max_seqlen)
        cu_i64 = cu.to(torch.int64)
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
        metadata = ops.generic_block_sparse_attention_grad_metadata(
            sparse_idx,
            sparse_count,
            q.shape[1],
            k.shape[1],
            q.shape[2],
            [1, _BLOCK_SIZE],
            max_seqlen_q=ctx.max_seqlen,
            max_seqlen_kv=ctx.max_seqlen,
            **attrs,
        )
        dq, dk, dv = ops.generic_block_sparse_attention_grad(
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
        return dq, dk, dv, None, None, None, None


def minimax_sparse_attention_npu(q, k, v, block_indices, cu_seqlens, max_seqlen, *, block_size, scale, dropout_p=0.0):
    if block_indices is None or dropout_p != 0.0:
        if dropout_p != 0.0:
            logger.warning_once("MiniMax split-KV has no dropout support; using eager attention for nonzero dropout.")
        return minimax_sparse_attention_eager(
            q, k, v, block_indices, cu_seqlens, max_seqlen, block_size=block_size, scale=scale, dropout_p=dropout_p
        )
    if block_size != _BLOCK_SIZE or q.shape[-1] != 128:
        raise ValueError("MiniMax split-KV/GBSA requires block_size=128 and attention head dimension=128.")
    if q.shape[0] == 0:
        return q + (k.sum() + v.sum()) * 0
    blocks = (min(max_seqlen, q.shape[0]) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    if (cu_seqlens.numel() - 1) * q.shape[1] * blocks > 1048576:
        raise ValueError("MiniMax GBSA metadata exceeds its B * Hq * J task limit (1048576).")
    if q.shape[0] * block_indices.shape[-1] > torch.iinfo(torch.int32).max:
        raise ValueError("MiniMax CSR edge capacity exceeds int32 offsets.")
    return _SplitKVGBSA.apply(q, k, v, block_indices, cu_seqlens, max_seqlen, scale)
