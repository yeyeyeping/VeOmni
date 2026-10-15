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

"""TND Split-KV forward paired with GBSAG backward for MiniMax training."""

import math

import torch
from torch.autograd.function import once_differentiable

from .ascendc_bindings import get_attention_ops, require_npu
from .eager import minimax_sparse_attention_eager
from .indices import build_gbsag_indices, build_k2q_csr, prepare_metadata, validate_selection


_GBSAG_ATTRIBUTES = {
    "layout_q": "TND",
    "layout_kv": "TND",
    "layout_sparse_pattern": 1,
    "mask_mode": 1,
    "softmax_precision": 0,
    "win_left": -1,
    "win_right": -1,
    "residual_block_mode": 0,
    "is_consistent_topk": False,
}


def _check_sparse_inputs(q, k, v, block_size, scale, dropout_p):
    require_npu()
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[0] != k.shape[0]:
        raise ValueError("MiniMax npu_cann requires TND Q/K/V with identical packed lengths and matching K/V shapes.")
    if (
        q.shape[-1] != 128
        or k.shape[-1] != 128
        or k.shape[1] <= 0
        or q.shape[1] <= 0
        or q.shape[1] % k.shape[1]
        or not 1 <= q.shape[1] // k.shape[1] <= 16
        or block_size != 128
    ):
        raise ValueError("MiniMax npu_cann requires head_dim=128, block_size=128 and an integral GQA ratio in [1,16].")
    if any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v)) or q.device.type != "npu":
        raise ValueError("MiniMax npu_cann requires BF16 Q/K/V on the same Ascend NPU.")
    if dropout_p != 0:
        raise ValueError("MiniMax npu_cann does not support attention dropout; set attention_dropout=0.")
    if not math.isfinite(scale):
        raise ValueError("MiniMax npu_cann requires a finite softmax scale.")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("MiniMax npu_cann GBSAG does not support deterministic algorithms.")


class _MsaAscendCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_indices, metadata, scale):
        ops = get_attention_ops()
        row_ptr, q_idx, slot_idx = build_k2q_csr(block_indices, metadata)
        out, lse = ops.forward(
            q,
            k,
            v,
            row_ptr,
            q_idx,
            slot_idx,
            metadata.lengths,
            metadata.lengths,
            num_key_value_heads=k.shape[1],
            scale_value=scale,
            block_size=metadata.block_size,
            top_k=block_indices.shape[-1],
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
            raise RuntimeError("MiniMax Split-KV returned an unexpected output or FP32 TND LSE.")
        ctx.save_for_backward(q, k, v, out, lse, block_indices)
        ctx.sequence_metadata = metadata
        ctx.scale = scale
        ctx.ops = ops
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, out, lse, block_indices = ctx.saved_tensors
        seq = ctx.sequence_metadata
        sparse_idx, sparse_count = build_gbsag_indices(block_indices, seq)
        block_shape = [1, seq.block_size]
        sequence_args = {"cu_seqlens_q": seq.cu_seqlens, "cu_seqlens_kv": seq.cu_seqlens}
        metadata = ctx.ops.metadata(
            sparse_idx,
            sparse_count,
            q.shape[1],
            k.shape[1],
            q.shape[-1],
            block_shape,
            max_seqlen_q=seq.max_seqlen,
            max_seqlen_kv=seq.max_seqlen,
            **sequence_args,
            **_GBSAG_ATTRIBUTES,
        )
        dq, dk, dv = ctx.ops.backward(
            q,
            k,
            v,
            dout.contiguous(),
            out,
            lse,
            sparse_idx,
            sparse_count,
            block_shape,
            metadata=metadata,
            softmax_scale=ctx.scale,
            **sequence_args,
            **_GBSAG_ATTRIBUTES,
        )
        return dq, dk, dv, None, None, None


def minimax_sparse_attention_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int | None,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Run sparse BF16 TND training, retaining the existing dense-layer path."""
    if block_indices is None:
        return minimax_sparse_attention_eager(
            q, k, v, None, cu_seqlens, max_seqlen, block_size=block_size, scale=scale, dropout_p=dropout_p
        )
    _check_sparse_inputs(q, k, v, block_size, scale, dropout_p)
    metadata = prepare_metadata(cu_seqlens, max_seqlen, q.shape[0], q.device, block_size)
    validate_selection(block_indices, metadata, k.shape[1])
    if q.shape[0] == 0:
        return q + (k.sum() + v.sum()) * 0
    return _MsaAscendCFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), block_indices.contiguous(), metadata, float(scale)
    )
