# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
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

"""MiniMax M3 eager indexer with NPU fused attention and a CPU reference, on BSND tensors.

The math is the Transformers ``MiniMaxM3VLIndexer`` / ``build_block_mask`` /
``eager_attention_forward`` reference. Inputs arrive as right-padded
``[B, S, N, D]`` rows, one sample per row, so no layout conversion happens
here: right padding sits after every real token, and causality alone keeps
real queries away from padded keys. Padded query rows always see at least
their own position (causal diagonal, local block), so they stay finite; their
outputs are discarded by the loss.

NPU attention runs ``npu_fusion_attention`` in ``BSND`` layout: dense causal
layers use its built-in causal mode, block-sparse layers pass one boolean mask
per KV group. GPU attention is intentionally unsupported in this private
training fork.
"""

import torch
import torch.nn.functional as F


# torch_npu's compressed causal mode (sparse_mode=3) takes this fixed 2048x2048
# upper-triangular mask regardless of sequence length; cache one per device.
_NPU_CAUSAL_MASK_SIZE = 2048
_NPU_CAUSAL_SPARSE_MODE = 3
_NPU_FULL_MASK_SPARSE_MODE = 1
_npu_causal_masks: dict[torch.device, torch.Tensor] = {}


def _npu_causal_mask(device: torch.device) -> torch.Tensor:
    mask = _npu_causal_masks.get(device)
    if mask is None:
        mask = torch.ones(_NPU_CAUSAL_MASK_SIZE, _NPU_CAUSAL_MASK_SIZE, dtype=torch.bool, device=device).triu(1)
        _npu_causal_masks[device] = mask
    return mask


def _select_blocks(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    block_size: int,
    topk_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Upstream indexer scoring on ``[B, G, S, D]`` / ``[B, 1, S, D]``; returns ``[B, G, S, topk]``."""
    batch, num_groups, seq_len, _ = index_q.shape
    positions = torch.arange(seq_len, device=index_q.device)
    num_key_blocks = -(-seq_len // block_size)
    pad = num_key_blocks * block_size - seq_len

    scores = torch.matmul(index_q.float(), index_k.float().transpose(-1, -2))
    token_future = positions[None, :] > positions[:, None]
    scores = scores.masked_fill(token_future, float("-inf"))
    if pad:
        scores = F.pad(scores, (0, pad), value=float("-inf"))
    block_scores = scores.view(batch, num_groups, seq_len, num_key_blocks, block_size).amax(dim=-1)

    if local_blocks > 0:
        query_block = positions // block_size
        local = torch.arange(local_blocks, device=index_q.device)
        local_idx = (query_block[:, None] - local[None, :]).clamp(min=0)
        block_scores.scatter_(-1, local_idx.expand(batch, num_groups, -1, -1), float("inf"))

    topk = min(topk_blocks, num_key_blocks)
    topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
    return topk_indices.masked_fill(topk_scores == float("-inf"), -1)


def _block_causal_drop(block_indices: torch.Tensor, *, block_size: int) -> torch.Tensor:
    """Expand ``[B, G, S, K]`` block indices into a ``[B, G, S, S]`` mask, True where attention is dropped."""
    batch, num_groups, seq_len, _ = block_indices.shape
    num_key_blocks = -(-seq_len // block_size)

    # `-1` slots land in a throwaway column that is dropped afterwards.
    safe = block_indices.long().masked_fill(block_indices < 0, num_key_blocks)
    selected = torch.zeros(
        (batch, num_groups, seq_len, num_key_blocks + 1), dtype=torch.bool, device=block_indices.device
    )
    selected.scatter_(-1, safe, True)
    block_keep = selected[..., :num_key_blocks].repeat_interleave(block_size, dim=-1)[..., :seq_len]

    positions = torch.arange(seq_len, device=block_indices.device)
    token_future = positions[None, :] > positions[:, None]
    return ~block_keep | token_future


def _npu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    drop: torch.Tensor | None,
    *,
    scale: float,
    dropout_p: float,
) -> torch.Tensor:
    """``npu_fusion_attention`` on BSND; ``drop`` is ``None`` (causal) or a ``[B, G, S, S]`` block mask."""
    import torch_npu

    def fused(q, k, v, atten_mask, sparse_mode):
        return torch_npu.npu_fusion_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            head_num=q.shape[2],
            input_layout="BSND",
            atten_mask=atten_mask,
            sparse_mode=sparse_mode,
            scale=scale,
            keep_prob=1.0 - dropout_p,
        )[0]

    if drop is None:
        return fused(query, key, value, _npu_causal_mask(query.device), _NPU_CAUSAL_SPARSE_MODE)

    # The fused kernel accepts a [B, 1, S, S] mask shared by all heads, not one
    # per KV group. Running each group separately avoids repeating the S x S
    # mask to every query head; under Ulysses a rank usually owns one group.
    num_groups = key.shape[2]
    group_size = query.shape[2] // num_groups
    outputs = [
        fused(
            query[:, :, group * group_size : (group + 1) * group_size],
            key[:, :, group : group + 1],
            value[:, :, group : group + 1],
            drop[:, group : group + 1].contiguous(),
            _NPU_FULL_MASK_SPARSE_MODE,
        )
        for group in range(num_groups)
    ]
    return outputs[0] if num_groups == 1 else torch.cat(outputs, dim=2)


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    drop: torch.Tensor | None,
    *,
    scale: float,
    dropout_p: float,
) -> torch.Tensor:
    """Upstream eager attention with local GQA on BSND; ``drop`` as in ``_npu_attention``."""
    seq_len = query.shape[1]
    group_size = query.shape[2] // key.shape[2]
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    key = key.repeat_interleave(group_size, dim=1)
    value = value.repeat_interleave(group_size, dim=1)
    if drop is None:
        positions = torch.arange(seq_len, device=query.device)
        drop = (positions[None, :] > positions[:, None])[None, None]
    else:
        drop = drop.repeat_interleave(group_size, dim=1)
    attention_mask = torch.zeros(drop.shape, dtype=query.dtype, device=query.device).masked_fill(
        drop, torch.finfo(query.dtype).min
    )

    attention_weights = torch.matmul(query, key.transpose(2, 3)) * scale
    attention_weights = attention_weights + attention_mask
    attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout_p > 0:
        attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)
    return torch.matmul(attention_weights, value).transpose(1, 2)


def minimax_sparse_indexer_eager(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    block_size: int,
    topk_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Select ``[B, S, G, topk_blocks]`` int32 key blocks per query and KV group; see the package docstring."""
    if index_k.shape[2] != 1:
        raise ValueError(f"MiniMax sparse indexer expects a single index key head, got {index_k.shape[2]}.")

    block_indices = _select_blocks(
        index_q.transpose(1, 2),
        index_k.transpose(1, 2),
        block_size=block_size,
        topk_blocks=topk_blocks,
        local_blocks=local_blocks,
    )
    # Short sequences have fewer key blocks than slots; the extra slots are empty.
    if block_indices.shape[-1] < topk_blocks:
        block_indices = F.pad(block_indices, (0, topk_blocks - block_indices.shape[-1]), value=-1)
    # The attention operator consumes [B, G, S, K]; hand over a view, not a copy.
    return block_indices.to(torch.int32).transpose(1, 2)


def minimax_sparse_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor | None,
    *,
    block_size: int | None,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Causal (block-sparse when ``block_indices`` is given) attention on BSND tensors; see the package docstring."""
    if q.device.type not in ("npu", "cpu"):
        raise NotImplementedError("MiniMax attention training supports NPU only; CPU is kept as a test reference.")

    num_query_heads, num_groups = q.shape[2], k.shape[2]
    if num_query_heads % num_groups != 0:
        raise ValueError(f"Query heads ({num_query_heads}) must be divisible by KV heads ({num_groups}).")
    if block_indices is not None and block_indices.shape[2] != num_groups:
        raise ValueError(
            f"MiniMax block_indices must have one selection per KV head ({num_groups}), got {block_indices.shape[2]}."
        )

    drop = None
    if block_indices is not None:
        drop = _block_causal_drop(block_indices.transpose(1, 2), block_size=block_size)

    attention = _npu_attention if q.device.type == "npu" else _reference_attention
    return attention(q, k, v, drop, scale=scale, dropout_p=dropout_p)
