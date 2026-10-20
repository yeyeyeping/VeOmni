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

"""Launch MiniMax projections and Ulysses communication without early waits."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ....distributed.sequence_parallel.ulysses import all_to_all_tensor
from ..attention.ulysses import _repeat_kv_heads_for_ulysses


class _AsyncHeadExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, wait, group):
        ctx.group = group
        return wait()

    @staticmethod
    def backward(ctx, grad_output):
        return all_to_all_tensor(grad_output, scatter_dim=0, gather_dim=1, group=ctx.group), None, None


def _start_head_exchange(tensor, group, ulysses_size, *, kv_group=False):
    if kv_group:
        tensor = _repeat_kv_heads_for_ulysses(tensor, ulysses_size)
    source = tensor.squeeze(0)
    wait = all_to_all_tensor(source, scatter_dim=1, gather_dim=0, group=group, async_op=True)

    def collect():
        # Anchor the inverse-exchange gradient to the original tensor. Keeping
        # source and wait alive also retains the asynchronous send buffers.
        return _AsyncHeadExchange.apply(source, wait, group).unsqueeze(0)

    return collect


@dataclass
class _PendingIndexKey:
    source: torch.Tensor
    output: torch.Tensor
    work: object

    def collect(self):
        self.work.wait()
        return self.output.unsqueeze(0)


def _start_index_key_gather(tensor, group, ulysses_size):
    # The indexer is frozen. Equal-sized, SP-padded shards allow a single
    # gather without the synchronous shape exchange used by gather_outputs.
    source = tensor.squeeze(0).contiguous()
    output = torch.empty((source.shape[0] * ulysses_size, *source.shape[1:]), dtype=source.dtype, device=source.device)
    work = dist.all_gather_into_tensor(output, source, group=group, async_op=True)
    return _PendingIndexKey(source, output, work)


def _rotary_one(tensor, position_embeddings):
    """Apply upstream partial RoPE to one [B,S,H,D] projection before exchange."""
    cos, sin = position_embeddings
    rotary_dim = min(cos.shape[-1], tensor.shape[-1])
    cos, sin = (part[..., :rotary_dim].unsqueeze(2) for part in (cos, sin))
    rotated, untouched = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    half = rotary_dim // 2
    rotate_half = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
    return torch.cat((rotated * cos + rotate_half * sin, untouched), dim=-1)


def start_minimax_ulysses_projections(attention, hidden_states, position_embeddings, *, group, ulysses_size):
    """Return deferred main Q/K/V exchanges and ready indexer Q/K tensors.

    All ranks must use the same schedule. Index K gather and Index Q exchange
    are launched first; their waits occur after the three main projections.
    Call the returned main collectors only after indexer selection, immediately
    before attention. Original projection/norm modules remain in autograd.
    """
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("MiniMax async Ulysses requires packed [1,T_local,D] hidden states.")
    if attention.config.num_attention_heads % ulysses_size:
        raise ValueError("MiniMax query heads must be divisible by ulysses_size.")
    heads = attention.config.num_key_value_heads
    if (heads >= ulysses_size and heads % ulysses_size) or (heads < ulysses_size and ulysses_size % heads):
        raise ValueError("MiniMax KV heads and ulysses_size must divide one another.")
    index_query_wait = index_key_wait = None
    indexer = attention.indexer
    if indexer is not None:
        with torch.no_grad():
            shape = (*hidden_states.shape[:2], -1, indexer.head_dim)
            index_key = indexer.k_norm(indexer.k_proj(hidden_states).view(shape))
            index_key = _rotary_one(index_key, position_embeddings)
            index_key_wait = _start_index_key_gather(index_key, group, ulysses_size)
            index_query = indexer.q_norm(indexer.q_proj(hidden_states).view(shape))
            index_query = _rotary_one(index_query, position_embeddings)
            index_query_wait = _start_head_exchange(index_query, group, ulysses_size, kv_group=True)

    shape = (*hidden_states.shape[:2], -1, attention.head_dim)
    query = attention.q_norm(attention.q_proj(hidden_states).view(shape))
    query_wait = _start_head_exchange(_rotary_one(query, position_embeddings), group, ulysses_size)
    key = attention.k_norm(attention.k_proj(hidden_states).view(shape))
    key_wait = _start_head_exchange(_rotary_one(key, position_embeddings), group, ulysses_size, kv_group=True)
    value = attention.v_proj(hidden_states).view(shape)
    value_wait = _start_head_exchange(value, group, ulysses_size, kv_group=True)

    index_query = index_key = None
    if indexer is not None:
        with torch.no_grad():
            index_query = index_query_wait()
            index_key = index_key_wait.collect()
    return (query_wait, key_wait, value_wait), index_query, index_key
