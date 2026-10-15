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

"""MiniMax M3 sparse attention (MSA) operators on right-padded BSND tensors.

The operator boundary follows the block-sparse kernel rather than the
Transformers reference code, so a kernel backend can replace the eager
implementation without touching the model. In this private fork, the existing
``eager`` entrypoint uses NPU fused attention; CPU retains the eager reference
and GPU attention raises ``NotImplementedError``. The indexer remains eager on
all devices, and no new configuration is required:

``indexer(index_q, index_k, *, block_size, topk_blocks, local_blocks)``
    ``index_q`` is ``[B, S, G, D]`` and ``index_k`` is ``[B, S, 1, D]`` (MQA),
    both after norm and RoPE. Returns ``[B, S, G, topk_blocks]`` int32 key-block
    indices, left-packed with ``-1`` padding (possibly a non-contiguous view).

``attention(q, k, v, block_indices, *, block_size, scale, dropout_p)``
    ``q`` is ``[B, S, Hq, D]`` and ``k``/``v`` are ``[B, S, G, D]`` without GQA
    repetition; query head ``h`` belongs to KV group ``h // (Hq // G)``.
    ``block_indices`` is the indexer output, or ``None`` for dense causal
    attention. Token-level causality inside a selected block is enforced by
    the operator. Returns ``[B, S, Hq, D]``.

Each row holds one sample, right-padded to the common length ``S``: the
MiniMax language model lays its packed input out this way once, before the
first decoder layer (``S`` is the full row length seen by this rank after the
Ulysses all-to-all). No varlen metadata is needed: every padded
position comes after every real token of its row, so causal masking alone
keeps real queries from attending to padding, and padded query rows produce
finite outputs that the loss ignores.

MiniMax generated modeling declares ``OpSlot("minimax_sparse_attention",
"indexer")`` and ``OpSlot("minimax_sparse_attention", "attention")`` and falls
back to the eager functions below when a slot is bound to ``eager`` (the only
implementation today). A kernel backend registers a ``KernelSpec`` for both
variants plus a ``minimax_sparse_attention_implementation`` config field; the
model code does not change.
"""

from .eager import minimax_sparse_attention_eager, minimax_sparse_indexer_eager


__all__ = ["minimax_sparse_attention_eager", "minimax_sparse_indexer_eager"]
