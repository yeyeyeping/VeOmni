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

"""MiniMax M3 sparse attention (MSA) operators on packed TND tensors.

The operator boundary follows the block-sparse kernel rather than the
Transformers reference code, so a kernel backend can replace the eager
implementation without touching the model:

``indexer(index_q, index_k, cu_seqlens, max_seqlen, *, block_size, topk_blocks, local_blocks)``
    ``index_q`` is ``[T, G, D]`` and ``index_k`` is ``[T, 1, D]`` (MQA), both
    after norm and RoPE. Returns ``[T, G, topk_blocks]`` int32 key-block
    indices, numbered within each sequence, left-packed with ``-1`` padding.

``attention(q, k, v, block_indices, cu_seqlens, max_seqlen, *, block_size, scale, dropout_p)``
    ``q`` is ``[T, Hq, D]`` and ``k``/``v`` are ``[T, G, D]`` without GQA
    repetition; query head ``h`` belongs to KV group ``h // (Hq // G)``.
    ``block_indices`` is the indexer output, or ``None`` for dense causal
    attention. Token-level causality inside a selected block is enforced by
    the operator. Returns ``[T, Hq, D]``.

``T`` is the full packed token count seen by this rank after the Ulysses
all-to-all, and ``cu_seqlens`` describes every packed sequence, including the
synthetic SP tail-padding sequence appended by the collator.

MiniMax generated modeling declares ``OpSlot("minimax_sparse_attention",
"indexer")`` and ``OpSlot("minimax_sparse_attention", "attention")`` and falls
back to the eager functions below when a slot is bound to ``eager``. On Ascend
NPU, ``npu_split_kv_gbsa`` replaces both slots with CANN operators exported by
``cann_ops_transformer.ops``. Its attention uses split-KV forward and GBSA
backward; dense layers and nonzero attention dropout retain the eager path.
"""

from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec
from .eager import minimax_sparse_attention_eager, minimax_sparse_indexer_eager


def _npu_indexer_factory():
    from .npu import minimax_sparse_indexer_npu

    return minimax_sparse_indexer_npu


def _npu_attention_factory():
    from .npu import minimax_sparse_attention_npu

    return minimax_sparse_attention_npu


for variant, factory in (("indexer", _npu_indexer_factory), ("attention", _npu_attention_factory)):
    KERNEL_REGISTRY.register(
        KernelSpec(
            name="npu_split_kv_gbsa",
            op_name="minimax_sparse_attention",
            variant=variant,
            factory=factory,
            hardware=HardwareRequirement(device_type="npu"),
            description=f"CANN MiniMax M3 {variant}: index score / split-KV forward with GBSA backward",
        )
    )


__all__ = ["minimax_sparse_attention_eager", "minimax_sparse_indexer_eager"]
