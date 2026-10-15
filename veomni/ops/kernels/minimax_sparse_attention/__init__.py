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
implementation without touching the model. In this private fork, the existing
``eager`` entrypoint uses NPU fused attention with a full mask; CPU retains the
eager reference and GPU attention raises ``NotImplementedError``. The indexer
remains eager by default. Set ``minimax_sparse_attention_implementation`` to
``npu_cann`` for AscendC index scores, Split-KV forward and GBSAG backward:

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
back to the eager functions below when a slot is bound to ``eager``. The
``npu_cann`` backend keeps Q/K/V packed and requires BF16, head_dim=128,
block_size=128, zero attention dropout and nondeterministic execution.
Dense layers retain the eager path. See docs/examples/minimax_m3_vl.md.
"""

from ...config.registry import BackendSpec, OpScope, OpSpec, register_op
from ...kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec
from .eager import minimax_sparse_attention_eager, minimax_sparse_indexer_eager


def _indexer_ascendc_factory():
    from .ascendc_bindings import get_index_score
    from .indexer_ascendc import minimax_sparse_indexer_ascendc

    get_index_score()
    return minimax_sparse_indexer_ascendc


def _attention_ascendc_factory():
    from .ascendc_bindings import get_attention_ops
    from .attention_ascendc import minimax_sparse_attention_ascendc

    get_attention_ops()
    return minimax_sparse_attention_ascendc


register_op(
    OpSpec(
        name="minimax_sparse_attention",
        config_field="minimax_sparse_attention_implementation",
        label="MiniMax sparse attention",
        scope=OpScope.PER_MODEL,
        default="eager",
        backends={
            "npu_cann": BackendSpec(
                entry="veomni.ops.kernels.minimax_sparse_attention.attention_ascendc:minimax_sparse_attention_ascendc",
                requires=("torch_npu", "cann_ops_transformer", "custom_ops"),
            ),
        },
    )
)

for variant, factory in (("indexer", _indexer_ascendc_factory), ("attention", _attention_ascendc_factory)):
    KERNEL_REGISTRY.register(
        KernelSpec(
            name="npu_cann",
            op_name="minimax_sparse_attention",
            variant=variant,
            factory=factory,
            hardware=HardwareRequirement(device_type="npu"),
            description=f"MiniMax TND AscendC {variant}: index scores / Split-KV forward + GBSAG backward",
        )
    )


__all__ = ["minimax_sparse_attention_eager", "minimax_sparse_indexer_eager"]
