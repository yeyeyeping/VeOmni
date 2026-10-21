# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Cross-variant helpers for ``moe_experts_lora``.

Not a registered kernel family. ``shared/`` here is the LoRA variant name,
not a helper package. NPU permute / EP / SwiGLU (or SwiGLU-OAI) / combine lives here so
each variant ``npu.py`` can own its own LoRA deltas.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from ..moe_experts.shared.swiglu import swiglu_oai


GateUpDelta = Callable[[Tensor, Tensor], Tensor]
DownDelta = Callable[[Tensor, Tensor], Tensor]


def eager_gated_activation(gate: Tensor, up: Tensor, swiglu_alpha: float | None, swiglu_limit: float | None) -> Tensor:
    """Eager ``silu(gate) * up``, or SwiGLU-OAI (MiniMax M3) when ``swiglu_alpha`` is set.

    ``swiglu_limit`` clamps ``gate`` to ``max=limit`` and ``up`` to ``[-limit, limit]``
    first, in the operator order of ``moe_experts/swiglu_oai/eager``.
    """
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    if swiglu_alpha is None:
        return F.silu(gate) * up
    return swiglu_oai(gate, up, swiglu_alpha)


def npu_non_ep_forward(
    num_experts: int,
    routing_weights: Tensor,
    selected_experts: Tensor,
    hidden_states: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    gate_up_delta: GateUpDelta,
    down_delta: DownDelta,
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
) -> Tensor:
    """NPU single-device fused MoE plus caller-supplied LoRA deltas."""
    import torch_npu

    from ..moe_experts.standard.npu import _apply_moe_activation, npu_group_gemm

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    selected_experts = selected_experts.to(torch.int32)
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, selected_experts)
    tokens_per_expert = torch.histc(selected_experts.to(torch.float32), bins=num_experts, min=0, max=num_experts).to(
        torch.int64
    )

    gate_up = npu_group_gemm(permuted_hidden_states, fc1_1_2_weight.transpose(1, 2), tokens_per_expert)
    gate_up = gate_up + gate_up_delta(permuted_hidden_states, tokens_per_expert)
    intermediate = _apply_moe_activation(gate_up, swiglu_limit, swiglu_alpha)
    output = npu_group_gemm(intermediate, fc2_weight.transpose(1, 2), tokens_per_expert)
    output = output + down_delta(intermediate, tokens_per_expert)
    return torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)


def npu_ep_forward(
    num_experts: int,
    routing_weights: Tensor,
    selected_experts: Tensor,
    hidden_states: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    gate_up_delta: GateUpDelta,
    down_delta: DownDelta,
    ep_group: dist.ProcessGroup | None = None,
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
) -> Tensor:
    """NPU expert-parallel fused MoE plus caller-supplied LoRA deltas."""
    from ..moe_experts.standard.npu import (
        _apply_moe_activation,
        alltoall_combine,
        alltoall_dispatch,
        dispatch_preprocess,
        npu_group_gemm,
    )

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    selected_experts = selected_experts.to(torch.int32)
    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
        dispatch_preprocess(selected_experts, num_experts, ep_group)
    )
    hidden_states, unpermute_indices = alltoall_dispatch(
        hidden_states,
        selected_experts,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )

    group_list = num_global_sum_tokens_per_local_expert.to(torch.int64)
    gate_up = npu_group_gemm(hidden_states, fc1_1_2_weight.transpose(1, 2), group_list)
    gate_up = gate_up + gate_up_delta(hidden_states, group_list)
    intermediate = _apply_moe_activation(gate_up, swiglu_limit, swiglu_alpha)
    output = npu_group_gemm(intermediate, fc2_weight.transpose(1, 2), group_list)
    output = output + down_delta(intermediate, group_list)
    return alltoall_combine(
        output,
        routing_weights,
        unpermute_indices,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )
