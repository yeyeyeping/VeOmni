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

import importlib
from types import SimpleNamespace

import pytest
import torch

from veomni.ops.dispatch import OpSlot


_MODELING_MODULES = (
    "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_gpu",
    "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_npu",
)


class _RecordingSlot:
    use_non_eager_impl = True

    def __init__(self, output):
        self.output = output
        self.args = None

    def __call__(self, *args):
        self.args = args
        return self.output


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_declares_gemma_style_rms_norm_slot(module_name):
    modeling = importlib.import_module(module_name)

    assert isinstance(modeling.veomni_rms_norm, OpSlot)
    assert modeling.veomni_rms_norm.op_name == "rms_norm"
    assert modeling.veomni_rms_norm.variant == "qwen3_5"


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_declares_swiglu_oai_moe_slot(module_name):
    modeling = importlib.import_module(module_name)

    assert isinstance(modeling.veomni_moe_experts_forward, OpSlot)
    assert modeling.veomni_moe_experts_forward.op_name == "moe_experts"
    assert modeling.veomni_moe_experts_forward.variant == "swiglu_oai"


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_model_classes_bind_their_matching_parallel_plan(module_name):
    modeling = importlib.import_module(module_name)

    text_plan = modeling.MiniMaxM3VLForCausalLM.get_parallel_plan(None)
    vlm_plan = modeling.MiniMaxM3SparseForConditionalGeneration.get_parallel_plan(None)

    assert set(text_plan.extra_parallel_plan["ep"]) == {
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    }
    assert set(vlm_plan.extra_parallel_plan["ep"]) == {
        "model.language_model.layers.*.mlp.experts.gate_up_proj",
        "model.language_model.layers.*.mlp.experts.down_proj",
    }


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_experts_preserve_swiglu_oai_eager_semantics(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=4,
        intermediate_size=3,
        swiglu_limit=0.7,
        swiglu_alpha=1.702,
    )
    experts = modeling.MiniMaxM3VLExperts(config)
    torch.manual_seed(0)
    experts.gate_up_proj.data.normal_(mean=0.0, std=0.1)
    experts.down_proj.data.normal_(mean=0.0, std=0.1)
    hidden_states = torch.randn(5, 4)
    top_k_index = torch.tensor([[0], [1], [0], [1], [0]])
    top_k_weights = torch.ones(5, 1)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))

    output = experts(hidden_states, top_k_index, top_k_weights)
    expected = torch.zeros_like(hidden_states)
    for expert_idx in range(config.num_local_experts):
        token_idx = torch.where(top_k_index[:, 0] == expert_idx)[0]
        gate_up = torch.nn.functional.linear(hidden_states[token_idx], experts.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=config.swiglu_limit)
        up = up.clamp(min=-config.swiglu_limit, max=config.swiglu_limit)
        activated = (up + 1.0) * gate * torch.sigmoid(config.swiglu_alpha * gate)
        expected[token_idx] = torch.nn.functional.linear(activated, experts.down_proj[expert_idx])

    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_experts_reject_eager_ep(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = SimpleNamespace(
        num_local_experts=4,
        hidden_size=4,
        intermediate_size=3,
        swiglu_limit=0.7,
        swiglu_alpha=1.702,
    )
    experts = modeling.MiniMaxM3VLExperts(config)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True, ep_size=2))

    with pytest.raises(RuntimeError, match="requires.*fused_triton or fused_npu"):
        experts(torch.randn(2, 4), torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1))


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_experts_require_expert_count_divisible_by_ep(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = SimpleNamespace(
        num_local_experts=3,
        hidden_size=4,
        intermediate_size=3,
        swiglu_limit=0.7,
        swiglu_alpha=1.702,
    )
    experts = modeling.MiniMaxM3VLExperts(config)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True, ep_size=2))

    with pytest.raises(ValueError, match="num_experts=3.*ep_size=2"):
        experts(torch.randn(2, 4), torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1))


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_experts_dispatch_fused_ep(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = SimpleNamespace(
        num_local_experts=4,
        hidden_size=4,
        intermediate_size=3,
        swiglu_limit=0.7,
        swiglu_alpha=1.702,
    )
    experts = modeling.MiniMaxM3VLExperts(config)
    hidden_states = torch.randn(2, 4)
    top_k_index = torch.zeros(2, 1, dtype=torch.long)
    top_k_weights = torch.ones(2, 1)
    output = torch.randn_like(hidden_states)
    slot = _RecordingSlot(output)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True, ep_size=2))
    monkeypatch.setattr(modeling, "veomni_moe_experts_forward", slot)

    assert experts(hidden_states, top_k_index, top_k_weights) is output
    assert slot.args[0] is experts
    assert slot.args[1] is hidden_states
    assert slot.args[2] is top_k_index
    assert slot.args[3] is top_k_weights


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_rms_norm_preserves_eager_fp32_semantics(module_name):
    modeling = importlib.import_module(module_name)
    norm = modeling.MiniMaxM3VLRMSNorm(8, eps=1e-6)
    norm.weight.data.copy_(torch.linspace(-0.1, 0.1, 8))
    hidden_states = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)

    output = norm(hidden_states)
    expected = hidden_states.float()
    expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + norm.eps)
    expected = expected * (1.0 + norm.weight.float())

    assert output.dtype == hidden_states.dtype
    assert torch.equal(output, expected.to(hidden_states.dtype))


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_rms_norm_dispatches_weight_and_eps(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    output = torch.randn(2, 4, 8)
    slot = _RecordingSlot(output)
    monkeypatch.setattr(modeling, "veomni_rms_norm", slot)

    norm = modeling.MiniMaxM3VLRMSNorm(8, eps=1e-5)
    hidden_states = torch.randn_like(output)

    assert norm(hidden_states) is output
    assert slot.args[0] is hidden_states
    assert slot.args[1] is norm.weight
    assert slot.args[2] == norm.eps
