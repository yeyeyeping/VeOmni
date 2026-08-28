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
