from types import SimpleNamespace

import pytest
import torch

from veomni.ops.kernel_registry import KERNEL_REGISTRY
from veomni.ops.kernels.moe.activation import swiglu_oai, swiglu_oai_backward


def test_swiglu_oai_moe_variant_registers_gpu_and_npu_backends():
    specs = KERNEL_REGISTRY._specs[("moe_experts", "swiglu_oai")]

    assert set(specs) == {"triton", "npu"}


def test_swiglu_oai_manual_backward_matches_autograd():
    torch.manual_seed(0)
    alpha = 1.702
    gate = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    up = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    grad_output = torch.randn_like(gate)

    output = swiglu_oai(gate, up, alpha)
    expected_grad_gate, expected_grad_up = torch.autograd.grad(output, (gate, up), grad_output)
    actual_grad_gate, actual_grad_up = swiglu_oai_backward(grad_output, gate.detach(), up.detach(), alpha)

    torch.testing.assert_close(actual_grad_gate, expected_grad_gate)
    torch.testing.assert_close(actual_grad_up, expected_grad_up)


def test_swiglu_oai_is_not_standard_swiglu():
    gate = torch.tensor([[0.5, -0.5]])
    up = torch.tensor([[0.25, -0.25]])

    output = swiglu_oai(gate, up, 1.702)
    standard = torch.nn.functional.silu(gate) * up

    assert not torch.equal(output, standard)


def test_swiglu_oai_clamp_backward_masks_saturated_gate_and_up_values():
    limit = 0.7
    alpha = 1.702
    raw_gate = torch.tensor([[-1.0, 0.2, 0.7, 1.0]], dtype=torch.float64, requires_grad=True)
    raw_up = torch.tensor([[-1.0, -0.7, 0.7, 1.0]], dtype=torch.float64, requires_grad=True)
    grad_output = torch.tensor([[0.3, -0.4, 0.5, -0.6]], dtype=torch.float64)

    gate = raw_gate.clamp(max=limit)
    up = raw_up.clamp(min=-limit, max=limit)
    output = swiglu_oai(gate, up, alpha)
    expected_grad_gate, expected_grad_up = torch.autograd.grad(output, (raw_gate, raw_up), grad_output)

    actual_grad_gate, actual_grad_up = swiglu_oai_backward(grad_output, gate.detach(), up.detach(), alpha)
    actual_grad_gate.masked_fill_(raw_gate.detach() > limit, 0)
    actual_grad_up.masked_fill_((raw_up.detach() < -limit) | (raw_up.detach() > limit), 0)

    torch.testing.assert_close(actual_grad_gate, expected_grad_gate)
    torch.testing.assert_close(actual_grad_up, expected_grad_up)


def test_swiglu_oai_matches_minimax_eager_gate_bitwise_in_bf16():
    """Every bf16 op rounds, so the fused activation must use the eager op order exactly."""
    modeling = pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    limit, alpha = 7.0, 1.702
    experts = modeling.MiniMaxM3VLExperts.__new__(modeling.MiniMaxM3VLExperts)
    experts.__dict__.update(vars(SimpleNamespace(swiglu_limit=limit, swiglu_alpha=alpha)))
    torch.manual_seed(0)
    gate_up = (4 * torch.randn(64, 2 * 256)).to(torch.bfloat16)

    gate, up = gate_up.chunk(2, dim=-1)
    output = swiglu_oai(gate.clamp(max=limit), up.clamp(min=-limit, max=limit), alpha)

    torch.testing.assert_close(output, experts._apply_gate(gate_up), rtol=0, atol=0)
