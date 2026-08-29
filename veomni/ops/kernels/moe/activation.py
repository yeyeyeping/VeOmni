import torch


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float) -> torch.Tensor:
    """Apply SwiGLU-OAI after any model-specific input clamping."""
    return (up + 1.0) * gate * torch.sigmoid(alpha * gate)


def swiglu_oai_backward(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return analytical gradients for the unclamped SwiGLU-OAI inputs."""
    sigmoid = torch.sigmoid(alpha * gate)
    glu = gate * sigmoid
    grad_gate = grad_output * (up + 1.0) * (sigmoid + alpha * gate * sigmoid * (1.0 - sigmoid))
    grad_up = grad_output * glu
    return grad_gate, grad_up
