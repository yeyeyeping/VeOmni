import torch
import torch_npu
from typing import Dict, Optional, Tuple

_CAUSAL_MASK_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}

def _causal_mask(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.triu(
            torch.ones(
                (2048, 2048),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


def _head_num(q: torch.Tensor, input_layout: str) -> int:
    if input_layout == "TND":
        return q.shape[1]
    if input_layout == "BSND":
        return q.shape[2]
    if input_layout == "BNSD":
        return q.shape[1]
    raise ValueError(f"unsupported input_layout: {input_layout!r}")


def npu_fa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_layout: str,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    actual_seq_qlen,
    actual_seq_kvlen,
    softmax_layout: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    atten_mask = _causal_mask(q.device) if causal else None
    sparse_mode = 2 if causal else 0 
    (
        block_out,
        block_max,
        block_sum,
        _,
        seed,
        offset,
        numels,
    ) = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=_head_num(q, input_layout),
        input_layout=input_layout,
        softmax_layout=softmax_layout,
        atten_mask=atten_mask,
        scale=softmax_scale,
        keep_prob=1.0 - dropout_p,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
    )
    return block_out, block_max, block_sum, (seed, offset, numels)


def npu_fa_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_layout: str,
    out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    rng_state: Tuple[int, int, int],
    actual_seq_qlen,
    actual_seq_kvlen,
    softmax_layout: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atten_mask = _causal_mask(q.device) if causal else None
    sparse_mode = 2 if causal else 0
     
    seed, offset, numels = rng_state
    dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad(
        q,
        k,
        v,
        dout,
        head_num=_head_num(q, input_layout),
        input_layout=input_layout,
        atten_mask=atten_mask,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=out,
        scale_value=softmax_scale,
        keep_prob=1.0 - dropout_p,
        seed=seed,
        offset=offset,
        numels=numels,
        softmax_layout=softmax_layout,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
    )
    return dq, dk, dv


def npu_ring_attention_update_tnd(
    prev_attn_out: Optional[torch.Tensor],
    prev_softmax_max: Optional[torch.Tensor],
    prev_softmax_sum: Optional[torch.Tensor],
    cur_attn_out: torch.Tensor,
    cur_softmax_max: torch.Tensor,
    cur_softmax_sum: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur_max = cur_softmax_max[:, :, 0]
    cur_sum = cur_softmax_sum[:, :, 0]

    if prev_attn_out is None:
        return cur_attn_out.to(torch.float32), cur_max, cur_sum

    softmax_max = torch.maximum(prev_softmax_max, cur_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_max - softmax_max)

    prev_sum_scaled = prev_softmax_sum * prev_scale
    cur_sum_scaled = cur_sum * cur_scale
    softmax_sum = prev_sum_scaled + cur_sum_scaled

    prev_out_scale = (prev_sum_scaled / softmax_sum).unsqueeze(-1)
    cur_out_scale = (cur_sum_scaled / softmax_sum).unsqueeze(-1)
    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    return attn_out, softmax_max, softmax_sum


__all__ = [
    "npu_fa_backward",
    "npu_fa_forward",
    "npu_ring_attention_update_tnd",
]
