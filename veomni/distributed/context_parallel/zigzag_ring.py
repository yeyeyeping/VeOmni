from typing import List, Optional, Tuple

import torch

from .attn_backend import (
    npu_fa_backward,
    npu_fa_forward,
    npu_ring_attention_update_tnd,
)
from .ringcomm import RingComm

def get_half_index(
    cu_seqlens: torch.Tensor, *, front: bool):
    if len(cu_seqlens) == 2:
        if front:
            return slice(None, cu_seqlens[-1] // 2)
        else:
            return slice(cu_seqlens[-1] // 2, None)
    index = torch.zeros((cu_seqlens[-1], ), dtype=bool, device=cu_seqlens.device)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end ) // 2
        index[start:end] = True
    return index


def zigzag_ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    half_index0,
    half_index1 ,
    softmax_scale: float,
    dropout_p: float = 0.0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List,
]:

    comm = RingComm(process_group)
    
    block_seq_len = q.shape[0] // 2
    q1 = q[half_index1].contiguous()

    global_out = None
    global_softmax_max = None
    global_softmax_sum = None
    next_k = next_v = None
    half_cu_seq_lens = cu_seqlens // 2
    
    actual_seqlen = tuple(cu_seqlens[1:].cpu().numpy().tolist())
    actual_half_seqlen = tuple(half_cu_seq_lens[1:].cpu().numpy().tolist())
    
    rng_states = [(0, 0, 0) for _ in range(comm.world_size)]

    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ):
        seqlen_q = q.shape[0]
        seqlen_kv = k.shape[0]
        cu_seqlens_q = actual_half_seqlen if seqlen_q == block_seq_len else actual_seqlen
        cu_seqlens_kv = actual_half_seqlen if seqlen_kv == block_seq_len else actual_seqlen
        return npu_fa_forward(
            q,
            k,
            v,
            input_layout="TND",
            softmax_layout="TND",
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            actual_seq_qlen=cu_seqlens_q,
            actual_seq_kvlen=cu_seqlens_kv,
        )

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        if step == 0:
            block_out, block_max, block_sum, rng_state = forward(
                q,
                k,
                v,
                causal=True,
            )
            global_out, global_softmax_max, global_softmax_sum = (
                npu_ring_attention_update_tnd(
                    global_out,
                    global_softmax_max,
                    global_softmax_sum,
                    block_out,
                    block_max,
                    block_sum,
                )
            )
        elif step <= comm.rank:
            k0 = k[half_index0].contiguous()
            v0 = v[half_index0].contiguous()
            block_out, block_max, block_sum, rng_state = forward(
                q,
                k0,
                v0,
                causal=False,
            )
            global_out, global_softmax_max, global_softmax_sum = (
                npu_ring_attention_update_tnd(
                    global_out,
                    global_softmax_max,
                    global_softmax_sum,
                    block_out,
                    block_max,
                    block_sum,
                )
            )
        else:
            block_out, block_max, block_sum, rng_state = forward(
                q1,
                k,
                v,
                causal=False,
            )
            updated_out, updated_max, updated_sum = npu_ring_attention_update_tnd(
                global_out[half_index1],
                global_softmax_max[half_index1],
                global_softmax_sum[half_index1],
                block_out,
                block_max,
                block_sum,
            )
            global_out[half_index1] = updated_out
            global_softmax_max[half_index1] = updated_max
            global_softmax_sum[half_index1] = updated_sum

        rng_states[step] = rng_state

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    return (
        global_out.to(q.dtype),
        global_softmax_max,
        global_softmax_sum,
        rng_states,
    )


def zigzag_ring_flash_attn_varlen_backward(
    process_group,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rng_states: List,
    half_index0,
    half_index1,
    softmax_scale: float,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    
    
    dq = dk = dv = None
    next_k = next_v = None
    next_dk = next_dv = None
    dk_comm_buffer = dv_comm_buffer = None
    
    dout = dout.contiguous()
    dout1 = dout[half_index1].contiguous()
    q1 = q[half_index1].contiguous()
    out1 = out[half_index1].contiguous()

    # The forward ring stores one lane as [T, N].  NPU backward requires the
    # original [T, N, 8] form.
    softmax_max1 = softmax_max[half_index1].unsqueeze(-1).expand(-1, -1, 8).contiguous()
    softmax_sum1 = softmax_sum[half_index1].unsqueeze(-1).expand(-1, -1, 8).contiguous()
    
    
    softmax_max = softmax_max.unsqueeze(-1).expand(-1, -1, 8).contiguous()
    softmax_sum = softmax_sum.unsqueeze(-1).expand(-1, -1, 8).contiguous()

    block_seq_len = q.shape[0] // 2
    half_cu_seqlens = cu_seqlens // 2
    
    actual_seqlen = tuple(cu_seqlens[1:].cpu().numpy().tolist())
    actual_half_seqlen = tuple(half_cu_seqlens[1:].cpu().numpy().tolist())
    
    def backward(
        dout, q, k, v, out, softmax_max, softmax_sum, causal, rng_state):
        seqlen_q = q.shape[0]
        seqlen_kv = k.shape[0]
        cu_seqlens_q = actual_half_seqlen if seqlen_q == block_seq_len else actual_seqlen
        cu_seqlens_kv = actual_half_seqlen if seqlen_kv == block_seq_len else actual_seqlen
        return npu_fa_backward(
            dout,
            q,
            k,
            v,
            input_layout="TND",
            softmax_layout="TND",
            out=out,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            rng_state=rng_state,
            actual_seq_qlen=cu_seqlens_q,
            actual_seq_kvlen=cu_seqlens_kv,
        )

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            dq_block, dk_block, dv_block = backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_max,
                softmax_sum,
                causal=True,
                rng_state=rng_states[step],
            )
            dq = dq_block.float().clone()
            dk = dk_block.float().clone()
            dv = dv_block.float().clone()
        else:
            if step <= kv_comm.rank:
                k0 = k[half_index0].contiguous()
                v0 = v[half_index0].contiguous()
                dq_block, dk_block, dv_block = backward(
                    dout,
                    q,
                    k0,
                    v0,
                    out,
                    softmax_max,
                    softmax_sum,
                    causal=False,
                    rng_state=rng_states[step],
                )
                dq += dq_block.float()
            else:
                dq_block, dk_block, dv_block = backward(
                    dout1,
                    q1,
                    k,
                    v,
                    out1,
                    softmax_max1,
                    softmax_sum1,
                    causal=False,
                    rng_state=rng_states[step],
                )
                dq[half_index1] += dq_block.float()

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[half_index0] += dk_block.float()
                dv[half_index0] += dv_block.float()
            else:
                dk += dk_block.float()
                dv += dv_block.float()

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


class ZigZagRingNPUFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        process_group,
        cu_seq_len: torch.Tensor,
        softmax_scale: Optional[float],
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        half_index0 = get_half_index(cu_seq_len, front=True)
        half_index1 = get_half_index(cu_seq_len, front=False)
        global_out, global_softmax_max, global_softmax_sum, rng_states = (
            zigzag_ring_flash_attn_varlen_forward(
                process_group,
                q,
                k,
                v,
                cu_seq_len,
                half_index0,
                half_index1,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
            )
        )

        is_half_index_tensor = isinstance(half_index0, torch.Tensor)
        ctx.is_half_index_tensor = is_half_index_tensor
        if is_half_index_tensor:
            ctx.save_for_backward(
                q,
                k,
                v,
                global_out,
                global_softmax_max,
                global_softmax_sum,
                cu_seq_len,
                half_index0,
                half_index1,
            )
        else:
            ctx.save_for_backward(
                q,
                k,
                v,
                global_out,
                global_softmax_max,
                global_softmax_sum,
                cu_seq_len,
            )
            ctx.half_index0 = half_index0
            ctx.half_index1 = half_index1

        ctx.group = process_group
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.rng_states = rng_states
        return global_out

    @staticmethod
    def backward(ctx, dout: torch.Tensor, *args):
        if ctx.is_half_index_tensor:
            (
                q,
                k,
                v,
                global_out,
                global_softmax_max,
                global_softmax_sum,
                cu_seq_len,
                half_index0,
                half_index1,
            ) = ctx.saved_tensors
        else:
            (
                q,
                k,
                v,
                global_out,
                global_softmax_max,
                global_softmax_sum,
                cu_seq_len,
            ) = ctx.saved_tensors
            half_index0 = ctx.half_index0
            half_index1 = ctx.half_index1

        dq, dk, dv = zigzag_ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            global_out,
            global_softmax_max,
            global_softmax_sum,
            cu_seq_len,
            ctx.rng_states,
            half_index0,
            half_index1,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
        )
        return dq, dk, dv, None, None, None, None
