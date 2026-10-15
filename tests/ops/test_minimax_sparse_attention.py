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

"""BSND eager MiniMax sparse attention against the upstream per-sample reference."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from veomni.ops.kernels.minimax_sparse_attention import (
    minimax_sparse_attention_eager,
    minimax_sparse_indexer_eager,
)
from veomni.utils.import_utils import is_torch_npu_available, is_transformers_version_greater_or_equal_to


pytestmark = pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.16.0"),
    reason="The upstream MiniMax M3 VL reference requires transformers>=5.16.0.",
)

# Rows are right-padded to a common length that exceeds the longest sample, the
# way the collator rounds rows up to the SP size.
_SEQUENCE_LENGTHS = (5, 1, 9, 3)
_PADDED_LENGTH = 10
_NUM_QUERY_HEADS = 4
_NUM_GROUPS = 2
_HEAD_DIM = 8
_BLOCK_SIZE = 2
_TOPK_BLOCKS = 3
_LOCAL_BLOCKS = 1


def _upstream():
    return importlib.import_module("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")


def _upstream_indexer():
    upstream = _upstream()
    config = upstream.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=_NUM_QUERY_HEADS,
        num_key_value_heads=_NUM_GROUPS,
        index_n_heads=_NUM_GROUPS,
        index_head_dim=_HEAD_DIM,
        index_block_size=_BLOCK_SIZE,
        index_topk_blocks=_TOPK_BLOCKS,
        index_local_blocks=_LOCAL_BLOCKS,
        head_dim=_HEAD_DIM,
        rotary_dim=_HEAD_DIM,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["minimax_m3_sparse"],
        mlp_layer_types=["dense"],
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    indexer = upstream.MiniMaxM3VLIndexer(config, layer_idx=0).eval()
    for norm in (indexer.q_norm, indexer.k_norm):
        torch.nn.init.normal_(norm.weight, std=0.5)
    return config, indexer, upstream.MiniMaxM3VLRotaryEmbedding(config)


def _pad_rows(tensors, length=_PADDED_LENGTH):
    """Stack ``[S_i, ...]`` samples into ``[B, length, ...]`` rows; padding is random, not zero.

    Random padding proves real positions never read it.
    """
    rows = []
    for tensor in tensors:
        garbage = torch.randn(length - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype)
        rows.append(torch.cat((tensor, garbage)))
    return torch.stack(rows)


def _project_index_qk(indexer, hidden_states, position_embeddings):
    """Upstream indexer projection + norm + RoPE, returned as ``[S, G, D]`` / ``[S, 1, D]`` per sample."""
    upstream = _upstream()
    seq_len = hidden_states.shape[1]
    index_q = indexer.q_norm(indexer.q_proj(hidden_states).view(1, seq_len, -1, indexer.head_dim)).transpose(1, 2)
    index_k = indexer.k_norm(indexer.k_proj(hidden_states).view(1, seq_len, 1, indexer.head_dim)).transpose(1, 2)
    cos, sin = position_embeddings
    index_q, index_k = upstream.apply_rotary_pos_emb(index_q, index_k, cos, sin)
    return index_q[0].transpose(0, 1), index_k[0].transpose(0, 1)


def _canonical_block_indices(block_indices):
    """Pad to ``topk_blocks`` slots and sort, since slot order and width are not part of the contract."""
    block_indices = F.pad(block_indices, (0, _TOPK_BLOCKS - block_indices.shape[-1]), value=-1)
    return block_indices.sort(dim=-1, descending=True).values


def _per_sample_inputs():
    config, indexer, rotary = _upstream_indexer()
    samples = []
    for seq_len in _SEQUENCE_LENGTHS:
        hidden_states = torch.randn(1, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        position_embeddings = rotary(hidden_states, position_ids)
        with torch.no_grad():
            reference_indices = indexer(hidden_states, position_embeddings, None, position_ids)
            index_q, index_k = _project_index_qk(indexer, hidden_states, position_embeddings)
        samples.append(
            SimpleNamespace(
                seq_len=seq_len,
                position_ids=position_ids,
                reference_indices=reference_indices,
                index_q=index_q,
                index_k=index_k,
                q=torch.randn(seq_len, _NUM_QUERY_HEADS, _HEAD_DIM),
                k=torch.randn(seq_len, _NUM_GROUPS, _HEAD_DIM),
                v=torch.randn(seq_len, _NUM_GROUPS, _HEAD_DIM),
            )
        )
    return indexer, samples


def _upstream_attention(indexer, sample, q, k, v, block_indices):
    """Upstream ``build_block_mask`` + ``eager_attention_forward`` on one unpadded sample."""
    upstream = _upstream()
    query, key, value = (tensor.transpose(0, 1).unsqueeze(0) for tensor in (q, k, v))
    if block_indices is None:
        positions = torch.arange(sample.seq_len)
        attention_mask = torch.zeros(1, 1, sample.seq_len, sample.seq_len).masked_fill(
            positions[None, :] > positions[:, None], torch.finfo(torch.float32).min
        )
    else:
        attention_mask = indexer.build_block_mask(
            block_indices, None, sample.seq_len, query.dtype, query.device, sample.position_ids
        )
    module = SimpleNamespace(num_key_value_groups=_NUM_QUERY_HEADS // _NUM_GROUPS, training=False)
    output, _ = upstream.eager_attention_forward(module, query, key, value, attention_mask, scaling=_HEAD_DIM**-0.5)
    return output[0]


def _padded_indexer(samples, length=_PADDED_LENGTH):
    return minimax_sparse_indexer_eager(
        _pad_rows([sample.index_q for sample in samples], length),
        _pad_rows([sample.index_k for sample in samples], length),
        block_size=_BLOCK_SIZE,
        topk_blocks=_TOPK_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
    )


def test_indexer_matches_upstream_per_sample():
    _, samples = _per_sample_inputs()
    block_indices = _padded_indexer(samples)

    assert block_indices.shape == (len(samples), _PADDED_LENGTH, _NUM_GROUPS, _TOPK_BLOCKS)
    assert block_indices.dtype == torch.int32
    for row, sample in enumerate(samples):
        expected = _canonical_block_indices(sample.reference_indices[0].transpose(0, 1))
        actual = _canonical_block_indices(block_indices[row, : sample.seq_len]).long()
        assert torch.equal(actual, expected)


# float64 guards padded query rows: a fully masked row turns into NaN once the
# float32 softmax sees the float64 dtype-min mask, and leaks into K/V gradients.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("sparse", [True, False], ids=["block_sparse", "dense_causal"])
def test_attention_matches_upstream_per_sample_forward_and_backward(sparse, dtype):
    indexer, samples = _per_sample_inputs()
    for sample in samples:
        sample.q, sample.k, sample.v = (getattr(sample, name).to(dtype) for name in ("q", "k", "v"))
    padded_q, padded_k, padded_v = (
        _pad_rows([getattr(sample, name) for sample in samples]).requires_grad_(True) for name in ("q", "k", "v")
    )
    # Padded rows select blocks like any query: the operator never sees lengths.
    block_indices = _padded_indexer(samples) if sparse else None
    output = minimax_sparse_attention_eager(
        padded_q,
        padded_k,
        padded_v,
        block_indices,
        block_size=_BLOCK_SIZE,
        scale=_HEAD_DIM**-0.5,
    )
    assert output.shape == padded_q.shape
    assert torch.isfinite(output).all()

    # The loss ignores padded positions, so their upstream gradient is zero.
    valid = torch.arange(_PADDED_LENGTH)[None, :] < torch.tensor(_SEQUENCE_LENGTHS)[:, None]
    grad_output = torch.randn_like(output) * valid[..., None, None]
    padded_grads = torch.autograd.grad(output, (padded_q, padded_k, padded_v), grad_output)

    # Upstream softmax runs in float32 for every input dtype, and padding changes
    # its reduction length, so float64 inputs only match to float32 precision.
    tolerance = dict(rtol=1.3e-6, atol=1e-5)
    for row, sample in enumerate(samples):
        q, k, v = (getattr(sample, name).clone().requires_grad_(True) for name in ("q", "k", "v"))
        reference_indices = sample.reference_indices if sparse else None
        expected = _upstream_attention(indexer, sample, q, k, v, reference_indices)
        grads = torch.autograd.grad(expected, (q, k, v), grad_output[row, : sample.seq_len])

        torch.testing.assert_close(output[row, : sample.seq_len], expected, **tolerance)
        for padded_grad, grad in zip(padded_grads, grads):
            assert torch.isfinite(padded_grad[row]).all()
            torch.testing.assert_close(padded_grad[row, : sample.seq_len], grad, **tolerance)
            assert not padded_grad[row, sample.seq_len :].any()


def test_indexer_output_feeds_attention_with_wider_topk_slots():
    """The indexer pads to ``topk_blocks`` slots even when the rows have fewer key blocks."""
    _, samples = _per_sample_inputs()
    short = [sample for sample in samples if sample.seq_len <= 3]
    block_indices = _padded_indexer(short, length=4)
    assert block_indices.shape[-1] == _TOPK_BLOCKS
    assert (block_indices[..., -1] == -1).all()

    output = minimax_sparse_attention_eager(
        _pad_rows([sample.q for sample in short], 4),
        _pad_rows([sample.k for sample in short], 4),
        _pad_rows([sample.v for sample in short], 4),
        block_indices,
        block_size=_BLOCK_SIZE,
        scale=_HEAD_DIM**-0.5,
    )
    assert torch.isfinite(output).all()


def test_rejects_multi_head_index_key():
    index_q = torch.randn(1, 4, _NUM_GROUPS, _HEAD_DIM)
    with pytest.raises(ValueError, match="single index key head"):
        minimax_sparse_indexer_eager(
            index_q, index_q, block_size=_BLOCK_SIZE, topk_blocks=_TOPK_BLOCKS, local_blocks=_LOCAL_BLOCKS
        )


def test_rejects_block_indices_not_grouped_by_kv_head():
    q = torch.randn(1, 4, _NUM_QUERY_HEADS, _HEAD_DIM)
    k = torch.randn(1, 4, _NUM_GROUPS, _HEAD_DIM)
    block_indices = torch.zeros(1, 4, _NUM_QUERY_HEADS, _TOPK_BLOCKS, dtype=torch.int32)
    with pytest.raises(ValueError, match="one selection per KV head"):
        minimax_sparse_attention_eager(q, k, k, block_indices, block_size=_BLOCK_SIZE, scale=_HEAD_DIM**-0.5)


def test_rejects_gpu_attention_before_allocating_mask():
    q = SimpleNamespace(device=torch.device("cuda"))
    with pytest.raises(NotImplementedError, match="NPU only"):
        minimax_sparse_attention_eager(q, None, None, None, block_size=None, scale=1.0)


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
@pytest.mark.parametrize("num_groups", [1, _NUM_GROUPS], ids=["one_kv_group", "two_kv_groups"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("sparse", [True, False], ids=["block_sparse", "dense_causal"])
def test_npu_attention_matches_cpu_forward_and_backward(dtype, sparse, num_groups):
    torch.manual_seed(42)
    batch, seq_len, head_dim = 3, 9, 64
    cpu_inputs = [
        torch.randn(batch, seq_len, heads, head_dim).to(dtype).float().requires_grad_()
        for heads in (_NUM_QUERY_HEADS, num_groups, num_groups)
    ]
    npu_inputs = [tensor.detach().to(device="npu", dtype=dtype).requires_grad_() for tensor in cpu_inputs]
    block_indices = None
    if sparse:
        block_indices = minimax_sparse_indexer_eager(
            cpu_inputs[1].detach(),
            cpu_inputs[1][:, :, :1].detach(),
            block_size=_BLOCK_SIZE,
            topk_blocks=_TOPK_BLOCKS,
            local_blocks=_LOCAL_BLOCKS,
        )
    kwargs = dict(block_size=_BLOCK_SIZE, scale=head_dim**-0.5)
    expected = minimax_sparse_attention_eager(*cpu_inputs, block_indices, **kwargs)
    actual = minimax_sparse_attention_eager(*npu_inputs, block_indices.to("npu") if sparse else None, **kwargs)
    grad_output = torch.randn_like(expected).to(dtype).float()
    expected_grads = torch.autograd.grad(expected, cpu_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, npu_inputs, grad_output.to(device="npu", dtype=dtype))
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    for result, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(result.float().cpu(), reference, atol=tolerance, rtol=tolerance)
