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

"""TND eager MiniMax sparse attention against the upstream per-sample BSND reference."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from veomni.ops.kernels.minimax_sparse_attention import (
    minimax_sparse_attention_eager,
    minimax_sparse_indexer_eager,
)
from veomni.ops.kernels.minimax_sparse_attention import npu as msa_npu
from veomni.ops.kernels.minimax_sparse_attention.eager import (
    _expand_selected_blocks,
    _pack,
    _packed_to_padded_indices,
    _unpack,
)
from veomni.utils.import_utils import is_torch_npu_available


# The last segment stands in for the collator's synthetic SP tail padding: it is
# an ordinary packed sequence and must not interact with the real samples.
_SEQUENCE_LENGTHS = (5, 1, 9, 3)
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


def _cu_seqlens(lengths):
    return F.pad(torch.tensor(lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))


def _project_index_qk(indexer, hidden_states, position_embeddings):
    """Upstream indexer projection + norm + RoPE, returned as ``[S, G, D]`` / ``[S, 1, D]``."""
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


def test_indexer_matches_upstream_per_sample():
    _, samples = _per_sample_inputs()
    block_indices = minimax_sparse_indexer_eager(
        torch.cat([sample.index_q for sample in samples]),
        torch.cat([sample.index_k for sample in samples]),
        _cu_seqlens(_SEQUENCE_LENGTHS),
        max(_SEQUENCE_LENGTHS),
        block_size=_BLOCK_SIZE,
        topk_blocks=_TOPK_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
    )

    assert block_indices.shape == (sum(_SEQUENCE_LENGTHS), _NUM_GROUPS, _TOPK_BLOCKS)
    assert block_indices.dtype == torch.int32
    expected = torch.cat([_canonical_block_indices(sample.reference_indices[0].transpose(0, 1)) for sample in samples])
    assert torch.equal(_canonical_block_indices(block_indices).long(), expected)


# float64 guards padded query rows: a fully masked row turns into NaN once the
# float32 softmax sees the float64 dtype-min mask, and leaks into K/V gradients.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("sparse", [True, False], ids=["block_sparse", "dense_causal"])
def test_attention_matches_upstream_per_sample_forward_and_backward(sparse, dtype):
    indexer, samples = _per_sample_inputs()
    for sample in samples:
        sample.q, sample.k, sample.v = (getattr(sample, name).to(dtype) for name in ("q", "k", "v"))
    packed_q, packed_k, packed_v = (
        torch.cat([getattr(sample, name) for sample in samples]).requires_grad_(True) for name in ("q", "k", "v")
    )
    block_indices = None
    if sparse:
        block_indices = torch.cat(
            [_canonical_block_indices(sample.reference_indices[0].transpose(0, 1)) for sample in samples]
        ).to(torch.int32)
    output = minimax_sparse_attention_eager(
        packed_q,
        packed_k,
        packed_v,
        block_indices,
        _cu_seqlens(_SEQUENCE_LENGTHS),
        max(_SEQUENCE_LENGTHS),
        block_size=_BLOCK_SIZE,
        scale=_HEAD_DIM**-0.5,
    )
    grad_output = torch.randn_like(output)
    packed_grads = torch.autograd.grad(output, (packed_q, packed_k, packed_v), grad_output)

    expected_outputs = []
    expected_grads = [[], [], []]
    offset = 0
    for sample in samples:
        q, k, v = (getattr(sample, name).clone().requires_grad_(True) for name in ("q", "k", "v"))
        reference_indices = sample.reference_indices if sparse else None
        expected = _upstream_attention(indexer, sample, q, k, v, reference_indices)
        grads = torch.autograd.grad(expected, (q, k, v), grad_output[offset : offset + sample.seq_len])
        expected_outputs.append(expected)
        for accumulated, grad in zip(expected_grads, grads):
            accumulated.append(grad)
        offset += sample.seq_len

    torch.testing.assert_close(output, torch.cat(expected_outputs))
    for packed_grad, grads in zip(packed_grads, expected_grads):
        assert torch.isfinite(packed_grad).all()
        torch.testing.assert_close(packed_grad, torch.cat(grads))


def test_indexer_output_feeds_attention_with_wider_topk_slots():
    """The indexer pads to ``topk_blocks`` slots even when the batch has fewer key blocks."""
    _, samples = _per_sample_inputs()
    short = [sample for sample in samples if sample.seq_len <= 3]
    lengths = [sample.seq_len for sample in short]
    block_indices = minimax_sparse_indexer_eager(
        torch.cat([sample.index_q for sample in short]),
        torch.cat([sample.index_k for sample in short]),
        _cu_seqlens(lengths),
        max(lengths),
        block_size=_BLOCK_SIZE,
        topk_blocks=_TOPK_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
    )
    assert block_indices.shape[-1] == _TOPK_BLOCKS
    assert (block_indices[..., -1] == -1).all()

    output = minimax_sparse_attention_eager(
        torch.cat([sample.q for sample in short]),
        torch.cat([sample.k for sample in short]),
        torch.cat([sample.v for sample in short]),
        block_indices,
        _cu_seqlens(lengths),
        max(lengths),
        block_size=_BLOCK_SIZE,
        scale=_HEAD_DIM**-0.5,
    )
    assert torch.isfinite(output).all()


def test_padded_layout_round_trips_sp_padding_segment():
    packed = torch.arange(14, dtype=torch.float32).view(7, 2)

    indices, padding = _packed_to_padded_indices(_cu_seqlens([2, 3, 2]), 3, 7, packed.device)
    padded = _unpack(packed, indices, padding, fill=-1)

    assert indices.tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert padding.tolist() == [[False, False, True], [False, False, False], [False, False, True]]
    assert padded.shape == (3, 3, 2)
    assert (padded[padding] == -1).all()
    assert torch.equal(_pack(padded, indices), packed)


def test_rejects_device_tensor_max_seqlen():
    q = torch.randn(4, _NUM_QUERY_HEADS, _HEAD_DIM)
    k = torch.randn(4, _NUM_GROUPS, _HEAD_DIM)
    with pytest.raises(TypeError, match="Python int"):
        minimax_sparse_attention_eager(
            q, k, k, None, _cu_seqlens([4]), torch.tensor(4), block_size=None, scale=_HEAD_DIM**-0.5
        )


def test_rejects_block_indices_not_grouped_by_kv_head():
    q = torch.randn(4, _NUM_QUERY_HEADS, _HEAD_DIM)
    k = torch.randn(4, _NUM_GROUPS, _HEAD_DIM)
    block_indices = torch.zeros(4, _NUM_QUERY_HEADS, _TOPK_BLOCKS, dtype=torch.int32)
    with pytest.raises(ValueError, match="one selection per KV head"):
        minimax_sparse_attention_eager(
            q, k, k, block_indices, _cu_seqlens([4]), 4, block_size=_BLOCK_SIZE, scale=_HEAD_DIM**-0.5
        )


@pytest.mark.parametrize(("seq_len", "block_size"), [(9, 2), (8, 4), (5, 8)], ids=["ragged", "exact", "one-block"])
def test_bfloat16_block_expansion_matches_boolean_expansion(seq_len, block_size):
    """The NPU-only bfloat16 expansion path must build exactly the boolean mask."""
    torch.manual_seed(0)
    num_key_blocks = -(-seq_len // block_size)
    selected = torch.rand(2, _NUM_GROUPS, seq_len, num_key_blocks) > 0.5
    kwargs = dict(block_size=block_size, seq_len=seq_len, heads_per_group=_NUM_QUERY_HEADS // _NUM_GROUPS)

    expected = _expand_selected_blocks(selected, via_bfloat16=False, **kwargs)
    actual = _expand_selected_blocks(selected, via_bfloat16=True, **kwargs)

    assert actual.dtype == torch.bool
    assert actual.shape == (2, _NUM_QUERY_HEADS, seq_len, seq_len)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("sparse", [True, False], ids=["block_sparse", "dense_causal"])
def test_npu_fused_attention_matches_cpu_reference_forward_and_backward(dtype, sparse):
    torch.manual_seed(42)
    lengths = _SEQUENCE_LENGTHS
    total_length, head_dim = sum(lengths), 64
    cu_seqlens = _cu_seqlens(lengths)
    cpu_inputs = [
        torch.randn(total_length, heads, head_dim).to(dtype).float().requires_grad_()
        for heads in (_NUM_QUERY_HEADS, _NUM_GROUPS, _NUM_GROUPS)
    ]
    npu_inputs = [tensor.detach().to(device="npu", dtype=dtype).requires_grad_() for tensor in cpu_inputs]
    block_indices = None
    if sparse:
        block_indices = minimax_sparse_indexer_eager(
            cpu_inputs[1].detach(),
            cpu_inputs[1][:, :1].detach(),
            cu_seqlens,
            max(lengths),
            block_size=_BLOCK_SIZE,
            topk_blocks=_TOPK_BLOCKS,
            local_blocks=_LOCAL_BLOCKS,
        )
    kwargs = dict(block_size=_BLOCK_SIZE, scale=head_dim**-0.5)
    expected = minimax_sparse_attention_eager(*cpu_inputs, block_indices, cu_seqlens, max(lengths), **kwargs)
    actual = minimax_sparse_attention_eager(
        *npu_inputs,
        block_indices.to("npu") if sparse else None,
        cu_seqlens,
        max(lengths),
        **kwargs,
    )
    grad_output = torch.randn_like(expected).to(dtype).float()
    expected_grads = torch.autograd.grad(expected, cpu_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, npu_inputs, grad_output.to(device="npu", dtype=dtype))
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    for result, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(result.float().cpu(), reference, atol=tolerance, rtol=tolerance)


def _index_scores(q, k, lengths):
    """Independent per-sample FP32 score reference, including CANN stride padding."""
    blocks = (max(lengths) + 127) // 128
    stride = (blocks + 15) // 16 * 16
    out = torch.full((q.shape[1], sum(lengths), stride), -torch.inf, device=q.device)
    offset = 0
    for length in lengths:
        if length == 0:
            continue
        logits = q[offset : offset + length].transpose(0, 1).float() @ k[offset : offset + length, 0].float().T
        positions = torch.arange(length, device=q.device)
        logits.masked_fill_(positions[None, :] > positions[:, None], -torch.inf)
        logits = F.pad(logits, (0, blocks * 128 - length), value=-torch.inf)
        out[:, offset : offset + length, :blocks] = logits.view(q.shape[1], length, blocks, 128).amax(-1)
        offset += length
    return out


def _attention_from_keep(q, k, v, keep, scale):
    """FP32 packed reference for checking CSR and backward-list connectivity."""
    group = q.shape[1] // k.shape[1]
    keys, values = (x.repeat_interleave(group, dim=1).transpose(0, 1) for x in (k, v))
    scores = (q.transpose(0, 1) @ keys.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep.repeat_interleave(group, dim=0), -torch.inf)
    lse = scores.logsumexp(-1).transpose(0, 1).unsqueeze(-1)
    return (scores.softmax(-1) @ values).transpose(0, 1), lse


class _FakeMSACannOps:
    """CPU math behind the CANN signatures; validates adapter/autograd wiring only."""

    def __init__(self):
        self.calls = []

    def msa_index_score(self, **kwargs):
        self.calls.append("score")
        assert kwargs["layout_key"] == "TND"
        assert kwargs["sparse_mode"] == 3
        assert kwargs["init_blocks"] == kwargs["local_blocks"] == 0
        assert kwargs["atten_mask"].shape == (2048, 2048)
        assert kwargs["atten_mask"].dtype == torch.int8
        assert kwargs["scale"] is None and kwargs["block_table"] is None
        assert kwargs["actual_seq_qlen"].dtype == torch.int32
        torch.testing.assert_close(kwargs["actual_seq_qlen"], kwargs["actual_seq_klen"])
        scores = _index_scores(kwargs["query"], kwargs["key"], kwargs["actual_seq_qlen"].diff().tolist())
        return scores.masked_fill(torch.isneginf(scores), torch.finfo(torch.float32).min)

    def k2q_csr(self, q2k, cu, cu_blocks, *, order_method, total_rows=None, max_kv=None, use_simt, q_global_offset):
        self.calls.append("csr")
        assert order_method == 1 and q_global_offset is True and use_simt is False
        assert q2k.dtype == cu.dtype == cu_blocks.dtype == torch.int32
        lengths = cu.diff().tolist()
        counts = [(length + 127) // 128 for length in lengths]
        assert cu_blocks.tolist() == [0] + torch.tensor(counts).cumsum(0).tolist()
        assert total_rows is None or total_rows == sum(counts)
        assert max_kv is None or max_kv == max(counts)
        self.csr_sizes = (total_rows, max_kv)
        self.rows = [(b, j) for j in range(max(counts)) for b, count in enumerate(counts) if j < count]
        self.q2k = q2k
        heads, total, budget = q2k.shape
        ptr = torch.zeros((heads, len(self.rows) + 1), dtype=torch.int32)
        queries = torch.full((heads, total * budget), -1, dtype=torch.int32)
        slots = torch.full_like(queries, -1)
        for h in range(heads):
            edges = []
            for row, (b, block) in enumerate(self.rows):
                for query in range(int(cu[b]), int(cu[b + 1])):
                    for slot in range(budget):
                        if q2k[h, query, slot] == block:
                            edges.append((query, slot))
                ptr[h, row + 1] = len(edges)
            if edges:
                queries[h, : len(edges)], slots[h, : len(edges)] = torch.tensor(edges, dtype=torch.int32).T
        return ptr, queries, slots

    def npu_minimax_sparse_attention_split_kv(self, **kwargs):
        self.calls.append("forward")
        assert kwargs["softmax_lse_flag"] is True and kwargs["inner_precise"] == 0
        assert kwargs["input_layout"] == "TND" and kwargs["block_table"] is None
        assert kwargs["block_size"] == 128
        q, k, v = (kwargs[name] for name in ("query", "key", "value"))
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        lengths = kwargs["actual_seq_lengths"]
        assert lengths.dtype == torch.int32
        torch.testing.assert_close(lengths, kwargs["actual_seq_lengths_kv"])
        cu = F.pad(lengths.cumsum(0), (1, 0))
        ptr, queries, slots = (kwargs[name] for name in ("k2q_row_ptr", "k2q_q_indices", "k2q_slot_indices"))
        keep = torch.zeros((k.shape[1], q.shape[0], k.shape[0]), dtype=torch.bool)
        for h in range(k.shape[1]):
            for row, (b, block) in enumerate(self.rows):
                for edge in range(int(ptr[h, row]), int(ptr[h, row + 1])):
                    query, slot = int(queries[h, edge]), int(slots[h, edge])
                    assert int(cu[b]) <= query < int(cu[b + 1])
                    assert self.q2k[h, query, slot] == block
                    begin = int(cu[b]) + block * 128
                    end = min(int(cu[b + 1]), begin + 128, query + 1)
                    keep[h, query, begin:end] = True
        self.forward_keep = keep
        return _attention_from_keep(q, k, v, keep, kwargs["scale_value"])

    def generic_block_sparse_attention_grad_metadata(self, idx, count, hq, hkv, dim, block_shape, **kwargs):
        self.calls.append("metadata")
        assert block_shape == [1, 128]
        assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_kv"] == idx.shape[-1]
        assert kwargs["cu_seqlens_q"].dtype == kwargs["cu_seqlens_kv"].dtype == torch.int64
        assert idx.dtype == count.dtype == torch.int32
        self.grad_attrs = {key: value for key, value in kwargs.items() if not key.startswith("max_seqlen_")}
        self.metadata = torch.zeros(198, dtype=torch.int32)
        return self.metadata

    def generic_block_sparse_attention_grad(self, q, k, v, dout, out, lse, idx, count, block_shape, **kwargs):
        self.calls.append("backward")
        assert dout.is_contiguous()
        assert kwargs.pop("metadata") is self.metadata and kwargs.pop("attn_mask") is None
        scale = kwargs.pop("softmax_scale")
        for key, value in self.grad_attrs.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(kwargs[key], value)
            else:
                assert kwargs[key] == value
        assert kwargs["layout_q"] == kwargs["layout_kv"] == "TND"
        assert kwargs["mask_mode"] == 1 and kwargs["softmax_precision"] == 0
        assert idx.shape[2] <= (k.shape[0] + 127) // 128  # Native GBSA TND tiling bound.
        keep = torch.zeros_like(self.forward_keep)
        cu = kwargs["cu_seqlens_q"]
        for b in range(idx.shape[0]):
            for h in range(idx.shape[1]):
                for block in range(idx.shape[2]):
                    size = int(count[b, h, block])
                    selected = idx[b, h, block, :size]
                    assert (selected[1:] > selected[:-1]).all()
                    assert (idx[b, h, block, size:] == -1).all()
                    for query in selected.tolist():
                        query += int(cu[b])
                        begin = int(cu[b]) + block * 128
                        end = min(int(cu[b + 1]), begin + 128, query + 1)
                        keep[h, query, begin:end] = True
        assert torch.equal(keep, self.forward_keep)
        with torch.enable_grad():
            inputs = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
            reference, reference_lse = _attention_from_keep(*inputs, keep, scale)
            torch.testing.assert_close(out, reference)
            torch.testing.assert_close(lse, reference_lse)
            return torch.autograd.grad(reference, inputs, dout)


def _install_fake_cann_ops(monkeypatch, fake):
    package = ModuleType("cann_ops_transformer")
    package.ops = fake
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", package)


@pytest.mark.parametrize("max_seqlen_padding", [0, 128])
@pytest.mark.parametrize("local_blocks", [0, 1, 3])
@pytest.mark.parametrize("topk_blocks", [2, 16], ids=["sparse_selection", "padded_slots"])
def test_cann_indexer_adapter_matches_eager_and_handles_finite_invalid_fill(
    monkeypatch, local_blocks, topk_blocks, max_seqlen_padding
):
    lengths = (127, 128, 129, 513, 3)
    torch.manual_seed(7)
    q = torch.randn(sum(lengths), 2, 128, dtype=torch.bfloat16)
    k = torch.randn(sum(lengths), 1, 128, dtype=torch.bfloat16)
    fake = _FakeMSACannOps()
    _install_fake_cann_ops(monkeypatch, fake)
    kwargs = dict(block_size=128, topk_blocks=topk_blocks, local_blocks=local_blocks)
    actual = msa_npu.minimax_sparse_indexer_npu(
        q, k, _cu_seqlens(lengths), max(lengths) + max_seqlen_padding, **kwargs
    )
    expected = minimax_sparse_indexer_eager(q, k, _cu_seqlens(lengths), max(lengths) + max_seqlen_padding, **kwargs)
    assert torch.equal(actual.sort(-1).values, expected.sort(-1).values)
    assert actual.dtype == torch.int32 and actual.shape == (sum(lengths), 2, topk_blocks)
    assert fake.calls == ["score"]
    positions = torch.cat([torch.arange(length) for length in lengths])
    assert ((actual == -1) | ((actual >= 0) & (actual <= positions[:, None, None] // 128))).all()
    ordered = actual.sort(-1).values
    assert ((ordered[..., 1:] != ordered[..., :-1]) | (ordered[..., 1:] == -1)).all()
    assert not ((actual[..., :-1] == -1) & (actual[..., 1:] != -1)).any()


@pytest.mark.parametrize("lengths", [(129, 257, 1), (0, 129, 0, 257, 1, 0), (1,), (128,)])
@pytest.mark.parametrize("max_seqlen_padding", [0, 256])
def test_cann_csr_to_gbsa_preserves_edges_without_host_reads_or_sort(monkeypatch, lengths, max_seqlen_padding):
    torch.manual_seed(29)
    max_seqlen = max(lengths) + max_seqlen_padding
    blocks = (max_seqlen + 127) // 128
    selected = torch.rand(sum(lengths), 2, blocks).topk(min(2, blocks), dim=-1).indices.to(torch.int32)
    positions = torch.cat([torch.arange(length) for length in lengths])
    selected.masked_fill_(selected > positions[:, None, None] // 128, -1)
    selected = F.pad(selected, (0, 4 - selected.shape[-1]), value=-1)
    # Include a head with no edges and repeated row pointers.
    selected[:, 1] = -1
    fake = _FakeMSACannOps()
    cu = _cu_seqlens(lengths)
    cu_blocks = F.pad(((cu.diff() + 127) // 128).cumsum(0, dtype=torch.int32), (1, 0))
    ptr, queries, _ = fake.k2q_csr(
        selected.permute(1, 0, 2).contiguous(), cu, cu_blocks, order_method=1, use_simt=False, q_global_offset=True
    )
    with monkeypatch.context() as guard:
        for name in ("cpu", "item", "tolist", "sort", "argsort"):
            guard.setattr(
                torch.Tensor, name, lambda *a, **kw: pytest.fail("CSR conversion must not read back or sort")
            )
        for name in ("sort", "argsort"):
            guard.setattr(torch, name, lambda *a, **kw: pytest.fail("CSR conversion must not sort"))
        idx, counts = msa_npu._csr_to_gbsa(ptr, queries, cu, max_seqlen)
    assert idx.shape == (len(lengths), 2, blocks, max_seqlen)
    assert idx.dtype == counts.dtype == torch.int32
    for b, length in enumerate(lengths):
        for h in range(2):
            for block in range(blocks):
                expected = torch.where((selected[int(cu[b]) : int(cu[b]) + length, h] == block).any(-1))[0]
                count = expected.numel()
                assert counts[b, h, block] == count
                torch.testing.assert_close(idx[b, h, block, :count], expected.to(torch.int32))
                assert (idx[b, h, block, count:] == -1).all()


@pytest.mark.parametrize(
    "lengths",
    [(129, 257, 1), (0, 129, 0, 257, 1), (128,), (0, 1, 0)],
    ids=["packed", "empty_segments", "single", "single_token"],
)
@pytest.mark.parametrize("max_seqlen_padding", [0, 128])
def test_cann_split_kv_gbsa_autograd_matches_eager_through_both_index_formats(
    monkeypatch, lengths, max_seqlen_padding
):
    torch.manual_seed(31)
    inputs = [torch.randn(heads, sum(lengths), 16).transpose(0, 1).requires_grad_() for heads in (4, 2, 2)]
    index_q, index_k = torch.randn(sum(lengths), 2, 16), torch.randn(sum(lengths), 1, 16)
    cu = _cu_seqlens(lengths)
    selected = minimax_sparse_indexer_eager(
        index_q, index_k, cu, max(lengths), block_size=128, topk_blocks=2, local_blocks=1
    )
    fake = _FakeMSACannOps()
    _install_fake_cann_ops(monkeypatch, fake)
    actual = msa_npu._SplitKVGBSA.apply(*inputs, selected, cu, max(lengths) + max_seqlen_padding, 0.17)
    torch.testing.assert_close(fake.q2k, selected.permute(1, 0, 2))
    block_counts = [(length + 127) // 128 for length in lengths]
    assert fake.csr_sizes == (sum(block_counts), max(block_counts))
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    expected = minimax_sparse_attention_eager(
        *reference_inputs, selected, cu, max(lengths), block_size=128, scale=0.17
    )
    grad = torch.randn(sum(lengths), 16, 4).transpose(1, 2)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad)
    for value, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)
    assert fake.calls == ["csr", "forward", "metadata", "backward"]


def test_cann_empty_pack_has_zero_gradients_without_loading_ops(monkeypatch):
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    q = torch.empty(0, 4, 128, dtype=torch.bfloat16, requires_grad=True)
    k = torch.empty(0, 2, 128, dtype=torch.bfloat16, requires_grad=True)
    v = torch.empty_like(k, requires_grad=True)
    selected = msa_npu.minimax_sparse_indexer_npu(
        k, k[:, :1], _cu_seqlens((0,)), 0, block_size=128, topk_blocks=16, local_blocks=1
    )
    output = msa_npu.minimax_sparse_attention_npu(q, k, v, selected, _cu_seqlens((0,)), 0, block_size=128, scale=0.2)
    for grad, source in zip(torch.autograd.grad(output.sum(), (q, k, v)), (q, k, v)):
        assert grad.shape == source.shape and grad.numel() == 0


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_cann_attention_dense_and_dropout_use_eager_without_loading_cann(monkeypatch, dropout):
    q = torch.randn(5, 4, 8, requires_grad=True)
    k = torch.randn(5, 2, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    indices = torch.zeros(5, 2, 3, dtype=torch.int32) if dropout else None
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    kwargs = dict(block_size=2 if indices is not None else None, scale=0.3, dropout_p=dropout)
    torch.manual_seed(19)
    actual = msa_npu.minimax_sparse_attention_npu(q, k, v, indices, _cu_seqlens((2, 3)), 3, **kwargs)
    torch.manual_seed(19)
    expected = minimax_sparse_attention_eager(q, k, v, indices, _cu_seqlens((2, 3)), 3, **kwargs)
    torch.testing.assert_close(actual, expected)
    assert all(torch.isfinite(grad).all() for grad in torch.autograd.grad(actual.sum(), (q, k, v)))


def test_cann_registry_binds_both_slots_and_gates_hardware(monkeypatch):
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops
    from veomni.ops import kernel_registry
    from veomni.ops.dispatch import OpSlot

    fields = OpsImplementationConfig.__dataclass_fields__
    assert fields["minimax_sparse_attention_implementation"].default == "eager"
    module = SimpleNamespace(
        indexer=OpSlot("minimax_sparse_attention", "indexer"),
        attention=OpSlot("minimax_sparse_attention", "attention"),
    )
    config = SimpleNamespace(minimax_sparse_attention_implementation="npu_split_kv_gbsa")
    monkeypatch.setattr(kernel_registry, "IS_NPU_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="requires device_type='npu'"):
        _bind_veomni_ops(module, config)
    monkeypatch.setattr(kernel_registry, "IS_NPU_AVAILABLE", True)
    assert _bind_veomni_ops(module, config)
    assert module.indexer.bound_kernel() is msa_npu.minimax_sparse_indexer_npu
    assert module.attention.bound_kernel() is msa_npu.minimax_sparse_attention_npu


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
@pytest.mark.parametrize("lengths", [(1025,), (0, 129, 257, 0, 513, 1)], ids=["single", "packed"])
def test_cann_native_csr_matches_sorted_reference(lengths):
    ops = pytest.importorskip("cann_ops_transformer.ops")
    if not callable(getattr(ops, "k2q_csr", None)):
        pytest.skip("The native CANN k2q_csr extension is not installed")
    torch.manual_seed(43)
    blocks = (max(lengths) + 127) // 128
    selected = torch.rand(2, sum(lengths), blocks).topk(min(4, blocks), dim=-1).indices.to(torch.int32)
    positions = torch.cat([torch.arange(length) for length in lengths])
    selected.masked_fill_(selected > positions[None, :, None] // 128, -1)
    selected[1] = -1
    cu = _cu_seqlens(lengths)
    counts = (cu.diff() + 127) // 128
    cu_blocks = F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
    kwargs = dict(order_method=1, use_simt=False, q_global_offset=True)
    expected = _FakeMSACannOps().k2q_csr(selected, cu, cu_blocks, **kwargs)
    inputs = [tensor.to("npu") for tensor in (selected, cu, cu_blocks)]
    for sizes in ({}, dict(total_rows=int(cu_blocks[-1]), max_kv=int(counts.max()))):
        actual = ops.k2q_csr(*inputs, **kwargs, **sizes)
        # Exact Q/slot order is required: backward now consumes rows without sorting.
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result.cpu(), reference)


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
@pytest.mark.parametrize("cu_device", ["cpu", "npu"])
@pytest.mark.parametrize("heads", [4, 32], ids=["gqa2", "gqa16"])
@pytest.mark.parametrize(
    ("lengths", "max_seqlen_padding"), [((129, 257, 1), 0), ((128,), 128)], ids=["packed", "padded_bound"]
)
def test_cann_split_kv_gbsa_numerical_forward_and_backward(heads, lengths, max_seqlen_padding, cu_device):
    ops = pytest.importorskip("cann_ops_transformer.ops")
    required = (
        "k2q_csr",
        "npu_minimax_sparse_attention_split_kv",
        "generic_block_sparse_attention_grad_metadata",
        "generic_block_sparse_attention_grad",
    )
    if any(not callable(getattr(ops, name, None)) for name in required):
        pytest.skip("The optional CANN MiniMax operator set is not installed")
    torch.manual_seed(37)
    cu = _cu_seqlens(lengths)
    cpu_inputs = [torch.randn(sum(lengths), h, 128).to(torch.bfloat16).float().requires_grad_() for h in (heads, 2, 2)]
    npu_inputs = [tensor.detach().to(device="npu", dtype=torch.bfloat16).requires_grad_() for tensor in cpu_inputs]
    selected = minimax_sparse_indexer_eager(
        torch.randn(sum(lengths), 2, 128),
        torch.randn(sum(lengths), 1, 128),
        cu,
        max(lengths),
        block_size=128,
        topk_blocks=2,
        local_blocks=1,
    )
    expected = minimax_sparse_attention_eager(*cpu_inputs, selected, cu, max(lengths), block_size=128, scale=0.11)
    actual = msa_npu.minimax_sparse_attention_npu(
        *npu_inputs,
        selected.to("npu"),
        cu.to(cu_device),
        max(lengths) + max_seqlen_padding,
        block_size=128,
        scale=0.11,
    )
    grad = torch.randn_like(expected).to(torch.bfloat16).float()
    expected_grads = torch.autograd.grad(expected, cpu_inputs, grad)
    actual_grads = torch.autograd.grad(actual, npu_inputs, grad.to(device="npu", dtype=torch.bfloat16))
    for value, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(value.float().cpu(), reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
def test_cann_indexer_numerical_scores_and_topk():
    ops = pytest.importorskip("cann_ops_transformer.ops")
    if not callable(getattr(ops, "msa_index_score", None)):
        pytest.skip("The optional CANN msa_index_score operator is not installed")
    torch.manual_seed(41)
    lengths = (129, 257, 1)
    q = torch.randn(sum(lengths), 2, 128).to(torch.bfloat16)
    k = torch.randn(sum(lengths), 1, 128).to(torch.bfloat16)
    cu = _cu_seqlens(lengths)
    q_npu, k_npu, cu_npu = q.to("npu"), k.to("npu"), cu.to("npu")
    actual = ops.msa_index_score(
        query=q_npu,
        key=k_npu,
        start_loc=torch.zeros(3, dtype=torch.int32, device="npu"),
        block_table=None,
        scale=None,
        atten_mask=torch.zeros((2048, 2048), dtype=torch.int8, device=q_npu.device),
        actual_seq_qlen=cu_npu,
        actual_seq_klen=cu_npu,
        layout_key="TND",
        sparse_mode=3,
        init_blocks=0,
        local_blocks=0,
    ).cpu()
    actual.masked_fill_(actual == torch.finfo(torch.float32).min, -torch.inf)
    expected = _index_scores(q, k, lengths)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    kwargs = dict(block_size=128, topk_blocks=16, local_blocks=1)
    selected = msa_npu.minimax_sparse_indexer_npu(q_npu, k_npu, cu, max(lengths), **kwargs).cpu()
    reference = minimax_sparse_indexer_eager(q, k, cu, max(lengths), **kwargs)
    assert torch.equal(selected.sort(-1).values, reference.sort(-1).values)
