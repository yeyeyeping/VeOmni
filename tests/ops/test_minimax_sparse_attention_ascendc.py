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

"""CPU indexing/autograd contracts and optional real-NPU tests for MiniMax MSA."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from veomni.ops.dispatch import OpSlot
from veomni.ops.kernel_registry import KERNEL_REGISTRY
from veomni.ops.kernels.minimax_sparse_attention import (
    ascendc_bindings,
    attention_ascendc,
    minimax_sparse_attention_eager,
    minimax_sparse_indexer_eager,
)
from veomni.ops.kernels.minimax_sparse_attention.indices import (
    build_gbsag_indices,
    build_k2q_csr,
    prepare_metadata,
    select_blocks,
    validate_selection,
)
from veomni.utils.import_utils import is_torch_npu_available


def _cu(lengths):
    return F.pad(torch.tensor(lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))


def _selection(lengths, groups=2, topk=5):
    result = torch.full((sum(lengths), groups, topk), -1, dtype=torch.int32)
    offset = 0
    for length in lengths:
        for q in range(length):
            for group in range(groups):
                candidates = list(range(q // 128, -1, -1))
                if group % 2:
                    candidates.reverse()
                selected = candidates[:topk]
                result[offset + q, group, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
        offset += length
    return result


def _rows(lengths):
    return [
        (b, j)
        for j in range((max(lengths, default=0) + 127) // 128)
        for b, length in enumerate(lengths)
        if j * 128 < length
    ]


@pytest.mark.parametrize("lengths", [(129, 0, 257, 3), (0, 1, 0), (0, 0), (256, 256)])
def test_sparse_adjacencies_preserve_ragged_boundaries_slots_and_sorted_local_q(lengths):
    selection = _selection(lengths)
    cu = _cu(lengths)
    metadata = prepare_metadata(cu, max(lengths), sum(lengths), selection.device)
    validate_selection(selection, metadata, 2)
    row_ptr, q_idx, slots = build_k2q_csr(selection, metadata)
    sparse_idx, counts = build_gbsag_indices(selection, metadata)

    rows = _rows(lengths)
    assert row_ptr.shape == (2, len(rows) + 1)
    assert q_idx.shape == slots.shape == (2, sum(lengths) * 5)
    assert sparse_idx.shape == (len(lengths), 2, (max(lengths) + 127) // 128, max(lengths))
    assert all(t.dtype == torch.int32 for t in (row_ptr, q_idx, slots, sparse_idx, counts))
    for group in range(2):
        for row, (b, block) in enumerate(rows):
            expected = [
                (q + int(cu[b]), slot)
                for q in range(lengths[b])
                for slot in range(selection.shape[-1])
                if selection[q + cu[b], group, slot] == block
            ]
            start, end = row_ptr[group, row : row + 2].tolist()
            assert list(zip(q_idx[group, start:end].tolist(), slots[group, start:end].tolist())) == expected
            local = [q - int(cu[b]) for q, _ in expected]
            assert counts[b, group, block] == len(local)
            assert sparse_idx[b, group, block, : len(local)].tolist() == local
            assert (sparse_idx[b, group, block, len(local) :] == -1).all()
        nnz = int(row_ptr[group, -1])
        assert (q_idx[group, nnz:] == -1).all()
        assert (slots[group, nnz:] == -1).all()
    for b, length in enumerate(lengths):
        assert (counts[b, :, (length + 127) // 128 :] == 0).all()


def test_host_metadata_does_not_follow_default_device():
    cu = _cu([129, 257, 3])
    with torch.device("meta"):
        metadata = prepare_metadata(cu, 257, 389, torch.device("cpu"))
    assert metadata.row_batch.tolist() == [0, 1, 2, 0, 1, 1]
    assert metadata.row_block.tolist() == [0, 0, 0, 1, 1, 2]
    assert torch.equal(metadata.cu_seqlens, cu)


@pytest.mark.parametrize(
    "cu,max_seqlen,total,error",
    [
        ([1, 4], 4, 4, ValueError),
        ([0, 4], 3, 4, ValueError),
        ([0, 4, 3], 4, 3, ValueError),
        ([0, 3], 4, 4, ValueError),
        ([0, 4], torch.tensor(4), 4, TypeError),
    ],
)
def test_rejects_invalid_packed_metadata(cu, max_seqlen, total, error):
    with pytest.raises(error):
        prepare_metadata(torch.tensor(cu, dtype=torch.int32), max_seqlen, total, torch.device("cpu"))


@pytest.mark.parametrize("invalid", ["duplicate", "future", "negative", "empty_row"])
def test_rejects_ambiguous_or_invalid_sparse_patterns(invalid):
    selection = _selection((129,), topk=3)
    metadata = prepare_metadata(_cu([129]), 129, 129, selection.device)
    if invalid == "duplicate":
        selection[0, 0, 1] = 0
    elif invalid == "future":
        selection[0, 0, 0] = 1
    elif invalid == "negative":
        selection[0, 0, 1] = -2
    else:
        selection[0, 0] = -1
    with pytest.raises(ValueError, match="unique causal block ids"):
        validate_selection(selection, metadata, 2)


@pytest.mark.parametrize("local_blocks", [0, 1, 2])
@pytest.mark.parametrize("topk", [2, 8])
def test_raw_score_topk_matches_eager_with_per_token_local_blocks(local_blocks, topk):
    torch.manual_seed(17)
    lengths = (129, 513, 3)
    q = torch.randn(sum(lengths), 2, 16)
    k = torch.randn(sum(lengths), 1, 16)
    cu = _cu(lengths)
    metadata = prepare_metadata(cu, max(lengths), q.shape[0], q.device)
    # Deliberately poison alignment padding and future blocks: the adapter must
    # exclude both even when they have the largest score.
    scores = torch.full((2, q.shape[0], 16), 1e6)
    for b, length in enumerate(lengths):
        start, end = cu[b : b + 2].tolist()
        raw = q[start:end].transpose(0, 1) @ k[start:end].transpose(0, 1).transpose(-1, -2)
        positions = torch.arange(length)
        raw.masked_fill_(positions[None, :] > positions[:, None], -float("inf"))
        raw = F.pad(raw, (0, metadata.num_blocks * 128 - length), value=-float("inf"))
        scores[:, start:end, : metadata.num_blocks] = raw.view(2, length, metadata.num_blocks, 128).amax(-1)
    scores.masked_fill_(torch.arange(16)[None, None, :] > (metadata.positions // 128)[None, :, None], 1e6)
    actual = select_blocks(scores, metadata, topk, local_blocks)
    expected = minimax_sparse_indexer_eager(
        q, k, cu, max(lengths), block_size=128, topk_blocks=topk, local_blocks=local_blocks
    )
    torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values)
    assert actual.dtype == torch.int32
    assert not actual.requires_grad


def _keep_masks(selection, lengths):
    masks = []
    offset = 0
    for length in lengths:
        selected = selection[offset : offset + length]
        blocks = torch.arange(length) // 128
        keep = (selected[:, :, :, None] == blocks[None, None, None, :]).any(2).permute(1, 0, 2)
        causal = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
        masks.append(keep & causal)
        offset += length
    return masks


def _reference(q, k, v, masks, lengths, scale):
    outputs, lses = [], []
    offset = 0
    for length, keep in zip(lengths, masks):
        query = q[offset : offset + length].transpose(0, 1).float()
        key, value = (
            t[offset : offset + length].transpose(0, 1).float().repeat_interleave(q.shape[1] // k.shape[1], 0)
            for t in (k, v)
        )
        mask = keep.repeat_interleave(q.shape[1] // k.shape[1], 0)
        logits = (query @ key.transpose(-1, -2) * scale).masked_fill(~mask, -float("inf"))
        outputs.append((logits.softmax(-1) @ value).transpose(0, 1).to(q.dtype))
        lses.append(logits.logsumexp(-1).transpose(0, 1).unsqueeze(-1))
        offset += length
    return torch.cat(outputs), torch.cat(lses)


def test_split_kv_gbsag_autograd_contract_on_cpu(monkeypatch):
    """Execute the real adapter with independent emulations of both opaque ops."""
    torch.manual_seed(37)
    lengths = (129, 257, 3)
    cu = _cu(lengths)
    selected = _selection(lengths, topk=2)
    seq = prepare_metadata(cu, max(lengths), sum(lengths), selected.device)
    scale = 0.17
    calls = {}

    def forward(q, k, v, row_ptr, q_idx, slot_idx, q_lengths, kv_lengths, **kwargs):
        assert q_lengths.tolist() == kv_lengths.tolist() == list(lengths)
        assert kwargs == dict(
            num_key_value_heads=2,
            scale_value=scale,
            block_size=128,
            top_k=2,
            block_table=None,
            inner_precise=0,
            softmax_lse_flag=True,
            input_layout="TND",
        )
        decoded = torch.full_like(selected, -1)
        for group in range(2):
            for row, (_, block) in enumerate(_rows(lengths)):
                start, end = row_ptr[group, row : row + 2].tolist()
                decoded[q_idx[group, start:end].long(), group, slot_idx[group, start:end].long()] = block
        torch.testing.assert_close(decoded, selected)
        out, lse = _reference(q, k, v, _keep_masks(decoded, lengths), lengths, scale)
        calls.update(out=out, lse=lse)
        return out, lse

    marker = torch.tensor([41], dtype=torch.int32)

    def metadata(idx, count, hq, g, dim, block_shape, **kwargs):
        assert (hq, g, dim, block_shape) == (4, 2, 128, [1, 128])
        assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_kv"] == max(lengths)
        assert torch.equal(kwargs["cu_seqlens_q"], cu)
        assert torch.equal(kwargs["cu_seqlens_kv"], cu)
        for name, value in attention_ascendc._GBSAG_ATTRIBUTES.items():
            assert kwargs[name] == value
        calls["metadata"] = True
        return marker

    def backward(q, k, v, dout, out, lse, idx, count, block_shape, **kwargs):
        assert kwargs["metadata"] is marker
        assert kwargs["softmax_scale"] == scale
        assert dout.is_contiguous()
        assert out.data_ptr() == calls["out"].data_ptr()
        assert lse.data_ptr() == calls["lse"].data_ptr()
        torch.testing.assert_close(out, calls["out"])
        torch.testing.assert_close(lse, calls["lse"])
        assert calls["metadata"] and block_shape == [1, 128]
        decoded = []
        for b, length in enumerate(lengths):
            keep = torch.zeros(2, length, length, dtype=torch.bool)
            for group in range(2):
                for block in range((length + 127) // 128):
                    queries = idx[b, group, block, : count[b, group, block]].long()
                    keep[group, queries, block * 128 : min((block + 1) * 128, length)] = True
            keep &= torch.arange(length)[None, :] <= torch.arange(length)[:, None]
            decoded.append(keep)
        with torch.enable_grad():
            inputs = [t.detach().requires_grad_() for t in (q, k, v)]
            expected, _ = _reference(*inputs, decoded, lengths, scale)
            return torch.autograd.grad(expected, inputs, dout)

    monkeypatch.setattr(
        attention_ascendc, "get_attention_ops", lambda: ascendc_bindings.AttentionOps(forward, metadata, backward)
    )
    # Contiguous copies must preserve gradients to noncontiguous upstream views.
    inputs = [torch.randn(sum(lengths), 128, heads, dtype=torch.bfloat16).requires_grad_() for heads in (4, 2, 2)]
    q, k, v = [t.transpose(1, 2) for t in inputs]
    actual = attention_ascendc._MsaAscendCFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), selected, seq, scale
    )
    expected, _ = _reference(q, k, v, _keep_masks(selected, lengths), lengths, scale)
    dout = torch.randn_like(inputs[0]).transpose(1, 2)
    actual_grads = torch.autograd.grad(actual, inputs, dout)
    expected_grads = torch.autograd.grad(expected, inputs, dout)
    torch.testing.assert_close(actual, expected)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference)


def test_dense_layer_retains_eager_autograd_without_loading_extensions(monkeypatch):
    def unexpected():
        pytest.fail("Dense layers must not load Split-KV/GBSAG.")

    monkeypatch.setattr(attention_ascendc, "get_attention_ops", unexpected)
    q, k, v = (torch.randn(7, heads, 8).requires_grad_() for heads in (4, 2, 2))
    cu = _cu([2, 3, 2])
    actual = attention_ascendc.minimax_sparse_attention_ascendc(q, k, v, None, cu, 3, block_size=None, scale=0.25)
    expected = minimax_sparse_attention_eager(q, k, v, None, cu, 3, block_size=None, scale=0.25)
    torch.testing.assert_close(actual, expected)
    assert all(torch.isfinite(grad).all() for grad in torch.autograd.grad(actual.sum(), (q, k, v)))


def test_backend_registration_config_and_binding(monkeypatch):
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops
    from veomni.ops import kernel_registry
    from veomni.utils import import_utils

    eager = dict(
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        rotary_pos_emb_vision_implementation="eager",
        swiglu_mlp_implementation="eager",
        load_balancing_loss_implementation="eager",
        cross_entropy_loss_implementation="eager",
        moe_implementation="eager",
    )
    assert OpsImplementationConfig(**eager).minimax_sparse_attention_implementation == "eager"
    monkeypatch.setattr(import_utils, "is_torch_npu_available", lambda: False)
    with pytest.raises(ValueError, match="requires Ascend NPU"):
        OpsImplementationConfig(**eager, minimax_sparse_attention_implementation="npu_cann")
    monkeypatch.setattr(import_utils, "is_torch_npu_available", lambda: True)
    config = OpsImplementationConfig(**eager, minimax_sparse_attention_implementation="npu_cann")
    with pytest.raises(ValueError, match="not supported on Ascend NPU"):
        OpsImplementationConfig(**eager, minimax_sparse_attention_implementation="unknown")
    monkeypatch.setattr(kernel_registry, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(ascendc_bindings, "get_index_score", lambda: None)
    monkeypatch.setattr(ascendc_bindings, "get_attention_ops", lambda: None)
    module = SimpleNamespace(
        indexer=OpSlot("minimax_sparse_attention", "indexer"),
        attention=OpSlot("minimax_sparse_attention", "attention"),
    )
    assert _bind_veomni_ops(module, config)
    assert module.indexer.use_non_eager_impl and module.attention.use_non_eager_impl
    for variant in ("indexer", "attention"):
        assert "npu_cann" in KERNEL_REGISTRY.list_available("minimax_sparse_attention", variant)
    config.minimax_sparse_attention_implementation = "eager"
    assert _bind_veomni_ops(module, config)
    assert not module.indexer.use_non_eager_impl and not module.attention.use_non_eager_impl


@pytest.mark.parametrize("failure", ["missing_package", "missing_operator"])
def test_missing_extension_errors_identify_dependency(monkeypatch, failure):
    monkeypatch.setattr(ascendc_bindings, "require_npu", lambda: None)

    def load(module):
        if failure == "missing_package":
            raise ModuleNotFoundError(module)
        return SimpleNamespace()

    monkeypatch.setattr(ascendc_bindings, "import_module", load)
    with pytest.raises(RuntimeError, match="msa_index_score"):
        ascendc_bindings._load_callable("cann_ops_transformer.ops.attention.msa_index_score", "msa_index_score")


def _require_extensions():
    pytest.importorskip("cann_ops_transformer", reason="Requires ops-transformer CANN bindings")
    pytest.importorskip("custom_ops", reason="Requires the separate Split-KV extension")


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
@pytest.mark.parametrize("local_blocks", [0, 1, 2])
def test_real_npu_indexer_matches_eager(local_blocks):
    from veomni.ops.kernels.minimax_sparse_attention.indexer_ascendc import minimax_sparse_indexer_ascendc

    _require_extensions()
    torch.manual_seed(53)
    lengths = (129, 513, 3)
    q = torch.randn(sum(lengths), 2, 128).bfloat16()
    k = torch.randn(sum(lengths), 1, 128).bfloat16()
    cu = _cu(lengths)
    kwargs = dict(block_size=128, topk_blocks=2, local_blocks=local_blocks)
    expected = minimax_sparse_indexer_eager(q.float(), k.float(), cu, max(lengths), **kwargs)
    actual = minimax_sparse_indexer_ascendc(q.to("npu"), k.to("npu"), cu, max(lengths), **kwargs)
    torch.testing.assert_close(actual.cpu().sort(-1).values, expected.sort(-1).values)


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
@pytest.mark.parametrize("group_ratio", [1, 2])
def test_real_npu_split_kv_gbsag_matches_reference(group_ratio):
    _require_extensions()
    torch.manual_seed(71)
    lengths = (129, 257, 3)
    selection = _selection(lengths, topk=2)
    cu = _cu(lengths)
    hq, groups = 2 * group_ratio, 2
    cpu = [torch.randn(sum(lengths), heads, 128).bfloat16().float().requires_grad_() for heads in (hq, groups, groups)]
    npu = [t.detach().to(device="npu", dtype=torch.bfloat16).requires_grad_() for t in cpu]
    scale = 0.13
    masks = _keep_masks(selection, lengths)
    expected, expected_lse = _reference(*cpu, masks, lengths, scale)
    actual = attention_ascendc.minimax_sparse_attention_ascendc(
        *npu, selection.to("npu"), cu, max(lengths), block_size=128, scale=scale
    )
    # Verify the exact forward LSE consumed by GBSAG as well as its output.
    seq = prepare_metadata(cu, max(lengths), sum(lengths), npu[0].device)
    row_ptr, q_idx, slot_idx = build_k2q_csr(selection.to("npu"), seq)
    out, lse = ascendc_bindings.get_attention_ops().forward(
        *npu,
        row_ptr,
        q_idx,
        slot_idx,
        seq.lengths,
        seq.lengths,
        num_key_value_heads=groups,
        scale_value=scale,
        block_size=128,
        top_k=2,
        block_table=None,
        inner_precise=0,
        softmax_lse_flag=True,
        input_layout="TND",
    )
    torch.testing.assert_close(out, actual)
    torch.testing.assert_close(lse.float().cpu(), expected_lse, atol=2e-2, rtol=2e-2)
    dout = torch.randn_like(expected).bfloat16().float()
    actual_grads = torch.autograd.grad(actual, npu, dout.to(device="npu", dtype=torch.bfloat16))
    expected_grads = torch.autograd.grad(expected, cpu, dout)
    for result, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(result.float().cpu(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
def test_real_npu_m3_sparse_and_dense_optimizer_steps(monkeypatch):
    import importlib

    pytest.importorskip("transformers", minversion="5.16.0")
    _require_extensions()
    from veomni.ops.kernels.cross_entropy import install_loss_mapping

    modeling = importlib.import_module(
        "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_npu"
    )
    for name, variant in (("veomni_msa_indexer", "indexer"), ("veomni_msa_attention", "attention")):
        slot = OpSlot("minimax_sparse_attention", variant)
        slot.bind("npu_cann")
        monkeypatch.setattr(modeling, name, slot)
    install_loss_mapping("eager")
    torch.manual_seed(91)
    config = modeling.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=16,
        dense_intermediate_size=32,
        shared_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        index_n_heads=2,
        index_head_dim=128,
        index_block_size=128,
        index_topk_blocks=2,
        index_local_blocks=1,
        head_dim=128,
        rotary_dim=128,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["minimax_m3_sparse", "full_attention"],
        mlp_layer_types=["dense", "dense"],
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling.MiniMaxM3VLForCausalLM(config).to(device="npu", dtype=torch.bfloat16).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    lengths = (129, 257, 3)
    cu = _cu(lengths)
    tokens = torch.randint(1, config.vocab_size, (1, sum(lengths)), device="npu")
    positions = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0).to("npu")
    before = model.lm_head.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(
            input_ids=tokens,
            labels=tokens,
            position_ids=positions,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=max(lengths),
            max_length_k=max(lengths),
            use_cache=False,
        )
        assert torch.isfinite(result.loss)
        result.loss.backward()
        for layer in model.model.layers:
            for projection in ("q_proj", "k_proj", "v_proj"):
                grad = getattr(layer.self_attn, projection).weight.grad
                assert grad is not None and torch.isfinite(grad).all()
        optimizer.step()
    assert not torch.equal(before, model.lm_head.weight.detach())
