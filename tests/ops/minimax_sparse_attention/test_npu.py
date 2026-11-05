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

"""Experimental CANN adapters, independent CPU references and optional native NPU parity."""

import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops.kernels.minimax_sparse_attention import minimax_sparse_attention_eager, minimax_sparse_indexer_eager
from veomni.ops.kernels.minimax_sparse_attention import npu as msa_npu
from veomni.utils.import_utils import is_torch_npu_available


def _cu_seqlens(lengths):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


def _index_scores(q, k, lengths):
    """Independent per-sample FP32 score reference, including CANN stride padding."""
    blocks = (max(lengths) + 127) // 128
    stride = (blocks + 15) // 16 * 16
    out = torch.full((q.shape[1], sum(lengths), stride), -torch.inf, device=q.device)
    offset = 0
    for length in lengths:
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
        self.forward_lengths = lengths.clone()
        return _attention_from_keep(q, k, v, keep, kwargs["scale_value"])

    def generic_block_sparse_attention_grad_metadata(self, idx, count, hq, hkv, dim, block_shape, **kwargs):
        self.calls.append("metadata")
        assert block_shape == [1, 128]
        assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_kv"] == idx.shape[-1]
        assert kwargs["cu_seqlens_q"].dtype == kwargs["cu_seqlens_kv"].dtype == torch.int64
        assert idx.dtype == count.dtype == torch.int32
        for name in ("seqused_q", "seqused_kv"):
            lengths = kwargs[name]
            assert isinstance(lengths, torch.Tensor)
            assert lengths.shape == (idx.shape[0],) and lengths.is_contiguous()
            torch.testing.assert_close(lengths, self.forward_lengths)
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
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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


@pytest.mark.parametrize("lengths", [(129, 257, 1), (1,), (128,)])
@pytest.mark.parametrize("max_seqlen_padding", [0, 256])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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
    [(129, 257, 1), (128,), (1,), (1, 2, 3) * 7],
    ids=["packed", "single", "single_token", "packed_21"],
)
@pytest.mark.parametrize("max_seqlen_padding", [0, 128])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_cann_dropout_uses_eager_without_loading_cann(monkeypatch):
    q = torch.randn(5, 4, 8, requires_grad=True)
    k = torch.randn(5, 2, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    indices = torch.zeros(5, 2, 3, dtype=torch.int32)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    kwargs = dict(block_size=2, scale=0.3, dropout_p=0.2)
    torch.manual_seed(19)
    actual = msa_npu.minimax_sparse_attention_npu(q, k, v, indices, _cu_seqlens((2, 3)), 3, **kwargs)
    torch.manual_seed(19)
    expected = minimax_sparse_attention_eager(q, k, v, indices, _cu_seqlens((2, 3)), 3, **kwargs)
    torch.testing.assert_close(actual, expected)
    assert all(torch.isfinite(grad).all() for grad in torch.autograd.grad(actual.sum(), (q, k, v)))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("variant", ["indexer", "attention"])
def test_cann_registry_gates_hardware_and_dependencies(monkeypatch, variant):
    from veomni.arguments import OpsImplementationConfig
    from veomni.ops import OP_REGISTRY, registry
    from veomni.ops.platform import NpuKernelRequirement

    assert OpsImplementationConfig.__dataclass_fields__["minimax_sparse_attention_implementation"].default == "eager"
    assert set(OP_REGISTRY.list_registered("minimax_sparse_attention", variant)) == {"eager", "npu"}
    monkeypatch.setattr(registry, "get_device_type", lambda: "cpu")
    assert OP_REGISTRY.list_available("minimax_sparse_attention", variant) == ["eager"]
    with pytest.raises(RuntimeError, match="not registered for device"):
        OP_REGISTRY.resolve("minimax_sparse_attention", variant, "npu")
    with pytest.raises(KeyError, match="Unknown op"):
        OP_REGISTRY.resolve("minimax_sparse_attention", variant, "missing")
    monkeypatch.setattr(registry, "get_device_type", lambda: "npu")
    monkeypatch.setattr(NpuKernelRequirement, "check", lambda self: None)
    monkeypatch.setattr(registry, "is_package_available", lambda name: False)
    with pytest.raises(RuntimeError, match="cann_ops_transformer"):
        OP_REGISTRY.resolve("minimax_sparse_attention", variant, "npu")
    monkeypatch.setattr(registry, "is_package_available", lambda name: True)
    entry = OP_REGISTRY.resolve("minimax_sparse_attention", variant, "npu")
    expected = msa_npu.minimax_sparse_indexer_npu if variant == "indexer" else msa_npu.minimax_sparse_attention_npu
    assert entry.wrapper is expected


@pytest.mark.skipif(not is_torch_npu_available(), reason="Requires Ascend NPU")
@pytest.mark.parametrize("lengths", [(1025,), (129, 257, 513, 1)], ids=["single", "packed"])
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
    ("lengths", "max_seqlen_padding"),
    [((129, 257, 1), 0), ((128,), 128), ((1, 2, 3) * 7, 0)],
    ids=["packed", "padded_bound", "packed_21"],
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
