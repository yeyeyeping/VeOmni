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
from veomni.ops.kernels.cross_entropy import install_loss_mapping
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


_MODELING_MODULES = (
    "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_gpu",
    "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_npu",
)

pytestmark = pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="MiniMax M3 VL generated modeling requires transformers>=5.12.0.",
)


class _RecordingSlot:
    use_non_eager_impl = True

    def __init__(self, output):
        self.output = output
        self.args = None

    def __call__(self, *args):
        self.args = args
        return self.output


class _RecordingVisionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, hidden_states, attention_mask, **kwargs):
        self.kwargs = {"attention_mask": attention_mask, **kwargs}
        return hidden_states


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_for_causal_lm_unpacks_veomni_loss_contract(module_name):
    modeling = importlib.import_module(module_name)
    install_loss_mapping("eager")
    config = modeling.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rotary_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["full_attention"],
        mlp_layer_types=["sparse"],
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling.MiniMaxM3VLForCausalLM(config)
    input_ids = torch.tensor([[1, 4, 5, 2]])
    labels = torch.tensor([[4, 5, 2, -100]])

    output = model(
        input_ids=input_ids,
        labels=labels,
        position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0),
        cu_seq_lens_q=torch.tensor([0, input_ids.shape[1]], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, input_ids.shape[1]], dtype=torch.int32),
        max_length_q=input_ids.shape[1],
        max_length_k=input_ids.shape[1],
        use_cache=False,
    )

    assert isinstance(output.loss, torch.Tensor)
    assert output.loss.ndim == 0
    assert output.logits.shape == (1, input_ids.shape[1], config.vocab_size)
    assert output.fused_linear_aux is None
    assert not hasattr(output, "aux_loss")
    assert not hasattr(output, "router_logits")
    assert "router_logits" not in model._can_record_outputs
    output.loss.backward()
    assert model.lm_head.weight.grad is not None


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_rejects_incompatible_router_aux_loss(module_name):
    modeling = importlib.import_module(module_name)
    config = modeling.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rotary_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["full_attention"],
        mlp_layer_types=["sparse"],
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling.MiniMaxM3VLForCausalLM(config)

    with pytest.raises(ValueError, match="router auxiliary loss is disabled"):
        model(
            input_ids=torch.tensor([[1, 4, 5, 2]]),
            output_router_logits=True,
            use_cache=False,
        )

    assert not hasattr(modeling, "load_balancing_loss_func")


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_text_model_keeps_decoder_packed_and_unpacks_only_attention(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = modeling.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rotary_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["minimax_m3_sparse"],
        mlp_layer_types=["dense"],
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling.MiniMaxM3VLTextModel(config)
    model.eval()
    decoder_inputs = []
    indexer_inputs = []
    attention_query_shapes = []

    def record_decoder_input(_, args, kwargs):
        decoder_inputs.append(
            {
                "shape": args[0].shape,
                "attention_mask": kwargs["attention_mask"],
                "position_ids": kwargs["position_ids"].detach().clone(),
                "extra_kwargs": set(kwargs) - {"attention_mask", "position_ids", "past_key_values"},
            }
        )

    def record_indexer_input(_, args, __):
        indexer_inputs.append(args[0].shape)

    eager_bsnd_attention_forward = modeling._eager_bsnd_attention_forward

    def record_attention(query, key, value, attention_mask, **kwargs):
        attention_query_shapes.append(query.shape)
        return eager_bsnd_attention_forward(query, key, value, attention_mask, **kwargs)

    monkeypatch.setattr(modeling, "_eager_bsnd_attention_forward", record_attention)

    assert model.layers[0].self_attn.indexer is not None

    first_hidden_states = torch.randn(1, 2, config.hidden_size)
    second_hidden_states = torch.randn(1, 3, config.hidden_size)
    packed_hidden_states = torch.cat((first_hidden_states, second_hidden_states), dim=1)
    packed_position_ids = torch.tensor([[0, 1, 0, 1, 2]])

    with torch.no_grad():
        expected = torch.cat(
            (
                model(
                    inputs_embeds=first_hidden_states,
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    position_ids=torch.arange(2).unsqueeze(0),
                    use_cache=False,
                ).last_hidden_state,
                model(
                    inputs_embeds=second_hidden_states,
                    attention_mask=torch.ones(1, 3, dtype=torch.long),
                    position_ids=torch.arange(3).unsqueeze(0),
                    use_cache=False,
                ).last_hidden_state,
            ),
            dim=1,
        )

        # The shared config controls ViT dispatch only. Packed MiniMax language
        # attention must keep using the M3-specific path below.
        model.config._attn_implementation = "veomni_flash_attention_2_with_sp"
        hook = model.layers[0].register_forward_pre_hook(record_decoder_input, with_kwargs=True)
        indexer_hook = model.layers[0].self_attn.indexer.register_forward_pre_hook(
            record_indexer_input, with_kwargs=True
        )
        attention_query_shapes.clear()
        output = model(
            inputs_embeds=packed_hidden_states,
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            position_ids=packed_position_ids,
            cu_seq_lens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_length_q=3,
            max_length_k=3,
            use_cache=False,
        )
        hook.remove()
        indexer_hook.remove()

    assert decoder_inputs[0]["shape"] == (1, 5, config.hidden_size)
    assert decoder_inputs[0]["position_ids"].tolist() == [[0, 1, 0, 1, 2]]
    assert decoder_inputs[0]["attention_mask"] is None
    assert indexer_inputs == [(1, 5, config.hidden_size)]
    assert attention_query_shapes == [(2, config.num_attention_heads, 3, config.head_dim)]
    assert "cu_seq_lens_q" not in decoder_inputs[0]["extra_kwargs"]
    assert "cu_seq_lens_k" not in decoder_inputs[0]["extra_kwargs"]
    assert "max_length_q" not in decoder_inputs[0]["extra_kwargs"]
    assert "max_length_k" not in decoder_inputs[0]["extra_kwargs"]
    assert "packed_to_padded_indices" in decoder_inputs[0]["extra_kwargs"]
    assert "packed_padding_mask" in decoder_inputs[0]["extra_kwargs"]
    assert output.last_hidden_state.shape == (1, 5, config.hidden_size)
    torch.testing.assert_close(output.last_hidden_state, expected)

    packed_grad_input = packed_hidden_states.detach().clone().requires_grad_(True)
    packed_grad_output = model(
        inputs_embeds=packed_grad_input,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        position_ids=packed_position_ids,
        cu_seq_lens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_length_q=3,
        max_length_k=3,
        use_cache=False,
    ).last_hidden_state
    (packed_grad,) = torch.autograd.grad(packed_grad_output.square().sum(), packed_grad_input)

    first_grad_input = first_hidden_states.detach().clone().requires_grad_(True)
    second_grad_input = second_hidden_states.detach().clone().requires_grad_(True)
    first_grad_output = model(
        inputs_embeds=first_grad_input,
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        position_ids=torch.arange(2).unsqueeze(0),
        use_cache=False,
    ).last_hidden_state
    second_grad_output = model(
        inputs_embeds=second_grad_input,
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        position_ids=torch.arange(3).unsqueeze(0),
        use_cache=False,
    ).last_hidden_state
    first_grad, second_grad = torch.autograd.grad(
        first_grad_output.square().sum() + second_grad_output.square().sum(),
        (first_grad_input, second_grad_input),
    )
    torch.testing.assert_close(packed_grad, torch.cat((first_grad, second_grad), dim=1))


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_text_model_single_sequence_preserves_packed_shape(module_name):
    """A single packed sample stays packed across every decoder layer."""
    modeling = importlib.import_module(module_name)
    config = modeling.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rotary_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["minimax_m3_sparse"],
        mlp_layer_types=["dense"],
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling.MiniMaxM3VLTextModel(config)
    model.eval()
    decoder_inputs = []

    def record_decoder_input(_, args, kwargs):
        decoder_inputs.append({"shape": args[0].shape})

    hook = model.layers[0].register_forward_pre_hook(record_decoder_input, with_kwargs=True)

    single_hidden_states = torch.randn(1, 5, config.hidden_size)
    single_position_ids = torch.arange(5).unsqueeze(0)
    with torch.no_grad():
        output = model(
            inputs_embeds=single_hidden_states,
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            position_ids=single_position_ids,
            cu_seq_lens_q=torch.tensor([0, 5], dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, 5], dtype=torch.int32),
            max_length_q=5,
            max_length_k=5,
            use_cache=False,
        )
    hook.remove()

    assert decoder_inputs[0]["shape"] == (1, 5, config.hidden_size)
    assert output.last_hidden_state.shape == (1, 5, config.hidden_size)

    # Single-sample packed output matches the no-FA-kwargs path (the fast path
    # is a pure pass-through, so dropping cu_seq_lens_q must give the same result).
    with torch.no_grad():
        reference = model(
            inputs_embeds=single_hidden_states,
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            position_ids=single_position_ids,
            use_cache=False,
        )
    torch.testing.assert_close(output.last_hidden_state, reference.last_hidden_state)


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_packed_layout_round_trips_sp_padding_segment(module_name):
    modeling = importlib.import_module(module_name)
    packed = torch.arange(14, dtype=torch.float32).view(1, 7, 2)

    packed_to_padded_indices, padding_mask = modeling._prepare_packed_layout(
        torch.tensor([0, 2, 5, 7], dtype=torch.int32),
        max_sequence_length=3,
        total_length=7,
        device=packed.device,
    )
    padded = modeling._unpack_to_bsnd(packed, packed_to_padded_indices, padding_mask)
    restored = modeling._pack_from_bsnd(padded, packed_to_padded_indices)

    assert packed_to_padded_indices.tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert padding_mask.tolist() == [
        [False, False, True],
        [False, False, False],
        [False, False, True],
    ]
    assert padded.shape == (3, 3, 2)
    assert torch.equal(restored, packed)


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_full_attention_packed_matches_separate_samples(module_name):
    modeling = importlib.import_module(module_name)
    config = modeling.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rotary_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["full_attention"],
        mlp_layer_types=["dense"],
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling.MiniMaxM3VLTextModel(config).eval()
    first = torch.randn(1, 2, config.hidden_size)
    second = torch.randn(1, 3, config.hidden_size)

    with torch.no_grad():
        expected = torch.cat(
            (
                model(
                    inputs_embeds=first,
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    position_ids=torch.arange(2).unsqueeze(0),
                    use_cache=False,
                ).last_hidden_state,
                model(
                    inputs_embeds=second,
                    attention_mask=torch.ones(1, 3, dtype=torch.long),
                    position_ids=torch.arange(3).unsqueeze(0),
                    use_cache=False,
                ).last_hidden_state,
            ),
            dim=1,
        )
        actual = model(
            inputs_embeds=torch.cat((first, second), dim=1),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            position_ids=torch.tensor([[0, 1, 0, 1, 2]]),
            cu_seq_lens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_length_q=3,
            max_length_k=3,
            use_cache=False,
        ).last_hidden_state

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_packed_layout_is_prepared_once_per_model_forward(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = modeling.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rotary_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["full_attention", "minimax_m3_sparse"],
        mlp_layer_types=["dense", "dense"],
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling.MiniMaxM3VLTextModel(config).eval()
    original_prepare = modeling._prepare_packed_layout
    calls = []

    def record_prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(modeling, "_prepare_packed_layout", record_prepare)
    with torch.no_grad():
        output = model(
            inputs_embeds=torch.randn(1, 5, config.hidden_size),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            position_ids=torch.tensor([[0, 1, 0, 1, 2]]),
            cu_seq_lens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_length_q=3,
            max_length_k=3,
            use_cache=False,
        )

    assert output.last_hidden_state.shape == (1, 5, config.hidden_size)
    assert len(calls) == 1


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_language_sp_communicates_before_unpack_and_restores_packed_output(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = modeling.MiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rotary_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["minimax_m3_sparse"],
        mlp_layer_types=["dense"],
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling.MiniMaxM3VLTextModel(config).eval()
    group = object()
    parallel_state = SimpleNamespace(
        ulysses_enabled=True,
        ulysses_size=2,
        ulysses_group=group,
        cp_enabled=False,
    )
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: parallel_state)
    calls = []

    def fake_gather(tensor, gather_dim, group):
        calls.append(("index_gather", tuple(tensor.shape), gather_dim, group))
        return torch.cat((tensor, tensor), dim=gather_dim)

    def fake_prepare(query, key, value, *, group, ulysses_size):
        calls.append(("main_prepare", tuple(query.shape), group, ulysses_size))
        query = torch.cat((query, query), dim=1)
        key = torch.cat((key, key), dim=1)
        value = torch.cat((value, value), dim=1)
        if key.shape[2] < ulysses_size:
            repeats = ulysses_size // key.shape[2]
            key = torch.repeat_interleave(key, repeats, dim=2)
            value = torch.repeat_interleave(value, repeats, dim=2)
        return (
            query[:, :, : query.shape[2] // ulysses_size],
            key[:, :, : key.shape[2] // ulysses_size],
            value[:, :, : value.shape[2] // ulysses_size],
            config.num_attention_heads,
        )

    def fake_restore(output, *, group):
        calls.append(("main_restore", tuple(output.shape), group))
        return torch.cat((output, output), dim=2)[:, :2]

    monkeypatch.setattr(modeling, "gather_outputs", fake_gather)
    monkeypatch.setattr(modeling, "prepare_ulysses_qkv", fake_prepare)
    monkeypatch.setattr(modeling, "restore_ulysses_output", fake_restore)

    decoder_shapes = []
    hook = model.layers[0].register_forward_hook(
        lambda _, args, output: decoder_shapes.append((args[0].shape, output.shape))
    )
    with torch.no_grad():
        output = model(
            inputs_embeds=torch.randn(1, 2, config.hidden_size),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            position_ids=torch.arange(2).unsqueeze(0),
            cu_seq_lens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, 2, 4], dtype=torch.int32),
            max_length_q=2,
            max_length_k=2,
            use_cache=False,
        )
    hook.remove()

    assert decoder_shapes == [((1, 2, config.hidden_size), (1, 2, config.hidden_size))]
    assert output.last_hidden_state.shape == (1, 2, config.hidden_size)
    assert [call[0] for call in calls] == ["index_gather", "index_gather", "main_prepare", "main_restore"]
    assert calls[0][1][1] == 2
    assert calls[2][1][1] == 2
    assert calls[3][1][1] == 4
    assert all(call[-1] is group or call[0] == "main_prepare" for call in calls)


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_metadata_preserves_visual_batch_boundaries_and_isolates_sp_padding(module_name):
    modeling = importlib.import_module(module_name)
    batch = {
        "image_grid_thw": torch.tensor([[1, 2, 2], [2, 1, 2]], dtype=torch.long),
        "video_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
    }

    modeling.collate_multimodal_metadata(
        batch,
        {"pixel_values": 4, "pixel_values_videos": 8},
    )

    metadata = batch["multimodal_metadata"]
    assert metadata["vit_image_cu_seqlens"].tolist() == [0, 4, 8, 12]
    assert metadata["vit_image_max_seqlen"] == 4
    assert metadata["vit_video_cu_seqlens"].tolist() == [0, 4, 12]
    assert metadata["vit_video_max_seqlen"] == 8


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
@pytest.mark.parametrize("precompute_attention_metadata", [False, True])
def test_minimax_m3_vl_vision_forward_threads_global_sp_metadata(
    module_name,
    precompute_attention_metadata,
    monkeypatch,
):
    modeling = importlib.import_module(module_name)
    config = modeling.MiniMaxM3VLVisionConfig(
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
    )
    vision_model = modeling.MiniMaxM3VLVisionModel(config)
    vision_model.config._attn_implementation = "veomni_flash_attention_2_with_sp"
    recording_layer = _RecordingVisionLayer()
    vision_model.layers = torch.nn.ModuleList([recording_layer])

    parallel_state = SimpleNamespace(
        sp_enabled=True,
        sp_size=2,
        ulysses_enabled=True,
        cp_enabled=False,
    )
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: parallel_state)

    def fake_sp_pad_and_slice(tensor, dim, pad_value, pad_scale):
        assert dim == 0
        assert pad_value == 0
        assert pad_scale == config.spatial_merge_size**2
        alignment = parallel_state.sp_size * pad_scale
        pad_size = (-tensor.shape[dim]) % alignment
        if pad_size > 0:
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat((tensor, pad), dim=dim)
        return tensor.narrow(dim, 0, tensor.shape[dim] // parallel_state.sp_size)

    monkeypatch.setattr(modeling, "sp_pad_and_slice", fake_sp_pad_and_slice)

    grid_thw = torch.tensor([[1, 2, 2], [2, 2, 2]], dtype=torch.long)
    vit_metadata = {"grid_thw_list": [[1, 2, 2], [2, 2, 2]]}
    if precompute_attention_metadata:
        vit_metadata.update(
            {
                "cu_seqlens": torch.tensor([0, 4, 12, 16], dtype=torch.int32),
                "max_seqlen": 8,
            }
        )

    pixel_values = torch.randn(8, config.num_channels * config.temporal_patch_size * config.patch_size**2)
    output = vision_model(
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
        vit_metadata=vit_metadata,
    )

    assert output.last_hidden_state.shape == (1, 8, config.hidden_size)
    assert recording_layer.kwargs["attention_mask"] is None
    assert recording_layer.kwargs["cu_seq_lens_q"].tolist() == [0, 4, 12, 16]
    assert recording_layer.kwargs["cu_seq_lens_k"].tolist() == [0, 4, 12, 16]
    assert recording_layer.kwargs["max_length_q"] == 8
    assert recording_layer.kwargs["max_length_k"] == 8
    cos, sin = recording_layer.kwargs["position_embeddings"]
    assert cos.shape[0] == sin.shape[0] == 8


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_allows_veomni_flash_attention_for_vision(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    transformers_modeling = importlib.import_module("transformers.modeling_utils")
    monkeypatch.setattr(transformers_modeling, "lazy_import_flash_attention", lambda *args, **kwargs: None)
    implementation = "veomni_flash_attention_2_with_sp"
    config = modeling.MiniMaxM3VLVisionConfig(
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
    )
    config._attn_implementation = implementation

    model = modeling.MiniMaxM3VLVisionModel(config)

    assert implementation in modeling.MiniMaxM3VLPreTrainedModel._compatible_flash_implementations
    assert model.config._attn_implementation == implementation


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
def test_minimax_m3_vl_model_classes_bind_their_matching_parallel_plan(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    text_config = SimpleNamespace(num_local_experts=4)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))

    text_plan = modeling.MiniMaxM3VLForCausalLM.get_parallel_plan(SimpleNamespace(config=text_config))
    vlm_plan = modeling.MiniMaxM3SparseForConditionalGeneration.get_parallel_plan(
        SimpleNamespace(config=SimpleNamespace(text_config=text_config))
    )

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
def test_minimax_m3_vl_parallel_plan_rejects_eager_ep(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = SimpleNamespace(
        num_local_experts=4,
    )
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True, ep_size=2))

    with pytest.raises(RuntimeError, match="requires.*fused_triton or fused_npu"):
        modeling._validate_minimax_m3_ep(config)


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_parallel_plan_requires_expert_count_divisible_by_ep(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = SimpleNamespace(
        num_local_experts=3,
    )
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True, ep_size=2))

    with pytest.raises(ValueError, match="num_experts=3.*ep_size=2"):
        modeling._validate_minimax_m3_ep(config)


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
