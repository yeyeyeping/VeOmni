from types import SimpleNamespace

import pytest
import torch

from veomni.models.transformers.minimax_m3_vl import parallel_plan
from veomni.models.transformers.minimax_m3_vl.checkpoint_tensor_converter import (
    MiniMaxM3VLCheckpointTensorConverter,
    convert_minimax_m3_vl_fqn_to_index_mapping,
    create_minimax_m3_vl_checkpoint_tensor_converter,
)


def test_minimax_public_dense_and_projector_mapping():
    converter = MiniMaxM3VLCheckpointTensorConverter(num_experts=2)
    tensor = torch.ones(3, 4)
    assert converter.convert("language_model.model.embed_tokens.weight", tensor).name == (
        "model.language_model.embed_tokens.weight"
    )
    assert converter.convert("patch_merge_mlp.linear_1.weight", tensor).name == (
        "model.multi_modal_projector.merge_linear_1.weight"
    )


def test_minimax_gate_up_and_expert_stacking():
    converter = MiniMaxM3VLCheckpointTensorConverter(num_experts=2)
    gate = torch.ones(3, 4)
    up = torch.full((3, 4), 2.0)
    assert converter.convert("language_model.model.layers.0.mlp.gate_proj.weight", gate) is None
    merged = converter.convert("language_model.model.layers.0.mlp.up_proj.weight", up)
    assert merged.name == "model.language_model.layers.0.mlp.gate_up_proj.weight"
    assert torch.equal(merged.tensor, torch.cat([gate, up], dim=0))

    for expert, value in enumerate((1.0, 2.0)):
        assert (
            converter.convert(
                f"language_model.model.layers.1.block_sparse_moe.experts.{expert}.w1.weight",
                torch.full((3, 4), value),
            )
            is None
        )
    for expert, value in enumerate((3.0, 4.0)):
        result = converter.convert(
            f"language_model.model.layers.1.block_sparse_moe.experts.{expert}.w3.weight",
            torch.full((3, 4), value),
        )
    assert result.name == "model.language_model.layers.1.mlp.experts.gate_up_proj"
    assert result.tensor.shape == (2, 6, 4)


def test_minimax_converter_factory_and_incomplete_checkpoint():
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(num_local_experts=3)))
    assert create_minimax_m3_vl_checkpoint_tensor_converter(model).num_experts == 3
    converter = MiniMaxM3VLCheckpointTensorConverter(num_experts=2)
    converter.convert("language_model.model.layers.0.mlp.gate_proj.weight", torch.ones(3, 4))
    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        converter.finalize()


def test_minimax_fqn_index_mapping_merges_layouts():
    mapping = {
        "language_model.model.layers.0.mlp.gate_proj.weight": 2,
        "language_model.model.layers.0.mlp.up_proj.weight": 3,
        "language_model.model.layers.0.mlp.down_proj.weight": 4,
        "patch_merge_mlp.linear_2.weight": 8,
    }
    converted = convert_minimax_m3_vl_fqn_to_index_mapping(mapping)
    assert converted["model.language_model.layers.0.mlp.gate_up_proj.weight"] == 2
    assert converted["model.language_model.layers.0.mlp.down_proj.weight"] == 4
    assert converted["model.multi_modal_projector.merge_linear_2.weight"] == 8


def test_minimax_parallel_plans_use_architecture_specific_prefixes():
    text_plan = parallel_plan.get_text_parallel_plan()
    vlm_plan = parallel_plan.get_vlm_parallel_plan()

    assert set(text_plan.extra_parallel_plan["ep"]) == {
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    }
    assert text_plan.extra_parallel_fsdp_no_shard_module["ep"] == {"model.layers.*.mlp.experts"}

    assert set(vlm_plan.extra_parallel_plan["ep"]) == {
        "model.language_model.layers.*.mlp.experts.gate_up_proj",
        "model.language_model.layers.*.mlp.experts.down_proj",
    }
    assert vlm_plan.extra_parallel_fsdp_no_shard_module["ep"] == {"model.language_model.layers.*.mlp.experts"}
