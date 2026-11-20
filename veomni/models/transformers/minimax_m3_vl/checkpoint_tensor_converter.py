# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
# Copyright 2026 verl contributors
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

"""
Runtime checkpoint tensor converter for MiniMax M3 VL public checkpoints.

The public MiniMaxAI/MiniMax-M3 safetensors index uses an older layout:

    language_model.model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
    language_model.model.layers.{i}.block_sparse_moe.experts.{j}.w{1,2,3}.weight
    language_model.model.layers.{i}.block_sparse_moe.shared_experts.{gate_proj,up_proj,down_proj}.weight
    language_model.model.layers.{i}.self_attn.index_{q,k}_{proj,norm}.weight
    language_model.model.layers.{i}.block_sparse_moe.e_score_correction_bias

The standalone MiniMax model uses the Transformers 5.16 checkpoint layout:

    model.language_model.layers.{i}.mlp.gate_up_proj.weight
    model.language_model.layers.{i}.mlp.experts.gate_up_proj
    model.language_model.layers.{i}.mlp.experts.down_proj
    model.language_model.layers.{i}.mlp.shared_experts.gate_up_proj.weight
    model.language_model.layers.{i}.self_attn.indexer.{q,k}_{proj,norm}.weight
    model.language_model.layers.{i}.mlp.gate.e_score_correction_bias

This converter handles the language-tower rename/merge path and the vision
projector split used by the public checkpoint. The public index stores the
first vision projector MLP as `multi_modal_projector.linear_{1,2}` and the
spatial merge MLP as `patch_merge_mlp.linear_{1,2}`; transformers>=5.16 folds
both into `model.multi_modal_projector`, naming the latter
`merge_linear_{1,2}`.
"""

import re
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import torch

from ...checkpoint.convert import ConvertedCheckpointTensor, export_weights


_EXPERT_PATTERN = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\.(?P<proj>w1|w2|w3)\.weight$"
)
_DENSE_MLP_PATTERN = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_SHARED_EXPERT_PATTERN = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.shared_experts\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_SPARSE_GATE_PATTERN = re.compile(r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.gate\.weight$")
_INDEXER_PATTERN = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.self_attn\.index_(?P<kind>q|k)_(?P<part>proj|norm)\.weight$"
)
_E_SCORE_CORRECTION_BIAS_PATTERN = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.e_score_correction_bias$"
)


def _map_plain_name(name: str) -> str:
    if name == "language_model.lm_head.weight":
        return "lm_head.weight"
    if name.startswith("language_model.model."):
        return "model.language_model." + name.removeprefix("language_model.model.")
    if name.startswith("vision_tower.vision_model."):
        suffix = name.removeprefix("vision_tower.vision_model.")
        suffix = suffix.removeprefix("encoder.")
        suffix = suffix.replace("embeddings.patch_embedding.", "embeddings.proj.")
        return "model.vision_tower." + suffix
    if name.startswith("multi_modal_projector."):
        return "model.multi_modal_projector." + name.removeprefix("multi_modal_projector.")
    if name.startswith("patch_merge_mlp."):
        suffix = name.removeprefix("patch_merge_mlp.")
        suffix = suffix.replace("linear_1.", "merge_linear_1.")
        suffix = suffix.replace("linear_2.", "merge_linear_2.")
        return "model.multi_modal_projector." + suffix
    return name


def _dense_mlp_prefix(layer: str) -> str:
    return f"model.language_model.layers.{layer}.mlp"


def _shared_experts_prefix(layer: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.shared_experts"


def _sparse_experts_prefix(layer: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts"


def _indexer_key(layer: str, kind: str, part: str) -> str:
    return f"model.language_model.layers.{layer}.self_attn.indexer.{kind}_{part}.weight"


def _e_score_correction_bias_key(layer: str) -> str:
    return f"{_dense_mlp_prefix(layer)}.gate.e_score_correction_bias"


def _check_shape(name, tensor, shape):
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"M3 export {name}: expected shape {tuple(shape)}, got {tuple(tensor.shape)}")


def _export_m3_tensor(name, tensor, config):
    """Validate and invert the public-to-VeOmni checkpoint conversion."""
    text = config.text_config
    hidden = text.hidden_size
    if name in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
        _check_shape(name, tensor, (text.vocab_size, hidden))
        target = (
            "language_model.lm_head.weight" if name == "lm_head.weight" else "language_model.model.embed_tokens.weight"
        )
    elif name == "model.language_model.norm.weight":
        _check_shape(name, tensor, (hidden,))
        target = "language_model.model.norm.weight"
    elif match := re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", name):
        layer, suffix = int(match[1]), match[2]
        if not 0 <= layer < text.num_hidden_layers:
            raise ValueError(f"M3 export: layer outside config: {name}")
        prefix = f"language_model.model.layers.{layer}."
        head_dim = getattr(text, "head_dim", hidden // text.num_attention_heads)
        attn_shapes = {
            "q_proj.weight": (text.num_attention_heads * head_dim, hidden),
            "k_proj.weight": (text.num_key_value_heads * head_dim, hidden),
            "v_proj.weight": (text.num_key_value_heads * head_dim, hidden),
            "o_proj.weight": (hidden, text.num_attention_heads * head_dim),
            "q_norm.weight": (head_dim,),
            "k_norm.weight": (head_dim,),
        }
        if suffix.startswith("mlp.experts."):
            match = re.fullmatch(r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
            if match is None or not 0 <= int(match[1]) < text.num_local_experts:
                raise ValueError(f"M3 export: unsupported expert parameter {name}")
            proj = match[2]
            inter = text.intermediate_size
            shape = (hidden, inter) if proj == "down_proj" else (inter, hidden)
            _check_shape(name, tensor, shape)
            public_proj = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[proj]
            yield prefix + f"block_sparse_moe.experts.{match[1]}.{public_proj}.weight", tensor.contiguous()
            return
        if suffix.startswith("mlp.shared_experts.") or suffix in ("mlp.gate_up_proj.weight", "mlp.down_proj.weight"):
            shared = suffix.startswith("mlp.shared_experts.")
            inter = text.shared_intermediate_size if shared else text.dense_intermediate_size
            mlp_prefix = prefix + ("block_sparse_moe.shared_experts." if shared else "mlp.")
            if suffix.endswith("gate_up_proj.weight"):
                _check_shape(name, tensor, (2 * inter, hidden))
                yield mlp_prefix + "gate_proj.weight", tensor[:inter].contiguous()
                yield mlp_prefix + "up_proj.weight", tensor[inter:].contiguous()
                return
            if not suffix.endswith("down_proj.weight"):
                raise ValueError(f"M3 export: unsupported MLP parameter {name}")
            _check_shape(name, tensor, (hidden, inter))
            target = mlp_prefix + "down_proj.weight"
        elif suffix == "mlp.gate.weight":
            _check_shape(name, tensor, (text.num_local_experts, hidden))
            target = prefix + "block_sparse_moe.gate.weight"
        elif suffix == "mlp.gate.e_score_correction_bias":
            _check_shape(name, tensor, (text.num_local_experts,))
            target = prefix + "block_sparse_moe.e_score_correction_bias"
        elif suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            _check_shape(name, tensor, (hidden,))
            target = prefix + suffix
        elif suffix.startswith("self_attn.indexer."):
            part = suffix.removeprefix("self_attn.indexer.")
            dim = text.index_head_dim
            shapes = {
                "q_proj.weight": (text.index_n_heads * dim, hidden),
                "k_proj.weight": (dim, hidden),
                "q_norm.weight": (dim,),
                "k_norm.weight": (dim,),
            }
            if part not in shapes:
                raise ValueError(f"M3 export: unsupported indexer parameter {name}")
            _check_shape(name, tensor, shapes[part])
            target = prefix + "self_attn.index_" + part
        elif suffix.startswith("self_attn.") and suffix.removeprefix("self_attn.") in attn_shapes:
            _check_shape(name, tensor, attn_shapes[suffix.removeprefix("self_attn.")])
            target = prefix + suffix
        else:
            raise ValueError(f"M3 export: unsupported language parameter {name}")
    elif name.startswith("model.vision_tower."):
        vision = config.vision_config
        dim, inter = vision.hidden_size, vision.intermediate_size
        suffix = name.removeprefix("model.vision_tower.")
        target = "vision_tower.vision_model." + suffix
        if suffix == "embeddings.proj.weight":
            _check_shape(
                name,
                tensor,
                (dim, vision.num_channels, vision.temporal_patch_size, vision.patch_size, vision.patch_size),
            )
            target = "vision_tower.vision_model.embeddings.patch_embedding.weight"
        elif suffix in ("pre_layrnorm.weight", "pre_layrnorm.bias"):
            _check_shape(name, tensor, (dim,))
        elif match := re.fullmatch(r"layers\.(\d+)\.(.+)\.(weight|bias)", suffix):
            if not 0 <= int(match[1]) < vision.num_hidden_layers:
                raise ValueError(f"M3 export: vision layer outside config: {name}")
            part, kind = match[2], match[3]
            shapes = {
                "self_attn.q_proj": (dim, dim),
                "self_attn.k_proj": (dim, dim),
                "self_attn.v_proj": (dim, dim),
                "self_attn.out_proj": (dim, dim),
                "mlp.fc1": (inter, dim),
                "mlp.fc2": (dim, inter),
                "layer_norm1": (dim,),
                "layer_norm2": (dim,),
            }
            if part not in shapes:
                raise ValueError(f"M3 export: unsupported vision parameter {name}")
            shape = shapes[part]
            _check_shape(name, tensor, shape[:1] if kind == "bias" else shape)
            target = "vision_tower.vision_model.encoder." + suffix
        else:
            raise ValueError(f"M3 export: unsupported vision parameter {name}")
    elif match := re.fullmatch(r"model\.multi_modal_projector\.(merge_)?linear_([12])\.(weight|bias)", name):
        merge, index, kind = match.groups()
        input_size = config.merged_hidden_size if merge else config.vision_config.hidden_size
        shape = (config.projector_hidden_size, input_size) if index == "1" else (hidden, config.projector_hidden_size)
        _check_shape(name, tensor, shape[:1] if kind == "bias" else shape)
        target = ("patch_merge_mlp." if merge else "multi_modal_projector.") + f"linear_{index}.{kind}"
    else:
        raise ValueError(f"M3 export: unsupported parameter {name}; refusing an unconverted weight")
    yield target, tensor.contiguous()


class MiniMaxM3VLCheckpointTensorConverter:
    # HF saves use fused Transformers keys; rollout consumers request the public M3 layout.
    use_export_weights_for_hf_save = False

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self._expert_buffer: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
        self._gate_up_buffer: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

    def export_weights(self, model: torch.nn.Module) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Stream public M3 weights without changing the model's state dict or dtypes.

        Like DeepSeek V4, delegate FSDP/EP reconstruction and expert splitting to
        the generic iterator. Only the checkpoint layout is model-specific here.
        VeOmni's native checkpoint writers do not call this export hook.
        """
        for name, tensor in export_weights(model, num_experts=self.num_experts):
            yield from _export_m3_tensor(name, tensor, model.config)

    def can_handle(self, name: str) -> bool:
        return (
            name.startswith("language_model.")
            or name.startswith("vision_tower.vision_model.")
            or name.startswith("multi_modal_projector.")
            or name.startswith("patch_merge_mlp.")
        )

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        if match := _EXPERT_PATTERN.match(name):
            layer = match.group("layer")
            expert_id = int(match.group("expert"))
            proj = match.group("proj")
            prefix = _sparse_experts_prefix(layer)
            if proj == "w2":
                return self._buffer_expert(prefix, "down_proj", expert_id, tensor)
            proj_name = "gate_proj" if proj == "w1" else "up_proj"
            return self._buffer_expert(prefix, proj_name, expert_id, tensor)

        if match := _DENSE_MLP_PATTERN.match(name):
            prefix = _dense_mlp_prefix(match.group("layer"))
            return self._convert_dense_or_shared(prefix, match.group("proj"), tensor)

        if match := _SHARED_EXPERT_PATTERN.match(name):
            prefix = _shared_experts_prefix(match.group("layer"))
            return self._convert_dense_or_shared(prefix, match.group("proj"), tensor)

        if match := _SPARSE_GATE_PATTERN.match(name):
            return ConvertedCheckpointTensor(f"{_dense_mlp_prefix(match.group('layer'))}.gate.weight", tensor)

        if match := _INDEXER_PATTERN.match(name):
            return ConvertedCheckpointTensor(
                _indexer_key(match.group("layer"), match.group("kind"), match.group("part")), tensor
            )

        if match := _E_SCORE_CORRECTION_BIAS_PATTERN.match(name):
            return ConvertedCheckpointTensor(_e_score_correction_bias_key(match.group("layer")), tensor)

        return ConvertedCheckpointTensor(_map_plain_name(name), tensor)

    def _buffer_expert(
        self, prefix: str, proj_name: str, expert_id: int, tensor: "torch.Tensor"
    ) -> Optional[ConvertedCheckpointTensor]:
        buf_key = (prefix, proj_name)
        self._expert_buffer.setdefault(buf_key, {})[expert_id] = tensor
        if len(self._expert_buffer[buf_key]) < self.num_experts:
            return None

        stacked = torch.stack([self._expert_buffer[buf_key][idx] for idx in range(self.num_experts)])
        del self._expert_buffer[buf_key]
        if proj_name == "down_proj":
            return ConvertedCheckpointTensor(f"{prefix}.down_proj", stacked)
        return self._buffer_gate_up(prefix, proj_name, stacked, weight_suffix=False)

    def _convert_dense_or_shared(
        self, prefix: str, proj_name: str, tensor: "torch.Tensor"
    ) -> Optional[ConvertedCheckpointTensor]:
        if proj_name == "down_proj":
            return ConvertedCheckpointTensor(f"{prefix}.down_proj.weight", tensor)
        return self._buffer_gate_up(prefix, proj_name, tensor, weight_suffix=True)

    def _buffer_gate_up(
        self, prefix: str, proj_name: str, tensor: "torch.Tensor", *, weight_suffix: bool
    ) -> Optional[ConvertedCheckpointTensor]:
        self._gate_up_buffer[prefix][proj_name] = tensor
        if "gate_proj" not in self._gate_up_buffer[prefix] or "up_proj" not in self._gate_up_buffer[prefix]:
            return None

        gate = self._gate_up_buffer[prefix].pop("gate_proj")
        up = self._gate_up_buffer[prefix].pop("up_proj")
        if not self._gate_up_buffer[prefix]:
            del self._gate_up_buffer[prefix]
        merged = torch.cat([gate, up], dim=-2)
        suffix = ".weight" if weight_suffix else ""
        return ConvertedCheckpointTensor(f"{prefix}.gate_up_proj{suffix}", merged)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        errors: List[str] = []
        if self._expert_buffer:
            errors.append(
                f"unflushed expert tensors: { {key: len(value) for key, value in self._expert_buffer.items()} }"
            )
        if self._gate_up_buffer:
            errors.append(
                f"unflushed gate/up tensors: { {key: list(value) for key, value in self._gate_up_buffer.items()} }"
            )
        if errors:
            raise RuntimeError(
                "MiniMaxM3VL checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors)
            )
        return []


def create_minimax_m3_vl_checkpoint_tensor_converter(model):
    config = model.config
    text_config = getattr(config, "text_config", config)
    return MiniMaxM3VLCheckpointTensorConverter(num_experts=text_config.num_local_experts)


def convert_minimax_m3_vl_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    converted: Dict[str, int] = {}
    gate_up_indices: Dict[str, List[int]] = defaultdict(list)
    down_indices: Dict[str, List[int]] = defaultdict(list)

    def collect_gate_up(prefix: str, shard_idx: int):
        gate_up_indices[prefix].append(shard_idx)

    def collect_down(prefix: str, shard_idx: int):
        down_indices[prefix].append(shard_idx)

    for fqn, shard_idx in fqn_to_index_mapping.items():
        if match := _EXPERT_PATTERN.match(fqn):
            prefix = _sparse_experts_prefix(match.group("layer"))
            if match.group("proj") == "w2":
                collect_down(prefix, shard_idx)
            else:
                collect_gate_up(prefix, shard_idx)
            continue

        if match := _DENSE_MLP_PATTERN.match(fqn):
            prefix = _dense_mlp_prefix(match.group("layer"))
            proj = match.group("proj")
            if proj == "down_proj":
                converted[f"{prefix}.down_proj.weight"] = shard_idx
            else:
                collect_gate_up(prefix, shard_idx)
            continue

        if match := _SHARED_EXPERT_PATTERN.match(fqn):
            prefix = _shared_experts_prefix(match.group("layer"))
            proj = match.group("proj")
            if proj == "down_proj":
                converted[f"{prefix}.down_proj.weight"] = shard_idx
            else:
                collect_gate_up(prefix, shard_idx)
            continue

        if match := _SPARSE_GATE_PATTERN.match(fqn):
            converted[f"{_dense_mlp_prefix(match.group('layer'))}.gate.weight"] = shard_idx
            continue

        if match := _INDEXER_PATTERN.match(fqn):
            converted[_indexer_key(match.group("layer"), match.group("kind"), match.group("part"))] = shard_idx
            continue

        if match := _E_SCORE_CORRECTION_BIAS_PATTERN.match(fqn):
            converted[_e_score_correction_bias_key(match.group("layer"))] = shard_idx
            continue

        converted[_map_plain_name(fqn)] = shard_idx

    for prefix, indices in down_indices.items():
        converted[f"{prefix}.down_proj"] = min(indices)
    for prefix, indices in gate_up_indices.items():
        suffix = ".weight" if not prefix.endswith(".experts") else ""
        converted[f"{prefix}.gate_up_proj{suffix}"] = min(indices)
    return converted
