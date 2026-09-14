# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
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
Patch configuration for MiniMax M3 VL transformers>=5.12.0 code generation.

The VeOmni integration keeps the upstream MiniMax M3 VL modeling body from
transformers, then adds VeOmni hooks for parallel plans, VLM collator metadata,
and FSDP-symmetric dummy vision execution. The public MiniMaxAI/MiniMax-M3
checkpoint uses an older language/MoE key layout, so runtime checkpoint
conversion is registered separately in checkpoint_tensor_converter.py. The
public index maps the spatial merge projector through `patch_merge_mlp` into
the generated `multi_modal_projector.merge_linear_{1,2}` parameters; full
public checkpoint loading still requires running the 59-shard load gate.

Regen command:
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config -o veomni/models/transformers/minimax_m3_vl/generated --diff
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import initialization as init
from transformers.modeling_utils import PreTrainedModel
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
    MiniMaxM3VLConfig,
    MiniMaxM3VLTextConfig,
)
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLAttention,
    MiniMaxM3VLCausalLMOutputWithPast,
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLExperts,
    MiniMaxM3VLRMSNorm,
    MiniMaxM3VLTopKRouter,
)

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad_and_slice
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.device import IS_NPU_AVAILABLE
from veomni.utils.model_outputs import CausalLMOutputWithLogProbs, FusedLinearAuxOutputMixin


config = PatchConfig(
    source_module="transformers.models.minimax_m3_vl.modeling_minimax_m3_vl",
    target_file="patched_modeling_minimax_m3_vl_gpu.py",
    description="MiniMax M3 VL with VeOmni parallel-plan hooks",
    transformers_version="5.12.0",
)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.ops.dispatch", names=["OpSlot"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["CausalLMOutputWithLogProbs", "FusedLinearAuxOutput", "FusedLinearAuxOutputMixin"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")
config.exclude_from_output("load_balancing_loss_func")
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_outputs", "slice_input_tensor", "sp_pad_and_slice"],
)
config.add_import(
    "veomni.ops.kernels.attention.ulysses",
    names=["prepare_ulysses_qkv", "restore_ulysses_output"],
)
config.add_import("veomni.models.transformers.attention_utils", names=["VARLEN_ATTENTION_TYPES"])
config.add_import("veomni.utils.device", names=["IS_NPU_AVAILABLE"])
config.add_post_import_block(
    """
veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
veomni_moe_experts_forward = OpSlot("moe_experts", "swiglu_oai")
"""
)


@config.replace_class(
    "MiniMaxM3VLPreTrainedModel",
    description="Allow VeOmni flash-attention implementations for the config-driven vision tower",
)
class PatchedMiniMaxM3VLPreTrainedModel(PreTrainedModel):
    config: MiniMaxM3VLConfig | MiniMaxM3VLTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM3VLDecoderLayer", "MiniMaxM3VLVisionEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": MiniMaxM3VLDecoderLayer,
        "attentions": MiniMaxM3VLAttention,
    }
    input_modalities = ("image", "video", "text")
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    _compatible_flash_implementations = [
        "MiniMaxAI/msa",
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
        "veomni_flash_attention_2_with_sp",
        "veomni_flash_attention_3_with_sp",
        "veomni_flash_attention_4_with_sp",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, MiniMaxM3VLExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, MiniMaxM3VLTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, MiniMaxM3VLRMSNorm):
            init.zeros_(module.weight)


@config.override_method(
    "MiniMaxM3VLRMSNorm.forward",
    description="Use VeOmni's Gemma-style fused RMSNorm backend when selected",
)
def minimax_m3_vl_rmsnorm_forward_patched(self, x):
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(x, self.weight, self.eps)

    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)


@config.add_helper
def collate_multimodal_metadata(batch, sp_pad):
    """Derive MiniMax ViT metadata on CPU inside the VeOmni collator.

    Each ``grid_thw`` entry is one logical visual sample with ``t * h * w``
    tokens. An SP-padding tail, when present, is appended as a separate
    synthetic sequence so neither another sample nor padding can affect the
    real visual tokens through varlen attention.
    """
    md = {}
    for modality, grid_key, pad_key in (
        ("image", "image_grid_thw", "pixel_values"),
        ("video", "video_grid_thw", "pixel_values_videos"),
    ):
        grid = batch.get(grid_key)
        if grid is None:
            continue
        grid_list = grid.tolist() if torch.is_tensor(grid) else grid
        if not grid_list:
            continue

        md[f"{modality}_grid_thw_list"] = grid_list
        cu_seqlens = [0]
        max_seqlen = 0
        for t, h, w in grid_list:
            seq_len = t * h * w
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
            max_seqlen = max(max_seqlen, seq_len)

        pad = sp_pad.get(pad_key, 0)
        if pad > 0:
            cu_seqlens.append(cu_seqlens[-1] + pad)
            max_seqlen = max(max_seqlen, pad)

        md[f"vit_{modality}_cu_seqlens"] = torch.tensor(
            cu_seqlens,
            dtype=torch.int32,
            device="cpu",
        )
        md[f"vit_{modality}_max_seqlen"] = max_seqlen
    if md:
        batch["multimodal_metadata"] = md


@config.add_helper
def _grid_thw_to_list(grid_thw, grid_thw_list):
    if grid_thw_list is not None:
        return grid_thw_list
    return grid_thw.tolist()


@config.add_helper
def _prepare_packed_layout(
    cu_seq_lens: torch.Tensor,
    max_sequence_length: int,
    total_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the reusable packed-to-BSND map and structural padding mask.

    ``cu_seq_lens`` already contains every token transported by sequence
    parallelism. In particular, the collator coalesces the SP tail padding into
    one synthetic sequence, so this helper deliberately does not distinguish it
    from a real sequence. ``padding_mask`` marks only the right-padding slots
    introduced while expanding the ragged packed stream to ``[B, Smax]``.
    """
    if not isinstance(max_sequence_length, int):
        raise TypeError(
            "MiniMax M3 VL packed max_length_q must be a Python int to avoid a device-to-host sync, "
            f"got {type(max_sequence_length).__name__}."
        )

    num_sequences = cu_seq_lens.numel() - 1
    # NPU keeps FlashAttention metadata on CPU, whereas the scatter indices must
    # live next to the tensors they index. This is a device copy, never a host
    # scalar read.
    device_cu_seq_lens = cu_seq_lens.to(device=device, dtype=torch.long, non_blocking=True)
    sequence_lengths = device_cu_seq_lens.diff()

    token_indices = torch.arange(total_length, device=device)
    sequence_indices = torch.repeat_interleave(
        torch.arange(num_sequences, device=device),
        sequence_lengths,
        output_size=total_length,
    )
    sequence_positions = token_indices - device_cu_seq_lens[sequence_indices]
    packed_to_padded_indices = sequence_indices * max_sequence_length + sequence_positions

    padding_mask = torch.arange(max_sequence_length, device=device).unsqueeze(0) >= sequence_lengths.unsqueeze(1)
    return packed_to_padded_indices, padding_mask


@config.add_helper
def _unpack_to_bsnd(
    packed_tensor: torch.Tensor,
    packed_to_padded_indices: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter ``[1, T, ...]`` packed data into right-padded ``[B, Smax, ...]`` data."""
    batch_size, max_sequence_length = padding_mask.shape
    trailing_shape = packed_tensor.shape[2:]
    padded_flat = packed_tensor.new_zeros(batch_size * max_sequence_length, *trailing_shape)
    padded_flat.index_copy_(0, packed_to_padded_indices, packed_tensor.squeeze(0))
    return padded_flat.view(batch_size, max_sequence_length, *trailing_shape)


@config.add_helper
def _pack_from_bsnd(padded_tensor: torch.Tensor, packed_to_padded_indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, Smax, ...]`` data back into its original ``[1, T, ...]`` packed order."""
    trailing_shape = padded_tensor.shape[2:]
    padded_flat = padded_tensor.reshape(-1, *trailing_shape)
    return padded_flat.index_select(0, packed_to_padded_indices).unsqueeze(0)


@config.add_helper
def _build_bsnd_causal_mask(padding_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build the dense reference mask for right-padded BSND self-attention."""
    sequence_length = padding_mask.shape[1]
    positions = torch.arange(sequence_length, device=padding_mask.device)
    causal_keep = positions.unsqueeze(0) <= positions.unsqueeze(1)
    key_keep = ~padding_mask[:, None, None, :]
    keep = causal_keep[None, None, :, :] & key_keep
    return torch.zeros(keep.shape, dtype=dtype, device=padding_mask.device).masked_fill(~keep, torch.finfo(dtype).min)


@config.add_helper
def _eager_bsnd_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    """Run reference attention using the post-Ulysses local GQA ratio."""
    local_num_key_value_groups = query.shape[1] // key.shape[1]
    if local_num_key_value_groups > 1:
        key = torch.repeat_interleave(key, local_num_key_value_groups, dim=1)
        value = torch.repeat_interleave(value, local_num_key_value_groups, dim=1)

    attention_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attention_weights = attention_weights + attention_mask
    attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_weights = F.dropout(attention_weights, p=dropout, training=training)
    return torch.matmul(attention_weights, value).transpose(1, 2).contiguous()


# ================================================================
# Patch: MiniMaxM3VLIndexer.forward
# 1. keep projection and RoPE on the local packed stream, then gather only the
#    small index Q/K tensors under Ulysses instead of the full hidden states
# 2. reuse the model-level packed layout to restore BSND and derive the
#    per-row arange position IDs required by block selection
# 3. mask only structural BSND padding; the collator's SP tail remains an
#    independent synthetic sequence carried through the packed transport
# ================================================================
@config.override_method(
    "MiniMaxM3VLIndexer.forward",
    description="Run the MiniMax indexer on temporary BSND views of packed training inputs",
)
def minimax_m3_vl_indexer_forward_patched(
    self,
    hidden_states,
    position_embeddings,
    past_key_values,
    position_ids=None,
    packed_to_padded_indices=None,
    packed_padding_mask=None,
):
    batch, q_len, _ = hidden_states.shape
    idx_q = self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)
    idx_q = self.q_norm(idx_q).transpose(1, 2)
    idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
    idx_k = self.k_norm(idx_k).transpose(1, 2)
    cos, sin = position_embeddings
    idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim])

    # --- Patch.1 ---
    if packed_to_padded_indices is not None:
        idx_q = idx_q.transpose(1, 2)
        idx_k = idx_k.transpose(1, 2)
        parallel_state = get_parallel_state()
        if parallel_state.ulysses_enabled:
            idx_q = gather_outputs(idx_q, gather_dim=1, group=parallel_state.ulysses_group)
            idx_k = gather_outputs(idx_k, gather_dim=1, group=parallel_state.ulysses_group)
        # --- Patch.1 ---

        # --- Patch.2 ---
        idx_q = _unpack_to_bsnd(idx_q, packed_to_padded_indices, packed_padding_mask).transpose(1, 2)
        idx_k = _unpack_to_bsnd(idx_k, packed_to_padded_indices, packed_padding_mask).transpose(1, 2)
        batch, _, q_len, _ = idx_q.shape
        position_ids = torch.arange(q_len, device=idx_q.device).unsqueeze(0).expand(batch, -1)
        # --- Patch.2 ---
    else:
        if past_key_values is not None:
            idx_k = past_key_values.layers[self.layer_idx].update_index(idx_k)
        if position_ids is None:
            position_ids = torch.arange(
                idx_k.shape[2] - idx_q.shape[2], idx_k.shape[2], device=idx_q.device
            ).unsqueeze(0)
        position_ids = (position_ids if position_ids.ndim > 1 else position_ids.unsqueeze(0)).expand(
            idx_q.shape[0], -1
        )

    k_len = idx_k.shape[2]
    num_key_blocks = -(-k_len // self.block_size)
    pad = num_key_blocks * self.block_size - k_len

    scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))
    # --- Patch.3 ---
    if packed_padding_mask is not None:
        scores = scores.masked_fill(packed_padding_mask[:, None, None, :], float("-inf"))
    # --- Patch.3 ---
    k_positions = torch.arange(k_len, device=idx_q.device)
    token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
    scores = scores.masked_fill(token_future, float("-inf"))
    if pad:
        scores = F.pad(scores, (0, pad), value=float("-inf"))
    scores = scores.view(batch, self.num_heads, q_len, num_key_blocks, self.block_size)
    block_scores = scores.amax(dim=-1).amax(dim=1)

    q_block = position_ids // self.block_size
    if self.local_blocks > 0:
        local = torch.arange(self.local_blocks, device=idx_q.device)
        local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
        block_scores.scatter_(-1, local_idx, float("inf"))

    topk = min(self.topk_blocks, num_key_blocks)
    topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
    block_indices = topk_indices.masked_fill(topk_scores == float("-inf"), -1)
    if packed_padding_mask is not None:
        block_indices = block_indices.masked_fill(packed_padding_mask[..., None], -1)
    return block_indices


# ================================================================
# Patch: MiniMaxM3VLIndexer.build_block_mask
# 1. accept a structural BSND padding mask with True=padding
# 2. compose block selection with token-level causality so future tokens in a
#    partially selected block remain inaccessible on the dense reference path
# ================================================================
@config.override_method(
    "MiniMaxM3VLIndexer.build_block_mask",
    description="Compose MiniMax block selection with BSND padding and causality",
)
def minimax_m3_vl_indexer_build_block_mask_patched(
    self,
    block_indices,
    attention_mask,
    key_length,
    dtype,
    device,
    position_ids,
    padding_mask=None,
):
    batch, q_len, _ = block_indices.shape
    num_key_blocks = -(-key_length // self.block_size)

    safe = block_indices.masked_fill(block_indices < 0, num_key_blocks)
    bias = block_indices.new_full((batch, q_len, num_key_blocks + 1), float("-inf"), dtype=dtype)
    bias.scatter_(-1, safe, 0.0)
    bias = bias[..., :num_key_blocks]
    block_keep = (bias == 0.0).repeat_interleave(self.block_size, dim=-1)[..., :key_length].unsqueeze(1)

    # --- Patch.1 ---
    if padding_mask is not None:
        key_keep = ~padding_mask[:, None, None, :]
    elif attention_mask is not None:
        key_keep = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    else:
        key_keep = True
    # --- Patch.1 ---

    # --- Patch.2 ---
    key_positions = torch.arange(key_length, device=device)
    causal_keep = key_positions[None, None, None, :] <= position_ids[:, None, :, None]
    keep = block_keep & key_keep & causal_keep
    # --- Patch.2 ---
    return torch.zeros(keep.shape, dtype=dtype, device=device).masked_fill(~keep, torch.finfo(dtype).min)


# ================================================================
# Patch: MiniMaxM3VLAttention.forward
# 1. keep decoder inputs/outputs packed and apply the precomputed RoPE before
#    any sequence-parallel communication
# 2. let the indexer gather only its small projections, while main Q/K/V use
#    Ulysses head/sequence all-to-all
# 3. temporarily restore BSND only around the MiniMax reference attention,
#    then repack before the inverse Ulysses exchange
# 4. keep language attention independent from the generic ViT-controlled
#    ``attn_implementation`` configuration
# ================================================================
@config.override_method(
    "MiniMaxM3VLAttention.forward",
    description="Run MiniMax language attention on temporary BSND views while decoder states stay packed",
)
def minimax_m3_vl_attention_forward_patched(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    packed_to_padded_indices=None,
    packed_padding_mask=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # --- Patch.1 ---
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # --- Patch.1 ---

    if packed_to_padded_indices is None:
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        position_ids = kwargs.get("position_ids")
        block_indices = None
        fallback_padding_mask = None
        if past_key_values is None and (attention_mask is None or attention_mask.ndim == 2):
            if attention_mask is None:
                fallback_padding_mask = torch.zeros(
                    hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
                )
            else:
                fallback_padding_mask = attention_mask == 0
        if self.indexer is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    key_states.shape[2] - query_states.shape[2], key_states.shape[2], device=query_states.device
                )
            position_ids = (position_ids if position_ids.ndim > 1 else position_ids.unsqueeze(0)).expand(
                query_states.shape[0], -1
            )
            block_indices = self.indexer(hidden_states, position_embeddings, past_key_values, position_ids)
            attention_mask = self.indexer.build_block_mask(
                block_indices,
                attention_mask,
                key_states.shape[2],
                query_states.dtype,
                query_states.device,
                position_ids,
                padding_mask=fallback_padding_mask,
            )
        elif fallback_padding_mask is not None:
            attention_mask = _build_bsnd_causal_mask(fallback_padding_mask, query_states.dtype)

        # --- Patch.4 ---
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            block_indices=block_indices,
            **kwargs,
        )
        # --- Patch.4 ---
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    # --- Patch.2 ---
    block_indices = None
    if self.indexer is not None:
        block_indices = self.indexer(
            hidden_states,
            position_embeddings,
            past_key_values=None,
            packed_to_padded_indices=packed_to_padded_indices,
            packed_padding_mask=packed_padding_mask,
        )

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    parallel_state = get_parallel_state()
    if parallel_state.ulysses_enabled:
        query_states, key_states, value_states, _ = prepare_ulysses_qkv(
            query_states,
            key_states,
            value_states,
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )
    # --- Patch.2 ---

    # --- Patch.3 ---
    query_states = _unpack_to_bsnd(query_states, packed_to_padded_indices, packed_padding_mask).transpose(1, 2)
    key_states = _unpack_to_bsnd(key_states, packed_to_padded_indices, packed_padding_mask).transpose(1, 2)
    value_states = _unpack_to_bsnd(value_states, packed_to_padded_indices, packed_padding_mask).transpose(1, 2)

    batch_size, _, sequence_length, _ = query_states.shape
    position_ids = torch.arange(sequence_length, device=query_states.device).unsqueeze(0).expand(batch_size, -1)
    if self.indexer is None:
        reference_mask = _build_bsnd_causal_mask(packed_padding_mask, query_states.dtype)
    else:
        reference_mask = self.indexer.build_block_mask(
            block_indices,
            attention_mask=None,
            key_length=key_states.shape[2],
            dtype=query_states.dtype,
            device=query_states.device,
            position_ids=position_ids,
            padding_mask=packed_padding_mask,
        )

    # --- Patch.4 ---
    attn_output = _eager_bsnd_attention_forward(
        query_states,
        key_states,
        value_states,
        reference_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        training=self.training,
    )
    # --- Patch.4 ---

    attn_output = _pack_from_bsnd(attn_output, packed_to_padded_indices)
    if parallel_state.ulysses_enabled:
        attn_output = restore_ulysses_output(attn_output, group=parallel_state.ulysses_group)
    # --- Patch.3 ---

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


# ================================================================
# Patch: MiniMaxM3VL3DRotaryEmbedding.forward
# 1. accept CPU-precomputed grid_thw_list from the collator so the vision
#    forward does not call `.tolist()` on a CUDA tensor in the hot path
# 2. keep the upstream tensor fallback for external callers that bypass
#    VeOmni's MainCollator
# ================================================================
@config.override_method(
    "MiniMaxM3VL3DRotaryEmbedding.forward",
    description="Consume collator-precomputed MiniMax vision grid lists when available",
)
def minimax_m3_vl_3d_rotary_embedding_forward_patched(self, grid_thw, device, dtype, grid_thw_list=None):
    # --- Patch.1 ---
    m = self.spatial_merge_size
    coords = []
    for t, h, w in _grid_thw_to_list(grid_thw, grid_thw_list):
        hi = torch.arange(h).unsqueeze(1).expand(-1, w)
        hi = hi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wi = torch.arange(w).unsqueeze(0).expand(h, -1)
        wi = wi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        ti = torch.arange(t).repeat_interleave(h * w)
        coords.append(torch.stack([ti, hi.repeat(t), wi.repeat(t)], dim=-1))
    # --- Patch.1 ---
    coords = torch.cat(coords).to(device=device, dtype=torch.float32)

    inv_freq = 1.0 / (
        self.theta ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32, device=device) / self.axis_dim)
    )
    freqs = torch.cat([coords[:, i : i + 1] * inv_freq for i in range(3)], dim=-1)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


# ================================================================
# Patch: MiniMaxM3VLVisionModel.forward
# 1. consume collator-precomputed grid / varlen-attention metadata
# 2. pad and slice 3D RoPE identically to the SP-sharded pixel rows
# 3. isolate the SP-padding tail and pass global sequence boundaries to every
#    vision attention layer; keep a runtime fallback for external callers
# 4. keep NPU cu_seqlens on CPU as required by its FA2 varlen interface
# ================================================================
@config.override_method(
    "MiniMaxM3VLVisionModel.forward",
    description="Add MiniMax vision SP RoPE and varlen-attention metadata handling",
)
def minimax_m3_vl_vision_model_forward_patched(self, pixel_values, image_grid_thw, **kwargs):
    r"""
    image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
        The temporal, height and width of each image's feature grid, used to build the vision 3D RoPE.
    """
    vit_metadata = kwargs.pop("vit_metadata", None) or {}
    precomputed_grid_thw_list = vit_metadata.get("grid_thw_list")
    precomputed_cu_seqlens = vit_metadata.get("cu_seqlens")
    precomputed_max_seqlen = vit_metadata.get("max_seqlen")

    embeds = self.embeddings(pixel_values).to(self.pre_layrnorm.weight.dtype)
    grid_thw_list = _grid_thw_to_list(image_grid_thw, precomputed_grid_thw_list)
    total_seq_len = sum(t * h * w for t, h, w in grid_thw_list)

    cos, sin = self.rotary_emb(
        image_grid_thw,
        device=embeds.device,
        dtype=embeds.dtype,
        grid_thw_list=grid_thw_list,
    )

    parallel_state = get_parallel_state()
    if parallel_state.cp_enabled:
        raise ValueError("MiniMax M3 VL does not support context parallelism; set cp_size=1.")

    attention_implementation = self.config._attn_implementation
    if (
        len(grid_thw_list) > 1
        and attention_implementation not in VARLEN_ATTENTION_TYPES
        and attention_implementation != "MiniMaxAI/msa"
    ):
        raise ValueError(
            "MiniMax M3 VL vision batches with multiple media inputs require a varlen flash-attention backend, "
            f"got {attention_implementation!r}."
        )
    if parallel_state.ulysses_enabled and (
        attention_implementation not in VARLEN_ATTENTION_TYPES or not attention_implementation.startswith("veomni_")
    ):
        raise ValueError(
            "MiniMax M3 VL vision Ulysses requires a VeOmni SP-aware varlen flash-attention backend, "
            f"got {attention_implementation!r}."
        )

    sp_pad_seq_len = 0
    if parallel_state.sp_enabled:
        merge_unit = self.config.spatial_merge_size**2
        cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=merge_unit)
        sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=merge_unit)
        sp_pad_seq_len = embeds.shape[0] * parallel_state.sp_size - total_seq_len

    if precomputed_cu_seqlens is not None:
        cu_seqlens = precomputed_cu_seqlens.to(
            embeds.device,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            non_blocking=True,
        )
    else:
        cu_seqlens_list = [0]
        fallback_max_seqlen = 0
        for t, h, w in grid_thw_list:
            seq_len = t * h * w
            cu_seqlens_list.append(cu_seqlens_list[-1] + seq_len)
            fallback_max_seqlen = max(fallback_max_seqlen, seq_len)
        if sp_pad_seq_len > 0:
            cu_seqlens_list.append(cu_seqlens_list[-1] + sp_pad_seq_len)
        cu_seqlens = torch.tensor(
            cu_seqlens_list,
            device=embeds.device,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )

    if precomputed_max_seqlen is not None:
        max_seqlen = precomputed_max_seqlen
    else:
        max_seqlen = max(fallback_max_seqlen, sp_pad_seq_len)

    if IS_NPU_AVAILABLE:
        cu_seqlens = cu_seqlens.cpu()

    hidden_states = self.pre_layrnorm(embeds).unsqueeze(0)
    for layer in self.layers:
        hidden_states = layer(
            hidden_states,
            attention_mask=None,
            position_embeddings=(cos, sin),
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            **kwargs,
        )
    return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=hidden_states[:, 0])


# ================================================================
# Patch: MiniMaxM3VLVisionModel.dummy_forward (new)
# 1. add dummy_forward so ranks without pixel_values can still run the
#    shared vision/projector parameters under FSDP
# 2. mirror MainCollator's SP layout and pass host-built metadata so dummy
#    execution follows the same vision-forward contract as real inputs
# ================================================================
@config.override_method(
    "MiniMaxM3VLVisionModel.dummy_forward",
    description="Provide MiniMax dummy vision forward for asymmetric FSDP batches",
)
def minimax_m3_vl_vision_dummy_forward_patched(self):
    # --- Patch.1 ---
    patch_size = self.config.patch_size
    temporal_patch_size = self.config.temporal_patch_size
    in_channels = getattr(self.config, "num_channels", getattr(self.config, "in_channels", 3))
    merge_size = self.config.spatial_merge_size
    t = 1
    h_base = 2 * merge_size
    w = 2 * merge_size
    parallel_state = get_parallel_state()
    h = h_base * parallel_state.sp_size if parallel_state.sp_enabled else h_base
    num_patches = t * h * w
    pixel_row_size = in_channels * temporal_patch_size * patch_size * patch_size

    weight = self.embeddings.proj.weight
    pixel_values = torch.zeros((num_patches, pixel_row_size), dtype=weight.dtype, device=weight.device)
    if parallel_state.sp_enabled:
        pixel_values = sp_pad_and_slice(
            pixel_values,
            dim=0,
            pad_value=0,
            pad_scale=merge_size**2,
        )
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long, device=weight.device)
    vit_metadata = {
        "grid_thw_list": [[t, h, w]],
        "cu_seqlens": torch.tensor([0, num_patches], dtype=torch.int32, device="cpu"),
        "max_seqlen": num_patches,
    }
    return self(pixel_values=pixel_values, image_grid_thw=grid_thw, vit_metadata=vit_metadata)
    # --- Patch.1 ---


@config.replace_class(
    "MiniMaxM3VLExperts",
    description="Remove the HF experts decorator and add SwiGLU-OAI fused MoE dispatch",
)
class PatchedMiniMaxM3VLExperts(nn.Module):
    """MiniMax M3 routed experts with VeOmni fused MoE dispatch."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.limit = config.swiglu_limit
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if veomni_moe_experts_forward.use_non_eager_impl:
            return veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)

        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return (up + 1.0) * glu


@config.add_helper
def _validate_minimax_m3_ep(text_config):
    """Validate static MiniMax EP requirements once while building the parallel plan."""
    parallel_state = get_parallel_state()
    if not parallel_state.ep_enabled:
        return
    if text_config.num_local_experts % parallel_state.ep_size != 0:
        raise ValueError(
            f"MiniMax M3 num_experts={text_config.num_local_experts} "
            f"must be divisible by ep_size={parallel_state.ep_size}."
        )
    if not veomni_moe_experts_forward.use_non_eager_impl:
        raise RuntimeError(
            "MiniMax M3 expert parallelism requires "
            "model.ops_implementation.moe_implementation=fused_triton or fused_npu."
        )


# ================================================================
# Patch: MiniMaxM3VLModel.forward
# 1. consume MiniMax multimodal_metadata and pass per-modality grid lists
#    to the vision tower
# 2. pop VeOmni data-pipeline helper masks/ids before dispatching to the
#    language model
# 3. run a zero-valued dummy vision/projector path when FSDP has no visual
#    input for a modality on this rank, keeping image/video call counts equal
#    across ranks and preventing asymmetric collectives from hanging
# ================================================================
@config.override_method(
    "MiniMaxM3VLModel.forward",
    description="Add MiniMax VLM metadata fast path and FSDP dummy vision branch",
)
def minimax_m3_vl_model_forward_patched(
    self,
    input_ids=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    **kwargs,
):
    r"""
    image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of each image's feature grid.
    video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of each video's feature grid.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.4 ---
    parallel_state = get_parallel_state()
    # --- Patch.4 ---

    # --- Patch.1 ---
    multimodal_metadata = kwargs.pop("multimodal_metadata", None) or {}
    image_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("image_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_image_cu_seqlens"),
            "max_seqlen": multimodal_metadata.get("vit_image_max_seqlen"),
        }
    }
    video_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("video_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_video_cu_seqlens"),
            "max_seqlen": multimodal_metadata.get("vit_video_max_seqlen"),
        }
    }
    # --- Patch.1 ---
    if parallel_state.sp_enabled:
        inputs_embeds = gather_outputs(inputs_embeds, gather_dim=1, group=parallel_state.sp_group)
    # --- Patch.2 ---
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    kwargs.pop("mm_token_type_ids", None)
    # --- Patch.2 ---

    # --- Patch.3 ---
    image_features = None
    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw, **image_vit_kwargs
        ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
        if parallel_state.sp_enabled:
            image_features = gather_outputs(image_features, gather_dim=0, group=parallel_state.sp_group)

    elif parallel_state.fsdp_enabled:
        fake_vision = self.vision_tower.dummy_forward()
        fake_features = self.multi_modal_projector(fake_vision.last_hidden_state.squeeze(0))
        inputs_embeds = inputs_embeds + fake_features.mean().to(inputs_embeds.device, inputs_embeds.dtype) * 0.0

    video_features = None
    if pixel_values_videos is not None:
        video_features = self.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **video_vit_kwargs
        ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
        if parallel_state.sp_enabled:
            video_features = gather_outputs(video_features, gather_dim=0, group=parallel_state.sp_group)
    elif parallel_state.fsdp_enabled:
        fake_vision = self.vision_tower.dummy_forward()
        fake_features = self.multi_modal_projector(fake_vision.last_hidden_state.squeeze(0))
        inputs_embeds = inputs_embeds + fake_features.mean().to(inputs_embeds.device, inputs_embeds.dtype) * 0.0
    # --- Patch.3 ---

    if image_mask is None or video_mask is None:
        image_mask, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds, image_features=image_features, video_features=video_features
        )
    else:
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    if image_features is not None:
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
    if video_features is not None:
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

    if parallel_state.sp_enabled:
        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=parallel_state.sp_group)

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    return MiniMaxM3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        image_hidden_states=image_features,
        video_hidden_states=video_features,
    )


# ================================================================
# Patch: MiniMaxM3VLTextModel.forward
# 1. keep every decoder layer in VeOmni's local packed layout and derive the
#    packed-to-BSND map once per model forward for all attention layers to reuse
# 2. generate RoPE once from the local packed position IDs; attention and the
#    indexer apply it before their respective Ulysses communications
# 3. consume global varlen metadata here so it cannot leak into the fixed
#    MiniMax language-attention implementation
# 4. reject router-logit capture because MiniMax's routing semantics are not
#    compatible with the upstream Switch-style auxiliary loss
# ================================================================
@config.override_method(
    "MiniMaxM3VLTextModel.forward",
    description="Keep decoder states packed and share a temporary BSND attention layout",
)
def minimax_m3_vl_text_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # --- Patch.4 ---
    output_router_logits = kwargs.pop("output_router_logits", None)
    if output_router_logits or self.config.output_router_logits:
        raise ValueError(
            "MiniMax M3 router auxiliary loss is disabled: the upstream Switch-style loss does not match "
            "MiniMax M3's sigmoid routing with e_score_correction_bias."
        )
    # --- Patch.4 ---

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # --- Patch.1 ---
    packed_cu_seq_lens = kwargs.pop("cu_seq_lens_q", None)
    packed_to_padded_indices = None
    packed_padding_mask = None
    if packed_cu_seq_lens is not None:
        kwargs.pop("cu_seq_lens_k", None)
        max_sequence_length = kwargs.pop("max_length_q", None)
        kwargs.pop("max_length_k", None)
        kwargs.pop("tail_padding_length", None)
        if past_key_values is not None:
            raise ValueError("MiniMax packed training does not support KV cache.")
        # The config default may still request cache during training. Packed
        # training has no decode state, so keep this path explicitly cache-free.
        use_cache = False

        parallel_state = get_parallel_state()
        if parallel_state.cp_enabled:
            raise ValueError("MiniMax M3 VL language attention supports Ulysses only; set cp_size=1.")
        total_length = inputs_embeds.shape[1] * parallel_state.ulysses_size
        packed_to_padded_indices, packed_padding_mask = _prepare_packed_layout(
            packed_cu_seq_lens,
            max_sequence_length,
            total_length,
            inputs_embeds.device,
        )
    # --- Patch.1 ---

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    hidden_states = inputs_embeds

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    if packed_to_padded_indices is not None:
        causal_mask = None
    elif isinstance(attention_mask, dict):
        causal_mask = next(iter(attention_mask.values()))
    else:
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    # --- Patch.2 ---
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
    # --- Patch.2 ---

    # --- Patch.3 ---
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            packed_to_padded_indices=packed_to_padded_indices,
            packed_padding_mask=packed_padding_mask,
            **kwargs,
        )
    # --- Patch.3 ---

    hidden_states = self.norm(hidden_states)

    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


# ================================================================
# Patch: MiniMaxM3VLForCausalLM.forward
# 1. slice hidden states before the LM head so eager and fused loss paths share
#    the same logits_to_keep behavior
# 2. unpack VeOmni's three-value causal-loss contract and preserve the fused
#    linear auxiliary payload on the returned ModelOutput
# 3. reject the upstream Switch-style router auxiliary loss because it applies
#    softmax routing without MiniMax's expert correction bias
# ================================================================
@config.override_method(
    "MiniMaxM3VLForCausalLM.forward",
    description="Unpack VeOmni causal LM loss tuple for MiniMax text-only training",
)
def minimax_m3_vl_for_causal_lm_forward_patched(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_router_logits=None,
    logits_to_keep=0,
    **kwargs,
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100. Tokens with indices set to -100 are ignored.
    """
    # --- Patch.3 ---
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )
    # --- Patch.3 ---

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        # --- Patch.3 ---
        output_router_logits=output_router_logits,
        # --- Patch.3 ---
        **kwargs,
    )

    # --- Patch.1 ---
    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]
    # --- Patch.1 ---

    # --- Patch.2 ---
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=None,
                labels=labels,
                vocab_size=self.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.2 ---

    return CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        fused_linear_aux=fused_linear_aux,
    )


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.forward",
    description="Unpack VeOmni causal LM loss tuple and route fused loss kernels when selected",
)
def minimax_m3_vl_sparse_for_conditional_generation_forward_patched(
    self,
    input_ids=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    logits_to_keep=0,
    **kwargs,
):
    r"""
    image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of each image's feature grid.
    video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of each video's feature grid.
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1 ---
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.1 ---

    return MiniMaxM3VLCausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        video_hidden_states=outputs.video_hidden_states,
    )


# Keep MiniMax's multimodal output fields while exposing VeOmni's fused-loss
# payload as a constructor field, so ModelOutput pytree flattening remains
# FSDP2-safe. This helper is emitted immediately after the upstream output
# class by patchgen and therefore works for both GPU and NPU artifacts.
@config.add_helper_after("MiniMaxM3VLCausalLMOutputWithPast")
@dataclass
class MiniMaxM3VLCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, MiniMaxM3VLCausalLMOutputWithPast):
    """MiniMaxM3VLCausalLMOutputWithPast plus VeOmni fused-loss payload."""


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_parallel_plan",
    description="Register MiniMax M3 VL expert parallel plan for the multimodal training path",
)
def minimax_m3_vl_get_parallel_plan_patched(self):
    from ..parallel_plan import get_vlm_parallel_plan

    _validate_minimax_m3_ep(self.config.text_config)
    return get_vlm_parallel_plan()


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_position_id_func",
    description="Use VeOmni's default 1-D packed-sequence position IDs for MiniMax M3 VL SFT data",
)
def minimax_m3_vl_get_position_id_func_patched(self):
    return None


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_metadata_collate_func",
    description="Expose MiniMax CPU-side vision grid metadata derivation to the VeOmni collator",
)
def minimax_m3_vl_get_metadata_collate_func_patched(self):
    return collate_multimodal_metadata  # noqa: F821 defined via add_helper


@config.override_method(
    "MiniMaxM3VLForCausalLM.get_parallel_plan",
    description="Register MiniMax M3 VL expert parallel plan for text-only reduced-layer smoke tests",
)
def minimax_m3_vl_text_get_parallel_plan_patched(self):
    from ..parallel_plan import get_text_parallel_plan

    _validate_minimax_m3_ep(self.config)
    return get_text_parallel_plan()
