# MiniMax M3 VL

MiniMax M3 VL is registered as `minimax_m3_vl` under VeOmni's transformers backend. The generated modeling files are based on `transformers==5.12.0`, because earlier transformers releases do not include `transformers.models.minimax_m3_vl`.

VeOmni's global `transformers-stable` dependency remains unchanged. Run MiniMax from an environment that overrides only that default group while retaining the appropriate accelerator extra.

GPU:

```bash
uv run --no-default-groups --extra gpu --with transformers==5.12.0 \
  torchrun --nproc_per_node=8 tasks/train_vlm.py \
  --config configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml
```

NPU (use `npu_aarch64` instead on ARM hosts):

```bash
uv run --no-default-groups --extra npu --with transformers==5.12.0 \
  torchrun --nproc_per_node=8 tasks/train_vlm.py \
  --config configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml
```

The `uv run` commands above are one-shot and do not activate a persistent environment. If you instead create and activate an accelerator environment with `transformers==5.12.0`, the device-independent helper runs the same training entry point:

```bash
NUM_PROCESSES=8 scripts/multimodal/train_minimax_m3_vl.sh
```

The public checkpoint can be referenced through either Hugging Face or ModelScope:

- Hugging Face: `MiniMaxAI/MiniMax-M3`
- ModelScope: `MiniMax/MiniMax-M3`

## Files

- `configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml`
- `veomni/models/transformers/minimax_m3_vl/configuration_minimax_m3_vl.py`
- `veomni/models/transformers/minimax_m3_vl/processing_minimax_m3_vl.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py`
- `veomni/models/transformers/minimax_m3_vl/parallel_plan.py`
- `veomni/models/transformers/minimax_m3_vl/checkpoint_tensor_converter.py`
- `scripts/multimodal/train_minimax_m3_vl.sh`

## Data Path

The `minimax_m3_vl` data transform reuses VeOmni's multimodal fetch and collate pipeline, then delegates image/video tensorization and conversation rendering to the MiniMax Hugging Face processor:

- `processor.image_processor(..., return_tensors="pt")` emits `pixel_values` and `image_grid_thw`.
- `processor.video_processor(..., return_metadata=True)` emits `pixel_values_videos`, `video_grid_thw`, and metadata used to expand MiniMax video timestamp tokens.
- `processor.apply_chat_template(..., tokenize=False)` applies the checkpoint's native MiniMax conversation protocol; VeOmni tokenizes the rendered string and masks the generation header, non-assistant turns, and visual placeholders in the labels.
- `MODEL_PROCESSOR_REGISTRY` maps both `MiniMaxM3VLProcessor` (upstream) and `MiniMaxVLProcessor` (the class in the checkpoint's bundled `processing_minimax.py`) to `veomni/models/transformers/minimax_m3_vl/processing_minimax_m3_vl.py`, so training always runs the transformers>=5.12 implementation the generated modeling and the parity test are written against, whatever the checkpoint's `auto_map` resolves to. That class also restores the chat template from the tokenizer when the checkpoint exposes none at processor level -- the bundled `MiniMaxVLProcessor.__init__` drops `**kwargs` and so loses the `chat_template.jinja` that `from_pretrained` passes through it, which otherwise surfaces as *"this processor does not have a chat template"* on the first sample.
- `MainCollator` packs `pixel_values`, `pixel_values_videos`, `image_grid_thw`, and `video_grid_thw` through the existing VLM collate rules.
- The MiniMax generated model exposes `get_metadata_collate_func()`, which converts packed `image_grid_thw` / `video_grid_thw` into `multimodal_metadata` grid lists on CPU. The vision tower consumes those lists to avoid calling `grid_thw.tolist()` inside the CUDA/NPU forward path.

MiniMax placeholder ids are preserved in `input_ids` so the upstream forward can scatter vision features by `config.image_token_id` and `config.video_token_id`. Labels for placeholder tokens are masked with `IGNORE_INDEX`.

## Current Scope

This recipe covers config loading, generated modeling import, MiniMax processor-shaped VLM samples, FSDP2 training, checkpoint conversion, multimodal metadata wiring, expert parallelism, and packed Ulysses sequence parallelism. Context parallelism is not supported, so keep `cp_size: 1`.

`model.ops_implementation.attn_implementation` controls the vision tower only. Keep it on a FlashAttention variant: ViT uses the varlen metadata to isolate multiple images/videos, and VeOmni's `*_with_sp` wrapper performs the ViT Ulysses all-to-all when `ulysses_size > 1`. The language tower deliberately ignores this generic setting for now. Each decoder layer keeps its input and output in local packed `[1, T_local, D]` form; its indexer gathers the small index Q/K tensors, while its main Q/K/V use Ulysses sequence/head exchange, temporarily unpack to padded BSND for MiniMax's sparse-attention reference math, then repack before the inverse exchange. RoPE is applied to local packed Q/K before communication, and the packed-to-BSND layout is built once per language-model forward and reused by every layer.

MiniMax's Gemma-style RMSNorm is wired to VeOmni's `rms_norm/qwen3_5` operator variant, selecting Liger on GPU and `torch_npu.npu_rms_norm` on NPU according to `model.ops_implementation.rms_norm_implementation`.

MiniMax routed experts use the `moe_experts/swiglu_oai` operator variant. It preserves the model's clamped `(up + 1) * gate * sigmoid(alpha * gate)` activation while reusing VeOmni's standard EP token dispatch. Enable EP with `fused_triton` on GPU or `fused_npu` on NPU:

```yaml
model:
  ops_implementation:
    moe_implementation: fused_triton  # use fused_npu on NPU
  accelerator:
    ep_size: 8
```

Router auxiliary load balancing is intentionally disabled for MiniMax M3. The upstream Switch-style loss recomputes routing with `softmax(router_logits)`, while MiniMax selects experts with `sigmoid(router_logits) + e_score_correction_bias`; these can select different experts. Keep `output_router_logits=false`. VeOmni raises an error if router-logit capture is requested until a MiniMax-specific balancing recipe is available.

The public MiniMax checkpoint stores separate per-expert `w1`/`w2`/`w3` tensors. Its runtime converter must first assemble the complete fused expert tensor, so this checkpoint layout is incompatible with `model.ep_sharded_stream_load=true`. Keep that option disabled unless the checkpoint has first been exported in the fused VeOmni/Hugging Face v5 layout.

To regenerate generated modeling files:

```bash
PYTHONPATH=$PWD uv run --no-project --with-editable ./patchgen-pkg --with transformers==5.12.0 \
  --with torch==2.7.1 --with packaging --with psutil --with einops \
  patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config \
  -o veomni/models/transformers/minimax_m3_vl/generated --diff

PYTHONPATH=$PWD uv run --no-project --with-editable ./patchgen-pkg --with transformers==5.12.0 \
  --with torch==2.7.1 --with packaging --with psutil --with einops \
  patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_npu_patch_gen_config \
  -o veomni/models/transformers/minimax_m3_vl/generated --diff
```
