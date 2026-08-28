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
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py`
- `veomni/models/transformers/minimax_m3_vl/parallel_plan.py`
- `veomni/models/transformers/minimax_m3_vl/checkpoint_tensor_converter.py`
- `scripts/multimodal/train_minimax_m3_vl.sh`

## Data Path

The `minimax_m3_vl` data transform reuses VeOmni's multimodal fetch and collate pipeline, then delegates image and video tensorization to the MiniMax Hugging Face processors:

- `processor.image_processor(..., return_tensors="pt")` emits `pixel_values` and `image_grid_thw`.
- `processor.video_processor(..., return_metadata=True)` emits `pixel_values_videos`, `video_grid_thw`, and metadata used to expand MiniMax video timestamp tokens.
- `MainCollator` packs `pixel_values`, `pixel_values_videos`, `image_grid_thw`, and `video_grid_thw` through the existing VLM collate rules.
- The MiniMax generated model exposes `get_metadata_collate_func()`, which converts packed `image_grid_thw` / `video_grid_thw` into `multimodal_metadata` grid lists on CPU. The vision tower consumes those lists to avoid calling `grid_thw.tolist()` inside the CUDA/NPU forward path.

MiniMax placeholder ids are preserved in `input_ids` so the upstream forward can scatter vision features by `config.image_token_id` and `config.video_token_id`. Labels for placeholder tokens are masked with `IGNORE_INDEX`.

## Current Scope

This recipe covers config loading, generated modeling import, MiniMax processor-shaped VLM samples, FSDP2 training, checkpoint conversion, and MiniMax multimodal metadata wiring. The current patch does not implement Ulysses sequence parallelism or VeOmni MoE token dispatch for expert parallelism, so keep `ulysses_size: 1`, `cp_size: 1`, and `ep_size: 1`. MiniMax's Gemma-style RMSNorm is wired to VeOmni's `rms_norm/qwen3_5` operator variant, selecting Liger on GPU and `torch_npu.npu_rms_norm` on NPU according to `model.ops_implementation.rms_norm_implementation`. The NPU generated file does not yet claim Ascend-specific RoPE, attention, MSA, or fused-MoE kernel replacements.

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
