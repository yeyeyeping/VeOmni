# MiniMax M3 VL

MiniMax M3 VL is registered as `minimax_m3_vl` under VeOmni's transformers backend. The generated modeling files are based on the pinned `transformers==5.16.1`: `transformers.models.minimax_m3_vl` first shipped in 5.12, but 5.16 changed the indexer to select key blocks per KV group and renamed the vision tower argument to `grid_thw`, and the VeOmni patches target that version.

GPU:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml
```

NPU:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/minimax_m3_vl/minimax_m3_vl_npu.yaml
```

The public checkpoint can be referenced through either Hugging Face or ModelScope:

- Hugging Face: `MiniMaxAI/MiniMax-M3`
- ModelScope: `MiniMax/MiniMax-M3`

## Files

- `configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml` (GPU)
- `configs/multimodal/minimax_m3_vl/minimax_m3_vl_npu.yaml` (NPU)
- `veomni/models/transformers/minimax_m3_vl/processing_minimax_m3_vl.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py`
- `veomni/models/transformers/minimax_m3_vl/parallel_plan.py`
- `veomni/models/transformers/minimax_m3_vl/checkpoint_tensor_converter.py`

## Data Path

The `minimax_m3_vl` data transform reuses VeOmni's multimodal fetch and collate pipeline, then delegates image/video tensorization and conversation rendering to the MiniMax Hugging Face processor:

- `processor.image_processor(..., return_tensors="pt")` emits `pixel_values` and `image_grid_thw`.
- `processor.video_processor(..., return_metadata=True)` emits `pixel_values_videos`, `video_grid_thw`, and metadata used to expand MiniMax video timestamp tokens.
- `processor.apply_chat_template(..., tokenize=False)` applies the checkpoint's native MiniMax conversation protocol; VeOmni tokenizes the rendered string and masks the generation header, non-assistant turns, and visual placeholders in the labels.
- `MODEL_PROCESSOR_REGISTRY` maps both `MiniMaxM3VLProcessor` (upstream) and `MiniMaxVLProcessor` (the class in the checkpoint's bundled `processing_minimax.py`) to `veomni/models/transformers/minimax_m3_vl/processing_minimax_m3_vl.py`, so training always runs the transformers>=5.16 implementation the generated modeling is written against, whatever the checkpoint's `auto_map` resolves to. Training never uses the checkpoint's bundled processor: the loader's initial `AutoProcessor` call (with `trust_remote_code=True`) still imports it once to read its class name, then VeOmni's processor replaces it, and that processor does not go through `AutoProcessor` / `ProcessorMixin.from_pretrained` at all. Instead it builds the tokenizer and the transformers-native `MiniMaxM3VLImageProcessor` / `MiniMaxM3VLVideoProcessor` by concrete class (each reads all its parameters from the checkpoint's own config files) and wires the chat template in explicitly. This sidesteps two ways the public `MiniMaxAI/MiniMax-M3` layout is unusable:
  - Its `preprocessor_config.json` declares the image/video processors *only* through a `trust_remote_code` `auto_map` (no `image_processor_type` / `video_processor_type`). Going through `AutoProcessor` would demand remote code — and once `get_model_processor` strips `trust_remote_code`, the loader's fallback masks the failure as a misleading *"does not appear to have a file named processor_config.json"*. Building the native classes by name needs no remote code.
  - The bundled `MiniMaxVLProcessor.__init__` drops `**kwargs` and so loses the `chat_template.jinja` that `from_pretrained` passes through it (checkpoints that keep the template in `tokenizer_config.json` land in the same state). VeOmni takes the template straight from the tokenizer, which parses whichever file carries it, so the template is never lost — which otherwise surfaces as *"this processor does not have a chat template"* on the first sample.
- `MainCollator` packs `pixel_values`, `pixel_values_videos`, `image_grid_thw`, and `video_grid_thw` through the existing VLM collate rules.
- The MiniMax generated model exposes `get_metadata_collate_func()`, which converts packed `image_grid_thw` / `video_grid_thw` into `multimodal_metadata` grid lists on CPU. The vision tower consumes those lists to avoid calling `grid_thw.tolist()` inside the CUDA/NPU forward path.

MiniMax placeholder ids are preserved in `input_ids`, and the data transform precomputes `image_mask` / `video_mask` from them; the patched forward scatters vision features with those masks and only falls back to `config.image_token_id` / `config.video_token_id` lookups when a caller passes no masks (under sequence parallelism it first gathers the sharded `input_ids`, as the Qwen-VL models do). Labels for placeholder tokens are masked with `IGNORE_INDEX`.

Conversations may contain `system`, `user`, and `assistant` turns; assistant reasoning written as `<mm:think>…</mm:think>` before the answer is kept in every turn, as the checkpoint template renders it. Assistant-only labels come from rendering the conversation turn by turn, and a sample whose rendering rewrites an earlier turn is rejected. Tool-call conversations are not supported: the checkpoint template requires structured `tool_calls` on the preceding assistant turn, which VeOmni's conversation format does not carry.

## Current Scope

This recipe covers config loading, generated modeling import, MiniMax processor-shaped VLM samples, FSDP2 training, checkpoint conversion, multimodal metadata wiring, expert parallelism, and packed Ulysses sequence parallelism. Only the multimodal `MiniMaxM3SparseForConditionalGeneration` is supported; the text-only `MiniMaxM3VLForCausalLM` is not generated. Context parallelism is not supported, so keep `cp_size: 1`.

`train.freeze_vit` and `train.vit_lr` apply to `model.vision_tower`; the `multi_modal_projector` stays trainable with the language model's learning rate.

The sparse-attention indexer (`self_attn.indexer.*`) is frozen and keeps its checkpoint values: its top-k block selection is not differentiable and no indexer loss is computed, so it would receive no gradient anyway. MiniMax has not published how the indexer is handled in post-training. LoRA has not been validated for this model; because LoRA matches `target_modules` by name suffix, targets such as `q_proj` / `k_proj` would also wrap the frozen indexer projections, so exclude them with a regex (string) `exclude_modules`, e.g. `exclude_modules: '.*\.indexer\..*'`; the list form matches name suffixes only.

`model.ops_implementation.attn_implementation` controls the vision tower only. Keep it on a FlashAttention variant: ViT uses the varlen metadata to isolate multiple images/videos, and VeOmni's `*_with_sp` wrapper performs the ViT Ulysses all-to-all when `ulysses_size > 1`. The language tower deliberately ignores this generic setting for now. Each decoder layer keeps its input and output in local packed `[1, T_local, D]` form and applies RoPE to local Q/K before communication. Under Ulysses, the main Q/K/V use the sequence/head all-to-all, the per-KV-group index Q is split across ranks exactly like K, and the single-head index K is all-gathered. Block selection and attention then run through the packed TND operators in `veomni/ops/kernels/minimax_sparse_attention/`, which take the standard `cu_seq_lens_q` / `max_length_q` varlen metadata. The current eager implementation right-pads internally for MiniMax's reference math; that padded layout never leaves the operator. On Ascend NPU the masked attention runs through `torch_npu.npu_fusion_attention`; on GPU it is the PyTorch reference, which materializes `[B, H, S, S]` masks and softmax buffers, so memory grows quadratically with the packed sequence length (the recipe's `max_seq_len: 4096` fits) until a GPU sparse-attention kernel is added. The language model accepts only packed input: VeOmni's collator (or verl's remove-padding path) supplies the varlen kwargs, and a caller without them may pass one sequence with batch size 1 and no padding, which is run as a single-segment pack (its `attention_mask` is not read). Padded `[B, S]` batches (e.g. verl `use_remove_padding=False`), KV cache / `generate`, and context parallelism are rejected with an explicit error.

MiniMax's Gemma-style RMSNorm is wired to VeOmni's `rms_norm/qwen3_5` operator variant, selecting Liger on GPU and `torch_npu.npu_rms_norm` on NPU according to `model.ops_implementation.rms_norm_implementation`.

## CANN Sparse Attention

The NPU example selects the CANN language attention backend explicitly:

```yaml
model:
  ops_implementation:
    minimax_sparse_attention_implementation: npu_split_kv_gbsa
```

The default for this field is `eager`. Set it to `eager` to run without the
custom MiniMax CANN operators. `attn_implementation` continues to select the
vision attention backend independently.

Install the CANN operator libraries and matching Torch extensions for the
target device before enabling this backend. The following callable names
must be exported directly under `cann_ops_transformer.ops`:

| Function | Use |
|---|---|
| `msa_index_score` | Index Q/K to FP32 block scores |
| `k2q_csr` | Native AscendC pipeline: selections to interleaved KV-to-Q CSR with ascending Q IDs |
| `npu_minimax_sparse_attention_split_kv` | Forward attention output and global FP32 LSE |
| `generic_block_sparse_attention_grad_metadata` | Backward task metadata |
| `generic_block_sparse_attention_grad` | Q/K/V gradients |

These signatures follow the ops-transformer wrappers analyzed at commit
`41121b97f9bf46e3c9575f366095390f13bbe280`. Some of those wrappers live under
`experimental/` upstream and need to be exported into the namespace above by
the installed extension package. VeOmni does not import `custom_ops`, fall
back to another namespace, or bundle those vendor kernels.

`k2q_csr` comes from `experimental/attention/minimax_build_k2q_csr`; install
both its CANN operator libraries and its Torch extension, exporting
`k2q_csr` under `cann_ops_transformer.ops`. Build the extension from the
ops-transformer repository root with
`bash build.sh --torch_extension --experimental --ops=minimax_build_k2q_csr --vendor_name=minimax_csr`,
then install the resulting `build_out/cann_ops_transformer_minimax_csr-*.whl`
alongside the regular `cann_ops_transformer` package. The operator's
`torch_extension/` directory is not a standalone pip project.
The Split-KV extension comes from
`experimental/attention/minimax_sparse_attention_split_kv/torch_ops_extension`.
Install it and the regular MSA/GBSA extensions, then install VeOmni's
metadata-only bridge from the repository root, into the same Python environment:

```bash
python -m pip install --upgrade --no-build-isolation --no-deps ./scripts/kernel/minimax_cann_bridge
```

The bridge exports `custom_ops:npu_minimax_sparse_attention_split_kv` via the
`cann_ops_transformer.ops` entry-point group. It does not compile or install
the operator libraries. Restart Python after installation so the CANN loader
discovers the entries. The separate native `minimax_build_k2q_csr` extension
is required; the Python `build_k2q_csr` helper is not used by this backend.

The indexer calls `msa_index_score` with `layout_key="TND"`,
`sparse_mode=3`, no block table or dequantization scale, an int8
`[2048,2048]` zero mask template, and `init_blocks=local_blocks=0`.
The checked kernel ignores mask tensor contents and computes right-down
causal bounds from the packed lengths. Its internal local boost uses one
`start_loc` per sequence, so VeOmni instead applies `index_local_blocks` per
query before TopK. The model's local-block setting is preserved.
FP32-minimum invalid scores are normalized to `-inf` and unused selection
slots are filled with `-1`.

The indexer guarantees unique causal block IDs with trailing `-1` slots.
The attention Function consumes that contract directly, without repeating
filtering, sorting or deduplication. `k2q_csr` receives `[Hkv,T,K]`
selections, int32 Q prefix sums and int32 KV-block-count prefix sums. It uses
`order_method=1`, `q_global_offset=True`, and `use_simt=False` to emit interleaved
KV-block rows with ascending global packed Q IDs. Split-KV consumes per-sequence
int32 lengths (not prefix sums) and
continuous TND Q/K/V, runs with `inner_precise=0`, and returns output plus
`softmax_lse_flag=True`. Backward reuses the saved Q/K/V/output/LSE, CSR row
pointers and Q IDs. It maps rows back to batch/block order, subtracts each
sequence's Q offset, and pads local-query lists to `[B,Hkv,J,Smax]`, then
calls metadata and GBSA Grad with `block_shape=[1,128]`, causal mask mode 1,
softmax precision 0, and the original model softmax scale. Backward does not
sort: it relies on ascending Q order from the native CSR MC implementation.
The model supplies packed prefix sums; GBSA receives their int64 device copy.
CSR slot IDs are only needed by forward and are not saved for backward.
The frozen indexer receives no gradients; double backward is not supported.

Supported sparse inputs have `index_block_size=128`, attention head dimension
128, index head dimension a positive multiple of 16 no larger than 128,
matching BF16 Q/K/V, and an integral Q-to-KV head ratio no larger than 16.
FP16 is accepted where supported by the installed split-KV implementation;
its Ascend 950 path requires BF16 for this non-quantized configuration.
Unsupported sparse shapes raise. Dense layers and nonzero attention dropout
retain the existing eager path. Packing isolates all samples, including a
synthetic sequence used for SP tail padding. Ulysses communication remains
in the model, outside these two OpSlots; context parallelism remains unsupported.

This backend still has material memory costs. Split-KV's FP32 output partials
scale as `T * Hq_local * K * D * 4` bytes, and backward query lists require
`B * Hkv_local * ceil(Smax/128) * Smax * 4` bytes, plus conversion temporaries.
The adapter uses the supplied Python `max_seqlen` and prefix sums without
copying device metadata to the host. When prefix sums are on CPU, it computes
`total_rows` and `max_kv` there; for one packed sequence these sizes can also
be derived from the token shape. Otherwise it leaves both sizes unset, and
the native CSR wrapper reads the device block-prefix sums once to infer them.
GBSA's sequence-capacity bound is clamped to the total packed token count,
as required by its TND tiling checks. The index-score TND extension also
calls `kv_len.cpu()` for output allocation, so this is not a synchronization-free
pipeline. Switching to native CSR removes the Python helper's edge sort and
the adapter's backward sort, but the GBSA rectangular output and gather
temporaries remain. Backward saves
`Hkv * (total_kv_rows + 1)` int32 row pointers in addition to the
`Hkv * T * K` int32 Q IDs that replace saved block selections. Measure the
complete pipeline and workspace before increasing sequence length.

Run the tests with the custom operators installed:

```bash
pytest -v tests/ops/test_minimax_sparse_attention.py -k cann
```

CPU contract tests check finite invalid fill, TopK, interleaved CSR, sorted
batch-local backward lists, and autograd wiring against eager math. Actual
NPU tests separately check exact native CSR Q/slot order, block scores, and
fixed-selection attention outputs/QKV gradients, including GQA ratios 2 and 16
and both CPU and NPU prefix sums. Those tests
skip when hardware or the optional operator set is absent; CPU success alone
does not establish vendor-kernel numerical agreement.

## Projection and Ulysses Overlap

Enable the MiniMax language forward pipeline with:

```yaml
model:
  accelerator:
    ulysses_size: 2
    enable_async: true
```

The NPU example enables this option. It is active only when Ulysses size is
greater than one; disabling `enable_async` retains the serial projection and
communication path. All ranks must use the same setting.

The asynchronous path starts the frozen Index K projection/Norm/RoPE and its
all-gather first, then starts Index Q projection/Norm/RoPE and its head/sequence
all-to-all. It subsequently runs each main Q/K/V projection and immediately
launches that tensor's all-to-all. Main Q/K normalization and partial RoPE
are applied locally before exchange, matching the serial math. The indexer
waits occur after these projections. Main Q/K/V waits are deferred until
after block selection, immediately before attention. This permits indexer
communication to overlap main projections, main communication to overlap
later projections, and pending main exchanges to overlap indexer selection.

Tensor ownership is unchanged: each attention rank still holds the full
sequence and its local head slice. Index Q is exchanged before selection,
Index K is gathered, and block indices require no additional exchange. The
single-head Index K uses one asynchronous equal-shard all-gather without a
host shape exchange; VeOmni's packed SP padding guarantees equal local token
counts.

The pipeline invokes the original Linear and Norm modules, preserving their
hooks, parameter gradients, and adapter behavior. A small autograd wrapper
connects each deferred main exchange to its inverse all-to-all in backward,
including KV-head repetition when Ulysses size exceeds the KV-head count.
The indexer stays frozen and runs under `no_grad`. This schedules forward
overlap; backward uses the inverse exchanges and normal projection autograd.

Tests in `tests/models/test_minimax_m3_vl_ops.py` check projection/launch/wait
ordering and partial RoPE, and compare serial versus asynchronous outputs,
input gradients and projection/normalization parameter gradients with two
processes. They cover sparse and dense layers, KV-head repetition, packed
boundaries, SP tail padding and non-reentrant checkpointing. CPU/Gloo checks
run without accelerators; NCCL/HCCL cases require two matching devices.
Scheduling and numerical parity do not establish a speedup: confirm actual
device overlap and exposed communication time with a target-device trace.

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

Exported Hugging Face weights (`save_hf_weights`) use the transformers-native layout: `model.language_model.*`, fused `mlp.experts.gate_up_proj` / `down_proj`, `self_attn.indexer.*`, and a `config.json` with `layer_types` / `index_*` fields. Transformers loads them directly, but vLLM and SGLang read the original MiniMax layout (`language_model.model.*`, per-expert `block_sparse_moe.experts.{j}.w{1,2,3}`, `index_{q,k}_proj`) and the legacy `sparse_attention_config` / `moe_layer_freq` config fields, so an export cannot be served by them without converting back. For the same reason, RL frameworks that sync actor weights into a vLLM/SGLang rollout (e.g. verl's VeOmni engine) are not supported yet: the checkpoint converter does not implement `export_weights`, and the rollout loader silently skips the unmatched names.

To regenerate generated modeling files:

```bash
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config \
  -o veomni/models/transformers/minimax_m3_vl/generated --diff
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_npu_patch_gen_config \
  -o veomni/models/transformers/minimax_m3_vl/generated --diff
```
