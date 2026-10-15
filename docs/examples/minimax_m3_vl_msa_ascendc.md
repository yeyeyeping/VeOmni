# MiniMax M3 AscendC MSA backend

Set `model.ops_implementation.minimax_sparse_attention_implementation: npu_cann`
to bind both MiniMax language MSA OpSlots. The implementation uses packed TND
throughout the sparse path:

```text
index Q/K -> msa_index_score -> per-token local blocks + top-k
         -> [T,G,K] sample-local KV block ids
         -> PyTorch CSR conversion -> Split-KV forward -> output + FP32 LSE
         -> GBSAG index conversion + metadata -> GBSAG backward -> dQ/dK/dV
```

No model, collator or Ulysses changes are needed. PA_BBND and paged KV caches
are not used. Dense layers (`block_indices=None`) retain the existing eager
implementation, including its internal padding and NPU fused attention.

## Installation

Use the accelerator environment and Transformers 5.16.0 described in
[the MiniMax recipe](minimax_m3_vl.md). Keep the existing repository-wide
Transformers pin unchanged.

Build and install the following OPP kernels from the compatible ops-transformer
checkout, following that checkout's CANN build instructions:

- `msa_index_score`
- `minimax_sparse_attention_split_kv`
- `generic_block_sparse_attention_grad`
- `generic_block_sparse_attention_grad_metadata`

Install both Python extension packages after preparing the CANN environment:

```bash
python -m pip install /path/to/ops-transformer/torch_extension
python -m pip install /path/to/ops-transformer/experimental/attention/minimax_sparse_attention_split_kv/torch_ops_extension
```

The second package provides `custom_ops.npu_minimax_sparse_attention_split_kv`;
it is separate from `cann_ops_transformer`. A package import alone does not
install its OPP kernels. Follow the kernel build's environment setup, including
its custom OPP library paths, before launching training.

The backend loads the public Python wrappers from:

```text
cann_ops_transformer.ops.attention.msa_index_score
cann_ops_transformer.ops.attention.generic_block_sparse_attention_grad
custom_ops
```

Missing imports fail at model binding with the dependency name. The backend
registers without importing CANN on CPU or GPU. Only Python callables are
cached; no live tensors are held in module-level caches.

## Configuration and supported inputs

```yaml
model:
  ops_implementation:
    minimax_sparse_attention_implementation: npu_cann
train:
  enable_full_determinism: false
```

The same override can be passed to the normal training entry point:

```bash
torchrun --nproc_per_node=8 tasks/train_vlm.py \
  --config configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml \
  --model.ops_implementation.minimax_sparse_attention_implementation npu_cann \
  --train.enable_full_determinism false
```

`attn_implementation` still selects vision attention. Configure the other
operators and parallelism normally; keep context parallel size at 1.

Initial sparse support is deliberately restricted:

| Property | Supported value |
|---|---|
| Q / K / V | BF16, same NPU device, head dimension 128 |
| Q shape | [T,Hq,128] |
| K / V shape | [T,G,128], no repeated GQA heads |
| GQA ratio | Hq divisible by G, with Hq/G in [1,16] |
| Index Q / K | BF16 [T,G,128] / [T,1,128], after existing norm and RoPE |
| KV block size | 128 |
| Lengths | Equal Q/K packed sample boundaries, including SP tail padding |
| Attention dropout | 0 |
| Deterministic algorithms | Disabled |
| Higher-order gradients | Unsupported |

Unsupported sparse inputs fail explicitly; they do not silently allocate a
dense mask. Patterns must contain unique causal block ids and at least one
visible block per token/group, with -1 used for invalid slots. The native
indexer produces this form. Empty packs bypass the kernels.

The GBSAG path computes gradients for main Q/K/V. Discrete top-k selection does
not send gradients into indexer projections. Training those projections needs
a separate differentiable auxiliary objective.

## Metadata and sparse indexing

The public two-OpSlot contract remains unchanged:

| Boundary | Length metadata | Sparse numbering |
|---|---|---|
| VeOmni indexer | cu_seqlens [B+1], max_seqlen as Python int | Returns [T,G,K] sample-local KV block ids |
| CANN index score with TND KV | Q/K prefix sums [B+1] | Returns raw [G,T,J_aligned] FP32 scores |
| Split-KV | Q/K lengths [B], obtained by diff(cu_seqlens) | CSR Q ids are globally packed; slots retain original top-k positions |
| GBSAG and metadata | Q/K prefix sums [B+1] and host maximum lengths | [B,G,J,C] lists contain sample-local Q ids |

Indexer local boosts are disabled inside `msa_index_score`. The adapter
applies local blocks separately for every query token, matching eager:

```text
query_block = query_position_within_sample // 128
local_block = max(query_block - local_offset, 0)
```

The CSR conversion is implemented in VeOmni with ordinary PyTorch operations,
so it can be validated on CPU and does not depend on the external CSR helper.
It follows Split-KV's block-major, batch-interleaved ordering. For lengths
[129,257,3], CSR rows are (sample, local block):

```text
(0,0), (1,0), (2,0), (0,1), (1,1), (1,2)
```

Edges within a row are sorted by global Q id. Forward and backward are built
from the same saved selection. Backward converts Q ids to sample-local values,
builds sorted GBSAG lists, then creates metadata for that exact pattern.

Both kernels use the same scale and causal convention. Split-KV runs with
`inner_precise=0` and `softmax_lse_flag=True`. The custom autograd function
saves the original output and FP32 [T,Hq,1] LSE for GBSAG; it does not recompute
the indexer or substitute a dense forward during backward.

GBSAG uses TND, `layout_sparse_pattern=1`, `block_shape=[1,128]`,
`mask_mode=1`, `softmax_precision=0`, `residual_block_mode=0`,
and `is_consistent_topk=False`.

## Memory and performance boundaries

The following auxiliary allocations can dominate long-context training:

```text
Indexer scores:            approximately 4 * G * T * J_aligned bytes
Split-KV FP32 partials:     approximately 4 * T * Hq * K * D bytes
GBSAG sparse indices:      approximately 4 * B * G * J * max_seqlen bytes
```

The current GBSAG interface requires list capacity C >= max_seqlen. This backend
therefore builds those lists and metadata during backward, saving only compact
[T,G,K] block ids in forward. It does not retain the forward CSR.

For T=128K, Hq=64, K=16 and D=128 on one rank, the Split-KV FP32 partial term
alone is about 64 GiB. Use the actual local head counts after Ulysses when
estimating memory. Packed TND does not remove these auxiliary workspaces.

CPU sequence metadata is accepted and copied to the tensor device. Supplying
device boundaries requires a host copy for size validation and allocation.
Selection validation also performs an explicit synchronization before the opaque
kernels. The CANN index-score wrapper can introduce another metadata sync.
This is an initial correctness-oriented integration, not a claim of a
synchronization-free or memory-optimal long-context path.

## Validation

The CPU-runnable suite tests ragged/empty sequences, SP-like tail segments,
both adjacency formats, top-k slots, local-block semantics, configuration and
OpSlot binding. It also executes the real custom autograd adapter with
independent CPU emulations of the forward/backward operator contracts.

```bash
python -m pytest -q tests/ops/test_minimax_sparse_attention_ascendc.py
python -m pytest -q tests/ops/test_minimax_sparse_attention.py tests/models/test_minimax_m3_vl_ops.py
```

On NPU with both extension packages installed, the new suite additionally
compares the real indexer with eager, tests Split-KV output/LSE plus GBSAG
dQ/dK/dV against a CPU FP32 sparse reference, and runs two optimizer steps
on a small M3 model with both sparse and dense layers. Hardware tests skip when NPU or
the optional extension packages are absent; inspect the skip summary before
treating a run as hardware validation:

```bash
python -m pytest -q -rs tests/ops/test_minimax_sparse_attention_ascendc.py
```

Existing MiniMax model/reference tests require Transformers >=5.16.0; the new
adjacency/adapter suite has no file-wide Transformers version gate and is
wired into NPU unit CI.

Hardware numerical acceptance, full M3 optimizer steps, Ulysses training
parity and target-length memory/performance still require execution on the
deployment environment. CPU emulation does not validate CANN kernel numerics.
