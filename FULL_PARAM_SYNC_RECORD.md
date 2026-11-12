# Full-parameter training source synchronization

Captured: 2026-11-12T02:27:19.660717+00:00

Source launcher: `/mnt/share/y00977881/script/vllm_new/multinode_16p_r3/train.sh`.
Default source: `/mnt/share/y00977881/project/veomni_verl_vllm3.1flash_r3/VeOmni`.
Target branch: `3.1flash/full_param`. Previous commit: `997bd80eb6f7326bbd6d72eb41a1ecb434cda7bf`.

Match server GBSA metadata precision 0, FSDP wrapping and expert LoRA activation implementation. Copy matching server tests and LoRA documentation.

This commit copies the server implementation; it does not independently fix or
validate its numerical behavior. In particular, CaMem uses pinned CPU backups,
and the server's GBSA metadata uses precision 0. No training or NPU tests run
during synchronization. Server files and running jobs are not modified.

Repository guidance, CI coverage, missing tool links and third-party build
patches are retained. Generated version/build metadata, compiled operators,
caches, logs and backups are not copied. Installed dependencies and compiled
operators are outside this source synchronization. The launcher may override
its default directory through `R3_CODE_ROOT`.

Changed/deleted source and test paths:

- `docs/key_features/lora.md`
- `tests/lora/test_veomni_lora_moe_native.py`
- `tests/ops/test_minimax_sparse_attention.py`
- `veomni/distributed/torch_parallelize.py`
- `veomni/lora/moe_layers.py`
- `veomni/lora/ops/npu_moe_group_gemm.py`
- `veomni/ops/kernels/minimax_sparse_attention/npu.py`

Retained repository-only differences:

- `.agents/knowledge/constraints.md`
- `.github/workflows/npu_unit_tests.yml`

Validation for the coordinated three-repository synchronization:

- All 3,588 selected production files match server Git-blob hashes; absent
  desktop-only runtime modules were removed. Python AST parsing passed.
- Server snapshot was read again before commit and remained unchanged.
- 74 CPU tests passed: attention 46, native MoE-LoRA 16, parameter refit 10,
  Geometry3k 2. Attention also skipped 19 hardware-dependent cases.
- VeOmni full quality commands passed (773 files formatted). Changed verl
  files passed Ruff. Ascend all-file static hooks passed except formatting
  of a copied test; that test was formatted and checked separately, and the
  changed-file long-function check passed.
- Two copied tests have formatting-only changes; production files retain
  the server bytes. Full checkpoint tests could not collect because the
  local CPU environment lacks Ray. NPU and multi-node tests were not run.
