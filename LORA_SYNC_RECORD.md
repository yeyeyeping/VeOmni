# LoRA source synchronization: 2026-11-08

Branch: `3.1flash/lora_expert`.

Source: `/mnt/share/l00991686/work/m31_r3_merged_gbsa1_20261107/VeOmni`.
The running launcher is `/mnt/share/l00991686/run_m31_4node_lora.sh`.
The three repositories must be used together; configuration still selects
LoRA, merged forward and R3 explicitly.

Includes R3, FP32 LoRA owners on expert FSDP meshes, GBSA metadata/backward
precision zero, and the frozen Linear transpose exclusion during activation
offload.

## Verification

- All 592 selected files under the production package match the
  server snapshot after CRLF/LF normalization. Python AST parsing passed.
- Existing full_param commits were fast-forwarded into the LoRA branch, then
  uncommitted runtime source changes were copied and reviewed.
- Source-only comparison covered tracked and untracked project files. No
  additional untracked project source was found. Existing local-only tests,
  documentation, submodule links and build patches were preserved.
- Logs, caches, compiled libraries, old backups and downloaded third-party
  build dependencies are not added to Git.
- VeOmni quality-equivalent Ruff 0.13.1 checks passed (773 formatted files).
  verl and Ascend changed Python files passed Ruff checks and formatting.
- WSL lacks make and pre-commit. The requested make quality command and
  Ascend format.sh ci were attempted; the full Ascend pre-commit suite could
  not run. No new CPU tensor or NPU runtime tests were run for this copy.
- Running training, server source, and original full_param workspaces were
  not modified.

Snapshot SHA-256:

```text
7d0e0343427de03ec516073c7e69e072628ce9229a4bbdef8d8f99bccbac130c
```

The complete source archive and per-file hash manifest are retained locally
in the parent `.lora-sync` directory as `source_snapshot.tar.gz` and
`source_verification.json`. The archive excludes runtime logs and compiled
artifacts but includes downloaded third-party source for traceability.
