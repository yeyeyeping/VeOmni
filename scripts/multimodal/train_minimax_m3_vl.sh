#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS:-8}}"
CONFIG_PATH="${CONFIG_PATH:-configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml}"

torchrun --nproc_per_node="${NUM_PROCESSES}" tasks/train_vlm.py --config "${CONFIG_PATH}"
