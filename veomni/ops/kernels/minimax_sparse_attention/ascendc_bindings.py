# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Lazy imports of the separately installed CANN and Split-KV extensions."""

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Callable

from ....utils.import_utils import is_torch_npu_available


def require_npu() -> None:
    if not is_torch_npu_available():
        raise RuntimeError("MiniMax npu_cann requires torch-npu and an Ascend NPU.")


def _load(module: str):
    require_npu()
    try:
        return import_module(module)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "MiniMax npu_cann requires the ops-transformer msa_index_score, GBSAG and "
            "GBSAG metadata bindings, plus the separate Split-KV custom_ops extension. "
            f"Could not load {module!r}; see docs/examples/minimax_m3_vl.md."
        ) from exc


@lru_cache(maxsize=1)
def get_index_score() -> Callable:
    return _load_callable("cann_ops_transformer.ops.attention.msa_index_score", "msa_index_score")


def _load_callable(module: str, name: str) -> Callable:
    function = getattr(_load(module), name, None)
    if not callable(function):
        raise RuntimeError(
            f"MiniMax npu_cann requires {module}.{name}; the installed extension does not expose it. "
            "Rebuild the matching ops-transformer bindings; see docs/examples/minimax_m3_vl_msa_ascendc.md."
        )
    return function


@dataclass(frozen=True)
class AttentionOps:
    forward: Callable
    metadata: Callable
    backward: Callable


@lru_cache(maxsize=1)
def get_attention_ops() -> AttentionOps:
    forward = _load_callable("custom_ops", "npu_minimax_sparse_attention_split_kv")
    grad_module = "cann_ops_transformer.ops.attention.generic_block_sparse_attention_grad"
    return AttentionOps(
        forward,
        _load_callable(grad_module, "generic_block_sparse_attention_grad_metadata"),
        _load_callable(grad_module, "generic_block_sparse_attention_grad"),
    )
