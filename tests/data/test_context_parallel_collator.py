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

from types import SimpleNamespace

import torch


def _feature():
    image_mask = torch.zeros(8, dtype=torch.bool)
    image_mask[[1, 6]] = True
    return {
        "input_ids": torch.arange(8),
        "labels": torch.arange(8),
        "attention_mask": torch.ones(8, dtype=torch.long),
        "mm_token_type_ids": torch.arange(8),
        "position_ids": torch.arange(8).repeat(3, 1),
        "image_mask": image_mask,
    }


def _build_collator(monkeypatch, *, cp_rank, ulysses_rank, hook=None):
    import veomni.data.data_collator as collator_module

    state = SimpleNamespace(
        cp_size=2,
        cp_rank=cp_rank,
        cp_enabled=True,
        ulysses_size=2,
        ulysses_rank=ulysses_rank,
        ulysses_enabled=True,
    )
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    return collator_module.ContextParallelCollator(context_parallel_collate_func=hook)


def test_context_parallel_collator_supports_mrope_and_visual_indices(monkeypatch):
    batch = _build_collator(monkeypatch, cp_rank=0, ulysses_rank=0)([_feature()])

    assert batch["position_ids"].shape == (1, 3, 2)
    assert batch["position_ids"][0, 0].tolist() == [0, 1]
    assert batch["mm_token_type_ids"].tolist() == [[0, 1]]
    assert batch["image_mask"].tolist() == [[False, True]]
    assert batch["image_embed_indices"].tolist() == [[-1, 0]]


def test_context_parallel_collator_preserves_global_visual_ordinals(monkeypatch):
    batch = _build_collator(monkeypatch, cp_rank=0, ulysses_rank=1)([_feature()])

    assert batch["position_ids"][0, 0].tolist() == [6, 7]
    assert batch["mm_token_type_ids"].tolist() == [[6, 7]]
    assert batch["image_mask"].tolist() == [[True, False]]
    assert batch["image_embed_indices"].tolist() == [[1, -1]]


def test_context_parallel_collator_invokes_model_visual_hook(monkeypatch):
    seen = {}

    def hook(batch, context):
        seen["context"] = context
        batch["multimodal_metadata"] = {"ok": True}

    batch = _build_collator(monkeypatch, cp_rank=1, ulysses_rank=0, hook=hook)([_feature()])

    assert seen["context"].cp_rank == 1
    assert seen["context"].ulysses_rank == 0
    assert batch["multimodal_metadata"] == {"ok": True}
