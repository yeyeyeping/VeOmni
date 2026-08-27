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

import pytest
import torch

from veomni.distributed.context_parallel_data import (
    ContextParallelCollateContext,
    collate_qwen3_vl_context_parallel,
    partition_contiguous_frames,
)
from veomni.distributed.sequence_parallel import sp_pad_and_slice


def test_partition_contiguous_frames_balances_patch_counts_without_reordering():
    frame_sizes = [16, 4, 4, 4]
    ranges = partition_contiguous_frames(frame_sizes, 2)

    assert ranges == [(0, 1), (1, 4)]
    assert [sum(frame_sizes[start:end]) for start, end in ranges] == [16, 12]


def test_partition_contiguous_frames_allows_empty_ranges():
    ranges = partition_contiguous_frames([16], 4)

    assert len(ranges) == 4
    assert sum(end - start for start, end in ranges) == 1
    assert all(left_end == right_start for (_, left_end), (right_start, _) in zip(ranges, ranges[1:]))


def test_sp_pad_and_slice_uses_explicit_group(monkeypatch):
    group = object()
    monkeypatch.setattr("veomni.distributed.sequence_parallel.data.dist.get_world_size", lambda value: 2)
    monkeypatch.setattr("veomni.distributed.sequence_parallel.data.dist.get_rank", lambda value: 1)

    local = sp_pad_and_slice(torch.arange(6), pad_scale=2, group=group)

    assert local.tolist() == [4, 5, 0, 0]


def test_qwen3_vl_context_parallel_layout_reconstructs_original_frame_order():
    spatial_merge_size = 2
    cp_size = 4
    ulysses_size = 2
    original_pixels = torch.arange(3 * 4 * 4, dtype=torch.float32).unsqueeze(-1)
    reconstructed_cp_parts = []

    for cp_rank in range(cp_size):
        ulysses_parts = []
        metadata = None
        for ulysses_rank in range(ulysses_size):
            batch = {
                "pixel_values_videos": original_pixels.clone(),
                "video_grid_thw": torch.tensor([[3, 4, 4]], dtype=torch.long),
            }
            collate_qwen3_vl_context_parallel(
                batch,
                ContextParallelCollateContext(cp_size, cp_rank, ulysses_size, ulysses_rank),
                spatial_merge_size=spatial_merge_size,
            )
            ulysses_parts.append(batch["pixel_values_videos"])
            metadata = batch["multimodal_metadata"]

        cp_pixels = torch.cat(ulysses_parts, dim=0)
        real_merged = metadata["vit_video_cp_real_merged_lengths"][cp_rank]
        reconstructed_cp_parts.append(cp_pixels[: real_merged * spatial_merge_size**2])

    reconstructed = torch.cat(reconstructed_cp_parts, dim=0)
    torch.testing.assert_close(reconstructed, original_pixels)


def test_qwen3_vl_context_parallel_reconstructs_mixed_uneven_modalities_with_empty_ranges():
    spatial_merge_size = 2
    cp_size = 3
    ulysses_size = 2
    original_images = torch.arange(20, dtype=torch.float32).unsqueeze(-1)
    original_videos = torch.arange(8, dtype=torch.float32).unsqueeze(-1) + 100
    reconstructed = {"image": [], "video": []}
    real_lengths = {}

    for cp_rank in range(cp_size):
        ulysses_parts = {"image": [], "video": []}
        metadata = None
        for ulysses_rank in range(ulysses_size):
            batch = {
                "pixel_values": original_images.clone(),
                "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.long),
                "pixel_values_videos": original_videos.clone(),
                "video_grid_thw": torch.tensor([[2, 2, 2]], dtype=torch.long),
            }
            collate_qwen3_vl_context_parallel(
                batch,
                ContextParallelCollateContext(cp_size, cp_rank, ulysses_size, ulysses_rank),
                spatial_merge_size=spatial_merge_size,
            )
            ulysses_parts["image"].append(batch["pixel_values"])
            ulysses_parts["video"].append(batch["pixel_values_videos"])
            metadata = batch["multimodal_metadata"]

        for modality in ("image", "video"):
            real_lengths[modality] = metadata[f"vit_{modality}_cp_real_merged_lengths"]
            cp_pixels = torch.cat(ulysses_parts[modality], dim=0)
            real_patches = real_lengths[modality][cp_rank] * spatial_merge_size**2
            reconstructed[modality].append(cp_pixels[:real_patches])

    assert 0 in real_lengths["image"]
    assert 0 in real_lengths["video"]
    torch.testing.assert_close(torch.cat(reconstructed["image"], dim=0), original_images)
    torch.testing.assert_close(torch.cat(reconstructed["video"], dim=0), original_videos)


def test_qwen3_vl_context_parallel_empty_rank_uses_discardable_dummy():
    batch = {
        "pixel_values": torch.arange(16, dtype=torch.float32).unsqueeze(-1),
        "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
    }
    collate_qwen3_vl_context_parallel(
        batch,
        ContextParallelCollateContext(cp_size=4, cp_rank=0, ulysses_size=2, ulysses_rank=0),
        spatial_merge_size=2,
    )

    metadata = batch["multimodal_metadata"]
    assert metadata["vit_image_cp_real_merged_lengths"][0] == 0
    assert metadata["image_grid_thw_list"] == [[1, 2, 2]]
    assert batch["pixel_values"].shape[0] == 4
    assert batch["pixel_values"].eq(0).all()


def test_qwen3_vl_context_parallel_rejects_misaligned_grid():
    batch = {
        "pixel_values": torch.zeros(12, 1),
        "image_grid_thw": torch.tensor([[1, 3, 4]], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="divisible by spatial_merge_size"):
        collate_qwen3_vl_context_parallel(
            batch,
            ContextParallelCollateContext(cp_size=2, cp_rank=0, ulysses_size=1, ulysses_rank=0),
            spatial_merge_size=2,
        )


def test_qwen3_vl_context_parallel_rejects_patch_count_mismatch():
    batch = {
        "pixel_values_videos": torch.zeros(15, 1),
        "video_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="describes 16"):
        collate_qwen3_vl_context_parallel(
            batch,
            ContextParallelCollateContext(cp_size=2, cp_rank=0, ulysses_size=1, ulysses_rank=0),
            spatial_merge_size=2,
        )
