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

from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class ContextParallelCollateContext:
    cp_size: int
    cp_rank: int
    ulysses_size: int
    ulysses_rank: int

    def __post_init__(self) -> None:
        if self.cp_size < 1 or not 0 <= self.cp_rank < self.cp_size:
            raise ValueError(f"Invalid CP coordinate: rank={self.cp_rank}, size={self.cp_size}.")
        if self.ulysses_size < 1 or not 0 <= self.ulysses_rank < self.ulysses_size:
            raise ValueError(f"Invalid Ulysses coordinate: rank={self.ulysses_rank}, size={self.ulysses_size}.")


def partition_contiguous_frames(frame_sizes: list[int], part_count: int) -> list[tuple[int, int]]:
    """Split ordered frames into contiguous ranges with approximately equal patch counts."""
    if part_count < 1:
        raise ValueError(f"part_count must be positive, got {part_count}.")
    if any(size <= 0 for size in frame_sizes):
        raise ValueError("Every frame must contain a positive number of patches.")

    prefix = [0]
    for size in frame_sizes:
        prefix.append(prefix[-1] + size)

    boundaries = [0]
    previous = 0
    total = prefix[-1]
    for part_idx in range(1, part_count):
        target = total * part_idx / part_count
        upper = max(previous, bisect_left(prefix, target, lo=previous))
        candidates = [upper]
        if upper > previous:
            candidates.append(upper - 1)
        boundary = min(candidates, key=lambda idx: (abs(prefix[idx] - target), idx))
        boundaries.append(boundary)
        previous = boundary
    boundaries.append(len(frame_sizes))
    return list(zip(boundaries, boundaries[1:]))


def _expand_frames(grid_thw_list: list[list[int]]) -> list[tuple[int, int, int, int]]:
    frames = []
    patch_offset = 0
    for temporal, height, width in grid_thw_list:
        frame_size = height * width
        for _ in range(temporal):
            frames.append((height, width, patch_offset, patch_offset + frame_size))
            patch_offset += frame_size
    return frames


def _partition_modality(
    batch: dict[str, Any],
    *,
    modality: str,
    grid_key: str,
    pixel_key: str,
    context: ContextParallelCollateContext,
    spatial_merge_size: int,
) -> None:
    grid = batch.get(grid_key)
    pixel_values = batch.get(pixel_key)
    if grid is None or pixel_values is None:
        return
    if not torch.is_tensor(grid) or grid.ndim != 2 or grid.shape[-1] != 3:
        raise ValueError(f"{grid_key} must be a tensor with shape [num_items, 3].")
    if not torch.is_tensor(pixel_values) or pixel_values.ndim < 1:
        raise ValueError(f"{pixel_key} must be a tensor with a patch dimension.")
    if spatial_merge_size < 1:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}.")

    grid_thw_list = grid.tolist()
    if not grid_thw_list:
        return

    for temporal, height, width in grid_thw_list:
        if temporal < 1 or height < 1 or width < 1:
            raise ValueError(f"Every {grid_key} entry must contain positive dimensions.")
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError(
                f"Every {grid_key} spatial dimension must be divisible by spatial_merge_size={spatial_merge_size}."
            )

    frames = _expand_frames(grid_thw_list)
    frame_sizes = [height * width for height, width, _, _ in frames]
    expected_patches = sum(frame_sizes)
    if pixel_values.shape[0] != expected_patches:
        raise ValueError(
            f"{pixel_key} has {pixel_values.shape[0]} patches, but {grid_key} describes {expected_patches}."
        )
    ranges = partition_contiguous_frames(frame_sizes, context.cp_size)
    merge_unit = spatial_merge_size**2

    real_merged_lengths = []
    padded_merged_lengths = []
    for frame_start, frame_end in ranges:
        real_patches = sum(frame_sizes[frame_start:frame_end])
        real_merged_lengths.append(real_patches // merge_unit)
        processing_patches = max(real_patches, merge_unit)
        alignment = context.ulysses_size * merge_unit
        padded_patches = ((processing_patches + alignment - 1) // alignment) * alignment
        padded_merged_lengths.append(padded_patches // merge_unit)

    frame_start, frame_end = ranges[context.cp_rank]
    local_frames = frames[frame_start:frame_end]
    if local_frames:
        patch_start = local_frames[0][2]
        patch_end = local_frames[-1][3]
        local_pixels = pixel_values[patch_start:patch_end]
        processing_grid = [[1, height, width] for height, width, _, _ in local_frames]
    else:
        local_pixels = pixel_values.new_zeros((merge_unit, *pixel_values.shape[1:]))
        processing_grid = [[1, spatial_merge_size, spatial_merge_size]]

    processing_patches = local_pixels.shape[0]
    padded_patches = padded_merged_lengths[context.cp_rank] * merge_unit
    pad_patches = padded_patches - processing_patches
    if pad_patches:
        padding = pixel_values.new_zeros((pad_patches, *pixel_values.shape[1:]))
        local_pixels = torch.cat((local_pixels, padding), dim=0)

    ulysses_chunk = local_pixels.shape[0] // context.ulysses_size
    ulysses_start = context.ulysses_rank * ulysses_chunk
    batch[pixel_key] = local_pixels.narrow(0, ulysses_start, ulysses_chunk).contiguous()
    batch[grid_key] = torch.tensor(processing_grid, dtype=grid.dtype, device=grid.device)

    cu_seqlens = [0]
    for _, height, width in processing_grid:
        cu_seqlens.append(cu_seqlens[-1] + height * width)
    if pad_patches:
        cu_seqlens.append(cu_seqlens[-1] + pad_patches)

    metadata = batch.setdefault("multimodal_metadata", {})
    metadata[f"{modality}_grid_thw_list"] = processing_grid
    metadata[f"vit_{modality}_cu_seqlens"] = torch.tensor(cu_seqlens, dtype=torch.int32, device="cpu")
    metadata[f"vit_{modality}_max_seqlen"] = max(
        (end - start for start, end in zip(cu_seqlens, cu_seqlens[1:])), default=0
    )
    metadata[f"vit_{modality}_context_parallel"] = True
    metadata[f"vit_{modality}_cp_real_merged_lengths"] = real_merged_lengths
    metadata[f"vit_{modality}_cp_padded_merged_lengths"] = padded_merged_lengths


def collate_qwen3_vl_context_parallel(
    batch: dict[str, Any],
    context: ContextParallelCollateContext,
    *,
    spatial_merge_size: int,
) -> None:
    """Partition Qwen3-VL vision inputs by frame across CP, then by patch across Ulysses."""
    _partition_modality(
        batch,
        modality="image",
        grid_key="image_grid_thw",
        pixel_key="pixel_values",
        context=context,
        spatial_merge_size=spatial_merge_size,
    )
    _partition_modality(
        batch,
        modality="video",
        grid_key="video_grid_thw",
        pixel_key="pixel_values_videos",
        context=context,
        spatial_merge_size=spatial_merge_size,
    )


def build_qwen3_vl_context_parallel_collate_func(spatial_merge_size: int) -> Callable:
    return partial(collate_qwen3_vl_context_parallel, spatial_merge_size=spatial_merge_size)
