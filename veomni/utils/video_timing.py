"""Source timestamps shared by video preprocessing and temporal position encoders."""

import math

import torch


def get_video_grid_timestamps(video_metadata, video_grid_thw, temporal_patch_size):
    """Return each temporal patch's first-frame time in source seconds.

    Metadata must describe already-sampled frames in source coordinates. Odd
    frame counts are padded by the video processor; that does not change the
    first frame of the final patch. Repeated source indices retain their time.
    """
    if len(video_metadata) != len(video_grid_thw):
        raise ValueError("Video metadata and video grids must have the same length.")
    if temporal_patch_size <= 0:
        raise ValueError("temporal_patch_size must be positive.")
    timestamps = []
    for metadata, grid in zip(video_metadata, video_grid_thw):
        fps = metadata["fps"] if isinstance(metadata, dict) else metadata.fps
        indices = metadata["frames_indices"] if isinstance(metadata, dict) else metadata.frames_indices
        if fps is None or not math.isfinite(fps) or fps <= 0 or indices is None:
            raise ValueError("Video timing requires source frame indices and a positive finite FPS.")
        indices = torch.as_tensor(indices, dtype=torch.float64)
        if indices.ndim != 1 or indices.numel() == 0 or not torch.isfinite(indices).all() or (indices < 0).any():
            raise ValueError("Video frame indices must be a nonempty finite nonnegative vector.")
        if (indices[1:] < indices[:-1]).any():
            raise ValueError("Video frame indices must be nondecreasing.")
        times = indices[::temporal_patch_size] / fps
        if len(times) != int(grid[0]):
            raise ValueError("Sampled frame indices do not match the processor's temporal grid.")
        timestamps.append(times)
    return timestamps
