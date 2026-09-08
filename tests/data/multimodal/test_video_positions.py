"""Temporal position regressions; no model weights or external media required."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

from veomni.models.transformers.qwen2_5vl.generated.patched_modeling_qwen2_5_vl_gpu import (
    Qwen2_5_VLForConditionalGeneration,
)
from veomni.utils.video_timing import get_video_grid_timestamps


def _position_func():
    config = Qwen2_5_VLConfig()
    config.vision_config.tokens_per_second = 25
    return Qwen2_5_VLForConditionalGeneration.get_position_id_func(SimpleNamespace(config=config))


@pytest.mark.parametrize("interval,expected", [(0.5, [0, 12, 25]), (1.5, [0, 37, 75])])
def test_fractional_legacy_video_interval(interval, expected):
    result = _position_func()(
        input_ids=torch.ones(1, 3, dtype=torch.long),
        mm_token_type_ids=torch.full((1, 3), 2),
        video_grid_thw=torch.tensor([[3, 2, 2]]),
        second_per_grid_ts=torch.tensor([interval]),
    )
    assert result["position_ids"][0, 0].tolist() == expected


def test_legacy_mixed_batch_positions_match_upstream():
    func = _position_func()
    kwargs = dict(
        input_ids=torch.ones(2, 8, dtype=torch.long),
        mm_token_type_ids=torch.tensor([[0, 0, 2, 2, 0, 1, 0, 0], [0, 0, 0, 2, 2, 0, 0, 0]]),
        attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]]),
        video_grid_thw=torch.tensor([[2, 2, 2], [2, 2, 2]]),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        second_per_grid_ts=torch.tensor([1, 2]),
    )
    positions, deltas = Qwen2_5_VLModel.get_rope_index(func.args[1], **kwargs)
    result = func(**kwargs)
    assert torch.equal(result["position_ids"], positions)
    assert torch.equal(result["rope_deltas"], deltas)


def test_source_times_preserve_repeats_and_multiple_videos():
    metadata = [
        {"fps": 30, "frames_indices": [0, 20, 40, 40, 40]},
        SimpleNamespace(fps=10, frames_indices=[10, 11, 25, 26]),
    ]
    times = get_video_grid_timestamps(metadata, [[3, 2, 2], [2, 2, 2]], 2)
    assert times[0].tolist() == pytest.approx([0, 4 / 3, 4 / 3])
    assert times[1].tolist() == [1, 2.5]
    assert metadata[0]["frames_indices"] == [0, 20, 40, 40, 40]
    result = _position_func()(
        input_ids=torch.ones(1, 6, dtype=torch.long),
        mm_token_type_ids=torch.tensor([[2, 2, 2, 0, 2, 2]]),
        video_grid_thw=torch.tensor([[3, 2, 2], [2, 2, 2]]),
        video_timestamps=times,
    )
    assert result["position_ids"][0, 0].tolist() == [0, 33, 33, 1, 27, 64]


def test_video_timing_rejects_mismatched_grid():
    with pytest.raises(ValueError, match="temporal grid"):
        get_video_grid_timestamps([{"fps": 2, "frames_indices": [0, 1, 2, 3]}], [[1, 2, 2]], 2)
