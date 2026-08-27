from types import SimpleNamespace

import numpy as np
import pytest
import torch


pytest.importorskip("av")

from veomni.data.multimodal import video_utils


class _FakeFrame:
    def __init__(self, index, materialized):
        self.index = index
        self.materialized = materialized

    def to_ndarray(self, format):
        assert format == "rgb24"
        self.materialized.append(self.index)
        return np.full((2, 2, 3), self.index, dtype=np.uint8)


class _FakeContainer:
    def __init__(self, frame_count, materialized):
        self.frame_count = frame_count
        self.materialized = materialized
        self.streams = SimpleNamespace(video=[SimpleNamespace(average_rate=10.0)])

    def decode(self, stream):
        assert stream is self.streams.video[0]
        return (_FakeFrame(index, self.materialized) for index in range(self.frame_count))

    def close(self):
        pass


def test_pyav_fallback_materializes_only_sampled_frames(monkeypatch):
    materialized = []
    open_count = 0

    def fake_open(source):
        nonlocal open_count
        open_count += 1
        return _FakeContainer(frame_count=100, materialized=materialized)

    monkeypatch.setattr(video_utils.av, "open", fake_open)
    monkeypatch.setattr(video_utils, "smart_resize", lambda video, **kwargs: video.float())

    video, audio, audio_fps, indices = video_utils._load_and_process_video_with_pyav(
        b"video", use_audio_in_video=False, fps=1.0, max_frames=8
    )

    assert open_count == 2
    assert materialized == indices.tolist()
    assert video.shape == (8, 3, 2, 2)
    assert video.dtype == torch.float32
    assert audio is None
    assert audio_fps is None
