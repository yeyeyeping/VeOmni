"""Video timing regressions independent of external sample files or model weights."""

import sys
from io import BytesIO
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from veomni.data.multimodal import video_utils


def _frames(count):
    # Pixel values identify the original frame, independently of returned metadata.
    return torch.arange(count, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 32, 32).clone()


@pytest.mark.parametrize("max_frames, expected", [(None, [0, 17, 34, 51, 68, 85, 102, 119]), (4, [0, 40, 79, 119])])
@pytest.mark.parametrize("container", ["video.mp4", b"video container"])
def test_container_metadata_preserves_source_time(monkeypatch, max_frames, expected, container):
    frames = _frames(120)

    class Decoder:
        def __init__(self, *args, **kwargs):
            self.metadata = SimpleNamespace(average_fps=30.0, num_frames=120)

        def get_frames_at(self, indices):
            return SimpleNamespace(data=frames[indices])

    decoders = ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = Decoder
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    monkeypatch.setattr(video_utils, "is_ffmpeg_available", lambda: True)
    kwargs = dict(fps=2.0, max_frames=max_frames, use_audio_in_video=False)
    videos, metadata, audios, _ = video_utils.fetch_videos_metadata([container], **kwargs)
    meta = metadata[0]
    assert meta["fps"] == 30.0
    assert meta["total_num_frames"] == 120
    assert meta["frames_indices"].tolist() == expected
    assert videos[0][:, 0, 0, 0].tolist() == expected
    assert (meta["frames_indices"] / meta["fps"]).tolist() == pytest.approx(np.array(expected) / 30.0)
    # Omni and DiT still receive the same two-value API and sampled frame data.
    legacy_videos, legacy_audios = video_utils.fetch_videos([container], **kwargs)
    assert torch.equal(legacy_videos[0], videos[0])
    assert legacy_audios == audios == [None]


@pytest.mark.parametrize("kind", ["array_dict", "bytes_dict", "pil_list", "bytes_list"])
@pytest.mark.parametrize(
    "sampling, expected", [({"max_frames": 4}, [0, 2, 3, 5]), ({"frames": 8}, [0, 1, 2, 3, 4, 5, 5, 5])]
)
def test_predecoded_timing_and_repeated_frame_padding(kind, sampling, expected):
    frames = _frames(6)
    images = [Image.fromarray(frame.permute(1, 2, 0).numpy()) for frame in frames]
    encoded = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded.append(buffer.getvalue())
    video = {
        "array_dict": {"video": frames.permute(0, 2, 3, 1).numpy(), "video_fps": 2.0},
        "bytes_dict": {"frames": encoded, "video_fps": 2.0},
        "pil_list": images,
        "bytes_list": encoded,
    }[kind]
    videos, metadata, _, _ = video_utils.fetch_videos_metadata([video], fps=2.0, **sampling)
    meta = metadata[0]
    assert meta["fps"] == 2.0
    assert meta["total_num_frames"] == 6
    assert meta["frames_indices"].tolist() == expected
    assert videos[0][:, 0, 0, 0].tolist() == expected


def test_dict_source_fps_differs_from_target():
    video = {"video": _frames(120).numpy(), "video_fps": 30.0}
    videos, metadata, _, _ = video_utils.fetch_videos_metadata([video], fps=2.0, max_frames=4)
    assert metadata[0]["fps"] == 30.0
    assert metadata[0]["total_num_frames"] == 120
    assert metadata[0]["frames_indices"].tolist() == [0, 40, 79, 119]
    assert videos[0][:, 0, 0, 0].tolist() == [0, 40, 79, 119]


def test_decoder_fallback_keeps_repeated_first_frame_time(monkeypatch):
    class Decoder:
        def __init__(self, *args, **kwargs):
            self.metadata = SimpleNamespace(average_fps=30.0, num_frames=120)

        def get_frames_at(self, indices):
            if indices != [0]:
                raise RuntimeError("End of stream")
            return SimpleNamespace(data=_frames(1))

    decoders = ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = Decoder
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    monkeypatch.setattr(video_utils, "is_ffmpeg_available", lambda: True)
    videos, metadata, _, _ = video_utils.fetch_videos_metadata(
        ["video.mp4"], fps=2, min_frames=4, frame_factor=2, use_audio_in_video=False
    )
    assert metadata[0]["fps"] == 30.0
    assert metadata[0]["total_num_frames"] == 120
    assert metadata[0]["frames_indices"].tolist() == [0, 0, 0, 0]
    assert videos[0].shape[0] == 4
