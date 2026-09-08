"""Real Omni processors and position helpers, with synthetic frames/audio/tokenizer."""

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen2VLVideoProcessor, WhisperFeatureExtractor

from veomni.data.data_transform import process_sample_qwen_omni
from veomni.models.transformers.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from veomni.models.transformers.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor


@pytest.fixture(params=["qwen25", "qwen3"])
def omni(request):
    special = dict(
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        audio_token="<|audio_pad|>",
        vision_bos_token="<|vision_start|>",
        vision_eos_token="<|vision_end|>",
        audio_bos_token="<|audio_start|>",
        audio_eos_token="<|audio_end|>",
    )
    words = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "user", "assistant", "system"] + list(special.values())
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=words[2:]
    )
    for name, token in special.items():
        setattr(tokenizer, name, token)
    if request.param == "qwen25":
        from transformers import Qwen2_5OmniThinkerConfig

        from veomni.models.transformers.qwen2_5_omni.generated.patched_modeling_qwen2_5_omni_gpu import (
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        cls, model_cls, config = (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
            Qwen2_5OmniThinkerConfig(),
        )
    else:
        from transformers import Qwen3OmniMoeThinkerConfig

        from veomni.models.transformers.qwen3_omni_moe.generated.patched_modeling_qwen3_omni_moe_gpu import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        cls, model_cls, config = (
            Qwen3OmniMoeProcessor,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerConfig(),
        )
        # Match the Qwen3-Omni processor's 13 audio tokens per second.
        config.position_id_per_seconds = 13

    class SmallProcessor(cls):
        def _merge_kwargs(self, *args, **kwargs):
            merged = super()._merge_kwargs(*args, **kwargs)
            pixels = getattr(self, "test_video_pixels", 784)
            merged["videos_kwargs"]["size"] = {"shortest_edge": pixels, "longest_edge": pixels}
            return merged

    template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% for item in message['content'] %}{% if item['type'] == 'video' %}"
        "{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
        "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}"
        "{{ '<|im_end|>\\n' }}{% endfor %}"
    )
    processor = SmallProcessor(
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen2VLVideoProcessor(),
        feature_extractor=WhisperFeatureExtractor(),
        tokenizer=tokenizer,
        chat_template=template,
    )
    for field, token_name in [
        ("image_token_id", "image_token"),
        ("video_token_id", "video_token"),
        ("audio_token_id", "audio_token"),
        ("vision_start_token_id", "vision_bos_token"),
        ("audio_start_token_id", "audio_bos_token"),
    ]:
        setattr(config, field, tokenizer.convert_tokens_to_ids(special[token_name]))
    # get_position_id_func only reads config and static helper methods.
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config, model.spatial_merge_size = config, 2
    return processor, model.get_position_id_func(), config, request.param


@pytest.mark.parametrize("with_audio", [False, True])
@pytest.mark.parametrize("side", [28, 56])
def test_training_omni_positions_use_source_time(omni, with_audio, side):
    processor, position_func, config, family = omni
    processor.video_processor.do_sample_frames = True
    processor.test_video_pixels = side**2
    video = {"video": np.zeros((150, 3, 28, 28), dtype=np.uint8), "video_fps": 30}
    if with_audio:
        video.update(audio=np.zeros(5 * 16000, dtype=np.float32), audio_fps=16000)
    sample = {
        "source": "qwen_omni_offline_av",
        "videos": [video],
        "conversations": [
            {"from": "human", "value": "<video>Describe."},
            {"from": "gpt", "value": "ok"},
        ],
    }
    result = process_sample_qwen_omni(sample, processor, position_func, fps=2, max_frames=4)[0]
    temporal = result["position_ids"][0, result["video_mask"]]
    expected = 3.3 * config.position_id_per_seconds
    if family == "qwen25":
        expected = int(expected)
    tokens_per_patch = (side // 28) ** 2
    assert (temporal - temporal[0]).tolist() == pytest.approx([0] * tokens_per_patch + [expected] * tokens_per_patch)
    assert result["video_grid_thw"].tolist() == [[2, side // 14, side // 14]]
    assert "video_timestamps" not in result
    assert result["input_ids"].shape == result["labels"].shape
    if with_audio and family == "qwen3":
        selected = result["audio_mask"] | result["video_mask"]
        times = result["position_ids"][0, selected]
        assert (times[1:] >= times[:-1]).all()


def test_processor_keeps_ragged_video_times(omni):
    processor, _, _, _ = omni
    videos = [torch.zeros(n, 3, 28, 28, dtype=torch.uint8) for n in [5, 4]]
    metadata = [
        {"fps": 10, "total_num_frames": 100, "frames_indices": [0, 10, 50, 50, 50]},
        {"fps": 20, "total_num_frames": 100, "frames_indices": [0, 10, 40, 60]},
    ]
    output = processor(
        text="<|vision_start|><|video_pad|><|vision_end|> " * 2,
        videos=videos,
        video_metadata=metadata,
        audios=[None, np.zeros(16000)],
        return_tensors="pt",
    )
    assert [times.tolist() for times in output["video_timestamps"]] == [[0, 5, 5], [0, 2]]
    assert output["video_grid_thw"][:, 0].tolist() == [3, 2]
    single = processor(
        text="<|vision_start|><|video_pad|><|vision_end|>",
        videos=videos[1:],
        video_metadata=metadata[1:],
        audios=[np.zeros(16000)],
        return_tensors="pt",
    )
    ids = output["input_ids"][0].tolist()
    first_end = ids.index(processor.tokenizer.convert_tokens_to_ids(processor.vision_eos_token)) + 1
    assert ids[first_end:] == single["input_ids"][0].tolist()


def test_training_mixed_videos_keep_independent_timelines(omni):
    processor, position_func, config, family = omni
    videos = [{"video": np.zeros((n, 3, 28, 28), dtype=np.uint8), "video_fps": 30} for n in [150, 300]]
    videos[1].update(audio=np.zeros(10 * 16000, dtype=np.float32), audio_fps=16000)
    sample = {
        "source": "qwen_omni_offline_av",
        "videos": videos,
        "conversations": [
            {"from": "human", "value": "Compare <video> with <video>."},
            {"from": "gpt", "value": "ok"},
        ],
    }
    result = process_sample_qwen_omni(sample, processor, position_func, fps=2, max_frames=4)[0]
    temporal = result["position_ids"][0, result["video_mask"]].reshape(2, 2)
    expected = [3.3 * config.position_id_per_seconds, 199 / 30 * config.position_id_per_seconds]
    if family == "qwen25":
        expected = [int(value) for value in expected]
    assert (temporal[:, 1] - temporal[:, 0]).tolist() == pytest.approx(expected)


def test_legacy_fps_matches_equivalent_source_metadata(omni):
    processor, _, _, _ = omni
    kwargs = dict(
        text="<|vision_start|><|video_pad|><|vision_end|>",
        videos=[torch.zeros(4, 3, 28, 28, dtype=torch.uint8)],
        audios=[np.zeros(2 * 16000)],
        return_tensors="pt",
    )
    legacy = processor(**kwargs, fps=4)
    explicit = processor(**kwargs, video_metadata=[{"fps": 4, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3]}])
    assert legacy["video_second_per_grid"].tolist() == [0.5]
    assert "video_timestamps" not in legacy
    assert "video_second_per_grid" not in explicit
    assert torch.equal(legacy["input_ids"], explicit["input_ids"])
