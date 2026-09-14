"""Parity checks between VeOmni's M3-VL text expansion and HF's processor.

The test is opt-in because loading the official processor requires a local HF
cache (or network access):

    MINIMAX_M3_VL_MODEL_ID=MiniMaxAI/MiniMax-M3 pytest tests/data/multimodal/test_minimax_m3_vl_hf_parity.py
"""

import os
from copy import deepcopy

import pytest
import torch
from PIL import Image

from veomni.data.chat_template import MiniMaxM3VLChatTemplate
from veomni.data.data_transform import process_sample_minimax_m3_vl
from veomni.data.multimodal.image_utils import fetch_images
from veomni.data.multimodal.video_utils import fetch_videos_metadata, save_video_tensors_to_file
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


MODEL_ID = os.environ.get("MINIMAX_M3_VL_MODEL_ID")


@pytest.mark.skipif(
    not MODEL_ID or not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="Set MINIMAX_M3_VL_MODEL_ID and install transformers>=5.12.0.",
)
def test_process_sample_minimax_m3_vl_matches_hf_processor(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    processor = transformers.MiniMaxM3VLProcessor.from_pretrained(
        MODEL_ID, local_files_only=True, trust_remote_code=False
    )
    template = MiniMaxM3VLChatTemplate(processor)

    image = torch.arange(3 * 56 * 56, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(3, 56, 56)
    image_path = tmp_path / "minimax_m3_vl_image.png"
    Image.fromarray(image.permute(1, 2, 0).numpy()).save(image_path)

    source_video = torch.stack([image.roll(frame, dims=-1) for frame in range(90)])
    video_path = tmp_path / "minimax_m3_vl_30fps.mp4"
    save_video_tensors_to_file(source_video, str(video_path), fps=30)
    monkeypatch.setattr(
        "veomni.data.multimodal.conv_preprocess",
        lambda source, conversations, **kwargs: [
            ["user", ("text", "image="), ("image", None), ("text", ";video="), ("video", None)],
            ["assistant", ("text", "Describe the media.")],
        ],
    )
    loaded_images = fetch_images([str(image_path)])
    loaded_videos, video_metadata, _, _ = fetch_videos_metadata(
        [str(video_path)], fps=2.0, max_frames=32, use_audio_in_video=False
    )

    assert video_metadata[0]["fps"] == 30.0
    assert video_metadata[0]["frames_indices"].tolist() == [0, 18, 36, 53, 71, 89]

    image_inputs = processor.image_processor(images=loaded_images, return_tensors="pt")
    video_inputs = processor.video_processor(
        videos=loaded_videos, video_metadata=video_metadata, return_tensors="pt", return_metadata=True
    )
    hf_image = processor.replace_image_token(deepcopy(image_inputs), image_idx=0)
    hf_video = processor.replace_video_token(deepcopy(video_inputs), video_idx=0)
    expected_messages = [
        {
            "role": "user",
            "content": f"image={hf_image};video={hf_video}",
        },
        {"role": "assistant", "content": "Describe the media."},
    ]
    expected_text = processor.apply_chat_template(
        expected_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    print(f"expected_text:\n{expected_text}")
    expected_ids = processor.tokenizer.encode(expected_text, add_special_tokens=False)

    # Capture the exact message payload sent by VeOmni to the native HF
    # template, while retaining the official renderer's output.
    calls = []
    rendered_texts = []
    native_apply = processor.apply_chat_template

    def capture_apply(messages, **kwargs):
        calls.append(deepcopy(messages))
        rendered = native_apply(messages, **kwargs)
        rendered_texts.append(rendered)
        return rendered

    monkeypatch.setattr(processor, "apply_chat_template", capture_apply)
    [actual] = process_sample_minimax_m3_vl(
        {
            "conversations": [{"from": "human", "value": "ignored"}],
            "images": [str(image_path)],
            "videos": [str(video_path)],
        },
        processor=processor,
        chat_template=template,
        fps=2.0,
        max_frames=32,
        use_audio_in_video=False,
    )
    print(f"actual_text:\n{rendered_texts[-1]}")
    assert [
        timestamp for timestamp in ("0.0 seconds", "1.2 seconds", "2.4 seconds") if timestamp in expected_text
    ] == [
        "0.0 seconds",
        "1.2 seconds",
        "2.4 seconds",
    ]
    assert calls[-1] == expected_messages
    assert actual["input_ids"].tolist() == expected_ids
    expected_image_tokens = int(image_inputs["image_grid_thw"][0].prod() // processor.image_processor.merge_size**2)
    expected_video_tokens = int(video_inputs["video_grid_thw"][0].prod() // processor.video_processor.merge_size**2)
    assert actual["image_mask"].sum().item() == expected_image_tokens
    assert actual["video_mask"].sum().item() == expected_video_tokens
    assert torch.all(actual["labels"][actual["image_mask"] | actual["video_mask"]] == -100)
    torch.testing.assert_close(actual["pixel_values"], image_inputs["pixel_values"])
    torch.testing.assert_close(actual["pixel_values_videos"], video_inputs["pixel_values_videos"])
    torch.testing.assert_close(actual["image_grid_thw"], image_inputs["image_grid_thw"])
    torch.testing.assert_close(actual["video_grid_thw"], video_inputs["video_grid_thw"])
