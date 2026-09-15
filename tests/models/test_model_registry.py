from types import SimpleNamespace

import pytest

from veomni.models.loader import MODEL_PROCESSOR_REGISTRY, get_model_class, get_model_config, get_model_processor
from veomni.models.transformers.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
from veomni.utils.helper import get_cache_dir
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


local_test_cases = [
    pytest.param("./tests/toy_config/qwen2vl_toy", True, False, ["config", "model", "processor"], ["model"]),
    pytest.param("./tests/toy_config/movqgan_toy", False, True, [], ["config", "model", "processor"]),
    pytest.param("./tests/toy_config/gpt_oss_toy", True, False, ["config", "model"], ["model"]),
]


def test_minimax_m3_text_config_preserves_partial_rotary_factor():
    config = MiniMaxM3VLTextConfig(head_dim=128, rotary_dim=64)

    assert config.partial_rotary_factor == 0.5
    assert config.rope_parameters == {
        "rope_theta": 5000000.0,
        "partial_rotary_factor": 0.5,
        "rope_type": "default",
    }


@pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="MiniMax M3 VL generated modeling requires transformers>=5.12.0.",
)
def test_minimax_m3_text_rope_uses_only_rotary_dim_channels():
    from veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_gpu import (
        MiniMaxM3VLRotaryEmbedding,
    )

    config = MiniMaxM3VLTextConfig(head_dim=128, rotary_dim=64)

    assert MiniMaxM3VLRotaryEmbedding(config).inv_freq.shape == (32,)


# Public MiniMax M3 checkpoints ship their own processing_minimax.py, so which
# class name AutoProcessor resolves depends on the checkpoint's auto_map. Both
# must route to VeOmni's processor.
@pytest.mark.parametrize("processor_class_name", ["MiniMaxM3VLProcessor", "MiniMaxVLProcessor"])
def test_minimax_m3_vl_processor_is_registered(processor_class_name):
    assert processor_class_name in MODEL_PROCESSOR_REGISTRY.valid_keys()


@pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="MiniMax M3 VL processor requires transformers>=5.12.0.",
)
@pytest.mark.parametrize("processor_class_name", ["MiniMaxM3VLProcessor", "MiniMaxVLProcessor"])
def test_minimax_m3_vl_processor_registry_resolves_to_veomni_class(processor_class_name):
    from veomni.models.transformers.minimax_m3_vl.processing_minimax_m3_vl import MiniMaxM3VLProcessor

    assert MODEL_PROCESSOR_REGISTRY[processor_class_name]() is MiniMaxM3VLProcessor


@pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="MiniMax M3 VL processor requires transformers>=5.12.0.",
)
@pytest.mark.parametrize(
    "processor_template, tokenizer_template, expected",
    [
        # A processor class that drops **kwargs in __init__ -- or a checkpoint
        # that keeps its template in tokenizer_config.json -- leaves the
        # processor with no template at all, and encode_messages would then die
        # on the first sample.
        pytest.param(None, "{{ tokenizer_template }}", "{{ tokenizer_template }}", id="recovered_from_tokenizer"),
        pytest.param("{{ processor_template }}", "{{ tokenizer_template }}", "{{ processor_template }}", id="kept"),
        pytest.param(None, None, None, id="nothing_to_recover"),
    ],
)
def test_minimax_m3_vl_processor_chat_template_fallback(processor_template, tokenizer_template, expected):
    from veomni.models.transformers.minimax_m3_vl.processing_minimax_m3_vl import _adopt_tokenizer_chat_template

    processor = SimpleNamespace(
        chat_template=processor_template, tokenizer=SimpleNamespace(chat_template=tokenizer_template)
    )
    _adopt_tokenizer_chat_template(processor)

    assert processor.chat_template == expected


@pytest.mark.parametrize(
    "config_path, is_hf_model, load_processor, hf_registered, veomni_registered", local_test_cases
)
def test_local_model_registry(monkeypatch, config_path, is_hf_model, load_processor, hf_registered, veomni_registered):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    if is_hf_model:
        save_path = get_cache_dir(config_path)
        hf_config = get_model_config(config_path)
        assert hf_config.__class__.__module__.startswith("transformers." if "config" in hf_registered else "veomni.")
        hf_config.save_pretrained(save_path)
        hf_model_class = get_model_class(hf_config)
        assert hf_model_class.__module__.startswith("transformers." if "model" in hf_registered else "veomni.")
        if load_processor:
            hf_processor = get_model_processor(config_path)
            assert hf_processor.__class__.__module__.startswith(
                "transformers." if "processor" in hf_registered else "veomni."
            )
            hf_processor.save_pretrained(save_path)

    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    save_path = get_cache_dir(config_path)
    veomni_config = get_model_config(config_path)
    assert veomni_config.__class__.__module__.startswith(
        "veomni." if "config" in veomni_registered else "transformers."
    )
    veomni_config.save_pretrained(save_path)
    veomni_model_class = get_model_class(veomni_config)
    assert veomni_model_class.__module__.startswith("veomni." if "model" in veomni_registered else "transformers.")
    if load_processor:
        veomni_processor = get_model_processor(config_path)
        assert veomni_processor.__class__.__module__.startswith(
            "veomni." if "processor" in veomni_registered else "transformers."
        )
        veomni_processor.save_pretrained(save_path)


remote_test_cases = [
    pytest.param("Qwen/Qwen2-VL-2B-Instruct", ["config", "model", "processor"], ["model"]),
]


@pytest.mark.xfail(reason="Remote path test may get too many requests error.")
@pytest.mark.parametrize("config_path, hf_registered, veomni_registered", remote_test_cases)
def test_remote_model_registry(monkeypatch, config_path, hf_registered, veomni_registered):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    save_path = get_cache_dir(config_path)
    hf_config = get_model_config(config_path)
    assert hf_config.__class__.__module__.startswith("transformers." if "config" in hf_registered else "veomni.")
    hf_config.save_pretrained(save_path)
    hf_model_class = get_model_class(hf_config)
    assert hf_model_class.__module__.startswith("transformers." if "model" in hf_registered else "veomni.")
    hf_processor = get_model_processor(config_path)
    assert hf_processor.__class__.__module__.startswith("transformers." if "processor" in hf_registered else "veomni.")
    hf_processor.save_pretrained(save_path)

    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    veomni_config = get_model_config(config_path)
    assert veomni_config.__class__.__module__.startswith(
        "veomni." if "config" in veomni_registered else "transformers."
    )
    veomni_config.save_pretrained(save_path)
    veomni_model_class = get_model_class(veomni_config)
    assert veomni_model_class.__module__.startswith("veomni." if "model" in veomni_registered else "transformers.")
    veomni_processor = get_model_processor(config_path)
    assert veomni_processor.__class__.__module__.startswith(
        "veomni." if "processor" in veomni_registered else "transformers."
    )
    veomni_processor.save_pretrained(save_path)
