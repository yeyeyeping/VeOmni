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
def test_minimax_m3_vl_processor_builds_from_native_subprocessors(monkeypatch):
    # VeOmni pins M3 to the transformers-native classes and the public checkpoint
    # ships an unusable bundled processor (drops **kwargs, declares sub-processors
    # only via a trust_remote_code auto_map). from_pretrained must therefore build
    # the tokenizer and native image/video processors by concrete class -- never
    # through AutoProcessor / ProcessorMixin.from_pretrained (which would drag in
    # the remote code and, once the loader strips trust_remote_code, surface as a
    # misleading "no processor_config.json"). The chat template must come straight
    # off the tokenizer, the one component that parses both chat_template.jinja and
    # a tokenizer_config.json template.
    import transformers

    from veomni.models.transformers.minimax_m3_vl import processing_minimax_m3_vl as mod

    calls = {}

    def _record(name, result):
        def _factory(path, **kwargs):
            calls[name] = {"path": path, "kwargs": kwargs}
            return result

        return _factory

    fake_tokenizer = SimpleNamespace(chat_template="{{ tokenizer_template }}")
    monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", staticmethod(_record("tokenizer", fake_tokenizer)))
    monkeypatch.setattr(
        transformers.MiniMaxM3VLImageProcessor, "from_pretrained", staticmethod(_record("image", "<image>"))
    )
    monkeypatch.setattr(
        transformers.MiniMaxM3VLVideoProcessor, "from_pretrained", staticmethod(_record("video", "<video>"))
    )

    # Fail loudly if the standard/remote path is ever touched.
    def _forbidden(*args, **kwargs):
        raise AssertionError("from_pretrained must not go through ProcessorMixin/AutoProcessor")

    monkeypatch.setattr(mod.HfMiniMaxM3VLProcessor, "from_pretrained", classmethod(_forbidden), raising=False)

    # Capture what the constructor receives instead of building a real processor.
    constructed = {}

    def _fake_init(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        constructed.update(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )

    monkeypatch.setattr(mod.MiniMaxM3VLProcessor, "__init__", _fake_init)

    mod.MiniMaxM3VLProcessor.from_pretrained("some/path", trust_remote_code=True, max_pixels=123)

    # The caller's trust_remote_code is stripped; every native class is loaded
    # with trust_remote_code=False so no bundled code can run.
    assert calls["tokenizer"]["kwargs"].get("trust_remote_code") is False
    assert calls["image"]["kwargs"].get("trust_remote_code") is False
    assert calls["video"]["kwargs"].get("trust_remote_code") is False
    assert calls["image"]["path"] == "some/path" and calls["image"]["kwargs"].get("max_pixels") == 123
    assert constructed["image_processor"] == "<image>"
    assert constructed["video_processor"] == "<video>"
    assert constructed["tokenizer"] is fake_tokenizer
    # The chat template is taken straight from the tokenizer, no separate recovery step.
    assert constructed["chat_template"] == "{{ tokenizer_template }}"


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
