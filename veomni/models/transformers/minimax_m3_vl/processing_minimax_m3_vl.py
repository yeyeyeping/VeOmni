# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
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
"""VeOmni's MiniMax M3 VL processor.

Registered in ``MODEL_PROCESSOR_REGISTRY`` under both the upstream class name
and the name the checkpoint's bundled ``processing_minimax.py`` uses, so a
MiniMax M3 run always gets this class regardless of which one ``AutoProcessor``
resolved from the checkpoint.
"""

from transformers import AutoTokenizer
from transformers.models.minimax_m3_vl.processing_minimax_m3_vl import (
    MiniMaxM3VLProcessor as HfMiniMaxM3VLProcessor,
)

from ....utils import logging


logger = logging.get_logger(__name__)


def _adopt_tokenizer_chat_template(processor) -> None:
    """Fall back to the tokenizer's chat template when the processor has none.

    ``ProcessorMixin.from_pretrained`` parses ``chat_template.jinja`` and hands
    it to the constructor as a ``chat_template`` kwarg, so a processor class
    whose ``__init__`` drops ``**kwargs`` silently loses it -- the bundled
    ``MiniMaxVLProcessor`` does exactly that. Checkpoints that keep the template
    in ``tokenizer_config.json`` instead of ``chat_template.jinja`` end up in the
    same state, because that file is only read by the tokenizer. Either way
    ``encode_messages`` dies on the first sample with "this processor does not
    have a chat template", even though the checkpoint ships one.

    The tokenizer parses both of those sources and keeps what it finds, so reuse
    its copy. Only fires when the processor carries no template of its own, so
    an M3 checkpoint that ships a processor-level template that deliberately
    differs from the tokenizer's is never overwritten.
    """
    if getattr(processor, "chat_template", None) is not None:
        return

    tokenizer_template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    if tokenizer_template is None:
        return

    processor.chat_template = tokenizer_template
    logger.warning_rank0(
        "[PROCESSOR] MiniMax M3 VL checkpoint exposes no processor-level chat template; reusing the "
        "tokenizer's. VeOmni's M3 data transform renders conversations through "
        "processor.apply_chat_template, which would otherwise refuse to run."
    )


class MiniMaxM3VLProcessor(HfMiniMaxM3VLProcessor):
    """Upstream M3 processor plus VeOmni's chat-template recovery.

    VeOmni pins MiniMax M3 to the transformers>=5.12 implementation -- that is
    what the generated modeling and the HF parity test are written against -- so
    this subclass also serves to keep the checkpoint's bundled processor code
    out of the training path.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            # Force ``trust_remote_code=False`` so a checkpoint that only offers
            # its sub-processors as remote code fails here immediately and
            # deterministically, instead of blocking on transformers' interactive
            # "run custom code? [y/N]" prompt in a TTY. VeOmni never wants the
            # bundled processor code, so this loses nothing.
            standard_kwargs = {**kwargs, "trust_remote_code": False}
            processor = super().from_pretrained(pretrained_model_name_or_path, **standard_kwargs)
        except Exception as exc:
            # The public MiniMaxAI/MiniMax-M3 checkpoint declares its image and
            # video processors *only* through a remote ``auto_map`` (its
            # ``preprocessor_config.json`` leaves ``image_processor_type`` /
            # ``video_processor_type`` unset), so ``ProcessorMixin.from_pretrained``
            # resolves them via ``AutoImageProcessor`` / ``AutoVideoProcessor`` and
            # demands ``trust_remote_code=True``. ``get_model_processor`` strips
            # that flag before it re-loads a registered processor, so the resolve
            # raises -- and the loader's outer ``except`` then masks it as a
            # misleading "no processor_config.json".
            #
            # VeOmni pins M3 to the transformers>=5.12 native classes anyway (the
            # generated modeling and the HF parity test are written against them),
            # so build those sub-processors by name, which needs no remote code.
            logger.warning_rank0(
                f"[PROCESSOR] MiniMax M3 VL processor could not load through the standard path "
                f"({type(exc).__name__}: {exc}); rebuilding from the transformers native "
                "image/video processors. This is expected for the public checkpoint, whose "
                "sub-processors are declared only via a trust_remote_code auto_map."
            )
            processor = cls._from_pretrained_native(pretrained_model_name_or_path, **kwargs)
        _adopt_tokenizer_chat_template(processor)
        return processor

    @classmethod
    def _from_pretrained_native(cls, pretrained_model_name_or_path, **kwargs):
        """Construct the processor from the native transformers sub-processors.

        Loads the image and video processors by their concrete class instead of
        through the Auto* registry, so the checkpoint's remote ``auto_map`` never
        triggers a ``trust_remote_code`` requirement. ``chat_template`` is read
        from ``chat_template.jinja`` when present; otherwise
        ``_adopt_tokenizer_chat_template`` fills it from the tokenizer.
        """
        from transformers import MiniMaxM3VLImageProcessor, MiniMaxM3VLVideoProcessor
        from transformers.utils import CHAT_TEMPLATE_FILE, cached_file

        # ``trust_remote_code`` only ever gated the sub-processor resolution we
        # are bypassing; the native classes take no such argument.
        kwargs.pop("trust_remote_code", None)

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, **kwargs)
        image_processor = MiniMaxM3VLImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        video_processor = MiniMaxM3VLVideoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)

        chat_template = None
        template_file = cached_file(
            pretrained_model_name_or_path,
            CHAT_TEMPLATE_FILE,
            _raise_exceptions_for_missing_entries=False,
        )
        if template_file is not None:
            with open(template_file, encoding="utf-8") as reader:
                chat_template = reader.read()

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )
