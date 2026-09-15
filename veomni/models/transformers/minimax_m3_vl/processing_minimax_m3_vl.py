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
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        _adopt_tokenizer_chat_template(processor)
        return processor
