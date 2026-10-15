# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""AscendC MSA index scores with VeOmni's per-token local/top-k semantics."""

import torch

from .ascendc_bindings import get_index_score, require_npu
from .indices import prepare_metadata, select_blocks


@torch.no_grad()
def minimax_sparse_indexer_ascendc(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int,
    topk_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    require_npu()
    if (
        index_q.ndim != 3
        or index_k.ndim != 3
        or index_q.shape[1] <= 0
        or index_k.shape != (index_q.shape[0], 1, index_q.shape[-1])
        or index_q.shape[-1] != 128
        or index_q.dtype != torch.bfloat16
        or index_k.dtype != index_q.dtype
        or index_k.device != index_q.device
        or index_q.device.type != "npu"
        or block_size != 128
    ):
        raise ValueError(
            "MiniMax npu_cann indexer requires BF16 TND Q=[T,G,128], K=[T,1,128] on NPU and block_size=128."
        )
    if topk_blocks <= 0 or local_blocks < 0:
        raise ValueError("MiniMax MSA requires topk_blocks > 0 and local_blocks >= 0.")
    metadata = prepare_metadata(cu_seqlens, max_seqlen, index_q.shape[0], index_q.device, block_size)
    if index_q.shape[0] == 0:
        return torch.empty((0, index_q.shape[1], topk_blocks), dtype=torch.int32, device=index_q.device)

    mask = torch.ones((2048, 2048), dtype=torch.int8, device=index_q.device).triu_(1)
    scores = get_index_score()(
        index_q.contiguous(),
        index_k.contiguous(),
        torch.zeros(metadata.batch_size, dtype=torch.int32, device=index_q.device),
        block_table=None,
        scale=None,
        atten_mask=mask,
        actual_seq_qlen=metadata.cu_seqlens,
        actual_seq_klen=metadata.cu_seqlens,
        layout_key="TND",
        sparse_mode=3,
        init_blocks=0,
        local_blocks=0,
    )
    return select_blocks(scores, metadata, topk_blocks, local_blocks)
