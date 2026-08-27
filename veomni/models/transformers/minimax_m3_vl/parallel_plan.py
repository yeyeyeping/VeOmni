from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan
from ....distributed.parallel_state import get_parallel_state


def get_parallel_plan():
    if get_parallel_state().ep_enabled:
        raise NotImplementedError(
            "MiniMax M3 VL does not yet support VeOmni expert parallelism. "
            "Keep train.accelerator.ep_size=1 until its expert forward is wired to VeOmni MoE token dispatch."
        )

    ep_plan = {
        "model.language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.language_model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    text_ep_plan = {
        "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    parallel_plan = ParallelPlan(
        extra_parallel_plan={
            "ep": ep_plan | text_ep_plan,
        }
    )
    return parallel_plan
