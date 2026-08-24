from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


class RingComm:
    def __init__(self, process_group: Optional[dist.ProcessGroup]):
        self._process_group = process_group
        self._ops = []
        self._reqs = None

        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)

        send_group_rank = (self.rank + 1) % self.world_size
        recv_group_rank = (self.rank - 1) % self.world_size

        if process_group is None:
            self.send_rank = send_group_rank
            self.recv_rank = recv_group_rank
        else:
            self.send_rank = dist.get_global_rank(process_group, send_group_rank)
            self.recv_rank = dist.get_global_rank(process_group, recv_group_rank)

    def send_recv(
        self,
        to_send: torch.Tensor,
        recv_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._reqs is not None:
            raise RuntimeError("cannot enqueue P2P operations before the previous batch is waited")

        result = torch.empty_like(to_send) if recv_tensor is None else recv_tensor
        if result.shape != to_send.shape or result.dtype != to_send.dtype:
            raise ValueError("receive buffer must match the send tensor's shape and dtype")

        self._ops.append(
            dist.P2POp(
                dist.isend,
                to_send,
                self.send_rank,
                group=self._process_group,
            )
        )
        self._ops.append(
            dist.P2POp(
                dist.irecv,
                result,
                self.recv_rank,
                group=self._process_group,
            )
        )
        return result

    def commit(self) -> None:
        if self._reqs is not None:
            raise RuntimeError("commit called twice without wait")
        if not self._ops:
            raise RuntimeError("commit called without queued P2P operations")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for request in self._reqs:
            request.wait()
        self._reqs = None
        self._ops = []


__all__ = ["RingComm"]
