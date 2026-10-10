from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @classmethod
    def set_up_dist_env(cls, dist_backend: str) -> "DistContext":
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if dist_backend == "gloo":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)

        # Init process group if world size > 0
        if world_size > 1:
            dist.init_process_group(
                backend=dist_backend, init_method="env://", device_id=device
            )

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
