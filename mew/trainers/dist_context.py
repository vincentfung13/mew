from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @classmethod
    def from_env(cls) -> "DistContext":
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=torch.device("cuda", local_rank),
        )
