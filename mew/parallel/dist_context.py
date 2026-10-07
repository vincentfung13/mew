from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
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
            is_main=(rank == 0),
            device=torch.device("cuda", local_rank),
        )

    @property
    def is_main(self) -> bool:
        return self.rank == 0
