import torch
import os
import numpy as np
import logging
from typing import Tuple

LOGGER = logging.getLogger(__name__)


class NumpyBatchLoader:
    def __init__(
        self,
        data: str | os.PathLike | np.ndarray,
        seq_len: int,
        batch_size: int,
        dtype=np.uint16,
        is_training: bool = True,
        seed_seq: np.random.SeedSequence | None = None,
    ):
        """
        Memory-optimized loader for massive memmap files.
        """
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.memmap(data, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_num = len(self.data)
        self.is_training = is_training

        if seed_seq is None:
            self.seed_seq = np.random.SeedSequence()
        else:
            self.seed_seq = seed_seq

        # Instead of every index, we track "chunks" or "offsets"
        # Total valid starting positions
        # (only effective in validation)
        self.num_samples = self.token_num - seq_len - 1
        # We calculate how many batches we can fit in one epoch
        self.num_batches = self.num_samples // batch_size

        LOGGER.info(
            f"Mapped {self.token_num} tokens. "
            f"Approx {self.num_batches} batches per epoch."
        )

    def get_batch(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sampling with replacement during training & training-time validation
        # indices = np.random.randint(
        #     0, len(self.data) - self.seq_len, size=self.batch_size
        # )
        child = self.seed_seq.spawn(1)[0]
        indices = (
            np.random.default_rng(child)
            .integers(0, len(self.data) - self.seq_len, self.batch_size)
            .tolist()
        )

        # Efficient batch construction
        x_list, y_list = [], []
        for i in indices:
            # Slicing memmap is O(1) memory as it returns a view
            x_list.append(self.data[i : i + self.seq_len])
            y_list.append(self.data[i + 1 : i + self.seq_len + 1])

        # Convert to torch directly from numpy views
        x = torch.from_numpy(np.stack(x_list)).to(device).long()
        y = torch.from_numpy(np.stack(y_list)).to(device).long()

        return x, y

    def resume(self, num_batches_drawn: int) -> None:
        self.seed_seq = np.random.SeedSequence(
            entropy=self.seed_seq.entropy,
            spawn_key=self.seed_seq.spawn_key,
            pool_size=self.seed_seq.pool_size,
            n_children_spawned=num_batches_drawn,
        )

    def num_batches_spawned(self) -> int:
        return self.seed_seq.n_children_spawned
