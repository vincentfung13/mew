import hydra
import logging
import timeit
from omegaconf import DictConfig
from tqdm import tqdm

import numpy as np
import torch

from mew.nn.utils import build_model
from mew.nn.functionals import cross_entropy
from mew.optimizers.adamw import AdamW

LOGGER = logging.getLogger(__name__)


def _validate_cfg(cfg: DictConfig) -> None:
    if cfg.profiling.warmup.run_optimizer_step:
        assert (
            cfg.profiling.warmup.run_backward
        ), "Cannot run opt step without running backward!"

    if cfg.profiling.exec.run_optimizer_step:
        assert (
            cfg.profiling.exec.run_backward
        ), "Cannot run opt step without running backward!"


@hydra.main(version_base=None, config_path="cfgs", config_name="profiling")
def main(cfg: DictConfig) -> None:
    _validate_cfg(cfg)

    # Init model
    model = build_model(cfg).to(cfg.device)
    LOGGER.info(f"Initialized model with config: {cfg.model}")

    # Init fake optim and fake data
    optim = AdamW(
        params=model.parameters(),
        # These params does not matter since we're
        # only doing profiling
        lr=0.01,
        weight_decay=0.01,
        betas=[0.9, 0.95],
        eps=0.01,
    )

    # Sample random batch (reused repeatedly)
    fake_data = torch.randint(
        low=0,
        high=cfg.model.vocab_size - 1,
        size=(cfg.data.batch_size, cfg.data.seq_len),
    ).to(cfg.device)
    fake_label = torch.randint(
        low=0,
        high=cfg.model.vocab_size - 1,
        size=(cfg.data.batch_size, cfg.data.seq_len),
    ).to(cfg.device)

    # Warmup
    LOGGER.info("Starting to warmup...")
    for _ in tqdm(list(range(cfg.profiling.warmup.steps))):
        optim.zero_grad()
        logits = model(fake_data)
        loss = cross_entropy(logits, fake_label)
        if cfg.profiling.warmup.run_backward:
            loss.backward()
        if cfg.profiling.warmup.run_optimizer_step:
            optim.step()

    # Wait for the warmup to finish
    if cfg.device == "cuda":
        torch.cuda.synchronize()

    # Exec
    latencies = []
    LOGGER.info("Starting to execute profiling...")
    for _ in tqdm(list(range(cfg.profiling.exec.steps))):
        optim.zero_grad()
        start = timeit.default_timer()
        logits = model(fake_data)
        loss = cross_entropy(logits, fake_label)
        if cfg.profiling.exec.run_backward:
            loss.backward()
        if cfg.profiling.exec.run_optimizer_step:
            optim.step()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        latencies.append(end - start)

    # Compute and log statistics
    latencies_array = np.array(latencies)
    avg_latency = np.mean(latencies_array)
    variance = np.var(latencies_array)
    p95 = np.percentile(latencies_array, 95)
    p99 = np.percentile(latencies_array, 99)

    LOGGER.info("Profiling Results:")
    LOGGER.info(f"  Average latency: {avg_latency:.6f}s")
    LOGGER.info(f"  Variance: {variance:.6f}s²")
    LOGGER.info(f"  P95 latency: {p95:.6f}s")
    LOGGER.info(f"  P99 latency: {p99:.6f}s")


if __name__ == "__main__":
    main()
