import hydra
import logging
import timeit
from typing import List
from omegaconf import DictConfig
from tqdm import tqdm

import numpy as np
import torch

from mew.nn.utils import build_model
from mew.nn.functionals import cross_entropy
from mew.optimizers.adamw import AdamW

LOGGER = logging.getLogger(__name__)


def attach_nvtx_hooks(
    model: torch.nn.Module,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach forward pre/post hooks that emit an NVTX range per submodule.

    Each module's forward is bracketed by ``range_push``/``range_pop`` so that
    Nsight Systems renders the modules as a hierarchy nested under whatever
    higher-level range (e.g. "forward") is open on the same thread. The range is
    labelled ``<ClassName>`` (e.g. ``SwiGLU``, ``CausalMultiHeadSelfAttn``).

    Returns the list of registered handles so the caller can remove them.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for _, module in model.named_modules():
        label = type(module).__name__

        def _pre_hook(_module, _inputs, name=label):
            torch.cuda.nvtx.range_push(name)

        def _post_hook(_module, _inputs, _output):
            torch.cuda.nvtx.range_pop()

        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))
    return handles


@hydra.main(version_base=None, config_path="cfgs", config_name="profiling")
def main(cfg: DictConfig) -> None:
    # Init model
    model = build_model(cfg, device=cfg.device)
    LOGGER.info(f"Initialized model with config: {cfg.model}")

    # Optionally attach per-module NVTX hooks so Nsight Systems shows a
    # hierarchy (e.g. SwiGLU / CausalMultiHeadSelfAttn nested under "forward").
    nvtx_handles: List[torch.utils.hooks.RemovableHandle] = []
    if cfg.profiling.nvtx.annotate_modules:
        nvtx_handles = attach_nvtx_hooks(model)
        LOGGER.info("Attached per-module NVTX hooks.")

    # AMP config
    try:
        amp_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.amp.dtype]
    except KeyError:
        raise ValueError(f"Unsupported AMP dtype: {cfg.amp.dtype}")

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
    with torch.autocast(
        device_type=cfg.device, dtype=amp_dtype, enabled=cfg.amp.enable
    ):
        for _ in tqdm(list(range(cfg.profiling.warmup_steps))):
            optim.zero_grad()
            logits = model(fake_data)
            loss = cross_entropy(logits, fake_label)
            loss.backward()
            optim.step()

    # Wait for the warmup to finish
    if cfg.device == "cuda":
        torch.cuda.synchronize()

    # Exec
    is_cuda = cfg.device == "cuda"

    def _now() -> float:
        # Synchronize before reading the clock so that each measured stage
        # only accounts for its own (asynchronous) CUDA work.
        if is_cuda:
            torch.cuda.synchronize()
        return timeit.default_timer()

    # Per-stage latency samples (in seconds).
    if is_cuda and cfg.profiling.nvtx.use_cudart_range:
        # Pair with `nsys profile --capture-range=cudaProfilerApi` so only the
        # measured (post-warmup) steps are recorded in the report.
        torch.cuda.profiler.start()
    timings: dict[str, list[float]] = {
        "forward": [],
        "backward": [],
        "optimizer_step": [],
        "total": [],
    }

    # Enable GPU memory profiling
    if cfg.profiling.memory_profiling.enable and cfg.device == "cuda":
        LOGGER.info("Enabling GPU memory profiling...")
        torch.cuda.memory._record_memory_history(enabled="all")

    LOGGER.info("Starting to execute profiling...")
    with torch.autocast(
        device_type=cfg.device, dtype=amp_dtype, enabled=cfg.amp.enable
    ):
        for _ in tqdm(list(range(cfg.profiling.exec_steps))):
            optim.zero_grad()

            total_start = _now()

            forward_start = total_start
            with torch.cuda.nvtx.range("forward"):
                logits = model(fake_data)
                loss = cross_entropy(logits, fake_label)
                forward_end = _now()
                timings["forward"].append(forward_end - forward_start)

            with torch.cuda.nvtx.range("backward"):
                backward_start = forward_end
                loss.backward()
                backward_end = _now()
                timings["backward"].append(backward_end - backward_start)

            with torch.cuda.nvtx.range("optimizer_step"):
                optimizer_start = _now()
                optim.step()
                optimizer_end = _now()
                timings["optimizer_step"].append(optimizer_end - optimizer_start)

            total_end = _now()
            timings["total"].append(total_end - total_start)

    if cfg.profiling.memory_profiling.enable and cfg.device == "cuda":
        torch.cuda.memory._dump_snapshot(cfg.profiling.memory_profiling.output_path)
        LOGGER.info(
            f"Mem profiling results dumped to {cfg.profiling.memory_profiling.output_path}..."
        )
        torch.cuda.memory._record_memory_history(enabled=None)

    if is_cuda and cfg.profiling.nvtx.use_cudart_range:
        torch.cuda.profiler.stop()

    # Remove NVTX hooks now that measurement is done.
    for handle in nvtx_handles:
        handle.remove()

    # Compute and log statistics as a table.
    LOGGER.info("Profiling Results:")
    header = (
        f"{'stage':<16}{'count':>8}{'mean (s)':>14}"
        f"{'var (s^2)':>16}{'p95 (s)':>14}{'p99 (s)':>14}"
    )
    LOGGER.info(header)
    LOGGER.info("-" * len(header))
    for stage in ("forward", "backward", "optimizer_step", "total"):
        samples = timings[stage]
        if not samples:
            continue
        arr = np.array(samples)
        LOGGER.info(
            f"{stage:<16}{len(arr):>8}{np.mean(arr):>14.6f}"
            f"{np.var(arr):>16.6e}{np.percentile(arr, 95):>14.6f}"
            f"{np.percentile(arr, 99):>14.6f}"
        )


if __name__ == "__main__":
    main()
