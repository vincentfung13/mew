import logging
import timeit
from contextlib import nullcontext
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from mew.optimizers.adamw import AdamW
from profiling.cases import build_profiling_case

LOGGER = logging.getLogger(__name__)


def _attach_nvtx_hooks(
    model: torch.nn.Module,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach forward hooks that emit an NVTX range per submodule."""
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


@hydra.main(version_base=None, config_path="configs", config_name="profiling")
def main(cfg: DictConfig) -> None:
    is_cuda = cfg.device == "cuda"
    output_dir = Path(cfg.profiling.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_profile_output_path = output_dir / f"{output_dir.name}.pkl"

    case = build_profiling_case(cfg, device=cfg.device)
    module = case.module
    LOGGER.info(
        "Initialized profiling target %s with case config: %s",
        cfg.case.name,
        cfg.case,
    )

    nvtx_handles: List[torch.utils.hooks.RemovableHandle] = []
    if is_cuda and cfg.profiling.nvtx.annotate_modules:
        nvtx_handles = _attach_nvtx_hooks(module)
        LOGGER.info("Attached per-module NVTX hooks.")

    if cfg.amp.enable:
        try:
            amp_dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[cfg.amp.dtype]
        except KeyError as error:
            raise ValueError(f"Unsupported AMP dtype: {cfg.amp.dtype}") from error
    else:
        amp_dtype = torch.bfloat16

    optim = AdamW(
        params=module.parameters(),
        lr=0.01,
        weight_decay=0.01,
        betas=[0.9, 0.95],
        eps=0.01,
    )

    def _now() -> float:
        if is_cuda:
            torch.cuda.synchronize()
        return timeit.default_timer()

    def _nvtx_range(name: str):
        if is_cuda:
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    def _profiling_step_once():
        timings = {}
        total_start = _now()
        if cfg.profiling.forward_only:
            with torch.no_grad(), _nvtx_range("forward"):
                module(*case.inputs)
            forward_end = _now()
            timings["forward"] = forward_end - total_start
        else:
            optim.zero_grad()
            with _nvtx_range("forward"):
                output = module(*case.inputs)
            loss = case.loss_fn(output)
            forward_end = _now()
            timings["forward"] = forward_end - total_start

            with _nvtx_range("backward"):
                backward_start = _now()
                loss.backward()
                backward_end = _now()
                timings["backward"] = backward_end - backward_start

            with _nvtx_range("optimizer_step"):
                optimizer_start = _now()
                optim.step()
                optimizer_end = _now()
                timings["optimizer_step"] = optimizer_end - optimizer_start
        total_end = _now()
        timings["total"] = total_end - total_start
        return timings

    LOGGER.info("Starting to warm up...")
    for _ in tqdm(range(cfg.profiling.warmup_steps)):
        with torch.autocast(
            device_type=cfg.device, dtype=amp_dtype, enabled=cfg.amp.enable
        ):
            _profiling_step_once()

    if is_cuda:
        torch.cuda.synchronize()

    if is_cuda and cfg.profiling.nvtx.use_cudart_range:
        torch.cuda.profiler.start()
    timings: dict[str, list[float]] = {
        "forward": [],
        "backward": [],
        "optimizer_step": [],
        "total": [],
    }

    if cfg.profiling.memory_profiling.enable and is_cuda:
        LOGGER.info("Enabling GPU memory profiling...")
        torch.cuda.memory._record_memory_history(enabled="all")

    LOGGER.info("Starting profiling execution...")
    for _ in tqdm(range(cfg.profiling.exec_steps)):
        with torch.autocast(
            device_type=cfg.device, dtype=amp_dtype, enabled=cfg.amp.enable
        ):
            step_timings = _profiling_step_once()
            for stage, duration in step_timings.items():
                timings[stage].append(duration)

    if cfg.profiling.memory_profiling.enable and is_cuda:
        torch.cuda.memory._dump_snapshot(str(memory_profile_output_path))
        LOGGER.info("Memory profiling results dumped to %s", memory_profile_output_path)
        torch.cuda.memory._record_memory_history(enabled=None)

    if is_cuda and cfg.profiling.nvtx.use_cudart_range:
        torch.cuda.profiler.stop()

    for handle in nvtx_handles:
        handle.remove()

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
        values = np.array(samples)
        LOGGER.info(
            f"{stage:<16}{len(values):>8}{np.mean(values):>14.6f}"
            f"{np.var(values):>16.6e}{np.percentile(values, 95):>14.6f}"
            f"{np.percentile(values, 99):>14.6f}"
        )


if __name__ == "__main__":
    main()
