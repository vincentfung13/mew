import logging
import timeit
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from mew.optimizers.adamw import AdamW
from profiling.cases import build_profiling_case
from profiling.protocols import parse_protocol, run_protocol

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
    protocol = parse_protocol(cfg.profiling.protocol)
    eager_module = case.module
    LOGGER.info(
        "Initialized profiling target %s with protocol %s and case config: %s",
        cfg.case.name,
        protocol.value,
        cfg.case,
    )

    nvtx_handles: List[torch.utils.hooks.RemovableHandle] = []
    module_annotation_enabled = is_cuda and cfg.profiling.nvtx.annotate_modules
    compile_disabled = not cfg.torch_compile.enable
    if module_annotation_enabled and compile_disabled:
        nvtx_handles = _attach_nvtx_hooks(eager_module)
        LOGGER.info("Attached per-module NVTX hooks.")
    elif is_cuda and cfg.profiling.nvtx.annotate_modules:
        LOGGER.info(
            "Skipping per-module NVTX hooks because they can cause graph breaks "
            "with torch.compile."
        )

    if cfg.torch_compile.enable:
        module = torch.compile(eager_module, mode=cfg.torch_compile.mode)
        LOGGER.info("Enabled torch.compile with mode=%s.", cfg.torch_compile.mode)
    else:
        module = eager_module

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
        params=eager_module.parameters(),
        lr=0.01,
        weight_decay=0.01,
        betas=[0.9, 0.95],
        eps=0.01,
    )

    def _now() -> float:
        if is_cuda:
            torch.cuda.synchronize()
        return timeit.default_timer()

    timings: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def _stage(name: str, *, collect_timings: bool):
        if collect_timings:
            start = _now()
        if is_cuda:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if is_cuda:
                torch.cuda.nvtx.range_pop()
            if collect_timings:
                timings[name].append(_now() - start)

    LOGGER.info("Starting to warm up...")
    with torch.autocast(
        device_type=cfg.device, dtype=amp_dtype, enabled=cfg.amp.enable
    ):
        run_protocol(
            protocol,
            steps=cfg.profiling.warmup_steps,
            module=module,
            case=case,
            optimizer=optim,
            stage=lambda name: _stage(name, collect_timings=False),
        )

    if is_cuda:
        torch.cuda.synchronize()

    if is_cuda and cfg.profiling.nvtx.use_cudart_range:
        torch.cuda.profiler.start()
    if cfg.profiling.memory_profiling.enable and is_cuda:
        LOGGER.info("Enabling GPU memory profiling...")
        torch.cuda.memory._record_memory_history(enabled="all")

    LOGGER.info("Starting profiling execution...")
    with torch.autocast(
        device_type=cfg.device, dtype=amp_dtype, enabled=cfg.amp.enable
    ):
        run_protocol(
            protocol,
            steps=cfg.profiling.exec_steps,
            module=module,
            case=case,
            optimizer=optim,
            stage=lambda name: _stage(name, collect_timings=True),
        )

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
    stage_order = (
        "forward",
        "forward_phase",
        "backward",
        "backward_phase",
        "optimizer_step",
        "total",
    )
    for stage in stage_order:
        samples = timings.get(stage, [])
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
