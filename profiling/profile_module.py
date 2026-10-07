"""
Profile one training step of the model in the training config.

The step is `mew.trainers.npt_trainer.TrainStep`, the same code `NPTTrainer`
runs, fed a fixed synthetic batch that is already on the device. The results
are therefore a compute-only upper bound for training: no data loading,
validation, logging or checkpointing.

Reports per-stage timing (forward, backward, optimizer step, total), achieved
TFLOP/s and MFU, and peak memory over the timed steps; optionally adds NVTX
ranges for Nsight Systems and dumps a CUDA memory snapshot.
"""

import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mew.perf.gpu_specs import load_gpu_specs, peak_tflops_per_second
from mew.perf.utils import compute_mfu, module_flops_per_token, peak_memory_stats
from mew.parallel.dist_context import DistContext
from mew.trainers.npt_trainer import TrainStep

LOGGER = logging.getLogger(__name__)

Stage = Callable[[str], AbstractContextManager]

STAGES = ("forward", "backward", "optimizer_step", "total")

# Model FLOPs of each stage as a multiple of the forward FLOPs. The optimizer
# step does no model FLOPs, so it has no MFU.
FORWARD_FLOP_MULTIPLIERS = {"forward": 1, "backward": 2, "total": 3}

AMP_DTYPES = {"bf16": torch.bfloat16}


@dataclass(frozen=True)
class StageSummary:
    stage: str
    count: int
    mean_s: float
    var_s2: float
    p95_s: float
    p99_s: float
    # Tokens per second of whole steps; only set for "total".
    tokens_per_s: float | None
    # None for stages without model FLOPs.
    achieved_tflops: float | None
    # None without model FLOPs or without a known peak (e.g. on CPU).
    mfu: float | None


def build_dist_context(device_type: str) -> DistContext:
    if device_type != "cuda":
        return DistContext(
            rank=0, local_rank=0, world_size=1, device=torch.device(device_type)
        )
    dist_context = DistContext.from_env()
    if dist_context.world_size > 1:
        raise NotImplementedError("Distributed profiling is not supported yet.")
    torch.cuda.set_device(dist_context.device)
    return dist_context


def resolve_peak_tflops(cfg: DictConfig, device: torch.device) -> float | None:
    # Same resolution as apps/launch_training.py: an explicit
    # trainer.perf.peak_tflops wins; otherwise look the GPU up in gpu_specs.
    # Undefined (None) on non-CUDA devices.
    if cfg.trainer.perf.peak_tflops is not None:
        return float(cfg.trainer.perf.peak_tflops)
    if device.type != "cuda":
        return None
    dtype = (
        AMP_DTYPES[cfg.trainer.amp.dtype] if cfg.trainer.amp.enable else torch.float32
    )
    return peak_tflops_per_second(device, dtype, load_gpu_specs(cfg.gpu_specs))


def synthetic_batch(
    cfg: DictConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (cfg.data.batch_size, cfg.data.seq_len)
    generator = torch.Generator().manual_seed(0)
    data = torch.randint(0, cfg.model.vocab_size, shape, generator=generator)
    target = torch.randint(0, cfg.model.vocab_size, shape, generator=generator)
    return data.to(device), target.to(device)


def run_steps(
    train_step: TrainStep,
    data: torch.Tensor,
    target: torch.Tensor,
    *,
    first_step: int,
    num_steps: int,
    grad_accumulation_steps: int,
    stage: Stage = nullcontext,
) -> None:
    # Mirrors the order of NPTTrainer.train(); the optimizer steps only on
    # gradient-accumulation boundaries, as in training.
    for step in range(first_step, first_step + num_steps):
        with stage("total"):
            with stage("forward"):
                loss = train_step.forward(data=data, target=target)
            with stage("backward"):
                train_step.backward(loss)
            if step % grad_accumulation_steps == 0:
                with stage("optimizer_step"):
                    train_step.optim_and_lr_scheduler_step()


def summarize(
    timings: Mapping[str, Sequence[float]],
    tokens_per_step: int,
    forward_flops_per_token: int,
    peak_tflops: float | None,
) -> list[StageSummary]:
    rows = []
    for stage in STAGES:
        samples = timings.get(stage, [])
        if not samples:
            continue
        values = np.asarray(samples, dtype=np.float64)
        mean_s = float(values.mean())

        tokens_per_s = tokens_per_step / mean_s if stage == "total" else None
        achieved_tflops = mfu = None
        multiplier = FORWARD_FLOP_MULTIPLIERS.get(stage)
        if multiplier is not None:
            stage_tokens_per_s = tokens_per_step / mean_s
            stage_flops_per_token = multiplier * forward_flops_per_token
            achieved_tflops = stage_tokens_per_s * stage_flops_per_token / 1e12
            if peak_tflops is not None:
                mfu = compute_mfu(
                    tokens_per_s=stage_tokens_per_s,
                    model_flops_per_token=stage_flops_per_token,
                    peak_tflops=peak_tflops,
                )

        rows.append(
            StageSummary(
                stage=stage,
                count=len(values),
                mean_s=mean_s,
                var_s2=float(values.var()),
                p95_s=float(np.percentile(values, 95)),
                p99_s=float(np.percentile(values, 99)),
                tokens_per_s=tokens_per_s,
                achieved_tflops=achieved_tflops,
                mfu=mfu,
            )
        )
    return rows


def format_table(rows: Sequence[StageSummary]) -> list[str]:
    def optional(value: float | None, fmt: str) -> str:
        return "-" if value is None else format(value, fmt)

    header = (
        f"{'stage':<16}{'count':>7}{'mean (s)':>12}{'var (s^2)':>13}"
        f"{'p95 (s)':>12}{'p99 (s)':>12}{'tokens/s':>14}{'TFLOP/s':>10}{'MFU':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.stage:<16}{row.count:>7}{row.mean_s:>12.6f}{row.var_s2:>13.3e}"
            f"{row.p95_s:>12.6f}{row.p99_s:>12.6f}"
            f"{optional(row.tokens_per_s, ',.0f'):>14}"
            f"{optional(row.achieved_tflops, '.2f'):>10}"
            f"{optional(row.mfu, '.2%'):>9}"
        )
    return lines


def artifact_path(output_dir: Path, suffix: str) -> Path:
    # Artifacts are named after their output directory.
    return output_dir / f"{output_dir.name}{suffix}"


def _attach_nvtx_hooks(
    model: torch.nn.Module,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach forward hooks that emit an NVTX range per submodule."""
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for _, module in model.named_modules():
        label = type(module).__name__

        def _pre_hook(_module, _inputs, name=label):
            torch.cuda.nvtx.range_push(name)

        def _post_hook(_module, _inputs, _output):
            torch.cuda.nvtx.range_pop()

        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))
    return handles


def profile(cfg: DictConfig) -> dict:
    """Run warmup and timed steps; log the results and write metrics.json."""
    dist_context = build_dist_context(cfg.device)
    device = dist_context.device
    is_cuda = device.type == "cuda"
    output_dir = Path(cfg.profiling.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("seed") is not None:
        torch.manual_seed(cfg.seed)

    train_step = TrainStep(cfg=cfg, dist_context=dist_context)
    data, target = synthetic_batch(cfg, device)
    tokens_per_step = data.numel()
    forward_flops_per_token = module_flops_per_token(
        module=train_step.model, seq_len=cfg.data.seq_len
    )
    peak_tflops = resolve_peak_tflops(cfg, device)
    grad_accumulation_steps = cfg.optim.grad_accumulation_steps
    LOGGER.info(
        "Profiling TrainStep on %s: %d tokens/step, %d forward FLOPs/token, peak %s TFLOP/s",
        device,
        tokens_per_step,
        forward_flops_per_token,
        peak_tflops,
    )

    nvtx_handles = []
    if is_cuda and cfg.profiling.nvtx.annotate_modules:
        nvtx_handles = _attach_nvtx_hooks(train_step.model)
        LOGGER.info("Attached per-module NVTX hooks.")

    def _now() -> float:
        if is_cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()

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
    run_steps(
        train_step,
        data,
        target,
        first_step=1,
        num_steps=cfg.profiling.warmup_steps,
        grad_accumulation_steps=grad_accumulation_steps,
        stage=lambda name: _stage(name, collect_timings=False),
    )

    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    if is_cuda and cfg.profiling.nvtx.use_cudart_range:
        torch.cuda.profiler.start()
    memory_profiling = is_cuda and cfg.profiling.memory_profiling.enable
    if memory_profiling:
        LOGGER.info("Enabling GPU memory profiling...")
        torch.cuda.memory._record_memory_history(enabled="all")

    LOGGER.info("Starting profiling execution...")
    run_steps(
        train_step,
        data,
        target,
        first_step=cfg.profiling.warmup_steps + 1,
        num_steps=cfg.profiling.exec_steps,
        grad_accumulation_steps=grad_accumulation_steps,
        stage=lambda name: _stage(name, collect_timings=True),
    )

    if memory_profiling:
        snapshot_path = artifact_path(output_dir, ".pkl")
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        LOGGER.info("Memory profiling results dumped to %s", snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
    if is_cuda and cfg.profiling.nvtx.use_cudart_range:
        torch.cuda.profiler.stop()
    peak_memory = peak_memory_stats(device)

    for handle in nvtx_handles:
        handle.remove()

    rows = summarize(timings, tokens_per_step, forward_flops_per_token, peak_tflops)
    LOGGER.info("Profiling Results:")
    for line in format_table(rows):
        LOGGER.info(line)
    for key, value in peak_memory.items():
        LOGGER.info("%s: %.2f", key, value)
    if nvtx_handles or memory_profiling:
        LOGGER.warning(
            "Timings include instrumentation overhead (per-module NVTX hooks or "
            "memory history recording). For clean MFU numbers, rerun with "
            "profiling.nvtx.annotate_modules=false profiling.memory_profiling.enable=false."
        )

    metrics = {
        "stages": [asdict(row) for row in rows],
        "peak_memory": peak_memory,
        "tokens_per_step": tokens_per_step,
        "forward_flops_per_token": forward_flops_per_token,
        "peak_tflops": peak_tflops,
        "device": str(device),
        "config": {
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "data": {"batch_size": cfg.data.batch_size, "seq_len": cfg.data.seq_len},
            "amp": OmegaConf.to_container(cfg.trainer.amp, resolve=True),
            "grad_accumulation_steps": grad_accumulation_steps,
            "warmup_steps": cfg.profiling.warmup_steps,
            "exec_steps": cfg.profiling.exec_steps,
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Metrics written to %s", metrics_path)
    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="profile_module")
def main(cfg: DictConfig) -> None:
    profile(cfg)


if __name__ == "__main__":
    main()
