"""
CPU tests for profiling/profile_module.py: the profiler runs the trainer's
TrainStep on a synthetic batch, with the training config composed into its own.

CUDA timing, NVTX, memory snapshots and Nsight capture still need a GPU for
end-to-end validation.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from mew.perf.utils import compute_mfu, module_flops_per_token
from mew.trainers.dist_context import DistContext
from mew.trainers.npt_trainer import TrainStep
from profiling.profile_module import (
    StageSummary,
    build_dist_context,
    profile,
    resolve_peak_tflops,
    run_steps,
    summarize,
    synthetic_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "profiling" / "configs"

TINY_CPU_OVERRIDES = [
    "device=cpu",
    "model.d_model=32",
    "model.d_ff=64",
    "model.num_heads=2",
    "model.num_transformer_layers=1",
    "model.vocab_size=64",
    "data.batch_size=2",
    "data.seq_len=8",
    "trainer.amp.enable=false",
    "profiling.warmup_steps=1",
    "profiling.exec_steps=3",
    "profiling.memory_profiling.enable=false",
]


def _compose(overrides=()):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="profile_module", overrides=list(overrides))


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    # hydra.searchpath (file://apps/cfgs) is relative to the working directory.
    monkeypatch.chdir(REPO_ROOT)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_is_the_training_config_plus_profiling_settings():
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "apps" / "cfgs"), version_base=None
    ):
        training = compose(config_name="training")
    cfg = _compose()

    for section in ("model", "data", "optim"):
        assert cfg[section] == training[section]
    assert cfg.trainer.amp == training.trainer.amp
    assert len(cfg.gpu_specs) == len(training.gpu_specs)
    assert {
        "output_dir",
        "warmup_steps",
        "exec_steps",
        "nvtx",
        "memory_profiling",
    } <= set(cfg.profiling)
    # Model-level only, no separate execution protocols or compile option.
    assert "case" not in cfg
    assert "protocol" not in cfg.profiling
    assert "torch_compile" not in cfg.profiling


def test_training_keys_override_the_profiled_model():
    cfg = _compose(["model.attn_impl=flash_triton", "data.batch_size=64"])
    assert cfg.model.attn_impl == "flash_triton"
    assert cfg.data.batch_size == 64


# ---------------------------------------------------------------------------
# Peak TFLOP/s and device
# ---------------------------------------------------------------------------


def test_explicit_peak_tflops_wins():
    cfg = _compose(["trainer.perf.peak_tflops=123.0"])
    assert resolve_peak_tflops(cfg, torch.device("cpu")) == 123.0


def test_peak_tflops_is_undefined_off_cuda():
    assert resolve_peak_tflops(_compose(), torch.device("cpu")) is None


@pytest.mark.parametrize(
    "amp_enable, expected_tflops",
    [(True, 989.4), (False, 66.9)],  # bf16 tensor cores vs. fp32 ("highest")
)
def test_peak_tflops_follows_gpu_and_amp(monkeypatch, amp_enable, expected_tflops):
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _: "NVIDIA H100 80GB HBM3"
    )
    cfg = _compose([f"trainer.amp.enable={str(amp_enable).lower()}"])
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        peak = resolve_peak_tflops(cfg, torch.device("cuda", 0))
    finally:
        torch.set_float32_matmul_precision(previous)
    assert peak == pytest.approx(expected_tflops)


def test_cpu_dist_context_is_single_process():
    dist_context = build_dist_context("cpu")
    assert dist_context.device == torch.device("cpu")
    assert (dist_context.rank, dist_context.world_size) == (0, 1)


# ---------------------------------------------------------------------------
# Running TrainStep
# ---------------------------------------------------------------------------


def _recording_stage(calls):
    @contextmanager
    def stage(name):
        calls.append(name)
        yield

    return stage


@pytest.mark.parametrize("grad_accumulation_steps", [1, 2])
def test_run_steps_runs_the_training_step_stages(grad_accumulation_steps):
    cfg = _compose(TINY_CPU_OVERRIDES)
    device = torch.device("cpu")
    train_step = TrainStep(cfg=cfg, dist_context=DistContext(0, 0, 1, device))
    data, target = synthetic_batch(cfg, device)
    calls = []

    run_steps(
        train_step,
        data,
        target,
        first_step=1,
        num_steps=4,
        grad_accumulation_steps=grad_accumulation_steps,
        stage=_recording_stage(calls),
    )

    assert calls.count("total") == 4
    assert calls.count("forward") == 4
    assert calls.count("backward") == 4
    # As in training, the optimizer only steps on accumulation boundaries.
    assert calls.count("optimizer_step") == 4 // grad_accumulation_steps
    assert calls[:3] == ["total", "forward", "backward"]


def test_synthetic_batch_matches_the_training_shape():
    cfg = _compose(TINY_CPU_OVERRIDES)
    data, target = synthetic_batch(cfg, torch.device("cpu"))
    assert data.shape == target.shape == (2, 8)
    assert int(data.max()) < cfg.model.vocab_size


# ---------------------------------------------------------------------------
# Summary math
# ---------------------------------------------------------------------------

TIMINGS = {
    "forward": [0.1, 0.1],
    "backward": [0.2, 0.2],
    "optimizer_step": [0.05],
    "total": [0.35, 0.35],
}


def _by_stage(rows):
    return {row.stage: row for row in rows}


def test_summary_counts_forward_backward_and_total_flops():
    # 1000 tokens/step at 1e9 forward FLOPs/token on a 100 TFLOP/s GPU.
    rows = _by_stage(summarize(TIMINGS, 1000, 1_000_000_000, peak_tflops=100.0))

    assert rows["forward"].achieved_tflops == pytest.approx(10.0)
    assert rows["forward"].mfu == pytest.approx(0.1)
    # Backward is 2x the forward FLOPs in 2x the time.
    assert rows["backward"].achieved_tflops == pytest.approx(10.0)
    assert rows["backward"].mfu == pytest.approx(0.1)
    # Total is 3x the forward FLOPs per token, the same as the trainer's MFU.
    total_tokens_per_s = 1000 / 0.35
    assert rows["total"].tokens_per_s == pytest.approx(total_tokens_per_s)
    assert rows["total"].mfu == pytest.approx(
        compute_mfu(total_tokens_per_s, 3_000_000_000, 100.0)
    )


def test_optimizer_step_has_no_model_flops():
    rows = _by_stage(summarize(TIMINGS, 1000, 1_000_000_000, peak_tflops=100.0))
    optimizer = rows["optimizer_step"]
    assert optimizer.count == 1
    assert optimizer.achieved_tflops is None
    assert optimizer.mfu is None
    assert optimizer.tokens_per_s is None


def test_unknown_peak_reports_tflops_without_mfu():
    rows = _by_stage(summarize(TIMINGS, 1000, 1_000_000_000, peak_tflops=None))
    assert rows["forward"].achieved_tflops == pytest.approx(10.0)
    assert all(row.mfu is None for row in rows.values())


def test_summary_keeps_stage_order_and_skips_missing_stages():
    timings = {"total": [0.3], "forward": [0.1], "backward": [0.2]}
    rows = summarize(timings, 1000, 1_000_000_000, peak_tflops=None)
    assert [row.stage for row in rows] == ["forward", "backward", "total"]
    assert all(isinstance(row, StageSummary) for row in rows)


# ---------------------------------------------------------------------------
# End to end on CPU
# ---------------------------------------------------------------------------


def test_profile_writes_metrics_on_cpu(tmp_path):
    output_dir = tmp_path / "tiny_cpu"
    cfg = _compose(TINY_CPU_OVERRIDES + [f"profiling.output_dir={output_dir}"])

    metrics = profile(cfg)

    written = json.loads((output_dir / "metrics.json").read_text())
    assert written == metrics
    stages = {row["stage"]: row for row in metrics["stages"]}
    assert set(stages) == {"forward", "backward", "optimizer_step", "total"}
    assert stages["total"]["count"] == 3
    assert stages["total"]["tokens_per_s"] > 0
    # No peak FLOP/s or CUDA memory stats on CPU.
    assert metrics["peak_tflops"] is None
    assert all(row["mfu"] is None for row in metrics["stages"])
    assert metrics["peak_memory"] == {}

    train_step = TrainStep(
        cfg=cfg, dist_context=DistContext(0, 0, 1, torch.device("cpu"))
    )
    assert metrics["tokens_per_step"] == 2 * 8
    assert metrics["forward_flops_per_token"] == module_flops_per_token(
        train_step.model, cfg.data.seq_len
    )
