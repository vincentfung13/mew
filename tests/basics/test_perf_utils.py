"""
Tests for the shared MFU and memory helpers in mew/perf/utils.py, used by both
`ThroughputMeter` (training) and the module profiler:

- `compute_mfu(tokens_per_s, model_flops_per_token, peak_tflops)`: achieved
  FLOP/s over peak FLOP/s. The peak is in TFLOP/s; this is the one place that
  converts it.
- `peak_memory_stats(device)`: peak allocated/reserved CUDA memory in GiB since
  the last `torch.cuda.reset_peak_memory_stats`, or `{}` for non-CUDA devices.
"""

import pytest
import torch

from mew.perf.utils import compute_mfu, peak_memory_stats

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_mfu_converts_tflops_once():
    # 1e6 tokens/s x 1e6 FLOPs/token = 1e12 FLOP/s = exactly a 1 TFLOP/s peak.
    assert compute_mfu(
        tokens_per_s=1e6, model_flops_per_token=1_000_000, peak_tflops=1.0
    ) == pytest.approx(1.0)


def test_mfu_matches_training_run_numbers():
    # The default training config on an H100 SXM: ~88.6k tokens/s at
    # 108,576,768 training FLOPs/token is ~0.97% of 989.4 TFLOP/s.
    mfu = compute_mfu(
        tokens_per_s=88_578, model_flops_per_token=108_576_768, peak_tflops=989.4
    )
    assert mfu == pytest.approx(0.00972, rel=1e-3)


@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_mfu_is_linear_in_throughput_and_inverse_in_peak(scale):
    base = compute_mfu(tokens_per_s=1e5, model_flops_per_token=1e8, peak_tflops=312.0)
    faster = compute_mfu(
        tokens_per_s=1e5 * scale, model_flops_per_token=1e8, peak_tflops=312.0
    )
    bigger_gpu = compute_mfu(
        tokens_per_s=1e5, model_flops_per_token=1e8, peak_tflops=312.0 * scale
    )
    assert faster == pytest.approx(base * scale)
    assert bigger_gpu == pytest.approx(base / scale)


def test_peak_memory_stats_is_empty_off_cuda():
    # No CUDA allocator stats for CPU: report nothing rather than zeros.
    assert peak_memory_stats(torch.device("cpu")) == {}


@requires_cuda
def test_peak_memory_stats_reports_gib_since_reset():
    device = torch.device("cuda", torch.cuda.current_device())
    num_bytes = 256 * 2**20

    torch.cuda.reset_peak_memory_stats(device)
    buffer = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    del buffer
    stats = peak_memory_stats(device)

    assert set(stats) == {"peak_mem_allocated_gib", "peak_mem_reserved_gib"}
    assert stats["peak_mem_allocated_gib"] >= num_bytes / 2**30
    assert stats["peak_mem_reserved_gib"] >= stats["peak_mem_allocated_gib"]
