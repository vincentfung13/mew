"""
Tests for `ThroughputMeter` in `mew/perf.py`, which turns a window
of training steps into tokens/s, MFU and per-GPU peak memory.

Interface under test:

    meter = ThroughputMeter(
        tokens_per_step,   # B * T processed by one step on this GPU
        flops_per_token,   # training FLOPs per token (already x3)
        peak_flops,        # this GPU's peak FLOP/s for the training dtype
        device="cuda",     # memory stats are only reported for CUDA
        clock=time.perf_counter,
    )
    meter.start()    # open a window: zero steps/time, reset CUDA peak stats
    meter.step()     # count one step
    meter.pause()    # stop the clock (e.g. around logging / validation)
    meter.resume()   # restart the clock
    meter.report()   # -> dict for the current window, does not reset it

`report()` keys: "tokens_per_s", "mfu", and on CUDA also
"peak_mem_allocated_gib" and "peak_mem_reserved_gib" (bytes / 2**30).

On CUDA the meter must `torch.cuda.synchronize()` before reading the clock;
the fake clock below makes the timing math testable on CPU.
"""

import pytest
import torch

from mew.perf import ThroughputMeter

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

TOKENS_PER_STEP = 128 * 256
FLOPS_PER_TOKEN = 108_576_768
PEAK_FLOPS = 312e12


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def _make_meter(clock, device="cpu"):
    return ThroughputMeter(
        tokens_per_step=TOKENS_PER_STEP,
        flops_per_token=FLOPS_PER_TOKEN,
        peak_flops=PEAK_FLOPS,
        device=device,
        clock=clock,
    )


def _run_steps(meter, clock, num_steps, seconds_per_step):
    for _ in range(num_steps):
        clock.advance(seconds_per_step)
        meter.step()


def test_tokens_per_second():
    clock = FakeClock()
    meter = _make_meter(clock)
    meter.start()
    _run_steps(meter, clock, num_steps=10, seconds_per_step=0.2)

    report = meter.report()
    assert report["tokens_per_s"] == pytest.approx(10 * TOKENS_PER_STEP / 2.0)


def test_mfu():
    clock = FakeClock()
    meter = _make_meter(clock)
    meter.start()
    _run_steps(meter, clock, num_steps=10, seconds_per_step=0.2)

    report = meter.report()
    expected = report["tokens_per_s"] * FLOPS_PER_TOKEN / PEAK_FLOPS
    assert report["mfu"] == pytest.approx(expected)
    assert 0.0 < report["mfu"] < 1.0


def test_paused_time_is_excluded():
    clock = FakeClock()
    meter = _make_meter(clock)
    meter.start()
    _run_steps(meter, clock, num_steps=5, seconds_per_step=0.2)

    meter.pause()
    clock.advance(100.0)  # e.g. validation + grad-norm logging
    meter.resume()

    _run_steps(meter, clock, num_steps=5, seconds_per_step=0.2)
    report = meter.report()
    assert report["tokens_per_s"] == pytest.approx(10 * TOKENS_PER_STEP / 2.0)


def test_start_opens_a_fresh_window():
    # The first window is slow (warmup); after start() only the new window counts.
    clock = FakeClock()
    meter = _make_meter(clock)
    meter.start()
    _run_steps(meter, clock, num_steps=10, seconds_per_step=5.0)

    meter.start()
    _run_steps(meter, clock, num_steps=4, seconds_per_step=0.5)
    report = meter.report()
    assert report["tokens_per_s"] == pytest.approx(4 * TOKENS_PER_STEP / 2.0)


def test_cpu_report_has_no_memory_stats():
    clock = FakeClock()
    meter = _make_meter(clock, device="cpu")
    meter.start()
    _run_steps(meter, clock, num_steps=1, seconds_per_step=1.0)
    assert set(meter.report()) == {"tokens_per_s", "mfu"}


@requires_cuda
def test_cuda_peak_memory_is_per_window():
    meter = _make_meter(clock=FakeClock(), device="cuda")
    num_bytes = 256 * 2**20

    meter.start()
    buffer = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")
    del buffer
    meter.step()
    first = meter.report()
    assert first["peak_mem_allocated_gib"] >= num_bytes / 2**30
    assert first["peak_mem_reserved_gib"] >= first["peak_mem_allocated_gib"]

    # start() must reset the peak. Only *allocated* is expected to drop: the
    # caching allocator keeps the freed 256 MiB block reserved, so the reserved
    # peak stays high until torch.cuda.empty_cache().
    meter.start()
    meter.step()
    second = meter.report()
    assert second["peak_mem_allocated_gib"] < first["peak_mem_allocated_gib"]
