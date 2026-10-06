"""
Tests for `ThroughputMeter` in `mew/perf/throughput_meter.py`, which turns a window
of training steps into tokens/s, MFU and per-GPU peak memory.

Interface under test:

    meter = ThroughputMeter(
        model_flops_per_token,  # training FLOPs per token (already x3)
        device_peak_flops,      # this GPU's peak FLOP/s for the training dtype
        device,                 # memory stats are only reported for CUDA
        clock=time.perf_counter,
    )
    meter.start()                # open a window: zero tokens, reset CUDA peak stats
    meter.step(num_tokens)       # count the tokens processed by one step
    meter.report()               # -> dict for the current window, does not reset it

`report()` keys: "tokens_per_s", "mfu", and on CUDA also
"peak_mem_allocated_gib" and "peak_mem_reserved_gib" (bytes / 2**30).

The meter measures end-to-end time: everything between start() and report()
counts, including data loading and logging.

On CUDA the meter must `torch.cuda.synchronize()` before reading the clock;
the fake clock below makes the timing math testable on CPU.
"""

import pytest
import torch

from mew.perf.throughput_meter import ThroughputMeter

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
        model_flops_per_token=FLOPS_PER_TOKEN,
        device_peak_flops=PEAK_FLOPS,
        device=device,
        clock=clock,
    )


def _run_steps(meter, clock, num_steps, seconds_per_step, tokens=TOKENS_PER_STEP):
    for _ in range(num_steps):
        clock.advance(seconds_per_step)
        meter.step(tokens)


def test_tokens_per_second():
    clock = FakeClock()
    meter = _make_meter(clock)
    meter.start()
    _run_steps(meter, clock, num_steps=10, seconds_per_step=0.2)

    report = meter.report()
    assert report["tokens_per_s"] == pytest.approx(10 * TOKENS_PER_STEP / 2.0)


def test_variable_tokens_per_step():
    # step(num_tokens) supports batches of different sizes (packing, a short
    # final batch): the rate is total tokens over total time.
    clock = FakeClock()
    meter = _make_meter(clock)
    meter.start()
    _run_steps(meter, clock, num_steps=2, seconds_per_step=0.5, tokens=1000)
    _run_steps(meter, clock, num_steps=1, seconds_per_step=1.0, tokens=4000)

    report = meter.report()
    assert report["tokens_per_s"] == pytest.approx((2 * 1000 + 4000) / 2.0)


def test_mfu():
    clock = FakeClock()
    meter = _make_meter(clock)
    meter.start()
    _run_steps(meter, clock, num_steps=10, seconds_per_step=0.2)

    report = meter.report()
    expected = report["tokens_per_s"] * FLOPS_PER_TOKEN / PEAK_FLOPS
    assert report["mfu"] == pytest.approx(expected)
    assert 0.0 < report["mfu"] < 1.0


def test_mfu_depends_on_elapsed_time():
    # Same tokens, twice the time -> half the MFU. Guards against computing
    # MFU from the token count alone (which has units of seconds, not a ratio).
    reports = []
    for seconds_per_step in (0.2, 0.4):
        clock = FakeClock()
        meter = _make_meter(clock)
        meter.start()
        _run_steps(meter, clock, num_steps=10, seconds_per_step=seconds_per_step)
        reports.append(meter.report())

    assert reports[1]["mfu"] == pytest.approx(reports[0]["mfu"] / 2)


def test_report_does_not_reset_the_window():
    clock = FakeClock()
    meter = _make_meter(clock)
    meter.start()
    _run_steps(meter, clock, num_steps=5, seconds_per_step=0.2)
    meter.report()

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


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_cpu_report_has_no_memory_stats(device):
    # Must hold on a GPU machine too: the meter's own device decides, not
    # torch.cuda.is_available().
    clock = FakeClock()
    meter = _make_meter(clock, device=device)
    meter.start()
    _run_steps(meter, clock, num_steps=1, seconds_per_step=1.0)
    assert set(meter.report()) == {"tokens_per_s", "mfu"}


@requires_cuda
def test_cuda_peak_memory_is_per_window():
    clock = FakeClock()
    meter = _make_meter(clock=clock, device="cuda")
    num_bytes = 256 * 2**20

    meter.start()
    buffer = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")
    del buffer
    _run_steps(meter, clock, num_steps=1, seconds_per_step=1.0)
    first = meter.report()
    assert first["peak_mem_allocated_gib"] >= num_bytes / 2**30
    assert first["peak_mem_reserved_gib"] >= first["peak_mem_allocated_gib"]

    # start() must reset the peak. Only *allocated* is expected to drop: the
    # caching allocator keeps the freed 256 MiB block reserved, so the reserved
    # peak stays high until torch.cuda.empty_cache().
    meter.start()
    _run_steps(meter, clock, num_steps=1, seconds_per_step=1.0)
    second = meter.report()
    assert second["peak_mem_allocated_gib"] < first["peak_mem_allocated_gib"]
