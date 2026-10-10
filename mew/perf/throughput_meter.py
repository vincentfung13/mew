from collections.abc import Callable
import time

import torch

from mew.perf.utils import compute_mfu, peak_memory_stats


class ThroughputMeter:
    def __init__(
        self,
        model_flops_per_token: int,
        # accepting None device_peak_tflops for CPU
        device_peak_tflops: float | None,
        device: str | torch.device,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.model_flops_per_token = model_flops_per_token
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.device_peak_tflops = device_peak_tflops
        self.total_tokens = 0
        self.w_start = None
        self.clock = clock

    def start(self):
        # Rest flops buffer
        self.total_tokens = 0

        if self.device.type == "cuda":
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(self.device)
            # Call cuda.synchronize
            torch.cuda.synchronize(self.device)

        # Record window start
        self.w_start = self.clock()

    def step(self, num_tokens: int):
        # Increment the total_flops_counter
        self.total_tokens += num_tokens

    def report(self) -> dict:
        assert (
            self.w_start is not None
        ), "Calling report() before start() is not permitted."
        throughput_report = {}
        if self.device.type == "cuda":
            # Synchronize and report performance
            torch.cuda.synchronize(self.device)

            # Record peak mem use
            throughput_report.update(peak_memory_stats(self.device))

        # record window end
        w_end = self.clock()

        # cal tokens_per_s
        tokens_per_s = self.total_tokens / (w_end - self.w_start)
        throughput_report["tokens_per_s"] = tokens_per_s

        # Calculate mfu
        if self.device_peak_tflops is not None:
            mfu = compute_mfu(
                tokens_per_s=tokens_per_s,
                model_flops_per_token=self.model_flops_per_token,
                peak_tflops=self.device_peak_tflops,
            )
            throughput_report["mfu"] = mfu

        return throughput_report
