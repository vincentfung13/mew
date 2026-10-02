import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from profiling.functions import (
    MODE_FLOP_MULTIPLIER,
    BenchMode,
    FunctionCase,
    Provider,
    build_function_case,
    parse_mode,
)

LOGGER = logging.getLogger(__name__)

# do_bench quantiles: median, then the 20th and 80th percentiles for the error band.
QUANTILES = [0.5, 0.2, 0.8]

# Resource failures that only invalidate a single (provider, sweep point): CUDA
# running out of device memory, or a Triton kernel needing more shared memory or
# registers than the GPU has.
# Triton is imported lazily because it ships no macOS wheels, which keeps the
# CPU-only helpers importable (and testable) there.
try:
    from triton.runtime.errors import OutOfResources
except ImportError:
    POINT_RESOURCE_ERRORS = (torch.cuda.OutOfMemoryError,)
else:
    POINT_RESOURCE_ERRORS = (torch.cuda.OutOfMemoryError, OutOfResources)


class ReferenceUnavailableError(RuntimeError):
    """The reference provider could not run, so a correctness check was skipped."""


def make_timed_fn(
    mode: BenchMode, provider: Provider, inputs: tuple[torch.Tensor, ...]
) -> tuple[Callable[[], Any], list[torch.Tensor] | None]:
    """Return the callable to time and the tensors whose grads do_bench resets.

    ``inputs`` must require grad for the backward modes. In ``bwd`` mode the
    forward runs once here, outside the timed region, and each timed call
    backpropagates through the same retained graph.
    """
    if mode is BenchMode.FWD:
        return (lambda: provider(*inputs)), None

    output = provider(*inputs)
    grad_output = torch.randn_like(output)
    if mode is BenchMode.BWD:
        return (lambda: output.backward(grad_output, retain_graph=True)), list(inputs)
    if mode is BenchMode.FWD_BWD:
        return (lambda: provider(*inputs).backward(grad_output)), list(inputs)
    raise AssertionError(f"Unhandled benchmark mode: {mode}")


def check_against_reference(
    case: FunctionCase,
    params: Mapping[str, Any],
    provider_name: str,
    reference_name: str,
    atol: float,
    rtol: float,
) -> None:
    """Assert that a provider's forward output matches the reference provider.

    Raises ``ReferenceUnavailableError`` if the reference hits a resource error,
    so the caller can skip the check. A resource error from the provider itself
    propagates unchanged.
    """
    inputs = case.make_inputs(params, False)
    with torch.no_grad():
        try:
            expected = case.providers[reference_name](params)(*inputs)
        except POINT_RESOURCE_ERRORS as error:
            raise ReferenceUnavailableError(
                f"{reference_name} failed with {type(error).__name__}"
            ) from error
        actual = case.providers[provider_name](params)(*inputs)
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        msg=lambda msg: f"{provider_name} does not match {reference_name} at {dict(params)}:\n{msg}",
    )


def _time_provider(
    case: FunctionCase,
    params: Mapping[str, Any],
    provider: str,
    mode: BenchMode,
    warmup_ms: float,
    rep_ms: float,
) -> tuple[float, float, float]:
    """Time one provider at one sweep point; returns (median, p20, p80) in ms.

    Kept separate from the sweep loop so that its tensors go out of scope as soon
    as it returns or raises.
    """
    import triton.testing

    requires_grad = mode is not BenchMode.FWD
    inputs = case.make_inputs(params, requires_grad)
    fn, grad_to_none = make_timed_fn(mode, case.providers[provider](params), inputs)
    # One explicit call so compilation (and autotuning) happens before do_bench
    # estimates how many repetitions to run.
    fn()
    ms, ms_low, ms_high = triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
        grad_to_none=grad_to_none,
        quantiles=QUANTILES,
    )
    return ms, ms_low, ms_high


def _to_tflops(ms: float, flops: float) -> float:
    return flops / (ms * 1e-3) / 1e12


@hydra.main(version_base=None, config_path="configs", config_name="bench_function")
def main(cfg: DictConfig) -> None:
    if cfg.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("bench_function requires a CUDA device for do_bench.")
    import triton.testing

    output_dir = Path(cfg.bench.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case = build_function_case(cfg, device=cfg.device)
    mode = parse_mode(cfg.bench.mode)
    providers = list(cfg.bench.providers)
    unknown = [name for name in providers if name not in case.providers]
    if unknown:
        raise ValueError(
            f"Unknown providers {unknown} for {cfg.function.name}. "
            f"Available: {', '.join(case.providers)}"
        )
    if cfg.bench.metric not in ("ms", "tflops"):
        raise ValueError(f"Unsupported metric: {cfg.bench.metric}. Use ms or tflops.")

    base_params = OmegaConf.to_container(cfg.function, resolve=True)
    x_name = cfg.bench.sweep.x_name
    if x_name not in base_params:
        raise ValueError(
            f"Sweep axis {x_name} is not a {cfg.function.name} parameter: "
            f"{', '.join(base_params)}"
        )

    plot_name = f"{cfg.function.name}_{mode.value}_{cfg.dtype}_{cfg.bench.metric}"
    LOGGER.info(
        "Benchmarking %s (%s) over %s=%s with providers %s and params: %s",
        cfg.function.name,
        mode.value,
        x_name,
        list(cfg.bench.sweep.x_vals),
        providers,
        base_params,
    )

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[x_name],
            x_vals=list(cfg.bench.sweep.x_vals),
            x_log=cfg.bench.sweep.x_log,
            line_arg="provider",
            line_vals=providers,
            line_names=providers,
            ylabel="ms" if cfg.bench.metric == "ms" else "TFLOP/s",
            plot_name=plot_name,
            args={},
        )
    )
    def bench(provider: str, **x_args):
        params = {**base_params, **x_args}
        failure = None
        try:
            if cfg.bench.check_correctness and provider != cfg.bench.reference_provider:
                try:
                    check_against_reference(
                        case,
                        params,
                        provider,
                        cfg.bench.reference_provider,
                        atol=cfg.function.atol,
                        rtol=cfg.function.rtol,
                    )
                except ReferenceUnavailableError as error:
                    LOGGER.warning(
                        "Skipped correctness check for %s at %s: %s.",
                        provider,
                        x_args,
                        error,
                    )
                torch.cuda.empty_cache()
            ms, ms_low, ms_high = _time_provider(
                case, params, provider, mode, cfg.bench.warmup_ms, cfg.bench.rep_ms
            )
        except POINT_RESOURCE_ERRORS as error:
            failure = type(error).__name__
        if failure is not None:
            # Released only after the except block, once the traceback (and the
            # tensors its frames reference) has been dropped.
            torch.cuda.empty_cache()
            LOGGER.warning(
                "%s failed at %s with %s; recording NaN.", provider, x_args, failure
            )
            return math.nan, math.nan, math.nan

        if cfg.bench.metric == "ms":
            return ms, ms_low, ms_high
        flops = case.forward_flops(params) * MODE_FLOP_MULTIPLIER[mode]
        # Faster times give higher throughput, so the band's bounds swap.
        return (
            _to_tflops(ms, flops),
            _to_tflops(ms_high, flops),
            _to_tflops(ms_low, flops),
        )

    bench.run(save_path=str(output_dir), print_data=True, show_plots=False)
    LOGGER.info("Wrote %s.csv and %s.png to %s", plot_name, plot_name, output_dir)


if __name__ == "__main__":
    main()
