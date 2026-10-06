"""
Peak dense matmul throughput of one GPU, used as the MFU denominator.

Values are NVIDIA datasheet numbers in TFLOP/s *without* 2:4 structured
sparsity (datasheets headline the sparse figure, which is 2x the dense one).
"half" is the bf16/fp16 tensor-core peak, "tf32" the TF32 tensor-core peak and
"fp32" the non-tensor-core FP32 peak.

Unknown GPUs raise instead of falling back to a default: a wrong peak gives a
plausible-looking but wrong MFU. Add a row here, or set
`trainer.perf.peak_tflops` in the config to bypass the lookup.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import torch

_HALF_DTYPES = (torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class GPUSpec:
    match: tuple[str, ...]
    half: float
    tf32: float
    fp32: float

    def __post_init__(self):
        # Reject empty match
        if len(self.match) == 0:
            raise ValueError("Empty match is not allowed")

        # convert peak flops to float
        for precision in ["half", "tf32", "fp32"]:
            peak = float(getattr(self, precision))
            if peak <= 0:
                raise ValueError(f"Peak value must be greater than 0, but it's {peak}!")
            object.__setattr__(self, precision, peak)

        # Normalize to lower case
        object.__setattr__(
            self, "match", tuple([_match.lower() for _match in self.match])
        )


def _precision_key(dtype: torch.dtype, fp32_matmul_precision: str) -> str:
    if dtype in _HALF_DTYPES:
        return "half"
    if dtype == torch.float32:
        # "highest" keeps fp32 matmuls on the FP32 units; "high" lets them use
        # TF32 tensor cores. "medium" may use bf16 internally, so its peak is
        # ambiguous and must be given explicitly.
        if fp32_matmul_precision == "highest":
            return "fp32"
        if fp32_matmul_precision == "high":
            return "tf32"
        raise ValueError(
            f"Ambiguous peak for fp32 matmul precision '{fp32_matmul_precision}'; "
            "set trainer.perf.peak_tflops explicitly."
        )
    raise ValueError(f"No peak TFLOP/s defined for dtype {dtype}.")


def peak_tflops_for(
    device_name: str,
    dtype: torch.dtype,
    gpu_specs: Sequence[GPUSpec],
    fp32_matmul_precision: str | None = None,
) -> float:
    """
    Peak dense TFLOP/s of the GPU called `device_name` (as reported by
    `torch.cuda.get_device_name`) for matmuls in `dtype`.

    `fp32_matmul_precision` only matters for fp32 and defaults to the global
    `torch.get_float32_matmul_precision()`.
    """
    if fp32_matmul_precision is None:
        fp32_matmul_precision = torch.get_float32_matmul_precision()
    precision_key = _precision_key(dtype, fp32_matmul_precision)

    name = device_name.lower()
    for gpu_spec in gpu_specs:
        if all(s in name for s in gpu_spec.match):
            return getattr(gpu_spec, precision_key)

    raise ValueError(
        f"Unknown GPU '{device_name}': add it to apps/cfgs/gpu_specs.yaml or set "
        "trainer.perf.peak_tflops explicitly."
    )


def peak_tflops_per_second(
    device: str | torch.device,
    dtype: torch.dtype,
    gpu_specs: Sequence[GPUSpec],
) -> float:
    """Peak dense TFLOP/s of the CUDA `device` for matmuls in `dtype`."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            f"Peak TFLOP/s is only defined for CUDA devices, got {device}."
        )
    return peak_tflops_for(torch.cuda.get_device_name(device), dtype, gpu_specs)
