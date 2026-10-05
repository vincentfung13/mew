from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from mew.nn.functionals import scaled_dot_product

Provider = Callable[..., torch.Tensor]


class BenchMode(str, Enum):
    FWD = "fwd"
    BWD = "bwd"
    FWD_BWD = "fwd_bwd"


# FLOPs of each mode relative to one forward pass. Backward is counted as 2.5x
# forward, following the FlashAttention papers' convention.
MODE_FLOP_MULTIPLIER = {
    BenchMode.FWD: 1.0,
    BenchMode.BWD: 2.5,
    BenchMode.FWD_BWD: 3.5,
}


def parse_mode(value: str) -> BenchMode:
    try:
        return BenchMode(value)
    except ValueError as error:
        supported = ", ".join(mode.value for mode in BenchMode)
        raise ValueError(
            f"Unsupported benchmark mode: {value}. Supported modes: {supported}"
        ) from error


@dataclass(frozen=True)
class FunctionCase:
    """A function workload that can be timed across interchangeable providers.

    ``make_inputs(params, requires_grad)`` builds fresh inputs for one sweep
    point, where ``params`` is the case config with the swept value applied.
    ``providers`` map a name to a callable taking ``*inputs``. ``forward_flops``
    returns the FLOPs of one forward pass for the given params.
    """

    make_inputs: Callable[[Mapping[str, Any], bool], tuple[torch.Tensor, ...]]
    providers: Mapping[str, Callable[[Mapping[str, Any]], Provider]]
    forward_flops: Callable[[Mapping[str, Any]], float]


def _attention_inputs(
    params: Mapping[str, Any], requires_grad: bool, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    q_shape = (
        params["batch_size"],
        params["num_heads"],
        params["seq_len"],
        params["d_head"],
    )
    kv_shape = (
        params["batch_size"],
        params["num_kv_heads"],
        params["seq_len"],
        params["d_head"],
    )
    return tuple(
        torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        for shape in (q_shape, kv_shape, kv_shape)
    )


def _flash_triton_provider(params: Mapping[str, Any]) -> Provider:
    # Imported lazily so that CPU-only environments can build the other providers.
    from mew.nn.flash_attention import FlashAttention

    is_causal = params["is_causal"]
    return lambda q, k, v: FlashAttention.apply(q, k, v, is_causal)


def _reference_provider(params: Mapping[str, Any]) -> Provider:
    def run(q, k, v):
        mask = None
        if params["is_causal"]:
            seq_len = q.size(-2)
            mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device)
            )
        return scaled_dot_product(q, k, v, mask=mask)

    return run


def _torch_sdpa_provider(params: Mapping[str, Any]) -> Provider:
    is_causal = params["is_causal"]
    enable_gqa = params["num_kv_heads"] != params["num_heads"]
    return lambda q, k, v: F.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, enable_gqa=enable_gqa
    )


def _attention_forward_flops(params: Mapping[str, Any]) -> float:
    # Two matmuls (QK^T and PV), each 2 * N^2 * D FLOPs per (batch, head).
    flops = (
        4.0
        * params["batch_size"]
        * params["num_heads"]
        * params["seq_len"] ** 2
        * params["d_head"]
    )
    # Causal masking skips roughly half of the score matrix.
    return flops / 2 if params["is_causal"] else flops


def _build_attention_case(device: str, dtype: torch.dtype) -> FunctionCase:
    return FunctionCase(
        make_inputs=lambda params, requires_grad: _attention_inputs(
            params, requires_grad, device, dtype
        ),
        providers={
            "flash_triton": _flash_triton_provider,
            "reference": _reference_provider,
            "torch_sdpa": _torch_sdpa_provider,
        },
        forward_flops=_attention_forward_flops,
    )


_FUNCTION_BUILDERS = {
    "attention": _build_attention_case,
}


def build_function_case(cfg: DictConfig, device: str) -> FunctionCase:
    """Build the workload selected by the Hydra ``function`` config group."""
    target = cfg.function.name
    try:
        builder = _FUNCTION_BUILDERS[target]
    except KeyError as error:
        supported = ", ".join(_FUNCTION_BUILDERS)
        raise ValueError(
            f"Unsupported benchmark function: {target}. Supported functions: {supported}"
        ) from error
    return builder(device, parse_dtype(cfg.dtype))


def parse_dtype(value: str) -> torch.dtype:
    try:
        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[value]
    except KeyError as error:
        raise ValueError(
            f"Unsupported dtype: {value}. Supported dtypes: fp32, fp16, bf16"
        ) from error
