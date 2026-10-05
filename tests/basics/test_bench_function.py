from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from profiling.bench_function import (
    ReferenceUnavailableError,
    check_against_reference,
    make_timed_fn,
)
from profiling.functions import (
    BenchMode,
    FunctionCase,
    build_function_case,
    parse_mode,
)

CONFIG_DIR = Path(__file__).parents[2] / "profiling" / "configs"


def _attention_cfg(**overrides):
    function = {
        "name": "attention",
        "batch_size": 2,
        "num_heads": 4,
        "num_kv_heads": 4,
        "seq_len": 8,
        "d_head": 16,
        "is_causal": True,
        "atol": 1.0e-4,
        "rtol": 1.0e-4,
    }
    function.update(overrides)
    return OmegaConf.create({"device": "cpu", "dtype": "fp32", "function": function})


def test_bench_function_config_composes():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="bench_function")

    assert cfg.function.name == "attention"
    assert cfg.bench.mode == "fwd"
    assert cfg.bench.sweep.x_name in cfg.function
    assert cfg.bench.reference_provider in cfg.bench.providers
    assert "case" not in cfg


def test_parse_mode_rejects_unknown_value():
    assert parse_mode("fwd_bwd") is BenchMode.FWD_BWD
    with pytest.raises(ValueError, match="Unsupported benchmark mode"):
        parse_mode("unknown")


@pytest.mark.parametrize("is_causal", [False, True])
def test_attention_providers_agree_on_cpu(is_causal):
    # flash_triton needs CUDA; the reference and torch_sdpa providers must agree.
    # MHA only: the reference implementation does not support GQA/MQA.
    cfg = _attention_cfg(is_causal=is_causal)
    case = build_function_case(cfg, device="cpu")
    params = OmegaConf.to_container(cfg.function)

    check_against_reference(
        case, params, "torch_sdpa", "reference", atol=1e-4, rtol=1e-4
    )


def test_attention_forward_flops_halves_for_causal():
    case = build_function_case(_attention_cfg(), device="cpu")
    params = OmegaConf.to_container(_attention_cfg().function)

    full = case.forward_flops({**params, "is_causal": False})
    causal = case.forward_flops({**params, "is_causal": True})

    assert full == 4 * 2 * 4 * 8**2 * 16
    assert causal == full / 2


@pytest.mark.parametrize("mode", list(BenchMode))
def test_make_timed_fn_runs_each_mode(mode):
    cfg = _attention_cfg()
    case = build_function_case(cfg, device="cpu")
    params = OmegaConf.to_container(cfg.function)
    requires_grad = mode is not BenchMode.FWD
    inputs = case.make_inputs(params, requires_grad)

    fn, grad_to_none = make_timed_fn(mode, case.providers["reference"](params), inputs)
    fn()
    fn()

    if mode is BenchMode.FWD:
        assert grad_to_none is None
        assert all(t.grad is None for t in inputs)
    else:
        assert grad_to_none == list(inputs)
        assert all(t.grad is not None for t in inputs)


def _raise_oom(*_inputs):
    raise torch.cuda.OutOfMemoryError("simulated out of memory")


def _case_with_providers(**providers):
    return FunctionCase(
        make_inputs=lambda _params, _requires_grad: (torch.ones(2),),
        providers={name: (lambda _params, fn=fn: fn) for name, fn in providers.items()},
        forward_flops=lambda _params: 0.0,
    )


def test_check_attributes_reference_oom_to_reference():
    case = _case_with_providers(reference=_raise_oom, candidate=lambda x: x)

    with pytest.raises(ReferenceUnavailableError, match="reference"):
        check_against_reference(
            case, {}, "candidate", "reference", atol=1e-4, rtol=1e-4
        )


def test_check_propagates_provider_oom_unchanged():
    case = _case_with_providers(reference=lambda x: x, candidate=_raise_oom)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        check_against_reference(
            case, {}, "candidate", "reference", atol=1e-4, rtol=1e-4
        )


def test_check_fails_on_mismatch():
    case = _case_with_providers(reference=lambda x: x, candidate=lambda x: x + 1)

    with pytest.raises(AssertionError, match="candidate does not match reference"):
        check_against_reference(
            case, {}, "candidate", "reference", atol=1e-4, rtol=1e-4
        )


def test_build_function_case_rejects_unknown_target():
    cfg = _attention_cfg(name="unknown")
    with pytest.raises(ValueError, match="Unsupported benchmark function"):
        build_function_case(cfg, device="cpu")


def test_torch_float_dtype_is_used():
    cfg = OmegaConf.merge(_attention_cfg(), {"dtype": "bf16"})
    case = build_function_case(cfg, device="cpu")
    params = OmegaConf.to_container(cfg.function)
    assert all(t.dtype == torch.bfloat16 for t in case.make_inputs(params, False))
