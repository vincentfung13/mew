"""
Tests for the GPU peak-FLOP/s lookup (the MFU denominator):

- `GPUSpec(match, half, tf32, fp32)` in mew/perf/gpu_specs.py. `__post_init__`
  normalises `match` to a lower-cased tuple, rejects an empty `match` (it would
  match every GPU), and requires positive numeric peaks (TFLOP/s).
- `peak_tflops_for(device_name, dtype, gpu_specs, fp32_matmul_precision=None)`
  and `peak_tflops_per_second(device, dtype, gpu_specs)`: the first spec whose
  `match` substrings all appear in the lower-cased device name wins.
- `load_gpu_specs(raw)` in apps/launch_training.py converts the entries of
  apps/cfgs/gpu_specs.yaml into GPUSpecs, naming the index of a bad entry.

Device names are the strings `torch.cuda.get_device_name()` reports, so the
lookup is tested on CPU without the hardware.
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from apps.launch_training import load_gpu_specs
from mew.perf.gpu_specs import GPUSpec, peak_tflops_for, peak_tflops_per_second

CFG_DIR = Path(__file__).resolve().parents[2] / "apps" / "cfgs"
H100_SXM = "NVIDIA H100 80GB HBM3"


@pytest.fixture(scope="module")
def real_specs():
    return load_gpu_specs(OmegaConf.load(CFG_DIR / "gpu_specs.yaml").gpu_specs)


def _spec(*match, half=1.0, tf32=2.0, fp32=3.0):
    return GPUSpec(match=tuple(match), half=half, tf32=tf32, fp32=fp32)


def _entry(match, half=1.0, tf32=2.0, fp32=3.0):
    return {"match": match, "half": half, "tf32": tf32, "fp32": fp32}


# ---------------------------------------------------------------------------
# The real table in apps/cfgs/gpu_specs.yaml
# ---------------------------------------------------------------------------


def test_training_config_composes_gpu_specs():
    # gpu_specs.yaml is in training.yaml's defaults list, so the launcher
    # finds the table at cfg.gpu_specs.
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="training")
    assert len(load_gpu_specs(cfg.gpu_specs)) > 0
    assert cfg.trainer.perf.peak_tflops is None


@pytest.mark.parametrize(
    "device_name, expected_tflops",
    [
        (H100_SXM, 989.4),
        ("NVIDIA H100 PCIe", 756.5),
        ("NVIDIA H100 NVL", 835.5),
        ("NVIDIA H800", 989.4),
        ("NVIDIA H800 PCIe", 756.5),
        ("NVIDIA H200", 989.4),
        ("NVIDIA H200 NVL", 835.5),
        ("NVIDIA H20", 148.0),
        ("NVIDIA A100-SXM4-80GB", 312.0),
        ("NVIDIA A100 80GB PCIe", 312.0),
        ("NVIDIA A800-SXM4-80GB", 312.0),
        ("NVIDIA L40S", 362.05),
    ],
)
def test_bf16_peak_by_device_name(real_specs, device_name, expected_tflops):
    peak = peak_tflops_for(device_name, torch.bfloat16, real_specs)
    assert peak == pytest.approx(expected_tflops)


@pytest.mark.parametrize(
    "device_name, generic_tflops",
    [
        ("NVIDIA H100 PCIe", 989.4),
        ("NVIDIA H200", 148.0),  # "h20" is a substring of "h200"
    ],
)
def test_real_table_orders_specific_names_first(
    real_specs, device_name, generic_tflops
):
    peak = peak_tflops_for(device_name, torch.bfloat16, real_specs)
    assert peak != pytest.approx(generic_tflops)


@pytest.mark.parametrize(
    "precision, expected_tflops",
    [("highest", 66.9), ("high", 494.7)],
)
def test_real_table_fp32_peak_follows_matmul_precision(
    real_specs, precision, expected_tflops
):
    peak = peak_tflops_for(
        H100_SXM, torch.float32, real_specs, fp32_matmul_precision=precision
    )
    assert peak == pytest.approx(expected_tflops)


# ---------------------------------------------------------------------------
# Lookup semantics, on small inline tables
# ---------------------------------------------------------------------------


def test_first_matching_spec_wins():
    specs = [_spec("h100", "pcie", half=7.0), _spec("h100")]
    assert peak_tflops_for("NVIDIA H100 PCIe", torch.bfloat16, specs) == 7.0
    assert peak_tflops_for(H100_SXM, torch.bfloat16, specs) == 1.0


def test_all_match_substrings_are_required():
    specs = [_spec("h100", "nvl", half=7.0), _spec("h100")]
    assert peak_tflops_for("NVIDIA H100 PCIe", torch.bfloat16, specs) == 1.0


def test_accepts_any_sequence():
    specs = (_spec("h100"),)
    assert peak_tflops_for(H100_SXM, torch.bfloat16, specs) == 1.0


def test_fp16_and_bf16_share_the_half_precision_peak():
    specs = [_spec("h100")]
    assert peak_tflops_for(H100_SXM, torch.float16, specs) == peak_tflops_for(
        H100_SXM, torch.bfloat16, specs
    )


def test_fp32_reads_global_matmul_precision_by_default():
    specs = [_spec("h100")]
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        assert peak_tflops_for(H100_SXM, torch.float32, specs) == 2.0
        torch.set_float32_matmul_precision("highest")
        assert peak_tflops_for(H100_SXM, torch.float32, specs) == 3.0
    finally:
        torch.set_float32_matmul_precision(previous)


def test_fp32_medium_precision_is_ambiguous():
    with pytest.raises(ValueError, match="medium"):
        peak_tflops_for(
            H100_SXM, torch.float32, [_spec("h100")], fp32_matmul_precision="medium"
        )


def test_unknown_gpu_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="NVIDIA GeForce RTX 3090"):
        peak_tflops_for("NVIDIA GeForce RTX 3090", torch.bfloat16, [_spec("h100")])


def test_unsupported_dtype_raises():
    with pytest.raises(ValueError, match="float8"):
        peak_tflops_for(H100_SXM, torch.float8_e4m3fn, [_spec("h100")])


def test_cuda_device_is_looked_up_by_its_name(monkeypatch):
    # Fake the device name so the CUDA path runs without a GPU.
    names = {}

    def fake_get_device_name(device):
        names["device"] = device
        return "NVIDIA H100 PCIe"

    monkeypatch.setattr(torch.cuda, "get_device_name", fake_get_device_name)
    specs = [_spec("h100", "pcie", half=7.0), _spec("h100")]
    assert peak_tflops_per_second("cuda:1", torch.bfloat16, specs) == 7.0
    assert names["device"] == torch.device("cuda", 1)


def test_non_cuda_device_raises():
    with pytest.raises(ValueError, match="cpu"):
        peak_tflops_per_second("cpu", torch.bfloat16, [_spec("h100")])


# ---------------------------------------------------------------------------
# GPUSpec validation and normalisation
# ---------------------------------------------------------------------------


def test_match_is_lower_cased_so_table_case_does_not_matter():
    spec = _spec("H100", "PCIe")
    assert spec.match == ("h100", "pcie")
    assert peak_tflops_for("NVIDIA H100 PCIe", torch.bfloat16, [spec]) == 1.0


def test_match_list_becomes_a_hashable_tuple():
    spec = GPUSpec(match=["h100"], half=1.0, tf32=2.0, fp32=3.0)
    assert spec.match == ("h100",)
    hash(spec)


def test_empty_match_is_rejected():
    # all() over an empty match is True, so it would match every GPU.
    with pytest.raises(ValueError):
        _spec()


@pytest.mark.parametrize(
    "peaks",
    [
        pytest.param(dict(half="fast"), id="non-numeric"),
        pytest.param(dict(tf32=0.0), id="zero"),
        pytest.param(dict(fp32=-1.0), id="negative"),
    ],
)
def test_invalid_peak_is_rejected(peaks):
    with pytest.raises(ValueError):
        _spec("h100", **peaks)


# ---------------------------------------------------------------------------
# load_gpu_specs (apps/launch_training.py)
# ---------------------------------------------------------------------------


def test_load_converts_config_entries():
    specs = load_gpu_specs(OmegaConf.create([_entry(["H100"], half=5.0)]))
    assert specs == [GPUSpec(match=("h100",), half=5.0, tf32=2.0, fp32=3.0)]


@pytest.mark.parametrize(
    "bad_entry",
    [
        pytest.param({"match": ["h20"], "half": 1.0, "fp32": 3.0}, id="missing-key"),
        pytest.param({**_entry(["h20"]), "fp64": 1.0}, id="unknown-key"),
        pytest.param(_entry([]), id="empty-match"),
        pytest.param(_entry(["h20"], half="fast"), id="non-numeric-peak"),
    ],
)
def test_load_names_the_malformed_entry(bad_entry):
    raw = OmegaConf.create([_entry(["h100"]), bad_entry])
    with pytest.raises(ValueError, match=r"entry 1\b"):
        load_gpu_specs(raw)
