from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from profiling.cases import build_profiling_case

CONFIG_DIR = Path(__file__).parents[2] / "profiling" / "configs"


@pytest.mark.parametrize("target", ["lm", "attention", "rmsnorm", "ffn"])
def test_profiling_case_config_composes_independently(target):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="profiling", overrides=[f"case={target}"])

    assert cfg.case.name == target
    assert cfg.torch_compile.enable is False
    assert cfg.torch_compile.mode == "default"
    assert cfg.profiling.protocol == "forward_only"
    assert "data" not in cfg
    assert "model" not in cfg


@pytest.mark.parametrize(
    "case_config",
    [
        {
            "name": "lm",
            "batch_size": 2,
            "seq_len": 4,
            "d_model": 8,
            "d_ff": 16,
            "num_heads": 2,
            "num_groups": None,
            "vocab_size": 16,
            "num_transformer_layers": 1,
            "rope_theta": 10000.0,
            "context_len": 4,
        },
        {
            "name": "attention",
            "batch_size": 2,
            "seq_len": 4,
            "d_model": 8,
            "num_heads": 2,
            "num_groups": None,
            "rope_theta": 10000.0,
            "max_seq_len": 4,
        },
        {
            "name": "rmsnorm",
            "batch_size": 2,
            "seq_len": 4,
            "d_model": 8,
            "eps": 1.0e-5,
        },
        {
            "name": "ffn",
            "batch_size": 2,
            "seq_len": 4,
            "d_model": 8,
            "d_ff": 16,
        },
    ],
    ids=["lm", "attention", "rmsnorm", "ffn"],
)
def test_profiling_case_runs_forward_and_backward(case_config):
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "case": case_config,
        }
    )

    case = build_profiling_case(cfg, device="cpu")
    output = case.module(*case.inputs)
    loss = case.loss_fn(output)
    loss.backward()

    assert output.shape[:2] == (2, 4)
    assert any(parameter.grad is not None for parameter in case.module.parameters())
