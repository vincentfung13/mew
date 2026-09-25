from dataclasses import dataclass
from typing import Callable

import torch
from omegaconf import DictConfig

from mew.nn.functionals import cross_entropy
from mew.nn.layers import RMSNorm, SwiGLU
from mew.nn.lm import TransformerLM
from mew.nn.transformers import CausalMultiHeadSelfAttn


@dataclass(frozen=True)
class ProfilingCase:
    """A module invocation and scalar objective used by the profiler."""

    module: torch.nn.Module
    inputs: tuple[torch.Tensor, ...]
    loss_fn: Callable[[torch.Tensor], torch.Tensor]


def _activation_input(cfg: DictConfig, device: str) -> torch.Tensor:
    return torch.randn(
        cfg.case.batch_size,
        cfg.case.seq_len,
        cfg.case.d_model,
        device=device,
    )


def _activation_loss(output: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean()


def _build_lm_case(cfg: DictConfig, device: str) -> ProfilingCase:
    module = TransformerLM(
        d_model=cfg.case.d_model,
        d_ff=cfg.case.d_ff,
        num_heads=cfg.case.num_heads,
        vocab_size=cfg.case.vocab_size,
        context_len=cfg.case.context_len,
        num_transformer_layers=cfg.case.num_transformer_layers,
        rope_theta=cfg.case.rope_theta,
        num_groups=cfg.case.num_groups,
    ).to(device)
    tokens = torch.randint(
        low=0,
        high=cfg.case.vocab_size,
        size=(cfg.case.batch_size, cfg.case.seq_len),
        device=device,
    )
    labels = torch.randint(
        low=0,
        high=cfg.case.vocab_size,
        size=tokens.shape,
        device=device,
    )
    return ProfilingCase(
        module=module,
        inputs=(tokens,),
        loss_fn=lambda logits: cross_entropy(logits, labels),
    )


def _build_attention_case(cfg: DictConfig, device: str) -> ProfilingCase:
    module = CausalMultiHeadSelfAttn(
        d_model=cfg.case.d_model,
        num_heads=cfg.case.num_heads,
        theta=cfg.case.rope_theta,
        max_seq_len=cfg.case.max_seq_len,
        device=device,
        num_groups=cfg.case.num_groups,
    ).to(device)
    return ProfilingCase(
        module=module,
        inputs=(_activation_input(cfg, device),),
        loss_fn=_activation_loss,
    )


def _build_rmsnorm_case(cfg: DictConfig, device: str) -> ProfilingCase:
    module = RMSNorm(d_model=cfg.case.d_model, eps=cfg.case.eps, device=device)
    return ProfilingCase(
        module=module,
        inputs=(_activation_input(cfg, device),),
        loss_fn=_activation_loss,
    )


def _build_ffn_case(cfg: DictConfig, device: str) -> ProfilingCase:
    module = SwiGLU(
        d_model=cfg.case.d_model,
        d_ff=cfg.case.d_ff,
        device=device,
    ).to(device)
    return ProfilingCase(
        module=module,
        inputs=(_activation_input(cfg, device),),
        loss_fn=_activation_loss,
    )


_CASE_BUILDERS = {
    "lm": _build_lm_case,
    "attention": _build_attention_case,
    "rmsnorm": _build_rmsnorm_case,
    "ffn": _build_ffn_case,
}


def build_profiling_case(cfg: DictConfig, device: str) -> ProfilingCase:
    """Build the workload selected by the Hydra ``case`` config group."""
    target = cfg.case.name
    try:
        builder = _CASE_BUILDERS[target]
    except KeyError as error:
        supported = ", ".join(_CASE_BUILDERS)
        raise ValueError(
            f"Unsupported profiling target: {target}. Supported targets: {supported}"
        ) from error
    return builder(cfg, device)
