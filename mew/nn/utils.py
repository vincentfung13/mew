from omegaconf import DictConfig

import torch

from mew.nn.lm import TransformerLM


def build_model(cfg: DictConfig, device: str = "cuda") -> torch.nn.Module:
    model = TransformerLM(
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        num_heads=cfg.model.num_heads,
        vocab_size=cfg.model.vocab_size,
        context_len=cfg.model.context_len,  # keep context len consistent with training
        num_transformer_layers=cfg.model.num_transformer_layers,
        rope_theta=cfg.model.rope_theta,
    ).to(device)
    return model
