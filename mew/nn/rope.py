from einops import rearrange, einsum

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.theta = theta

        # ks & inverse_freq -> (d_k_half)
        # inds -> (max_seq_len)
        ks = torch.arange(1, d_k // 2 + 1, device=device).float()
        self.inverse_freq = 1.0 / (theta ** ((2 * ks - 2) / d_k))
        inds = torch.arange(0, max_seq_len, device=device).float()

        # Init sin and cos cache
        # compute outer product
        # freqs -> (max_seq_len, d_k_half)
        freqs = einsum(inds, self.inverse_freq, "seq_len, d_k_half -> seq_len d_k_half")
        if device is not None:
            freqs = freqs.to(device)

        # Cache sin and cos
        self.register_buffer("sin", freqs.sin())
        self.register_buffer("cos", freqs.cos())

    def _update_cache(self, max_seq_len: int):
        inds = torch.arange(0, max_seq_len, device=self.sin.device).float()
        freqs = einsum(
            inds,
            self.inverse_freq.to(self.sin.device),
            "seq_len, d_k_half -> seq_len d_k_half",
        )
        self.register_buffer("sin", freqs.sin())
        self.register_buffer("cos", freqs.cos())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x -> (... seq_len d_k)
        # token_positions -> (... seq_len)

        # if token_positions has elements larger than the cached freq length,
        if token_positions.max() >= self.sin.size(0):
            # dynamically extend sin and cos
            max_seq_len = int(token_positions.max().item()) + 1
            self._update_cache(max_seq_len)

        # Select sin and cos based on token positions
        sin_selected = self.sin[token_positions]  # (... seq_len, d_k_half)
        cos_selected = self.cos[token_positions]  # (... seq_len, d_k_half)

        # Rearrange x into pairs: (..., seq_len, d_k) -> (..., seq_len, d_k_half, 2)
        x_rearr = rearrange(
            x,
            "... seq_len (d_k_half d2) -> ... seq_len d_k_half d2",
            d_k_half=self.d_k // 2,
            d2=2,
        )

        # Extract the two components of each pair
        x0 = x_rearr[..., 0]  # (..., seq_len, d_k_half)
        x1 = x_rearr[..., 1]  # (..., seq_len, d_k_half)

        # Apply rotation matrix: [cos -sin] [x0]   = [x0*cos - x1*sin]
        #                        [sin  cos] [x1]     [x0*sin + x1*cos]
        x0_new = x0 * cos_selected - x1 * sin_selected
        x1_new = x0 * sin_selected + x1 * cos_selected

        # Stack back and rearrange to original shape
        return rearrange(
            [x0_new, x1_new], "d2 ... seq_len d_k_half -> ... seq_len (d_k_half d2)"
        )
