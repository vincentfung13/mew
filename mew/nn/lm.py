import torch
import torch.nn as nn

from mew.nn.layers import Linear, Embedding, RMSNorm
from mew.nn.transformers import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_kv_heads: int,
        vocab_size: int,
        context_len: int,
        num_transformer_layers: int,
        rope_theta: float,
        attn_impl: str = "naive",
    ):
        super().__init__()

        # Init layers
        self.token_embeddings = Embedding(num_weights=vocab_size, embedding_dim=d_model)

        # Transformer layers
        layers = []
        for _ in range(num_transformer_layers):
            layers.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    d_ff=d_ff,
                    theta=rope_theta,
                    max_seq_len=context_len,
                    attn_impl=attn_impl,
                )
            )
        self.layers = nn.Sequential(*layers)

        # Final norm and linear
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch seq_len)

        # embedding lookup
        x = self.token_embeddings(x)  # (batch seq_len d_model)

        # run through transformer blocks
        x = self.layers(x)  # (batch seq_len d_model)

        # final layer norm, lm head
        x = self.lm_head(self.ln_final(x))  # (batch seq_len vocab_size)

        return x
