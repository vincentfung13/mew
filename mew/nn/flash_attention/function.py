import math
import torch
import triton

from mew.nn.flash_attention.kernels.forward import flash_fwd_kernel
from mew.nn.flash_attention.kernels.backward import (
    flash_bwd_kernel_dkv,
    flash_bwd_kernel_dq,
)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,  # (batch, n_q_heads, n_q, d_head)
        K: torch.Tensor,  # (batch, n_kv_heads, n_kv, d_head)
        V: torch.Tensor,  # (batch, n_kv_heads, n_kv, d_head)
        is_causal: bool = False,
        cfg: dict = {"Q_TILE_SIZE": 16, "K_TILE_SIZE": 16},
    ):
        batch, n_q_heads, n_q, d = Q.size()
        batch, n_kv_heads, n_kv, _ = K.size()
        scale = math.sqrt(d)
        assert torch.cuda.is_available()

        # Init output buffer:
        # O should match the dtype of Q (avoid fp16 <-> fp32 mismatch);
        # L always needs to be fp32, because backward uses it as an exponent offset in exp(S - L)
        # where a rounding error in L turns into a relative error in every recomputed probability.
        O = torch.empty_like(Q, device="cuda")
        L = torch.empty((batch, n_q_heads, n_q), dtype=torch.float32, device="cuda")

        # Launch triton kernel (launch grid is [n_queries, batch * n_q_heads])
        flash_fwd_kernel[(triton.cdiv(n_q, cfg["Q_TILE_SIZE"]), batch * n_q_heads)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            O.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            N_QUERIES=n_q,
            N_KEYS=n_kv,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            scale=scale,
            dim=d,
            Q_TILE_SIZE=cfg["Q_TILE_SIZE"],
            K_TILE_SIZE=cfg["K_TILE_SIZE"],
            IS_CAUSAL=is_causal,
        )

        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.Q_TILE_SIZE = cfg["Q_TILE_SIZE"]
        ctx.K_TILE_SIZE = cfg["K_TILE_SIZE"]
        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        # Retrieve saved tensors from forward
        L, Q, K, V, O = ctx.saved_tensors
        batch, n_q_heads, n_q, d = Q.size()
        batch, n_kv_heads, n_kv, _ = K.size()
        scale = ctx.scale

        # Init result pointers
        dQ = torch.empty_like(Q, device="cuda")
        dK = torch.empty_like(K, device="cuda")
        dV = torch.empty_like(V, device="cuda")

        # Pre-compute row sum for dO to simplify softmax grad
        # dO -> (batch, n_q_heads, n_q, dim)
        D = (O.float() * dO).sum(axis=-1)

        # Launch triton kernels - two outer loops
        flash_bwd_kernel_dq[(triton.cdiv(n_q, ctx.Q_TILE_SIZE), batch * n_q_heads)](
            Q,
            K,
            V,
            L,  # (batch, n_q_heads, n_q)
            D,  # (batch, n_q_heads, n_q)
            dO,  # (batch, n_q_heads, n_q, d)
            dQ,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            D.stride(0),
            D.stride(1),
            D.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dO.stride(3),
            N_QUERIES=n_q,
            N_KEYS=n_kv,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            scale=scale,
            dim=d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            IS_CAUSAL=ctx.is_causal,
        )

        flash_bwd_kernel_dkv[(triton.cdiv(n_kv, ctx.K_TILE_SIZE), batch * n_kv_heads)](
            Q,
            K,
            V,
            L,
            D,  # (batch, n_q_heads, n_q)
            dO,  # (batch, n_q_heads, n_q, d)
            dK,
            dV,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            D.stride(0),
            D.stride(1),
            D.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dO.stride(3),
            N_QUERIES=n_q,
            N_KEYS=n_kv,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            scale=scale,
            dim=d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            IS_CAUSAL=ctx.is_causal,
        )

        return dQ, dK, dV, None, None
