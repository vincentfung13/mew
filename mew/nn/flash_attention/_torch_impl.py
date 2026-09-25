import math

import torch
from einops import einsum

TILE_SIZE_Q = 16
TILE_SIZE_KV = 16


class FlashAttentionTorchImpl(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,  # (batch, n_q, d)
        K: torch.Tensor,  # (batch, n_kv, d)
        V: torch.Tensor,  # (batch, n_kv, d)
        is_causal: bool = False,
    ):
        # batch size
        batch = Q.size()[0]

        # model dim
        d = Q.size()[-1]

        # number of qs and kvs
        n_q = Q.size(1)
        n_kv = K.size()[1]
        num_blk_q = math.ceil(n_q / TILE_SIZE_Q)
        num_blk_kv = math.ceil(n_kv / TILE_SIZE_KV)

        # output buffer
        O_acc = torch.empty_like(Q)
        L = torch.empty((batch, n_q))

        for i in range(num_blk_q):
            # Load Q_i
            head_q = i * TILE_SIZE_Q
            tail_q = min(n_q, (i + 1) * TILE_SIZE_Q)
            Q_i = Q[:, head_q:tail_q]  # (batch, TILE_SIZE_Q, d)
            O_i = torch.empty_like(Q_i)  # (batch, TILE_SIZE_Q, d)
            l_i = torch.zeros((batch, min(tail_q - head_q, TILE_SIZE_Q)))
            m_i = torch.full((batch, min(tail_q - head_q, TILE_SIZE_Q)), float("-inf"))

            # Load K_j, V_j
            for j in range(num_blk_kv):
                head_kv = j * TILE_SIZE_KV
                tail_kv = min(n_kv, (j + 1) * TILE_SIZE_KV)
                K_j = K[:, head_kv:tail_kv]  # (batch, TILE_SIZE_KV, d)
                V_j = V[:, head_kv:tail_kv]  # (batch, TILE_SIZE_KV, d)

                # Compute tile-of pre-softmax attn
                Q_i_shape = "batch TILE_SIZE_Q d"
                K_j_shape = "batch TILE_SIZE_KV d"
                S_ij_shape = "batch TILE_SIZE_Q TILE_SIZE_KV"
                S_ij = einsum(
                    Q_i, K_j, f"{Q_i_shape}, {K_j_shape} -> {S_ij_shape}"
                ) / math.sqrt(d)

                # Compute local max and partial softmax denominator
                _new_max = torch.maximum(
                    m_i,
                    torch.max(S_ij, dim=-1)[0],  # (batch, TILE_SIZE_Q, 1)
                )
                S_ij = S_ij - _new_max[:, :, None]
                P_ij = S_ij.exp()  # (batch, TILE_SIZE_Q, TILE_SIZE_KV)

                # Compute running exp sum, calibrate with new local max
                l_i = l_i * (m_i - _new_max).exp() + P_ij.sum(dim=-1)

                # Aggregate & calibrate local output
                O_i = O_i * (m_i - _new_max).exp()[:, :, None] + einsum(
                    P_ij, V_j, f"{S_ij_shape}, {K_j_shape} -> {Q_i_shape}"
                )  # (batch, TILE_SIZE_Q, d)

                # update running max
                m_i = _new_max

            # Normalize O_i, l_i and write to O, L
            O_acc[:, head_q:tail_q] = O_i / l_i[:, :, None]

            # -logits.max only works if we're normalizing with full softmax,
            # to cal the real log exp sum for backward, we need to add back m_i
            L[:, head_q:tail_q] = m_i + l_i.log()

        ctx.save_for_backward(L, Q, K, V, O_acc)

        return O_acc

    @staticmethod
    def backward(ctx):
        raise NotImplementedError
