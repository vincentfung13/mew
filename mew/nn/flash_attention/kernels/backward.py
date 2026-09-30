import triton
import triton.language as tl


@triton.jit
def flash_bwd_kernel_dkv(
    # pointers to:
    # Q -> (batch, n_q, d)
    # K -> (batch, n_kv, d)
    # V -> (batch, n_kv, d)
    Q_ptr,
    K_ptr,
    V_ptr,
    # pointers to:
    # L -> (batch, n_q) max_logits + log_exp_sum pre_computed in forward
    L_ptr,
    # D -> (batch, n_q) row sum of O * dO, precomputed to simply softmax backward
    D_ptr,
    # dO -> (batch, n_q, d)
    dO_ptr,
    # pointers to:
    # dK -> (batch, n_kv, d)
    # dV -> (batch, n_kv, d)
    dK_ptr,
    dV_ptr,
    # strides
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kq,
    stride_kd,
    stride_vb,
    stride_vq,
    stride_vd,
    stride_lb,
    stride_lq,
    stride_dob,
    stride_doq,
    stride_dod,
    N_QUERIES,
    N_KEYS,
    scale,
    dim: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    kv_tile_ind = tl.program_id(0)
    batch_ind = tl.program_id(1)

    # Create blk pointers for inputs
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_ind * stride_qb,
        shape=(N_QUERIES, dim),
        strides=(stride_qq, stride_qd),
        # always load from the first position from:
        # Q_ptr + batch_ind * stride_qb
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, dim),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_ind * stride_kb,
        shape=(N_KEYS, dim),
        strides=(stride_kq, stride_kd),
        offsets=(kv_tile_ind * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, dim),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_ind * stride_vb,
        shape=(N_KEYS, dim),
        strides=(stride_vq, stride_vd),
        offsets=(kv_tile_ind * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, dim),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_ind * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_ind * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_ind * stride_dob,
        shape=(N_QUERIES, dim),
        strides=(stride_doq, stride_dod),
        # always load from the first position from:
        # dO_ptr + batch_ind * stride_dob
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, dim),
        order=(1, 0),
    )

    # Create blk pointers for outputs
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_ind * stride_kb,
        shape=(N_KEYS, dim),
        strides=(stride_kq, stride_kd),
        offsets=(kv_tile_ind * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, dim),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_ind * stride_vb,
        shape=(N_KEYS, dim),
        strides=(stride_vq, stride_vd),
        offsets=(kv_tile_ind * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, dim),
        order=(1, 0),
    )

    # Load K_j and V_j
    K_j = tl.load(
        K_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (K_TILE_SIZE, dim)
    V_j = tl.load(
        V_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (K_TILE_SIZE, dim)
    K_offsets = kv_tile_ind * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # (K_TILE_SIZE,)
    K_is_valid = K_offsets[None, :] < N_KEYS

    # Init buffer for output
    dK_j = tl.zeros((K_TILE_SIZE, dim), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, dim), dtype=tl.float32)

    # Load query and compute grad tile by tile
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(
            Q_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (Q_TILE_SIZE, dim)
        L_i = tl.load(
            L_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (Q_TILE_SIZE,)
        D_i = tl.load(
            D_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (Q_TILE_SIZE,)
        dO_i = tl.load(
            dO_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (Q_TILE_SIZE, dim)
        Q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)  # (Q_TILE_SIZE, )
        Q_is_valid = Q_offsets[:, None] < N_QUERIES

        # To handle partial tile, each (q_ind, k_ind/v_ind) is valid
        QK_is_valid = Q_is_valid & K_is_valid

        # Recompute S_i and P_i
        S_ij = tl.dot(Q_i, tl.trans(K_j)) / scale  # (Q_TILE_SIZE, K_TILE_SIZE)
        if IS_CAUSAL:
            # Upper triangular causal mask
            mask = (
                Q_offsets[:, None] >= K_offsets[None, :]
            )  # (Q_TILE_SIZE, K_TILE_SIZE)
            S_ij = tl.where(mask & QK_is_valid, S_ij, -1e6)
        else:
            S_ij = tl.where(QK_is_valid, S_ij, -1e6)
        P_ij = tl.exp(S_ij - L_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Compute and aggregate dV_j
        dV_j += tl.dot(tl.trans(P_ij), dO_i)  # (K_TILE_SIZE, dim)

        # Compute Jacobian dP_ij and then dS_ij
        dP_ij = tl.dot(dO_i, tl.trans(V_j))  # (Q_TILE_SIZE, K_TILE_SIZE)
        dS_ij = P_ij * (dP_ij - D_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        dK_j += tl.dot(tl.trans(dS_ij), Q_i) / scale  # (K_TILE_SIZE, dim)

        # Advance the pointers
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))

    # Save dK_j and dV_j to buffer
    tl.store(dK_block_ptr, dK_j, boundary_check=(0,))
    tl.store(dV_block_ptr, dV_j, boundary_check=(0,))


@triton.jit
def flash_bwd_kernel_dq(
    # pointers to:
    # Q -> (batch, n_q, d)
    # K -> (batch, n_kv, d)
    # V -> (batch, n_kv, d)
    Q_ptr,
    K_ptr,
    V_ptr,
    # pointers to:
    # L -> (batch, n_q) max_logits + log_exp_sum pre_computed in forward
    L_ptr,
    # D -> (batch, n_q) row sum of O * dO, precomputed to simply softmax backward
    D_ptr,
    # dO -> (batch, n_q, d)
    dO_ptr,
    # pointer to:
    # dQ -> (batch, n_q, d)
    dQ_ptr,
    # strides
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kq,
    stride_kd,
    stride_vb,
    stride_vq,
    stride_vd,
    stride_lb,
    stride_lq,
    stride_dob,
    stride_doq,
    stride_dod,
    N_QUERIES,
    N_KEYS,
    scale,
    dim: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_ind = tl.program_id(0)
    batch_ind = tl.program_id(1)

    # Init input blk ptrs
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_ind * stride_qb,
        shape=(N_QUERIES, dim),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_ind * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, dim),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_ind * stride_kb,
        shape=(N_KEYS, dim),
        strides=(stride_kq, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, dim),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_ind * stride_vb,
        shape=(N_KEYS, dim),
        strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, dim),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_ind * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_ind * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_ind * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_ind * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_ind * stride_dob,
        shape=(N_QUERIES, dim),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_ind * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, dim),
        order=(1, 0),
    )

    # Init output blk ptr
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_ind * stride_qb,
        shape=(N_QUERIES, dim),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_ind * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, dim),
        order=(1, 0),
    )

    # Load Q_i, L_i, D_i and dO_i
    Q_i = tl.load(
        Q_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (Q_TILE_SIZE, dim)
    L_i = tl.load(
        L_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (Q_TILE_SIZE,)
    D_i = tl.load(
        D_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (Q_TILE_SIZE,)
    dO_i = tl.load(
        dO_block_ptr, boundary_check=(0,), padding_option="zero"
    )  # (Q_TILE_SIZE, dim)
    Q_offsets = query_tile_ind * Q_TILE_SIZE + tl.arange(
        0, Q_TILE_SIZE
    )  # (Q_TILE_SIZE,)
    Q_is_valid = Q_offsets[:, None] < N_QUERIES

    # Init buffer for output
    dQ_i = tl.zeros((Q_TILE_SIZE, dim), dtype=tl.float32)

    # Load kv and compute grad tile by tile
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load K_j and V_j
        K_j = tl.load(
            K_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (K_TILE_SIZE, dim)
        V_j = tl.load(
            V_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (K_TILE_SIZE, dim)
        K_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # (K_TILE_SIZE, )
        K_is_valid = K_offsets[None, :] < N_KEYS
        # To handle partial tile, each (q_ind, k_ind/v_ind) is valid
        QK_is_valid = Q_is_valid & K_is_valid

        # Recompute S_i and P_i
        S_ij = tl.dot(Q_i, tl.trans(K_j)) / scale  # (Q_TILE_SIZE, K_TILE_SIZE)
        if IS_CAUSAL:
            # Upper triangular causal mask
            mask = (
                Q_offsets[:, None] >= K_offsets[None, :]
            )  # (Q_TILE_SIZE, K_TILE_SIZE)
            S_ij = tl.where(mask & QK_is_valid, S_ij, -1e6)
        else:
            S_ij = tl.where(QK_is_valid, S_ij, -1e6)
        P_ij = tl.exp(S_ij - L_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Compute Jacobian dP_ij and then dS_ij
        dP_ij = tl.dot(dO_i, tl.trans(V_j))  # (Q_TILE_SIZE, K_TILE_SIZE)
        dS_ij = P_ij * (dP_ij - D_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        dQ_i += tl.dot(dS_ij, K_j) / scale  # (Q_TILE_SIZE, dim)

        # Advance block ptrs
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_i, boundary_check=(0,))
