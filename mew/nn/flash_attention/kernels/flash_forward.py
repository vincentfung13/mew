import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    # pointers to:
    # Q -> (batch, n_q, d)
    # K -> (batch, n_kv, d)
    # V -> (batch, n_kv, d)
    Q_ptr, K_ptr, V_ptr,
    # O -> (batch, n_q, d)
    # L -> (batch, n_q)
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kq, stride_kd,
    stride_vb, stride_vq, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr
):
    query_tile_ind = tl.program_id(0)
    batch_ind = tl.program_id(1)

    # Init input blk ptrs
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_ind * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_ind * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_ind * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kq, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_ind * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    # Init output buffer ptrs
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_ind * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_ind * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_ind * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
