import pytest
import torch
from einops import einsum

from .adapters import get_flashattention_autograd_function_triton


def _attention_and_lse(q, k, v, is_causal=False):
    """
    Reference attention. q is (B, H_q, N, D), k/v are (B, H_kv, N, D);
    output is (B, H_q, N, D) and L is (B, H_q, N).

    For GQA (H_kv < H_q), query head h attends to kv head h // (H_q // H_kv),
    matching the "(num_groups num_grouped_queries)" grouping in mew.
    """
    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    d = q.shape[-1]
    scale = 1 / (d**0.5)
    S = einsum(q, k, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        S = torch.where(
            torch.arange(n_queries, device=S.device)[:, None]
            >= torch.arange(n_keys, device=S.device)[None, :],
            S,
            -1e6,
        )
    P = torch.softmax(S, dim=-1)
    o = einsum(P, v, "... q k, ... k d -> ... q d")
    L = torch.logsumexp(S, dim=-1)
    return o, L


def _make_attn_inputs(device=None, non_contiguous=False, n_kv_heads=None):
    """
    Returns (q, k, v, do, leaves). q is (B, H_q, N, D) and k/v are
    (B, H_kv, N, D), as fed to the kernel; leaves are the tensors that
    receive .grad. n_kv_heads defaults to H_q (plain multi-head attention).

    If non_contiguous, the leaves are allocated as (B, N, H, D) and q/k/v are
    their .transpose(1, 2) views, so the kernel sees non-contiguous strides.
    """
    torch.random.manual_seed(0)
    batch_size = 2
    n_queries = 128
    n_keys = 128
    n_heads = 4
    if n_kv_heads is None:
        n_kv_heads = n_heads
    D = 64
    shape_q = (batch_size, n_heads, n_queries, D)
    shape_k = (batch_size, n_kv_heads, n_keys, D)

    def make(shape):
        if non_contiguous:
            b, h, n, d = shape
            leaf = torch.randn(b, n, h, d, device=device, requires_grad=True)
            return leaf, leaf.transpose(1, 2)
        leaf = torch.randn(*shape, device=device, requires_grad=True)
        return leaf, leaf

    q_leaf, q = make(shape_q)
    k_leaf, k = make(shape_k)
    v_leaf, v = make(shape_k)
    do = torch.randn(*shape_q, device=device)

    if non_contiguous:
        assert not q.is_contiguous()
    return q, k, v, do, (q_leaf, k_leaf, v_leaf)


def _test_flash_forward_pass(
    impl, device="cpu", is_causal=False, non_contiguous=False, n_kv_heads=None
):
    q, k, v, _do, _ = _make_attn_inputs(device, non_contiguous, n_kv_heads)
    o = impl(q, k, v, is_causal)
    assert o.shape == q.shape

    # Extract L from the saved tensors
    assert (
        o.grad_fn.saved_tensors is not None
    ), "No saved tensors found in the output tensor. Make sure your autograd forward is saving them using ctx.save_for_backward."
    expected_l_shape = tuple(q.shape[:3])  # (B, H, N_q)
    maybe_ls = [t for t in o.grad_fn.saved_tensors if t.shape == expected_l_shape]

    assert (
        len(maybe_ls) == 1
    ), f"Expected one tensor of shape {expected_l_shape} in saved tensors, but found {len(maybe_ls)}. The tests require you to save exactly one tensor of this shape, corresponding to the log-sum-exp of the attention scores."
    lse = maybe_ls[0]

    o_ref, lse_ref = _attention_and_lse(q, k, v, is_causal)

    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("non_contiguous", [False, True])
# n_kv_heads: 4 = MHA (H_q = 4), 2 = GQA (2 query heads per kv head), 1 = MQA
@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
def test_flash_forward_pass_triton(is_causal, non_contiguous, n_kv_heads):
    _test_flash_forward_pass(
        get_flashattention_autograd_function_triton().apply,
        device="cuda",
        is_causal=is_causal,
        non_contiguous=non_contiguous,
        n_kv_heads=n_kv_heads,
    )


def flash_backward_results(
    impl, is_causal, device=None, non_contiguous=False, n_kv_heads=None
):
    q, k, v, do, leaves = _make_attn_inputs(device, non_contiguous, n_kv_heads)
    impl(q, k, v, is_causal).backward(do)
    return tuple(t.grad for t in leaves)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("non_contiguous", [False, True])
@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
def test_flash_backward_triton(is_causal, non_contiguous, n_kv_heads):
    dq_expected, dk_expected, dv_expected = flash_backward_results(
        lambda *args: _attention_and_lse(*args)[0],
        is_causal,
        device="cuda",
        non_contiguous=non_contiguous,
        n_kv_heads=n_kv_heads,
    )
    dq, dk, dv = flash_backward_results(
        get_flashattention_autograd_function_triton().apply,
        is_causal,
        device="cuda",
        non_contiguous=non_contiguous,
        n_kv_heads=n_kv_heads,
    )

    torch.testing.assert_close(dq_expected, dq, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dk_expected, dk, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dv_expected, dv, rtol=1e-2, atol=1e-2)
