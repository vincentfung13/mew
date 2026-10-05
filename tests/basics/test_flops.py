"""
Tests for the model-FLOP accounting used as the MFU numerator.

Interface under test:
- `flops_per_token(self, seq_len) -> int` on every module that owns a matmul
  (`Linear`, `SwiGLU`, `CausalMultiHeadSelfAttn`). It returns *forward* FLOPs
  per token; the trainer multiplies by 3 for a training step.
- `module_flops_per_token(module, seq_len)` in `mew/perf.py`. A module that
  defines `flops_per_token` is counted via that method and its children are
  NOT visited (so an MoE layer or a looping parent can account for how often
  its children actually run). Modules without the method recurse into their
  children, unless they own parameters directly: then the aggregator raises a
  `TypeError` naming the fully qualified class (`<module>.<qualname>`), so a
  new layer cannot be silently left out and the message points at the class
  to fix. Parameter-owning modules without matmuls (Embedding, RMSNorm) report 0
  explicitly.

Conventions:
- Only matmuls are counted (2 FLOPs per multiply-accumulate). Norms,
  activations, RoPE, softmax, the embedding lookup and the loss are excluded.
- Causal attention counts half of the T x T score matrix, matching
  `_attention_forward_flops` in profiling/functions.py.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from mew.nn.functionals import cross_entropy
from mew.nn.layers import Embedding, Linear, RMSNorm, SwiGLU
from mew.nn.lm import TransformerLM
from mew.nn.transformers import CausalMultiHeadSelfAttn
from mew.perf import module_flops_per_token

D_MODEL = 64
NUM_HEADS = 4
D_HEAD = D_MODEL // NUM_HEADS
D_FF = 172
VOCAB_SIZE = 128
NUM_LAYERS = 2
SEQ_LEN = 32
BATCH_SIZE = 2

GQA_KV_HEADS = [
    pytest.param(NUM_HEADS, id="mha"),
    pytest.param(2, id="gqa2"),
    pytest.param(1, id="mqa"),
]


def _attn_forward_flops(d_model, num_kv_heads, d_head, seq_len, is_causal=True):
    # q and output projections are d_model x d_model; k and v project to the
    # (possibly smaller) GQA width.
    projections = 2 * (2 * d_model * d_model + 2 * d_model * d_head * num_kv_heads)
    # QK^T and PV: each 2 * T * d_model per token for the full T x T matrix,
    # halved for causal masking.
    scores = 4 * seq_len * d_model
    if is_causal:
        scores //= 2
    return projections + scores


def _lm_forward_flops(d_model, d_ff, num_kv_heads, d_head, vocab_size, layers, T):
    per_layer = _attn_forward_flops(d_model, num_kv_heads, d_head, T) + (
        6 * d_model * d_ff
    )
    lm_head = 2 * d_model * vocab_size
    return layers * per_layer + lm_head


def _build_lm(num_kv_heads=NUM_HEADS, attn_impl="naive"):
    torch.manual_seed(0)
    return TransformerLM(
        d_model=D_MODEL,
        d_ff=D_FF,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        vocab_size=VOCAB_SIZE,
        context_len=SEQ_LEN,
        num_transformer_layers=NUM_LAYERS,
        rope_theta=10000.0,
        attn_impl=attn_impl,
    )


def _measured_flops_per_token(fn, num_tokens):
    with FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops() / num_tokens


# ---------------------------------------------------------------------------
# Per-module formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [16, 256])
def test_linear_flops_per_token(seq_len):
    layer = Linear(in_features=48, out_features=80)
    assert layer.flops_per_token(seq_len) == 2 * 48 * 80


@pytest.mark.parametrize("seq_len", [16, 256])
def test_swiglu_flops_per_token(seq_len):
    layer = SwiGLU(D_MODEL, D_FF)
    assert layer.flops_per_token(seq_len) == 6 * D_MODEL * D_FF


@pytest.mark.parametrize("seq_len", [16, 256])
@pytest.mark.parametrize("num_kv_heads", GQA_KV_HEADS)
def test_attention_flops_per_token(num_kv_heads, seq_len):
    layer = CausalMultiHeadSelfAttn(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        theta=10000.0,
        max_seq_len=seq_len,
    )
    expected = _attn_forward_flops(D_MODEL, num_kv_heads, D_HEAD, seq_len)
    assert layer.flops_per_token(seq_len) == expected


@pytest.mark.parametrize("num_kv_heads", GQA_KV_HEADS)
def test_non_causal_attention_counts_full_score_matrix(num_kv_heads):
    # TransformerLM is always causal, so this is the only coverage of the
    # is_causal=False branch.
    layer = CausalMultiHeadSelfAttn(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        theta=10000.0,
        max_seq_len=SEQ_LEN,
        is_causal=False,
    )
    expected = _attn_forward_flops(
        D_MODEL, num_kv_heads, D_HEAD, SEQ_LEN, is_causal=False
    )
    assert layer.flops_per_token(SEQ_LEN) == expected


def test_parameter_owning_layers_without_matmuls_report_zero():
    assert (
        Embedding(num_weights=VOCAB_SIZE, embedding_dim=D_MODEL).flops_per_token(
            SEQ_LEN
        )
        == 0
    )
    assert RMSNorm(d_model=D_MODEL).flops_per_token(SEQ_LEN) == 0


def test_attention_flops_grow_linearly_with_seq_len():
    # Only the score term depends on T: doubling T adds exactly 2 * T * d.
    layer = CausalMultiHeadSelfAttn(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        theta=10000.0,
        max_seq_len=2 * SEQ_LEN,
    )
    delta = layer.flops_per_token(2 * SEQ_LEN) - layer.flops_per_token(SEQ_LEN)
    assert delta == 2 * SEQ_LEN * D_MODEL


# ---------------------------------------------------------------------------
# Whole-model aggregation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_kv_heads", GQA_KV_HEADS)
def test_lm_flops_match_closed_form(num_kv_heads):
    # Embedding and RMSNorm must contribute nothing; lm_head must be included.
    model = _build_lm(num_kv_heads=num_kv_heads)
    expected = _lm_forward_flops(
        D_MODEL, D_FF, num_kv_heads, D_HEAD, VOCAB_SIZE, NUM_LAYERS, SEQ_LEN
    )
    assert module_flops_per_token(model, SEQ_LEN) == expected


@pytest.mark.parametrize("attn_impl", ["naive", "torch_sdpa", "flash_triton"])
def test_lm_flops_independent_of_attn_impl(attn_impl):
    # Model FLOPs describe the math, not the kernel: recomputation in the
    # flash backward or the materialised T x T matrix in naive don't count.
    reference = module_flops_per_token(_build_lm(attn_impl="naive"), SEQ_LEN)
    assert module_flops_per_token(_build_lm(attn_impl=attn_impl), SEQ_LEN) == reference


def test_default_training_config_flops():
    # Dimensions of apps/cfgs/training.yaml at the time of writing (MHA).
    # Pinned so an accidental change to a formula shows up as a concrete number.
    model = TransformerLM(
        d_model=512,
        d_ff=1344,
        num_heads=16,
        num_kv_heads=16,
        vocab_size=10000,
        context_len=256,
        num_transformer_layers=4,
        rope_theta=10000.0,
    )
    forward = module_flops_per_token(model, 256)
    assert forward == 36_192_256
    assert 3 * forward == 108_576_768  # ~108.6 MFLOPs per training token


def test_naive_lm_matches_flop_counter():
    # Independent check against PyTorch's dispatcher-level matmul count, so a
    # module that forgets to report shows up. FlopCounterMode cannot see inside
    # the flash_triton kernel, hence naive. Naive attention computes the full
    # T x T score matrix and masks it afterwards, so the counter also sees the
    # causal half we deliberately leave out.
    model = _build_lm()
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    measured = _measured_flops_per_token(lambda: model(tokens), BATCH_SIZE * SEQ_LEN)
    masked_half = NUM_LAYERS * 2 * SEQ_LEN * D_MODEL
    assert measured == module_flops_per_token(model, SEQ_LEN) + masked_half


def test_training_step_is_three_times_forward():
    # Every matmul's backward computes grads w.r.t. both inputs, each costing
    # as much as the forward matmul.
    model = _build_lm()
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    num_tokens = BATCH_SIZE * SEQ_LEN

    forward = _measured_flops_per_token(lambda: model(tokens), num_tokens)
    train_step = _measured_flops_per_token(
        lambda: cross_entropy(model(tokens), targets).backward(), num_tokens
    )
    assert train_step == 3 * forward


# ---------------------------------------------------------------------------
# Aggregator semantics, on toy modules (future architectures)
# ---------------------------------------------------------------------------


class _Leaf(nn.Module):
    def __init__(self, flops_per_position: int):
        super().__init__()
        self.flops_per_position = flops_per_position

    def flops_per_token(self, seq_len: int) -> int:
        # Depends on seq_len so the test checks it is passed through.
        return self.flops_per_position * seq_len


class _TopKMoE(nn.Module):
    """Stand-in MoE layer: E experts exist, but each token runs only k."""

    def __init__(self, num_experts: int, k: int, router_flops: int):
        super().__init__()
        self.experts = nn.ModuleList([_Leaf(10) for _ in range(num_experts)])
        self.k = k
        self.router_flops = router_flops

    def flops_per_token(self, seq_len: int) -> int:
        return self.router_flops + self.k * self.experts[0].flops_per_token(seq_len)


class _Looped(nn.Module):
    """Stand-in weight-shared block: one module run `num_loops` times."""

    def __init__(self, num_loops: int):
        super().__init__()
        self.block = _Leaf(7)
        self.num_loops = num_loops

    def flops_per_token(self, seq_len: int) -> int:
        return self.num_loops * self.block.flops_per_token(seq_len)


def test_aggregator_sums_through_plain_containers():
    model = nn.Sequential(_Leaf(1), nn.Sequential(_Leaf(2), nn.ReLU()), _Leaf(3))
    assert module_flops_per_token(model, seq_len=5) == (1 + 2 + 3) * 5


def test_aggregator_reporting_parent_hides_children():
    # Summing every module would count all 8 experts (8 * 10 * T); the MoE
    # layer knows only k = 2 of them run per token.
    model = nn.Sequential(_TopKMoE(num_experts=8, k=2, router_flops=3), _Leaf(1))
    seq_len = 4
    assert module_flops_per_token(model, seq_len) == (3 + 2 * 10 * seq_len) + seq_len


def test_aggregator_respects_looping_parent():
    # nn.Module.modules() would yield the shared block once; the parent knows
    # it runs num_loops times.
    model = nn.Sequential(_Looped(num_loops=3))
    assert module_flops_per_token(model, seq_len=2) == 3 * 7 * 2


class _UnreportedLayer(nn.Module):
    """A layer with weights that forgot to implement flops_per_token."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))


class _ReportingParentOfUnreported(nn.Module):
    """Reports for its children, so their missing methods are irrelevant."""

    def __init__(self):
        super().__init__()
        self.inner = _UnreportedLayer()

    def flops_per_token(self, seq_len: int) -> int:
        return 11


def test_aggregator_rejects_parameter_owner_without_method():
    # The error must name the offending class so the missing method is easy to
    # add, even when the layer is nested deep in the model.
    model = nn.Sequential(nn.Sequential(_Leaf(1), _UnreportedLayer()))
    with pytest.raises(TypeError, match="_UnreportedLayer"):
        module_flops_per_token(model, seq_len=4)


def test_aggregator_rejects_unreported_torch_module():
    # torch.nn layers don't know the protocol either; they must not be
    # silently counted as 0. The bare class name "Linear" would be ambiguous
    # with mew.nn.layers.Linear, so the defining module must be included.
    model = nn.Sequential(_Leaf(1), nn.Sequential(nn.Linear(4, 4)))
    with pytest.raises(TypeError, match=r"torch\.nn\.modules\.linear\.Linear"):
        module_flops_per_token(model, seq_len=4)


def test_aggregator_does_not_check_below_reporting_parent():
    model = nn.Sequential(_ReportingParentOfUnreported(), _Leaf(1))
    assert module_flops_per_token(model, seq_len=4) == 11 + 4
