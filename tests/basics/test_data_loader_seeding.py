"""
Tests for the seeded sampling streams of `NumpyBatchLoader` in
`mew/data_loaders/numpy_batch_loader.py`, and for resuming them through
`save_checkpoint` / `load_checkpoint` in `mew/trainers/utils.py`.

Interface under test:

    root = np.random.SeedSequence([seed, rank])   # built by NPTTrainer
    train_seq, val_seq = root.spawn(2)
    loader = NumpyBatchLoader(data, seq_len, batch_size, seed_seq=train_seq)
    loader.get_batch(device)          # each call spawns one child of seed_seq
    loader.num_batches_spawned()      # -> number of batches drawn so far
    loader.resume(num_batches_drawn)  # fast-forward the stream to that count

All tests run on CPU in a single process.
"""

import numpy as np
import torch

from mew.data_loaders.numpy_batch_loader import NumpyBatchLoader
from mew.trainers.utils import load_checkpoint, save_checkpoint

SEED = 42
DATA = np.arange(1000)
SEQ_LEN = 8
BATCH_SIZE = 4
DEVICE = "cpu"


def _seed_seqs(seed: int = SEED, rank: int = 0):
    # Mirrors NPTTrainer: one root per rank, split into train and val streams
    train_seq, val_seq = np.random.SeedSequence([seed, rank]).spawn(2)
    return train_seq, val_seq


def _loader(seed_seq=None) -> NumpyBatchLoader:
    return NumpyBatchLoader(
        data=DATA, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, seed_seq=seed_seq
    )


def _draw(loader: NumpyBatchLoader, n: int) -> list[torch.Tensor]:
    # x only: with DATA = arange, x fully determines the sampled offsets
    return [loader.get_batch(DEVICE)[0] for _ in range(n)]


def _assert_same(a: list[torch.Tensor], b: list[torch.Tensor]):
    assert len(a) == len(b)
    for i, (x_a, x_b) in enumerate(zip(a, b)):
        assert torch.equal(x_a, x_b), f"batch {i} differs"


def test_same_seed_and_rank_reproduces():
    train_a, _ = _seed_seqs()
    train_b, _ = _seed_seqs()
    _assert_same(_draw(_loader(train_a), 5), _draw(_loader(train_b), 5))


def test_ranks_draw_different_batches():
    rank0, _ = _seed_seqs(rank=0)
    rank1, _ = _seed_seqs(rank=1)
    batches0 = _draw(_loader(rank0), 5)
    batches1 = _draw(_loader(rank1), 5)
    assert all(not torch.equal(x0, x1) for x0, x1 in zip(batches0, batches1))


def test_train_and_val_streams_differ():
    train_seq, val_seq = _seed_seqs()
    train = _draw(_loader(train_seq), 5)
    val = _draw(_loader(val_seq), 5)
    assert all(not torch.equal(t, v) for t, v in zip(train, val))


def test_val_draws_do_not_shift_train_stream():
    train_ref, _ = _seed_seqs()
    reference = _draw(_loader(train_ref), 6)

    train_seq, val_seq = _seed_seqs()
    train_loader, val_loader = _loader(train_seq), _loader(val_seq)
    interleaved = []
    for _ in range(6):
        interleaved += _draw(train_loader, 1)
        _draw(val_loader, 2)
    _assert_same(reference, interleaved)


def test_num_batches_spawned_counts_draws():
    train_seq, _ = _seed_seqs()
    loader = _loader(train_seq)
    assert loader.num_batches_spawned() == 0
    _draw(loader, 3)
    assert loader.num_batches_spawned() == 3


def test_resume_continues_stream():
    train_ref, _ = _seed_seqs()
    reference = _draw(_loader(train_ref), 10)

    # First process: draw 4 batches, record the count
    train_seq, _ = _seed_seqs()
    loader = _loader(train_seq)
    _draw(loader, 4)
    count = loader.num_batches_spawned()

    # New process: fresh stream from the same seed and rank, fast-forwarded
    train_seq, _ = _seed_seqs()
    resumed = _loader(train_seq)
    resumed.resume(count)
    assert resumed.num_batches_spawned() == count
    _assert_same(reference[4:], _draw(resumed, 6))


def test_resume_keeps_stream_identity():
    train_seq, _ = _seed_seqs()
    loader = _loader(train_seq)
    loader.resume(7)
    assert loader.seed_seq.entropy == train_seq.entropy
    assert loader.seed_seq.spawn_key == train_seq.spawn_key
    assert loader.seed_seq.pool_size == train_seq.pool_size


def test_resume_twice_through_checkpoints(tmp_path):
    # Uninterrupted reference: 9 train batches, 3 val batches (one per segment)
    train_ref, val_ref = _seed_seqs()
    ref_train_loader, ref_val_loader = _loader(train_ref), _loader(val_ref)
    ref_train, ref_val = [], []
    for _ in range(3):
        ref_train += _draw(ref_train_loader, 3)
        ref_val += _draw(ref_val_loader, 1)

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Three "processes", each saving a checkpoint the next one resumes from.
    # Resuming twice catches a count that stops updating after the first resume.
    got_train, got_val = [], []
    ckpt_path = None
    for segment in range(3):
        train_seq, val_seq = _seed_seqs()
        train_loader, val_loader = _loader(train_seq), _loader(val_seq)
        if ckpt_path is not None:
            iteration = load_checkpoint(
                src=ckpt_path,
                model=model,
                optimizer=optimizer,
                train_data_loader=train_loader,
                val_data_loader=val_loader,
            )
            assert iteration == 3 * segment

        got_train += _draw(train_loader, 3)
        got_val += _draw(val_loader, 1)

        ckpt_path = tmp_path / f"checkpoint_{segment}.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=3 * (segment + 1),
            output_path=ckpt_path,
            train_batch_spawned=train_loader.num_batches_spawned(),
            val_batch_spawned=val_loader.num_batches_spawned(),
        )

    _assert_same(ref_train, got_train)
    _assert_same(ref_val, got_val)

    # Counts are plain ints, so torch.load's default weights_only=True accepts them
    final = torch.load(ckpt_path)
    assert final["train_batch_spawned"] == 9
    assert final["val_batch_spawned"] == 3


def test_default_seed_is_fresh_entropy():
    # Without a seed_seq each loader draws fresh OS entropy, so two loaders differ
    # (test_get_batch in test_data.py relies on this to see varied batches)
    assert not torch.equal(_draw(_loader(), 1)[0], _draw(_loader(), 1)[0])
