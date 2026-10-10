"""
Tests for `_sync_throughput_report` in `mew/trainers/npt_trainer.py`, which turns
each rank's `ThroughputMeter.report()` into logging keys:

    world_size == 1:  perf/<metric>                  (this rank's value)
    world_size  > 1:  perf/<metric>                  (mean across ranks)
                      perf/min/<metric>, perf/max/<metric>
                      perf/local/<metric>            (this rank's own value)

The multi-rank case runs 2 CPU processes on the gloo backend.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mew.parallel.dist_context import DistContext
from mew.trainers.npt_trainer import _sync_throughput_report

# One report per rank. peak_alloc_gib is an int on purpose: report values may be
# ints or floats, and both must survive being packed into one tensor.
RANK_REPORTS = [
    {"tokens_per_s": 1000.0, "mfu": 0.40, "peak_alloc_gib": 10},
    {"tokens_per_s": 3000.0, "mfu": 0.20, "peak_alloc_gib": 14},
]


def _expected_synced(rank: int) -> dict:
    expected = {}
    for metric in RANK_REPORTS[0]:
        values = [report[metric] for report in RANK_REPORTS]
        expected[f"perf/{metric}"] = sum(values) / len(values)
        expected[f"perf/min/{metric}"] = min(values)
        expected[f"perf/max/{metric}"] = max(values)
        expected[f"perf/local/{metric}"] = RANK_REPORTS[rank][metric]
    return expected


def test_single_process_passes_report_through():
    dist_context = DistContext(
        rank=0, local_rank=0, world_size=1, device=torch.device("cpu")
    )
    synced = _sync_throughput_report(
        throughput_report=RANK_REPORTS[0], dist_context=dist_context
    )
    assert synced == {f"perf/{k}": v for k, v in RANK_REPORTS[0].items()}


def test_two_ranks_report_mean_min_max_and_local(tmp_path):
    world_size = len(RANK_REPORTS)
    # File rendezvous in tmp_path: no fixed port to collide with other tests
    init_method = f"file://{tmp_path / 'pg_init'}"
    mp.spawn(
        _two_ranks_worker,
        args=(world_size, init_method),
        nprocs=world_size,
        join=True,
    )


def _two_ranks_worker(rank: int, world_size: int, init_method: str):
    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    try:
        dist_context = DistContext(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        synced = _sync_throughput_report(
            throughput_report=RANK_REPORTS[rank], dist_context=dist_context
        )

        expected = _expected_synced(rank)
        assert set(synced) == set(expected)
        for key, value in expected.items():
            # Values travel as float32 tensors
            assert synced[key] == pytest.approx(value, rel=1e-6), key
            assert isinstance(synced[key], float), f"{key} is not a plain float"
    finally:
        dist.barrier()
        dist.destroy_process_group()
