# Agent Instructions

This repository contains a custom implementation of a GPT-like language model developed from scratch.

## 0. Purpose: This Is a Learning Project

This repo exists so the user can learn how a GPT-like model works by implementing it themselves, by hand. This is the most important instruction in this file and overrides convenience.

- **NEVER modify anything under `mew/`. This is a strict rule with no exceptions.** All changes to `mew/` are made by the user. This covers model/layer code (`mew/nn/`), tokenization logic (`mew/tokenization/`), optimizers (`mew/optimizers/`), data loaders (`mew/data_loaders/`), generation logic (`mew/generators/`), training loops (`mew/trainers/`), and every other file in `mew/`.
  - Do not create, edit, rename, or delete files in `mew/`, not even for a one-line bug fix, a typo, a lint fix, or a refactor.
  - Do not run commands that change `mew/` indirectly, e.g. `git checkout`/`restore`/`stash`/`reset`/`clean` that touch the user's uncommitted work.
  - The only exception is formatting with `uvx black`, which is allowed on `mew/` (still subject to Section 0.1).
  - If a change to `mew/` is needed, describe it (what, where, and why) and let the user make it.
- **Instead, give suggestions, hints, and explanations.** Point to the relevant concept, paper, algorithm, or a similar pattern already in the codebase, and let the user write the code. Ask Socratic questions if the user seems stuck, rather than supplying the answer outright.
- **Debugging and review are read-only.** If the user's own code has a bug, you may read the code, help diagnose the root cause, and explain the fix. The user applies the fix.
- **Non-core work outside `mew/` may be implemented directly** when the user asks, e.g. Hydra configs, scripts under `apps/`, profiling tools under `profiling/`, reusable agent skills under `skills/`, tests, or documentation. When unsure whether something counts as "core," ask.

## 0.1 Commands: Let the User Run the Experiments

- Read-only inspection commands (e.g. `ls`, `grep`, `find`, `git status`, `git diff`, `git log`) may be run freely.
- Prefer that the user runs tests, profiling jobs, benchmarks, and training runs themselves, since doing so is part of the hands-on learning. Propose the exact command and what to look for in the output instead of running it.
- Ask before running anything that changes state: formatters, installs (`uv pip install`, `uv sync`), git operations that modify history or the working tree, deleting files, or long-running GPU jobs.
- The "always run" instructions in later sections describe which checks to *propose*; the user decides whether to run them.

## 1. Background

The codebase provides the core building blocks to train and run inference on a neural probabilistic language model. It includes custom tokenization (BPE), data loading, neural network layers (Transformers, RoPE), optimizers (AdamW with learning rate scheduling), text generation, training loops, and hand-written distributed training (DDP, with FSDP, TP, PP, CP and EP planned in `docs/ROADMAP.md`). It also includes a standalone toolkit for profiling the model's training step and benchmarking individual functions. The project allows users to understand, experiment with, and measure the fundamental components of modern generative AI models.

## 2. High-Level Design and Modules

The architecture is separated into the core library, application entry points, and supporting tooling:

- **`@mew/`** **(Core Library):**
  - `mew/data_loaders/`: Handles batching and loading data for training (e.g., `numpy_batch_loader`).
  - `mew/generators/`: Contains logic for autoregressive text generation (e.g., `conditional_generator`).
  - `mew/nn/`: Implements the neural network architecture, including Transformer blocks, linear layers, and rotary positional embeddings (RoPE).
  - `mew/optimizers/`: Provides optimization algorithms like AdamW and custom learning rate scheduling.
  - `mew/tokenization/`: Contains the custom Byte-Pair Encoding (BPE) tokenizer and text processing utilities.
  - `mew/trainers/`: Implements the training loops and utilities for training the language model (e.g., `NPTTrainer`).
  - `mew/parallel/`: Distributed training. `dist_context.py` (`DistContext`: rank, local rank, world size and device from the `torchrun` environment; process-group init and shutdown), `ddp.py` (the DDP wrapper: per-parameter async all-reduce from post-accumulate-grad hooks, waited on through an autograd engine callback, plus `no_sync()`), and `comm.py` (collective helpers).
  - `mew/perf/`: FLOPs-per-token accounting (`utils.module_flops_per_token`), the GPU peak table (`gpu_specs.py`), `compute_mfu`/peak-memory helpers, and `ThroughputMeter`.
- **`@apps/`** **(Application Layer):**
  - Contains high-level scripts to execute workflows using the `mew` library.
  - `apps/cfgs/`: Stores Hydra configurations for tokenization, training, and inference.
  - `apps/launch_training.py` & `apps/tokenization.py`: Entry points for launching model training and running the data tokenization pipelines. `launch_training.py` runs on one GPU or under `torchrun`; see Section 7.
  - `apps/cfgs/gpu_specs.yaml`: Peak dense TFLOP/s per GPU model, composed into `training.yaml` as `cfg.gpu_specs` and used as the MFU denominator.
- **`@profiling/`** **(Performance Toolkit):**
  - `profiling/profile_module.py`: Hydra entry point that *profiles* the model's training step (where time and memory go). It runs the trainer's `TrainStep` (`mew/trainers/npt_trainer.py`) on a synthetic batch and reports per-stage timing (forward, backward, optimizer step, total), achieved TFLOP/s and MFU, and peak memory, with optional NVTX annotation and CUDA memory snapshots.
  - `profiling/bench_function.py`: Hydra entry point that *benchmarks* a single function (how fast it is) across providers and a swept shape, using `triton.testing.do_bench` and `perf_report`. Layer-level comparisons belong here.
  - `profiling/functions.py`: Defines function workloads (currently `attention`), their providers (`flash_triton`, `reference`, `torch_sdpa`), FLOP counts, and the `fwd`/`bwd`/`fwd_bwd` modes.
  - `profiling/configs/`: `profile_module.yaml` composes `apps/cfgs/training.yaml` (through `hydra.searchpath`) and adds only profiling settings; `bench_function.yaml` holds benchmark settings, with per-function parameters in `function/`.
  - `profiling/examples/`: Provides an LM sweep and an attention benchmark sweep, with optional Nsight Systems capture.
- **`@skills/`** **(Reusable Analysis Workflows):**
  - `skills/pytorch-memory-report/`: Renders and interprets trusted PyTorch CUDA memory snapshots as interactive HTML reports.
- **`@docs/`** **(Planning):**
  - `docs/ROADMAP.md`: The distributed-training roadmap (phases, parity tests, benchmarks against PyTorch/TorchTitan).

## 3. Package Management

- **Always use** **`uv`** for package management and running the code.
- Example: Use `uv run <script.py>` to execute code or `uv pip install <package>` for managing dependencies to ensure a fast, reliable, and reproducible Python environment.

## 4. Code Formatting and Linting

- **Always format the code with** **`black`.**
- **Check for lint errors with** **`flake8`**, but strictly ignore the "line too long" error (`E501`).
- **Scope:** Only apply `uvx black` formatting and `uvx flake8` linting to `@mew/`, `@apps/`, and `@tests/`. Do not run them on other directories or files in the repository.
- After changes under `@mew/`, `@apps/`, or `@tests/`, **propose** styling and lint checks and run them only once approved (Section 0.1), e.g.:
  ```bash
  uvx black mew/ apps/ tests/
  uvx flake8 mew/ apps/ tests/
  ```

## 5. Testing

- After any code change, **propose** running the test suite (`uv run pytest`) to make sure nothing regresses, and run it only once approved (Section 0.1).
- Distributed code is tested on CPU with the gloo backend through `torch.multiprocessing.spawn`: `tests/systems/test_ddp.py` (DDP wrapper parity against single-process training, through `tests/systems/adapters.py`) and `tests/systems/test_trainer_dist_logging.py` (cross-rank throughput reporting). Prefer a `file://` rendezvous in `tmp_path` over a fixed `MASTER_PORT` in new tests. Per-rank data seeding and checkpoint resume are tested in `tests/basics/test_data_loader_seeding.py`. NCCL and multi-GPU behavior still need a GPU machine.
- The module profiler has CPU-compatible tests under `tests/basics/test_profile_module.py` (config composition, peak resolution, stage running, summary math, an end-to-end CPU run), and the shared MFU/memory helpers under `tests/basics/test_perf_utils.py`; function benchmark configs, providers, and modes are tested on CPU in `tests/basics/test_bench_function.py`. CUDA, `do_bench`, the `flash_triton` provider, and Nsight behavior still require an appropriate GPU environment for end-to-end validation.

## 6. Profiling and Benchmarking Conventions

- Distinguish the two tools: `profile_module` *profiles* (explains where time and memory go inside the model's training step); `bench_function` *benchmarks* (compares how fast functions run across providers and shapes). Use benchmarking to find what is slow and profiling to explain why. `profile_module` is model-level only; layer-level work goes to `bench_function`.
- Run the module profiler from the repository root (its `hydra.searchpath` is relative to the working directory) with `uv run python -m profiling.profile_module`. Model, data shape, AMP, optimizer and GPU peak come from the training config, so override them with the training keys (`model.*`, `data.*`, `trainer.*`); keep only warmup/exec steps, NVTX, memory capture and the output directory under `profiling.*`.
- Keep the profiler on the trainer's code path: it must run `TrainStep` exactly as `NPTTrainer` does (forward, backward, optimizer step on accumulation boundaries) rather than reimplementing a training step. Execution options such as `torch.compile` belong in `TrainStep`, not in the profiler.
- Profiler MFU counts model FLOPs per stage as forward ×1, backward ×2, total ×3 (the optimizer step has none), using `module_flops_per_token` and the same `compute_mfu` and GPU table as the trainer. Its numbers are a compute-only upper bound for training; the gap to the trainer's MFU is input-pipeline and logging overhead.
- Run the function benchmark with `uv run python -m profiling.bench_function`. Keep function dimensions and correctness tolerances in `profiling/configs/function/<name>.yaml`; keep the mode, metric, providers, sweep axis, `do_bench` budgets, and output path in `profiling/configs/bench_function.yaml`.
- Add new function workloads by registering a builder in `profiling/functions.py`. Every provider of a function must accept the same inputs and compute the same result, so the correctness check against `bench.reference_provider` stays meaningful.
- Preserve the three benchmark modes: `fwd` builds no autograd graph, `bwd` times only backward on a retained graph (forward runs once outside the timed region), and `fwd_bwd` times both. Backward modes must pass the inputs as `grad_to_none` so gradient accumulation is not timed.
- Benchmark outputs (`.csv`, `.png`) belong under `bench.output_dir`. Do not commit them unless the user explicitly requests it.
- Per-module NVTX hooks and memory-history recording add CPU overhead to profiler timings; measure MFU with both disabled. If `torch.compile` is added to `TrainStep`, do not attach per-module NVTX hooks to compiled runs, because the hooks can introduce graph breaks.
- Profiling artifacts belong under the configured output directory. Do not commit generated `.pkl` memory snapshots or `.nsys-rep` files unless the user explicitly requests it.
- Treat PyTorch memory snapshots as untrusted pickle data unless their provenance is known. Never load or render a snapshot from an untrusted source.
- When interpreting memory reports, separate measured observations from hypotheses and do not infer a leak from allocation traffic alone.

## 7. Distributed Training Conventions

- Launch multi-GPU training with `uv run torchrun --standalone --nproc_per_node=<N> apps/launch_training.py parallel.backend=nccl ...`. A non-null `parallel.backend` requires a `torchrun` launch (checked by `dist.is_torchelastic_launched()`); a plain `uv run` launch keeps `parallel.backend=null` and runs without a process group.
- `data.batch_size` is the per-GPU micro batch. The global batch is `data.batch_size × optim.grad_accumulation_steps × world_size`; when comparing with a single-GPU run, state which of these was held fixed.
- `TrainStep` keeps `self.model` as the raw module (checkpoints without a `module.` prefix, FLOP counting, NVTX, grad-norm logging) and calls through a separate handle that is the DDP wrapper only when `world_size > 1`. Gradient sync is skipped with `no_sync()` on non-boundary micro-batches; the decision lives in `TrainStep`, the mechanism in the wrapper.
- Side effects run on rank 0 only: config and tokenizer copies, checkpoints (followed by a barrier), and the W&B run. Collectives (loss all-reduce, throughput all-gather, barriers) must be called on every rank, never inside an `is_main` branch, or the run hangs.
- Every rank shares rank 0's `run_id` (`sync_run_id` in `apps/launch_training.py` broadcasts it, because Hydra resolves `${now:...}` per process), so all ranks write into one `save_dir`. Each rank logs to `<job>_rank<R>.log`, log lines carry `[rank<R>]`, and non-main ranks' console handlers are raised to WARNING.
- Logged metrics: losses are averaged across ranks; `perf/<metric>` is the cross-rank mean (so it matches the single-GPU key), with `perf/min/*`, `perf/max/*` and `perf/local/*` alongside. Prefer SUM-then-divide and `all_gather` over `ReduceOp.AVG`, which gloo lacks.
- Data sampling is per rank and counter-based: the trainer derives train/val `np.random.SeedSequence` streams from `[seed, rank]`, each batch spawns one child, and checkpoints store the spawn counts so `resume` fast-forwards exactly. `seed` is required. Resume a DDP run with the same world size.
- Correctness before speed: validate a DDP change against the single-GPU run (same global batch, matching loss curve, identical gradients across ranks) before optimizing it.
