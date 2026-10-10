# mew 

<p align="center">
    <img src="resources/logo.jpg" width="30%" alt="mew logo">
</p>

A custom implementation of a GPT-like language model developed entirely from scratch. This project provides the fundamental building blocks for training and running inference on an autoregressive neural probabilistic language model.

## Features

- **Custom Tokenization**: Byte-Pair Encoding (BPE) implementation.
- **Efficient Data Loading**: Memory-optimized Numpy batch loaders capable of handling massive memmap files.
- **Neural Network Architecture**: Custom Transformer blocks, Rotary Positional Embeddings (RoPE), and linear layers built with PyTorch.
- **Optimizers**: Custom AdamW optimizer with learning rate scheduling.
- **Generators**: Autoregressive text generation logic.
- **Trainer**: Training loops with experiment tracking (W&B integration), seeded and exactly resumable data sampling.
- **Distributed Training**: A hand-written DistributedDataParallel (DDP) wrapper with gradient accumulation (`no_sync`), launched with `torchrun` for single-node multi-GPU training.
- **Configuration Management**: Hydra-based configuration for easy parameter sweeping and experiment management.
- **Performance Profiling**: MFU, tokens/s and peak-memory logging during training; per-stage timing, MFU, CUDA memory snapshots and NVTX/Nsight Systems profiling of the training step; function benchmarks across implementations and shapes.

## Project Structure

The repository is organized into a core library, an application layer, and supporting performance tooling:

### 1. `@mew/` (Core Library)
The core engine behind the language model:
- `mew/data_loaders/`: Efficient batching and data loading logic (`numpy_batch_loader`).
- `mew/generators/`: Text generation utilities (`conditional_generator`).
- `mew/nn/`: Neural network architectures, modules, and layers (Transformers, RoPE).
- `mew/optimizers/`: Custom optimizers and schedulers (AdamW, LR scheduling).
- `mew/tokenization/`: BPE tokenizer and text processing tools.
- `mew/trainers/`: Implementations of the training loops (e.g., `NPTTrainer`).
- `mew/parallel/`: Distributed training: `DistContext` (rank, world size and device from the `torchrun` environment, process-group setup), the DDP wrapper (`ddp.py`), and collective helpers (`comm.py`).
- `mew/perf/`: FLOPs-per-token accounting, the GPU peak-throughput table, MFU and peak-memory helpers, and the `ThroughputMeter` used by the trainer.

### 2. `@apps/` (Application Layer)
High-level scripts and configurations:
- `apps/cfgs/`: Hydra configuration files (`training.yaml`, `inference.yaml`, `tokenization.yaml`, and `gpu_specs.yaml`, the per-GPU peak TFLOP/s table used for MFU).
- `apps/launch_training.py`: Entry point for launching model training, on one GPU or under `torchrun`.
- `apps/tokenization.py`: Entry point for running the data tokenization pipelines.

### 3. `@profiling/` (Performance Toolkit)
Standalone profiling workloads and configurations:
- `profiling/profile_module.py`: Hydra entry point that profiles the training step of the configured model (the trainer's own `TrainStep`) with per-stage timing, MFU, peak memory, CUDA memory snapshots, and NVTX-annotated runs.
- `profiling/bench_function.py`: Hydra entry point that benchmarks a single function across providers and shapes with Triton's `do_bench` and `perf_report`.
- `profiling/functions.py`: Function workloads and their providers (e.g. Triton FlashAttention, the eager reference, and PyTorch SDPA).
- `profiling/configs/`: The profiler config (which composes `apps/cfgs/training.yaml`) and the benchmark configs.
- `profiling/examples/`: An LM profiling sweep and an attention benchmark sweep, optionally captured with Nsight Systems.

The repository also includes `skills/pytorch-memory-report/`, which renders an interactive HTML report from a trusted PyTorch CUDA memory snapshot, and [docs/ROADMAP.md](docs/ROADMAP.md), the plan for distributed training (DDP, then FSDP, TP, PP, CP and EP).

## Setup and Installation

This project strictly uses **`uv`** for fast and reliable Python package management. 

1. Ensure you have `uv` installed.
2. Install the project and its dependencies:
   ```bash
   uv sync
   ```

By default this resolves packages against the public PyPI index. If you're on a network where only an internal mirror is reachable, copy `uv.toml.example` to `uv.toml` (gitignored) and fill in your mirror's URL — uv picks it up automatically, no other changes needed. Note that switching indexes and re-running `uv sync`/`uv lock` may change `uv.lock`, since the resolved package graph can differ between indexes.

## Usage

You can run the application scripts using `uv run`. 

**Tokenization:**
```bash
# 1. Train a BPE tokenizer
uv run apps/tokenization.py \
    task_name=train_bpe \
    training.vocab_size=10000 \
    training.input_path='./data/TinyStoriesV2-GPT4-train.txt' \
    training.save_dir='./data/tinystories_bpe_tokenizer'

# 2. Tokenize the training and vaidation file
uv run apps/tokenization.py \
    task_name=tokenize_file \
    file_tokenization.input_path='./data/TinyStoriesV2-GPT4-train.txt' \
    file_tokenization.tokenizer_path='./data/tinystories_bpe_tokenizer' \
    file_tokenization.save_path='./data/tiny_stories_train.tokens.uint16.npy' \
    file_tokenization.num_workers=12

uv run apps/tokenization.py \
    task_name=tokenize_file \
    file_tokenization.input_path='./data/TinyStoriesV2-GPT4-valid.txt' \
    file_tokenization.tokenizer_path='./data/tinystories_bpe_tokenizer' \
    file_tokenization.save_path='./data/tiny_stories_val.tokens.uint16.npy' \
    file_tokenization.num_workers=12
```

**Training:**
```bash
EXP_PREFIX="tinystories_ablation_$(date +%Y%m%d_%H%M%S)"
uv run apps/launch_training.py \
    run_name="${EXP_PREFIX}_mha" \
    trainer.total_steps=10000 \
    data.train_file='./data/tiny_stories_train.tokens.uint16.npy' \
    data.val_file='./data/tiny_stories_val.tokens.uint16.npy' \
    data.batch_size=128 \
    data.seq_len=256 \
    data.tokenizer_path='./data/tinystories_bpe_tokenizer' \
    model.d_model=512 \
    model.d_ff=1344 \
    model.num_heads=16 \
    model.num_kv_heads=16
```
*Note: The launch scripts use Hydra, so you can override configurations via the CLI (e.g., `uv run apps/launch_training.py wandb.enable=True`).*

Each run writes its config copy, tokenizer copy, checkpoints and log files to `checkpoints/<run_name>/<run_id>/` (`run_id` defaults to the launch time). `seed` is required: it seeds weight init and the data loaders' sampling streams, whose positions are saved in checkpoints so a resumed run draws the same batches as an uninterrupted one. Resume with `trainer.resume=true trainer.resume_checkpoint_path=<path to checkpoint>`.

**Distributed training (single node, multiple GPUs):**

Launch the same script with `torchrun`, one process per GPU, and set the backend:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 apps/launch_training.py \
    parallel.backend=nccl \
    run_name=ddp_pilot \
    data.batch_size=64
```

- `--nproc_per_node=gpu` uses every visible GPU. `parallel.backend` must stay `null` (the default) for a plain `uv run` launch.
- `data.batch_size` is the **per-GPU** micro batch, so the global batch is `data.batch_size × optim.grad_accumulation_steps × world_size`. Halving `data.batch_size` on 2 GPUs (as above) keeps the global batch, and hence the loss curve, comparable with a 1-GPU run.
- `trainer.total_steps` counts loop iterations, so each step processes `world_size` times more tokens than on one GPU.
- Rank 0 writes the config, tokenizer copy, checkpoints and the W&B run; all ranks share rank 0's `run_id`, and each writes its own `launch_training_rank<R>.log`. The console shows rank 0's output plus warnings from every rank.
- Logged `train/loss` and `val/loss` are averaged across ranks. `perf/<metric>` is the cross-rank mean, with `perf/min/*` and `perf/max/*` to spot a slow rank; each rank's own numbers are under `perf/local/*` in its log file.
- Each rank samples its own batches from a stream derived from `[seed, rank]`, so resume a DDP run with the same number of GPUs.

**Profiling:**

Training logs `perf/tokens_per_s`, `perf/mfu` and per-GPU peak memory at every log step. To see where the time and memory of a training step go, profile it. The profiler runs the trainer's own step on a synthetic batch, using the training config, so override the training keys (run from the repository root):

```bash
uv run python -m profiling.profile_module \
    model.attn_impl=flash_triton \
    data.batch_size=64 \
    profiling.nvtx.annotate_modules=false \
    profiling.memory_profiling.enable=false
```

It reports forward, backward, optimizer-step and total timings with achieved TFLOP/s and MFU, plus peak memory, and writes `metrics.json` under `profiling.output_dir`. Leave memory profiling on (the default) to also dump a `.pkl` CUDA memory snapshot. Because no data loading happens, its MFU is an upper bound for the trainer's.

Run the LM sweep without Nsight Systems capture:

```bash
USE_NSYS=0 ./profiling/examples/run_lm_sweep.sh
```

Benchmark a single function (how fast it is) across providers and a swept shape. This writes a CSV and a plot to `bench.output_dir`:

```bash
uv run python -m profiling.bench_function \
    bench.mode=fwd_bwd \
    bench.metric=tflops \
    'bench.sweep.x_vals=[512,1024,2048,4096]'
```

See [profiling/README.md](profiling/README.md) for the difference between profiling and benchmarking, how profiler MFU is counted, benchmark modes and providers, Nsight Systems capture, artifact naming, and memory-report instructions. Only open memory snapshots from trusted sources because they use Python's pickle format.

## Development Guidelines

- **Formatting**: Always format the code using `black`.
- **Linting**: Check for lint errors using `flake8`, but ignore the "line too long" error (`E501`).

```bash
uvx black mew/ apps/ tests/
uvx flake8 --ignore=E501 mew/ apps/ tests/
uv run pytest
```

The distributed tests (`tests/systems/test_ddp.py`, `tests/systems/test_trainer_dist_logging.py`) spawn CPU processes on the gloo backend, so they run without GPUs.

See [AGENTS.md](AGENTS.md) for more details regarding instructions for AI agents and code contributors.
