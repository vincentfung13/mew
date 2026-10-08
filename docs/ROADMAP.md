# mew: Distributed Training Roadmap

As of Oct 8, 2026. Source of truth: https://claude.ai/code/artifact/15cff9ce-c901-46e0-9284-5c610c817109

## Goal

In about 8 months, mew becomes a from-scratch stack covering 4D-parallel pretraining (DP × TP × PP × CP, plus EP for MoE) and post-training (SFT and GRPO). Every component ships with a cost model, a parity test, a benchmark against TorchTitan or an established framework, and a write-up that explains the gap.

Principles for every milestone:

- **Predict, then measure.** Write the cost model first (memory, comm volume, expected step time), then compare it with what you measured. The gap and its explanation are the deliverable.
- **Correctness before speed.** Each feature lands with a parity test against the single-GPU run (loss curve and gradients within tolerance) before any optimization.
- **Compare against the industrial version.** Every stage gets a side-by-side with the PyTorch built-in (DDP, FSDP2, DTensor TP, `torch.distributed.pipelining`) or TorchTitan on the same hardware.
- **Publish as you go.** One short write-up per phase, so the story builds even if you stop early.
- **Stop polishing at "explainable".** Matching TorchTitan's MFU is not the bar; explaining the remaining gap is.

## Where mew is today

`feature/data_parallel` (as of commit `e2a9b17`) has a correct naive DDP with buffer broadcast and good measurement plumbing. The next step is turning it into a fast DDP you can benchmark.

Already in place:

- `DistributedDataParallel` wrapper: rank-0 weight broadcast, a per-parameter async `all_reduce` fired from `register_post_accumulate_grad_hook`, a wait queued via the autograd engine callback, and `no_sync()` for gradient accumulation.
- FLOPs-per-token accounting, `ThroughputMeter`, GPU spec table and MFU logging in the trainer.
- Triton FlashAttention (fwd + bwd) with bf16 AMP and parity tests.
- `tests/systems/` already holds `test_ddp.py`, `test_fsdp.py` and `test_sharded_optimizer.py` stubs, which line up with Phases 1–2 below.

Fixed in the latest commit: the `DistContext.from_env` `is_main` bug, and parameter/buffer broadcast now lives in `comm.py` as `broadcast_tensors`. Still open:

- **One collective per parameter.** Fine for correctness, slow at scale: hundreds of small NCCL calls, each paying latency. Bucketing is the first real optimization (Phase 1).
- **No handling for unused parameters.** A parameter that gets no grad never fires its hook; with bucketing this becomes a hang. Decide now: assert all params are used, or track readiness per bucket.
- **Buffer sync is init-only.** That is correct for RoPE's deterministic buffers; keep the per-forward sync TODO for buffers that change during training, and cover it in a test.
- **`comm.py` is a start.** Grow it into the collectives layer for every later phase: wrap all-reduce, reduce-scatter, all-gather, all-to-all and send/recv, and log bytes and time per call.
- **Gloo fallback divides before reducing.** Fine, but keep the NCCL and Gloo paths numerically identical in the parity test.

## Milestones

Ten phases over about 34 weeks: seven for distributed pretraining, then two for post-training, assuming roughly 15 hours a week alongside your job. Each phase ends with a parity test, a benchmark against the PyTorch or TorchTitan equivalent, and a short write-up. Do most development on 1–2 local or cheap GPUs; rent 8–32 GPUs only for benchmark sessions.

| Phase | Weeks | Focus |
| --- | --- | --- |
| 0 | 1 | Harness |
| 1 | 2–4 | Fast DDP |
| 2 | 5–8 | ZeRO and FSDP |
| 3 | 9–12 | Tensor + sequence parallelism |
| 4 | 13–16 | Pipeline parallelism |
| 5 | 17–19 | Context parallelism |
| 6 | 20–23 | MoE + expert parallelism |
| 7 | 24–26 | Checkpoints + capstone |
| 8 | 27–29 | SFT |
| 9 | 30–34 | GRPO |

Phases 1–7 build the parallel training stack; Phases 8–9 reuse it for SFT and GRPO, where rollout generation becomes the new bottleneck to measure.

### Phase 0 — Harness (week 1)

- **Build:** `torchrun` single- and multi-node launch; fixed benchmark configs (about 125M, 1B and 7B-shaped models); a single-GPU reference run with saved losses and grads; comm logging in `comm.py` (bytes and time per collective).
- **Done when:** any parallel config can be checked against the reference with one pytest command, and every run logs tokens/s, MFU, peak memory and comm time.

### Phase 1 — Fast DDP (weeks 2–4)

- **Build:** gradient buckets (about 25 MB, reverse registration order), comm/compute overlap, bf16 gradient reduction option, unused-param handling, `no_sync` kept working.
- **Measure:** step time and scaling efficiency at 1, 2, 4, 8 GPUs and across 2 nodes; naive vs bucketed vs `torch.nn.parallel.DDP`; bucket-size sweep; Nsight trace showing the overlap.
- **Write-up 1:** "Why naive DDP is slow": latency vs bandwidth terms, the ring all-reduce cost (2(N−1)/N × bytes), and what bucketing and overlap recover.

### Phase 2 — ZeRO and FSDP (weeks 5–8)

- **Build:** ZeRO-1 (sharded optimizer state: reduce-scatter grads, all-gather params), ZeRO-2, then ZeRO-3/FSDP with per-transformer-block units, forward/backward prefetch, mixed-precision params, activation checkpointing.
- **Measure:** peak memory vs your formula (params, grads, Adam states, activations) at each stage; largest model trainable on one 8×80 GB node; your FSDP vs FSDP2 throughput.
- **Write-up 2:** a memory model you can derive on a whiteboard, validated by snapshots from your memory-report tool.

### Phase 3 — Tensor and sequence parallelism (weeks 9–12)

- **Build:** column/row-parallel linears for attention and MLP, vocab-parallel embedding and cross-entropy, sequence parallelism for norms, a device mesh so TP composes with FSDP (2D).
- **Measure:** TP = 1, 2, 4, 8 inside a node vs TP = 2 across nodes; the all-reduce/all-gather volume per layer vs your prediction; your TP vs DTensor TP.
- **Write-up 3:** why TP stays inside the NVLink domain, with your own numbers.

### Phase 4 — Pipeline parallelism (weeks 13–16)

- **Build:** stage partitioning, P2P send/recv, GPipe, then 1F1B, then interleaved 1F1B (virtual stages). Read zero-bubble and DualPipe; implement one only if time allows.
- **Measure:** bubble fraction vs (p − 1)/(m + p − 1) across microbatch counts; activation memory per schedule; a full 3D (DP × TP × PP) run on 16–32 GPUs.
- **Write-up 4:** choosing a 3D layout for a given model and cluster, with a small calculator script that predicts step time and the measured result next to it.

### Phase 5 — Context parallelism (weeks 17–19)

- **Build:** ring attention on top of your Triton FlashAttention (merging partial results with log-sum-exp), causal load balancing (zig-zag chunk assignment).
- **Measure:** max sequence length and throughput at 8K–128K tokens with CP = 1, 2, 4, 8; comm hidden vs exposed.
- **Write-up 5:** long-context training and why load balancing matters for causal masks.

### Phase 6 — MoE and expert parallelism (weeks 20–23)

- **Build:** top-k router with load-balancing loss, capacity factor and token dropping, all-to-all dispatch/combine, grouped GEMM for experts, EP composed with DP.
- **Measure:** all-to-all time vs expert count and EP degree; load imbalance over training; dense vs MoE at equal active parameters.
- **Write-up 6:** where MoE training time goes, read against the DeepSeek-V3 report.

### Phase 7 — Production concerns and capstone (weeks 24–26)

- **Build:** sharded, async distributed checkpointing with resharding on load (save at one layout, resume at another); fault injection (kill a rank mid-run, resume); a bitwise-determinism test; optionally FP8 linears.
- **Capstone:** train a roughly 1B-parameter model on a few billion tokens across 16–32 GPUs with your best 4D layout, side by side with TorchTitan on the same hardware and config.
- **Write-up 7:** the capstone report: MFU vs TorchTitan, a breakdown of the gap, and the run log (spikes, restarts, what broke).

### Phase 8 — SFT (weeks 27–29)

A 1B model pretrained on a few billion tokens is too weak for post-training results to mean much. So this phase starts by loading an open small base model (for example a 0.5B–1.5B Llama- or Qwen-style checkpoint) into mew's own modules, and keeps your capstone model as a second baseline.

- **Build:** a weight-conversion script into mew's Transformer (add QK-norm or biases only if the chosen model needs them); chat template and special tokens in the tokenizer; loss masking on prompt tokens; sequence packing with document-aware masking, using variable-length support in your Triton FlashAttention; SFT trainer on top of your FSDP/TP stack.
- **Experiments:** packing vs padding (tokens/s and wasted compute); prompt masking on vs off; learning rate and epoch sweeps on a small instruction dataset; validate the loaded model's logits against the reference implementation first.
- **Write-up 8:** what packing buys you, and how your SFT throughput compares with a TRL or TorchTune run on the same model and data.

### Phase 9 — GRPO (weeks 30–34)

- **Build:** batched sampling with a KV cache (extending your generator); verifiable reward functions (math answers such as GSM8K, or an arithmetic puzzle task); group-relative advantages, the clipped policy loss and an optional KL term to a frozen reference; log-prob recomputation under the training layout; weight sync from trainer to rollout, first colocated on the same GPUs, then with separate rollout ranks.
- **Experiments:** reward and response-length curves over training; group size and KL-coefficient sweeps; standard GRPO vs the length-normalization fix from Dr. GRPO; time split between rollout, log-prob and update steps; colocated vs split rollout throughput.
- **Write-up 9:** where RL time goes. Generation usually dominates, so this ties the post-training work back to the systems story; compare against verl on the same model and task.

## Compute plan

Develop small, benchmark big: write and debug on 1–2 GPUs (or 2 processes on one GPU with Gloo), then rent multi-GPU nodes in short, scripted sessions. Budget roughly 3,500–5,000 H100-hours, or about $9,000–17,000 at $2.50–3.50 per GPU-hour from GPU-focused clouds (an 8×H100 node runs about $20–28 per hour; the big clouds cost 2–4× more). Prices move quickly, so check rates before each session.

| Phases | Hardware | Why | Rough GPU-hours | Rough cost (USD) |
| --- | --- | --- | --- | --- |
| 0–2 | 1×8 GPU node, plus 2 nodes once for DDP scaling | NVLink for intra-node; one inter-node test | 400–600 | 1,000–2,100 |
| 3–4 | 2–4 nodes with InfiniBand | TP across vs within nodes; 3D runs | 800–1,200 | 2,000–4,200 |
| 5–6 | 1–2 nodes | Long context and all-to-all show up on 8–16 GPUs | 400–600 | 1,000–2,100 |
| 7 | 2–4 nodes | Capstone run and TorchTitan comparison | 1,200–1,600 | 3,000–5,600 |
| 8 | 1 node | SFT on a sub-2B model fits on 8 GPUs | 200–300 | 500–1,100 |
| 9 | 1 node | GRPO rollouts plus updates; colocated vs split | 500–700 | 1,300–2,500 |

Multi-node rentals with InfiniBand are often sold only as reserved clusters or at higher rates, so get quotes for Phases 3–4 and 7 specifically.

Ways to keep cost down:

- Insist on InfiniBand or RoCE for multi-node rentals; plain-Ethernet clusters give misleading inter-node numbers.
- Script every benchmark session end to end (setup, run, upload traces, tear down), so a booked node never sits idle.
- Apply for research or open-source compute credits from cloud and GPU providers; a public repo with write-ups is a strong application.
- Use a few short steps per benchmark config; only the capstone needs a long run.

## Reading list by phase

Read each item just before its phase, and cite it in that phase's write-up.

| Phase | Read |
| --- | --- |
| All | Hugging Face, *The Ultra-Scale Playbook*; the TorchTitan paper and repo |
| 1 | PyTorch Distributed: Experiences on Accelerating Data Parallel Training (the PyTorch DDP paper) |
| 2 | ZeRO (Rajbhandari et al.); the PyTorch FSDP paper; FSDP2 design notes |
| 3 | Megatron-LM (Shoeybi et al.); Reducing Activation Recomputation in Large Transformer Models (sequence parallelism) |
| 4 | GPipe; Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (interleaved 1F1B); Zero Bubble Pipeline Parallelism |
| 5 | Ring Attention with Blockwise Transformers; the Llama 3 paper's context-parallelism section |
| 6 | Switch Transformers; GShard; DeepSeek-V3 technical report (DualPipe, FP8, all-to-all overlap) |
| 7 | MegaScale (ByteDance); Llama 3 paper, infrastructure and reliability sections; the OPT-175B logbook |
| 8 | InstructGPT (Ouyang et al.); Tülu 3 report (SFT data and recipes) |
| 9 | DeepSeekMath (introduces GRPO); DeepSeek-R1; Understanding R1-Zero-Like Training (Dr. GRPO); HybridFlow (the verl paper) |
