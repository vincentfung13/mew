#!/bin/bash

# Benchmark example for single-head attention across head dims, causal masking,
# sequence lengths, and precisions.
#
# Each run sweeps seq_len over powers of two from 128 to 65536 with perf_report, so
# one invocation per (head dim, causal, precision, mode) covers every sequence
# length. Providers that run out of memory at long sequences (typically the eager
# reference, which materializes the full N x N score matrix) are reported as NaN
# for those points.
#
# A run that fails for any other reason (e.g. a kernel compile error or a failed
# correctness check) does not stop the sweep. Failed runs are listed at the end,
# each run's full output is kept in <run dir>/bench.log, and the script exits
# non-zero if any run failed.
#
# Environment overrides:
#   OUTPUT_DIR  root directory for results (default: benchmarks)
#   METRIC      ms or tflops (default: ms)
set -uo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-benchmarks}
METRIC=${METRIC:-ms}

BATCH_SIZE=1
NUM_HEADS=1
D_HEADS=(64 128 256)
CAUSALS=(true)
SEQ_LENS="[128,512,2048,8192,32768,65536]"
DTYPES=(bf16)
MODES=(fwd bwd fwd_bwd)
PROVIDERS="[flash_triton,reference,torch_sdpa]"

case "$METRIC" in
    ms | tflops) ;;
    *)
        echo "METRIC must be one of: ms, tflops" >&2
        exit 2
        ;;
esac

mkdir -p "$OUTPUT_DIR"

failed_runs=()
num_runs=0

echo "Starting single-head attention benchmark sweep..."

for d_head in "${D_HEADS[@]}"; do
    for is_causal in "${CAUSALS[@]}"; do
        causal_tag="causal"
        if [ "$is_causal" = "false" ]; then
            causal_tag="noncausal"
        fi

        for dtype in "${DTYPES[@]}"; do
            # bf16 carries ~3 significant digits, so loosen the correctness tolerance.
            tolerance=1.0e-2
            if [ "$dtype" = "bf16" ]; then
                tolerance=2.0e-2
            fi

            for mode in "${MODES[@]}"; do
                run_name="attention_b${BATCH_SIZE}_h${NUM_HEADS}_d${d_head}_${causal_tag}_${dtype}_${mode}"
                run_output_dir="${OUTPUT_DIR}/${run_name}"

                echo ""
                echo "=========================================="
                echo "Benchmarking: $run_name"
                echo "Output dir: $run_output_dir"
                echo "=========================================="

                num_runs=$((num_runs + 1))
                mkdir -p "$run_output_dir"

                # torch_sdpa is the correctness reference so that long sequences,
                # where the eager reference runs out of memory, are still checked.
                if ! uv run python -m profiling.bench_function \
                    function=attention \
                    function.batch_size="$BATCH_SIZE" \
                    function.num_heads="$NUM_HEADS" \
                    function.num_kv_heads="$NUM_HEADS" \
                    function.d_head="$d_head" \
                    function.is_causal="$is_causal" \
                    function.atol="$tolerance" \
                    function.rtol="$tolerance" \
                    dtype="$dtype" \
                    bench.mode="$mode" \
                    bench.metric="$METRIC" \
                    "bench.providers=$PROVIDERS" \
                    bench.reference_provider=torch_sdpa \
                    bench.sweep.x_name=seq_len \
                    "bench.sweep.x_vals=$SEQ_LENS" \
                    bench.sweep.x_log=true \
                    bench.output_dir="$run_output_dir" \
                    2>&1 | tee "${run_output_dir}/bench.log"; then
                    echo "FAILED: $run_name (see ${run_output_dir}/bench.log)" >&2
                    failed_runs+=("$run_name")
                fi
            done
        done
    done
done

echo ""
echo "Single-head attention benchmark sweep completed:" \
    "$((num_runs - ${#failed_runs[@]}))/${num_runs} runs succeeded."

if [ "${#failed_runs[@]}" -gt 0 ]; then
    echo "Failed runs:" >&2
    for run_name in "${failed_runs[@]}"; do
        echo "  - ${run_name} (${OUTPUT_DIR}/${run_name}/bench.log)" >&2
    done
    exit 1
fi
