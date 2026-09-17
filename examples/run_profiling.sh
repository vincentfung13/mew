#!/bin/bash

# Profiling sweep across model sizes and execution modes.
#
# Set USE_NSYS=1 (default) to capture an Nsight Systems report per setting,
# or USE_NSYS=0 to run the plain profiling script (per-stage timing table only,
# no nsys report). Example:
#   USE_NSYS=0 ./examples/run_profiling.sh
set -euo pipefail

USE_NSYS=${USE_NSYS:-1}
AMP_DTYPE=${AMP_DTYPE:-bf16}
OUTPUT_DIR=${OUTPUT_DIR:-profiles}

mkdir -p "$OUTPUT_DIR"

run_profiling() {
    local model_size=$1
    local d_model=$2
    local d_ff=$3
    local num_heads=$4
    local num_transformer_layers=$5
    local amp_enable=$6
    local forward_only=$7
    local amp_dtype=$8

    local precision_tag="fp32"
    if [ "$amp_enable" = "true" ]; then
        precision_tag="amp_${amp_dtype}"
    fi

    local step_tag="full_step"
    if [ "$forward_only" = "true" ]; then
        step_tag="forward_only"
    fi

    local run_name="${model_size}_${precision_tag}_${step_tag}"
    local run_output_dir="${OUTPUT_DIR}/${run_name}"
    local report_path="${run_output_dir}/nsys_report"
    local mem_profile_output_path="mem_profile.pkl"

    mkdir -p "$run_output_dir"

    echo ""
    echo "=========================================="
    echo "Profiling: $run_name"
    echo "Output dir: $run_output_dir"
    echo "=========================================="

    if [ "$USE_NSYS" -eq 1 ]; then
        # --capture-range=cudaProfilerApi makes nsys record only the region
        # between torch.cuda.profiler.start()/stop() (the post-warmup exec
        # loop), so warmup steps are excluded from the report.
        # --capture-range-end=stop lets the app keep running after the range
        # ends so it prints the per-stage timing table and exits cleanly.
        # The profiling.nvtx flags enable per-module NVTX ranges and that
        # cudaProfilerStart/Stop bracket.
        uv run nsys profile \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --output="$report_path" \
            --force-overwrite=true \
            -- python apps/run_profiling.py \
            model.d_model="$d_model" \
            model.d_ff="$d_ff" \
            model.num_heads="$num_heads" \
            model.num_groups=null \
            model.num_transformer_layers="$num_transformer_layers" \
            profiling.nvtx.annotate_modules=true \
            profiling.nvtx.use_cudart_range=true \
            profiling.output_dir="$run_output_dir" \
            profiling.memory_profiling.enable=true \
            profiling.memory_profiling.output_path="$mem_profile_output_path" \
            profiling.forward_only="$forward_only" \
            amp.enable="$amp_enable" \
            amp.dtype="$amp_dtype"
    else
        # No nsys capture: skip the cudaProfilerStart/Stop bracket (nothing is
        # listening) but keep the module NVTX ranges (cheap no-ops off-profiler).
        uv run python apps/run_profiling.py \
            model.d_model="$d_model" \
            model.d_ff="$d_ff" \
            model.num_heads="$num_heads" \
            model.num_groups=null \
            model.num_transformer_layers="$num_transformer_layers" \
            profiling.nvtx.annotate_modules=true \
            profiling.nvtx.use_cudart_range=false \
            profiling.output_dir="$run_output_dir" \
            profiling.memory_profiling.enable=true \
            profiling.memory_profiling.output_path="$mem_profile_output_path" \
            profiling.forward_only="$forward_only" \
            amp.enable="$amp_enable" \
            amp.dtype="$amp_dtype"
    fi
}

echo "Starting profiling sweep..."

MODEL_SPECS=(
    "small 768 3072 12 12"
    "medium 1024 4096 16 24"
    "large 1280 5120 20 36"
    "xl 2560 10240 20 36"
    # "10b 4608 12288 36 50"
)

AMP_ENABLES=(true false)
FORWARD_ONLY_FLAGS=(true false)

for model_spec in "${MODEL_SPECS[@]}"; do
    read -r model_size d_model d_ff num_heads num_transformer_layers <<< "$model_spec"
    for amp_enable in "${AMP_ENABLES[@]}"; do
        for forward_only in "${FORWARD_ONLY_FLAGS[@]}"; do
            run_profiling \
                "$model_size" \
                "$d_model" \
                "$d_ff" \
                "$num_heads" \
                "$num_transformer_layers" \
                "$amp_enable" \
                "$forward_only" \
                "$AMP_DTYPE"
        done
    done
done

echo ""
echo "=========================================="
echo "Profiling sweep completed!"
echo "=========================================="
