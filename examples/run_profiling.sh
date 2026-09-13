#!/bin/bash

# Profiling sweep across different model sizes.
#
# Set USE_NSYS=1 (default) to capture an Nsight Systems report per model size,
# or USE_NSYS=0 to run the plain profiling script (per-stage timing table only,
# no nsys report). Example:
#   USE_NSYS=0 ./examples/run_profiling.sh
USE_NSYS=${USE_NSYS:-1}

run_profiling() {
    local model_size=$1
    local d_model=$2
    local d_ff=$3
    local num_heads=$4
    local num_transformer_layers=$5

    echo ""
    echo "=========================================="
    echo "Profiling: $model_size"
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
            --output=report_${model_size} \
            --force-overwrite=true \
            -- python apps/run_profiling.py \
            model.d_model=$d_model \
            model.d_ff=$d_ff \
            model.num_heads=$num_heads \
            model.num_groups=null \
            model.num_transformer_layers=$num_transformer_layers \
            profiling.nvtx.annotate_modules=true \
            profiling.nvtx.use_cudart_range=true
    else
        # No nsys capture: skip the cudaProfilerStart/Stop bracket (nothing is
        # listening) but keep the module NVTX ranges (cheap no-ops off-profiler).
        uv run python apps/run_profiling.py \
            model.d_model=$d_model \
            model.d_ff=$d_ff \
            model.num_heads=$num_heads \
            model.num_groups=null \
            model.num_transformer_layers=$num_transformer_layers \
            profiling.nvtx.annotate_modules=true \
            profiling.nvtx.use_cudart_range=false
    fi

    if [ $? -ne 0 ]; then
        echo "Error running profiling for $model_size"
        exit 1
    fi
}

echo "Starting profiling sweep..."

# small: d_model=768, d_ff=3072, num_heads=12, num_transformer_layers=12
run_profiling "small" 768 3072 12 12

# medium: d_model=1024, d_ff=4096, num_heads=16, num_transformer_layers=24
run_profiling "medium" 1024 4096 16 24

# large: d_model=1280, d_ff=5120, num_heads=20, num_transformer_layers=36
run_profiling "large" 1280 5120 20 36

# xl: d_model=2560, d_ff=10240, num_heads=20, num_transformer_layers=36
run_profiling "xl" 2560 10240 20 36

# 10b: d_model=4608, d_ff=12288, num_heads=36, num_transformer_layers=50
# run_profiling "10b" 4608 12288 36 50

echo ""
echo "=========================================="
echo "Profiling sweep completed!"
echo "=========================================="
