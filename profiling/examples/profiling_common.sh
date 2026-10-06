#!/bin/bash

# Shared helpers for the profiling sweep examples. This file does not run a sweep.
#
# profile_module profiles the training step of apps/cfgs/training.yaml; each run
# passes training-config overrides (model.*, data.*, trainer.amp.*).

USE_NSYS=${USE_NSYS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-profiles}

mkdir -p "$OUTPUT_DIR"

run_profiling() {
    local run_name=$1
    shift 1
    local overrides=("$@")

    local run_output_dir="${OUTPUT_DIR}/${run_name}"
    # Nsight adds the .nsys-rep extension to this output prefix.
    local report_path="${run_output_dir}/${run_name}"

    mkdir -p "$run_output_dir"

    echo ""
    echo "=========================================="
    echo "Profiling: $run_name"
    echo "Output dir: $run_output_dir"
    echo "=========================================="

    if [ "$USE_NSYS" -eq 1 ]; then
        uv run nsys profile \
            --cuda-memory-usage=true \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --output="$report_path" \
            --force-overwrite=true \
            -- python -m profiling.profile_module \
            "${overrides[@]}" \
            profiling.nvtx.annotate_modules=true \
            profiling.nvtx.use_cudart_range=true \
            profiling.output_dir="$run_output_dir" \
            profiling.memory_profiling.enable=true
    else
        uv run python -m profiling.profile_module \
            "${overrides[@]}" \
            profiling.nvtx.annotate_modules=true \
            profiling.nvtx.use_cudart_range=false \
            profiling.output_dir="$run_output_dir" \
            profiling.memory_profiling.enable=true
    fi
}

# Runs one model spec once per AMP setting in AMP_ENABLES.
run_all_precisions() {
    local name=$1
    shift 1
    local overrides=("$@")

    for amp_enable in "${AMP_ENABLES[@]}"; do
        local precision_tag="fp32"
        if [ "$amp_enable" = "true" ]; then
            precision_tag="amp_bf16"
        fi
        run_profiling \
            "lm_${name}_${precision_tag}" \
            "${overrides[@]}" \
            trainer.amp.enable="$amp_enable"
    done
}
