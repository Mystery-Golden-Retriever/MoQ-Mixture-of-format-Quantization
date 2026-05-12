#!/bin/bash
set -e

MODEL="Qwen/Qwen2.5-1.5B"

run_zs() {
    local OUT_DIR="$1"
    shift
    echo "Evaluating zero-shot for ${OUT_DIR}"
    /opt/homebrew/Caskroom/miniforge/base/envs/moq/bin/python scripts/run_evaluation.py \
        --model ${MODEL} \
        --output-dir "${OUT_DIR}" \
        --zero-shot hellaswag piqa \
        --batch-size 16 \
        "$@"
}

echo "Starting zero-shot re-evaluation for missing models..."

# Phase 1: Intermediate MSE
run_zs "results/Qwen2.5-1.5B_wt_imse_4bit_eval" --weight-format-map results/Qwen2.5-1.5B_wt_imse_4bit/weight_format_map.json

# Phase 1: Static
for FMT in int4 int4_aciq mxfp4 nf4 nvfp4; do
    run_zs "results/Qwen2.5-1.5B_wt_static_${FMT}_eval" --weight-format-map results/Qwen2.5-1.5B_wt_static_${FMT}/weight_format_map.json
done

# Phase 1: Cosine
for BITS in 4 8; do
    run_zs "results/Qwen2.5-1.5B_wt_cosine_${BITS}bit_eval" --weight-format-map results/Qwen2.5-1.5B_wt_cosine_${BITS}bit/weight_format_map.json
done

# Phase 2: E2E
for BITS in 4 6 8; do
    run_zs "results/Qwen2.5-1.5B_wt_e2e_${BITS}bit_eval" --weight-format-map results/Qwen2.5-1.5B_wt_e2e_${BITS}bit/weight_format_map.json
done

# Phase 3: Combined
run_zs "results/Qwen2.5-1.5B_combined_a8_w8_eval" \
    --format-map results/Qwen2.5-1.5B_moq_8bit/format_map.json \
    --weight-format-map results/Qwen2.5-1.5B_wt_imse_8bit/weight_format_map.json
run_zs "results/Qwen2.5-1.5B_combined_a8_w4_eval" \
    --format-map results/Qwen2.5-1.5B_moq_8bit/format_map.json \
    --weight-format-map results/Qwen2.5-1.5B_wt_imse_4bit/weight_format_map.json
run_zs "results/Qwen2.5-1.5B_combined_a6_w8_eval" \
    --format-map results/Qwen2.5-1.5B_moq_6bit/format_map.json \
    --weight-format-map results/Qwen2.5-1.5B_wt_imse_8bit/weight_format_map.json
run_zs "results/Qwen2.5-1.5B_combined_a6_w6_eval" \
    --format-map results/Qwen2.5-1.5B_moq_6bit/format_map.json \
    --weight-format-map results/Qwen2.5-1.5B_wt_imse_6bit/weight_format_map.json

# Finally collect results
/opt/homebrew/Caskroom/miniforge/base/envs/moq/bin/python scripts/collect_results.py

echo "Done re-evaluating zero-shot!"
