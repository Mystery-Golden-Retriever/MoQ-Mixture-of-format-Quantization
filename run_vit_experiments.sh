#!/bin/bash
set -e

# Define ANSI colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

MODEL="google/vit-base-patch16-224"
MODEL_SHORT="vit"
DEVICE="mps" # Change to 'cuda' if on NVIDIA

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  Starting Complete ViT Automated Pipeline (ImageNet)           ${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# Helper functions to avoid repeating identical commands
run_calibration() {
    local OUT_DIR=$1
    shift
    if [ ! -f "${OUT_DIR}/format_map.json" ] && [ ! -f "${OUT_DIR}/weight_format_map.json" ]; then
        echo -e "${GREEN}>> Running Calibration: ${OUT_DIR}${NC}"
        python scripts/run_calibration.py "$@"
    else
        echo -e "${YELLOW}>> Skipping Calibration (format_map already exists): ${OUT_DIR}${NC}"
    fi
}

run_evaluation() {
    local EVAL_DIR=$1
    shift
    if [ ! -f "${EVAL_DIR}/eval_results.json" ]; then
        echo -e "${GREEN}>> Running Evaluation: ${EVAL_DIR}${NC}"
        python scripts/run_evaluation.py "$@"
    else
        echo -e "${YELLOW}>> Skipping Evaluation (eval_results.json already exists): ${EVAL_DIR}${NC}"
    fi
}

# =====================================================================
# PHASE 1: Activation-Only Quantization (Baselines + MoQ)
# =====================================================================
echo -e "\n${BLUE}--- Phase 1: Activation-Only Quantization ---${NC}"

for BITS in 8 6 4 3 2; do
    # 1. MoQ E2E (Activation)
    OUT_DIR="results/${MODEL_SHORT}_moq_${BITS}bit"
    EVAL_DIR="${OUT_DIR}_eval"
    run_calibration "${OUT_DIR}" \
        --config configs/vit_moq.yaml \
        --model ${MODEL} \
        --bits ${BITS} \
        --strategy moq_end_to_end \
        --output-dir ${OUT_DIR} \
        --device ${DEVICE}
    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --format-map ${OUT_DIR}/format_map.json \
        --image-dataset imagenet \
        --batch-size 32 \
        --output-dir ${EVAL_DIR}

    # 2. Intermediate MSE (Activation)
    OUT_DIR="results/${MODEL_SHORT}_intermediate_mse_${BITS}bit"
    EVAL_DIR="${OUT_DIR}_eval"
    run_calibration "${OUT_DIR}" \
        --config configs/vit_moq.yaml \
        --model ${MODEL} \
        --bits ${BITS} \
        --strategy intermediate_mse \
        --output-dir ${OUT_DIR} \
        --device ${DEVICE}
    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --format-map ${OUT_DIR}/format_map.json \
        --image-dataset imagenet \
        --batch-size 32 \
        --output-dir ${EVAL_DIR}
done

# =====================================================================
# PHASE 2: Weight-Only Quantization (Baselines + MoQ)
# =====================================================================
echo -e "\n${BLUE}--- Phase 2: Weight-Only Quantization ---${NC}"

for BITS in 8 6 4; do
    # 1. MoQ E2E (Weight)
    OUT_DIR="results/${MODEL_SHORT}_wt_e2e_${BITS}bit"
    EVAL_DIR="${OUT_DIR}_eval"
    run_calibration "${OUT_DIR}" \
        --config configs/vit_moq.yaml \
        --model ${MODEL} \
        --weight-only \
        --weight-bits ${BITS} \
        --weight-strategy moq_end_to_end \
        --output-dir ${OUT_DIR} \
        --device ${DEVICE}
    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --weight-format-map ${OUT_DIR}/weight_format_map.json \
        --image-dataset imagenet \
        --batch-size 32 \
        --output-dir ${EVAL_DIR}

    # 2. Intermediate MSE (Weight)
    OUT_DIR="results/${MODEL_SHORT}_wt_imse_${BITS}bit"
    EVAL_DIR="${OUT_DIR}_eval"
    run_calibration "${OUT_DIR}" \
        --config configs/vit_moq.yaml \
        --model ${MODEL} \
        --weight-only \
        --weight-bits ${BITS} \
        --weight-strategy intermediate_mse \
        --output-dir ${OUT_DIR} \
        --device ${DEVICE}
    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --weight-format-map ${OUT_DIR}/weight_format_map.json \
        --image-dataset imagenet \
        --batch-size 32 \
        --output-dir ${EVAL_DIR}
done

# =====================================================================
# PHASE 3: Combined Activation + Weight Evaluation
# =====================================================================
echo -e "\n${BLUE}--- Phase 3: Combined Activation + Weight Evaluation ---${NC}"

for A_BITS in 8 6 4 3 2; do
    for W_BITS in 8 6 4; do
        EVAL_DIR="results/${MODEL_SHORT}_combined_a${A_BITS}_w${W_BITS}_eval"
        if [ -f "results/${MODEL_SHORT}_moq_${A_BITS}bit/format_map.json" ] && [ -f "results/${MODEL_SHORT}_wt_e2e_${W_BITS}bit/weight_format_map.json" ]; then
            run_evaluation "${EVAL_DIR}" \
                --model ${MODEL} \
                --format-map results/${MODEL_SHORT}_moq_${A_BITS}bit/format_map.json \
                --weight-format-map results/${MODEL_SHORT}_wt_e2e_${W_BITS}bit/weight_format_map.json \
                --image-dataset imagenet \
                --batch-size 32 \
                --output-dir ${EVAL_DIR}
        fi
    done
done

echo -e "\n${GREEN}All ViT experiments completed! Run 'python scripts/collect_results.py' to aggregate.${NC}"
