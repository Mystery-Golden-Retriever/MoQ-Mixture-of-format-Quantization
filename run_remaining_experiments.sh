#!/bin/bash
set -e

# =====================================================================
# MoQ Weight Quantization Experiments
# =====================================================================
# Follows the approved experiment plan:
#   Phase 1: Quick wins (IntermediateMSE weight-only + format deep-dive)
#   Phase 2: Full MoQ E2E weight calibration
#   Phase 3: Combined activation + weight evaluation
# =====================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODEL="Qwen/Qwen2.5-1.5B"
MODEL_SHORT="Qwen2.5-1.5B"
DEVICE="auto"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  MoQ Weight Quantization Experiment Pipeline                    ${NC}"
echo -e "${BLUE}  Model: ${MODEL}                                               ${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# Helper: skip if format map already exists
run_calibration() {
    local OUT_DIR="$1"
    shift
    if [ -f "${OUT_DIR}/weight_format_map.json" ]; then
        echo -e "${YELLOW}>> Skipping (weight_format_map.json exists): ${OUT_DIR}${NC}"
        return 0
    fi
    echo -e "${GREEN}>> Running: $@${NC}"
    python scripts/run_calibration.py "$@"
}

run_evaluation() {
    local OUT_DIR="$1"
    shift
    if [ -f "${OUT_DIR}/eval_results.json" ]; then
        echo -e "${YELLOW}>> Skipping (eval_results.json exists): ${OUT_DIR}${NC}"
        return 0
    fi
    echo -e "${GREEN}>> Running evaluation: $@${NC}"
    python scripts/run_evaluation.py "$@"
}

# =====================================================================
# PHASE 1: Quick Wins (~40 min)
# =====================================================================
echo -e "\n${BLUE}================================================================${NC}"
echo -e "${BLUE}  PHASE 1: Quick Wins (IntermediateMSE + Format Deep-Dive)       ${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# ----- Group A: Weight-Only, IntermediateMSE (A1, A3, A5) -----
echo -e "${GREEN}--- Group A: Weight-Only IntermediateMSE ---${NC}"

for BITS in 8 6 4; do
    OUT_DIR="results/${MODEL_SHORT}_wt_imse_${BITS}bit"
    EVAL_DIR="${OUT_DIR}_eval"

    run_calibration "${OUT_DIR}" \
        --config configs/qwen_moq.yaml \
        --model ${MODEL} \
        --weight-only \
        --weight-bits ${BITS} \
        --weight-strategy intermediate_mse \
        --output-dir ${OUT_DIR} \
        --device ${DEVICE}

    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --weight-format-map ${OUT_DIR}/weight_format_map.json \
        --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
        --batch-size 16 \
        --compare-fp \
        --output-dir ${EVAL_DIR}
done

# ----- Group D: 4-bit Format Deep-Dive (D1-D5) -----
echo -e "\n${GREEN}--- Group D: 4-bit Format Deep-Dive (Static) ---${NC}"

FORMATS_4BIT="int4 int4_aciq mxfp4 nvfp4 nf4"
FORMAT_NAMES=("int4" "int4_aciq" "mxfp4" "nvfp4" "nf4")
STATIC_ARGS=("--weight-static-format int4"
              "--weight-static-format int4"
              "--weight-static-format mxfp4"
              "--weight-static-format nvfp4"
              "--weight-static-format nf4")

# D1: INT4 static
OUT_DIR="results/${MODEL_SHORT}_wt_static_int4"
EVAL_DIR="${OUT_DIR}_eval"
run_calibration "${OUT_DIR}" \
    --config configs/qwen_moq.yaml \
    --model ${MODEL} \
    --weight-only \
    --weight-bits 4 \
    --weight-strategy static \
    --weight-static-format int4 \
    --output-dir ${OUT_DIR} \
    --device ${DEVICE}
run_evaluation "${EVAL_DIR}" \
    --model ${MODEL} \
    --weight-format-map ${OUT_DIR}/weight_format_map.json \
    --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
    --batch-size 16 \
    --output-dir ${EVAL_DIR}

# D2: INT4+ACIQ static
OUT_DIR="results/${MODEL_SHORT}_wt_static_int4_aciq"
EVAL_DIR="${OUT_DIR}_eval"
run_calibration "${OUT_DIR}" \
    --config configs/qwen_moq.yaml \
    --model ${MODEL} \
    --weight-only \
    --weight-bits 4 \
    --weight-strategy static \
    --weight-static-format int4_aciq \
    --output-dir ${OUT_DIR} \
    --device ${DEVICE}
run_evaluation "${EVAL_DIR}" \
    --model ${MODEL} \
    --weight-format-map ${OUT_DIR}/weight_format_map.json \
    --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
    --batch-size 16 \
    --output-dir ${EVAL_DIR}

# D3: MXFP4 static
OUT_DIR="results/${MODEL_SHORT}_wt_static_mxfp4"
EVAL_DIR="${OUT_DIR}_eval"
run_calibration "${OUT_DIR}" \
    --config configs/qwen_moq.yaml \
    --model ${MODEL} \
    --weight-only \
    --weight-bits 4 \
    --weight-strategy static \
    --weight-static-format mxfp4 \
    --output-dir ${OUT_DIR} \
    --device ${DEVICE}
run_evaluation "${EVAL_DIR}" \
    --model ${MODEL} \
    --weight-format-map ${OUT_DIR}/weight_format_map.json \
    --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
    --batch-size 16 \
    --output-dir ${EVAL_DIR}

# D4: NVFP4 static
OUT_DIR="results/${MODEL_SHORT}_wt_static_nvfp4"
EVAL_DIR="${OUT_DIR}_eval"
run_calibration "${OUT_DIR}" \
    --config configs/qwen_moq.yaml \
    --model ${MODEL} \
    --weight-only \
    --weight-bits 4 \
    --weight-strategy static \
    --weight-static-format nvfp4 \
    --output-dir ${OUT_DIR} \
    --device ${DEVICE}
run_evaluation "${EVAL_DIR}" \
    --model ${MODEL} \
    --weight-format-map ${OUT_DIR}/weight_format_map.json \
    --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
    --batch-size 16 \
    --output-dir ${EVAL_DIR}

# D5: NF4 static
OUT_DIR="results/${MODEL_SHORT}_wt_static_nf4"
EVAL_DIR="${OUT_DIR}_eval"
run_calibration "${OUT_DIR}" \
    --config configs/qwen_moq.yaml \
    --model ${MODEL} \
    --weight-only \
    --weight-bits 4 \
    --weight-strategy static \
    --weight-static-format nf4 \
    --output-dir ${OUT_DIR} \
    --device ${DEVICE}
run_evaluation "${EVAL_DIR}" \
    --model ${MODEL} \
    --weight-format-map ${OUT_DIR}/weight_format_map.json \
    --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
    --batch-size 16 \
    --output-dir ${EVAL_DIR}

# ----- Group C: Cosine ablation (C3, C6) -----
echo -e "\n${GREEN}--- Group C: Cosine Strategy Ablation ---${NC}"

for BITS in 8 4; do
    OUT_DIR="results/${MODEL_SHORT}_wt_cosine_${BITS}bit"
    EVAL_DIR="${OUT_DIR}_eval"

    run_calibration "${OUT_DIR}" \
        --config configs/qwen_moq.yaml \
        --model ${MODEL} \
        --weight-only \
        --weight-bits ${BITS} \
        --weight-strategy cosine \
        --output-dir ${OUT_DIR} \
        --device ${DEVICE}

    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --weight-format-map ${OUT_DIR}/weight_format_map.json \
        --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
        --batch-size 16 \
        --output-dir ${EVAL_DIR}
done


# =====================================================================
# PHASE 2: Full MoQ E2E Weight Calibration (~2 hours)
# =====================================================================
echo -e "\n${BLUE}================================================================${NC}"
echo -e "${BLUE}  PHASE 2: MoQ E2E Weight Calibration                           ${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# Group A: Weight-Only, MoQ E2E (A2, A4, A6)
for BITS in 8 6 4; do
    OUT_DIR="results/${MODEL_SHORT}_wt_e2e_${BITS}bit"
    EVAL_DIR="${OUT_DIR}_eval"

    run_calibration "${OUT_DIR}" \
        --config configs/qwen_moq.yaml \
        --model ${MODEL} \
        --weight-only \
        --weight-bits ${BITS} \
        --weight-strategy moq_end_to_end \
        --output-dir ${OUT_DIR} \
        --device ${DEVICE}

    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --weight-format-map ${OUT_DIR}/weight_format_map.json \
        --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
        --batch-size 16 \
        --output-dir ${EVAL_DIR}
done


# =====================================================================
# PHASE 3: Combined Activation + Weight Evaluation (~40 min)
# =====================================================================
echo -e "\n${BLUE}================================================================${NC}"
echo -e "${BLUE}  PHASE 3: Combined Activation + Weight Evaluation               ${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# Uses EXISTING activation format maps + BEST weight format maps
# Best weight strategy TBD after Phase 1/2 — default to IntermediateMSE

# B1: Act=8bit MoQ E2E + Wt=8bit
echo -e "${GREEN}--- B1: Act 8-bit + Wt 8-bit ---${NC}"
EVAL_DIR="results/${MODEL_SHORT}_combined_a8_w8_eval"
if [ -f "results/${MODEL_SHORT}_moq_8bit/format_map.json" ] && [ -f "results/${MODEL_SHORT}_wt_imse_8bit/weight_format_map.json" ]; then
    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --format-map results/${MODEL_SHORT}_moq_8bit/format_map.json \
        --weight-format-map results/${MODEL_SHORT}_wt_imse_8bit/weight_format_map.json \
        --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
        --batch-size 16 \
        --output-dir ${EVAL_DIR}
else
    echo -e "${YELLOW}>> Skipping B1: missing prerequisite format maps${NC}"
fi

# B2: Act=8bit MoQ E2E + Wt=4bit
echo -e "${GREEN}--- B2: Act 8-bit + Wt 4-bit ---${NC}"
EVAL_DIR="results/${MODEL_SHORT}_combined_a8_w4_eval"
if [ -f "results/${MODEL_SHORT}_moq_8bit/format_map.json" ] && [ -f "results/${MODEL_SHORT}_wt_imse_4bit/weight_format_map.json" ]; then
    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --format-map results/${MODEL_SHORT}_moq_8bit/format_map.json \
        --weight-format-map results/${MODEL_SHORT}_wt_imse_4bit/weight_format_map.json \
        --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
        --batch-size 16 \
        --output-dir ${EVAL_DIR}
else
    echo -e "${YELLOW}>> Skipping B2: missing prerequisite format maps${NC}"
fi

# B3: Act=6bit MoQ E2E + Wt=8bit
echo -e "${GREEN}--- B3: Act 6-bit + Wt 8-bit ---${NC}"
EVAL_DIR="results/${MODEL_SHORT}_combined_a6_w8_eval"
if [ -f "results/${MODEL_SHORT}_moq_6bit/format_map.json" ] && [ -f "results/${MODEL_SHORT}_wt_imse_8bit/weight_format_map.json" ]; then
    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --format-map results/${MODEL_SHORT}_moq_6bit/format_map.json \
        --weight-format-map results/${MODEL_SHORT}_wt_imse_8bit/weight_format_map.json \
        --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
        --batch-size 16 \
        --output-dir ${EVAL_DIR}
else
    echo -e "${YELLOW}>> Skipping B3: missing prerequisite format maps${NC}"
fi

# B4: Act=6bit MoQ E2E + Wt=6bit
echo -e "${GREEN}--- B4: Act 6-bit + Wt 6-bit ---${NC}"
EVAL_DIR="results/${MODEL_SHORT}_combined_a6_w6_eval"
if [ -f "results/${MODEL_SHORT}_moq_6bit/format_map.json" ] && [ -f "results/${MODEL_SHORT}_wt_imse_6bit/weight_format_map.json" ]; then
    run_evaluation "${EVAL_DIR}" \
        --model ${MODEL} \
        --format-map results/${MODEL_SHORT}_moq_6bit/format_map.json \
        --weight-format-map results/${MODEL_SHORT}_wt_imse_6bit/weight_format_map.json \
        --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa \
        --batch-size 16 \
        --output-dir ${EVAL_DIR}
else
    echo -e "${YELLOW}>> Skipping B4: missing prerequisite format maps${NC}"
fi


# =====================================================================
# Summary
# =====================================================================
echo -e "\n${BLUE}================================================================${NC}"
echo -e "${BLUE}  All weight quantization experiments completed!                 ${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Results directories:${NC}"
echo "  Phase 1 (Quick):    results/${MODEL_SHORT}_wt_imse_*"
echo "  Phase 1 (Deep):     results/${MODEL_SHORT}_wt_static_*"
echo "  Phase 1 (Cosine):   results/${MODEL_SHORT}_wt_cosine_*"
echo "  Phase 2 (E2E):      results/${MODEL_SHORT}_wt_e2e_*"
echo "  Phase 3 (Combined): results/${MODEL_SHORT}_combined_*"
echo ""
echo "To collect all results into a table, run:"
echo "  python scripts/collect_results.py"
