#!/bin/bash
set -e

# Define ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  Starting MoQ Automated Pipeline (Calibration + Evaluation)    ${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# Target bit configurations
BITS_LIST="2 3 4 6 8"

# -------------------------------------------------------------------------
# 1. Vision Model (ViT) Pipeline [ALREADY COMPLETED]
# -------------------------------------------------------------------------
# echo -e "${GREEN}[1/2] Processing Vision Model: google/vit-base-patch16-224${NC}"
# 
# for BITS in $BITS_LIST; do
#     echo -e "${BLUE}>>> Running ViT Pipeline for ${BITS}-bit${NC}"
#     OUT_DIR="results/vit_moq_${BITS}bit"
#     
#     echo ">> Running Calibration..."
#     python scripts/run_calibration.py \
#         --config configs/vit_moq.yaml \
#         --bits ${BITS} \
#         --output-dir ${OUT_DIR} \
#         --device auto
# 
#     echo ">> Running Evaluation (ImageNet Accuracy)..."
#     python scripts/run_evaluation.py \
#         --model google/vit-base-patch16-224 \
#         --format-map ${OUT_DIR}/format_map.json \
#         --image-dataset imagenet \
#         --batch-size 32 \
#         --compare-fp \
#         --output-dir ${OUT_DIR}_eval
# done
# echo -e "Vision Model Pipeline completed.\n"


# -------------------------------------------------------------------------
# 2. NLP Model (Llama-3) Pipeline  [DISABLED FOR NOW]
# -------------------------------------------------------------------------
# echo -e "${GREEN}[2/2] Processing NLP Model: meta-llama/Meta-Llama-3-8B${NC}"
# 
# for BITS in $BITS_LIST; do
#     echo -e "${BLUE}>>> Running Llama Pipeline for ${BITS}-bit${NC}"
#     OUT_DIR="results/llama_moq_${BITS}bit"
# 
#     echo ">> Running Calibration..."
#     python scripts/run_calibration.py \
#         --config configs/llama_moq.yaml \
#         --bits ${BITS} \
#         --output-dir ${OUT_DIR} \
#         --device auto
# 
#     echo ">> Running Evaluation (PPL + Zero-shot)..."
#     python scripts/run_evaluation.py \
#         --model meta-llama/Meta-Llama-3-8B \
#         --format-map ${OUT_DIR}/format_map.json \
#         --ppl-dataset wikitext2 \
#         --zero-shot hellaswag piqa \
#         --batch-size 16 \
#         --compare-fp \
#         --output-dir ${OUT_DIR}_eval
# done
# 
# echo -e "NLP Model Pipeline completed.\n"


# -------------------------------------------------------------------------
# 3. NLP Model (Qwen) Pipeline
# -------------------------------------------------------------------------
# Supported Qwen Models:
# - Qwen/Qwen2.5-0.5B
# - Qwen/Qwen2.5-1.5B
# - Qwen/Qwen2.5-3B
QWEN_MODELS="Qwen/Qwen2.5-1.5B"

echo -e "${GREEN}[3/3] Processing NLP Model (Qwen Series)${NC}"

for MODEL in $QWEN_MODELS; do
    # Extract the short name (e.g., Qwen2.5-1.5B) for folder naming
    MODEL_SHORT=$(echo $MODEL | cut -d '/' -f 2)
    
    for BITS in $BITS_LIST; do
        echo -e "${BLUE}>>> Running ${MODEL_SHORT} Pipeline for ${BITS}-bit${NC}"
        OUT_DIR="results/${MODEL_SHORT}_moq_${BITS}bit"

        if [ ! -f "${OUT_DIR}/format_map.json" ]; then
            echo ">> Running Calibration..."
            python scripts/run_calibration.py \
                --config configs/qwen_moq.yaml \
                --model ${MODEL} \
                --bits ${BITS} \
                --output-dir ${OUT_DIR} \
                --device auto
        else
            echo ">> Skipping Calibration for ${BITS}-bit (format_map.json already exists)"
        fi

        echo ">> Running Evaluation (PPL + Zero-shot)..."
        python scripts/run_evaluation.py \
            --model ${MODEL} \
            --format-map ${OUT_DIR}/format_map.json \
            --ppl-dataset wikitext2 \
            --zero-shot hellaswag piqa \
            --batch-size 16 \
            --compare-fp \
            --output-dir ${OUT_DIR}_eval
    done
done

echo -e "Qwen Model Pipeline completed.\n"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  All pipelines finished successfully!                          ${NC}"
echo -e "${BLUE}  Results are available in:                                     ${NC}"
echo -e "${BLUE}    - results/*_eval/eval_results.json                          ${NC}"
echo -e "${BLUE}================================================================${NC}"
