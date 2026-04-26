#!/bin/bash
set -e

# Define ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  Starting Baseline Evaluation Pipeline (Qwen 1.5B)             ${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# Target configurations
MODEL="Qwen/Qwen2.5-1.5B"
MODEL_SHORT="Qwen2.5-1.5B"
BITS_LIST="6 8"
STRATEGY_LIST="static intermediate_mse"

echo -e "${GREEN}Processing Baseline for Model: ${MODEL}${NC}"

for STRATEGY in $STRATEGY_LIST; do
    echo -e "\n${BLUE}================================================================${NC}"
    echo -e "${BLUE}  Running Strategy: ${STRATEGY} ${NC}"
    echo -e "${BLUE}================================================================${NC}"
    
    for BITS in $BITS_LIST; do
        echo -e "\n${GREEN}>>> Evaluating ${MODEL_SHORT} with ${STRATEGY} for ${BITS}-bit${NC}"
        OUT_DIR="results/${MODEL_SHORT}_${STRATEGY}_${BITS}bit"

        if [ ! -f "${OUT_DIR}/format_map.json" ]; then
            echo ">> Running Calibration (${STRATEGY})..."
            python scripts/run_calibration.py \
                --config configs/qwen_moq.yaml \
                --model ${MODEL} \
                --bits ${BITS} \
                --strategy ${STRATEGY} \
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

echo -e "\n${GREEN}Baseline Pipeline completed successfully!${NC}"
