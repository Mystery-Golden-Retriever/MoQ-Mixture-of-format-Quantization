# `scripts/` — Entry-Point Scripts

CLI entry points for running MoQ calibration, evaluation, and paper reproduction experiments.

---

## Scripts

### `run_calibration.py` — End-to-End Calibration

Reads a YAML config, loads the model, runs MoQ calibration, and saves the resulting format map.

```bash
python scripts/run_calibration.py --config configs/llama_moq.yaml [--device auto]
```

**Pipeline:**
1. Load model + tokenizer from HuggingFace
2. Auto-detect quantizable layers via model adapter
3. Load calibration data (WikiText-2 by default)
4. Generate candidate format set
5. Run MoQ greedy calibration with selected strategy
6. Save `format_map.json` to the output directory
7. Optionally run PPL evaluation

**Output:** `results/*/format_map.json`

### `run_evaluation.py` — Evaluation-Only

Loads a pre-calibrated `format_map.json` and evaluates the quantized model.

```bash
python scripts/run_evaluation.py \
    --model meta-llama/Meta-Llama-3-8B \
    --format-map results/llama_moq/format_map.json \
    --ppl-dataset wikitext2 c4 \
    --zero-shot hellaswag piqa arc_easy \
    --compare-fp \
    --output-dir results/eval
```

**Features:**
- Multiple PPL benchmarks in one run
- Zero-shot tasks via `lm-eval`
- `--compare-fp` flag runs full-precision evaluation as baseline
- Saves results to `eval_results.json`

### `reproduce_paper.py` — Paper Reproduction (OHR)

Reproduces the core MoQ experiment: exhaustive search for ground truth, followed by format selection with all strategies at varying calibration sizes.

```bash
python scripts/reproduce_paper.py \
    --model google/vit-base-patch16-224 \
    --bits 4 8 \
    --calib-sizes 64 128 320 640 \
    --calib-dataset cifar10 \
    --max-layers 6 \
    --output-dir results/paper_reproduction
```

**Pipeline:**
1. Load model (auto-detects ViT/Llama/BERT adapter)
2. For each bit budget:
   - Run **exhaustive search** (all formats × all layers) → ground-truth accuracy table
   - Run 4 baseline strategies (Static INT, Static FP, Intermediate MSE, Cosine Distance)
   - Run MoQ end-to-end strategy with varying calibration sizes
   - Compute OHR for each strategy
3. Save results and print comparison table

**Flags:**
- `--max-layers N` — Limit layers for quick testing
- `--calib-dataset` — Override dataset (default: `cifar10` for vision, `wikitext2` for NLP)

**Output:** `results/paper_reproduction/paper_reproduction_results.json`
