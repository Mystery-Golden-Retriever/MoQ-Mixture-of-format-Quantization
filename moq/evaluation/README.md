# `moq/evaluation/` — Evaluation Pipeline

This module provides standardized evaluation metrics for measuring quantized model quality.

---

## Files

### `ppl_evaluator.py` — Perplexity Evaluator

`PPLEvaluator` computes perplexity on standard language modeling benchmarks using a sliding-window approach.

```python
from moq.evaluation import PPLEvaluator

evaluator = PPLEvaluator(model, tokenizer, seq_len=2048)
ppl = evaluator.evaluate("wikitext2")
print(f"Perplexity: {ppl:.2f}")
```

**Supported datasets:**

| Dataset | Config |
|---|---|
| WikiText-2 | `"wikitext2"` |
| C4 (validation) | `"c4"` |
| Penn Treebank | `"ptb"` |

**Parameters:**
- `seq_len` — Context window length (default 2048)
- `stride` — Sliding window stride (`None` = non-overlapping, matching quantization paper conventions)
- `max_samples` — Limit dataset size for quick testing

Also supports `evaluate_from_text(text)` for custom datasets.

### `zero_shot_runner.py` — Zero-Shot Evaluation

`ZeroShotRunner` wraps the `lm-evaluation-harness` library for standardized zero-shot benchmarks.

```python
from moq.evaluation import ZeroShotRunner

runner = ZeroShotRunner(model, tokenizer, batch_size=32)
results = runner.run(["hellaswag", "piqa", "arc_easy"])
# → {"hellaswag": 0.78, "piqa": 0.81, "arc_easy": 0.72}
```

**Default tasks:** HellaSwag, WinoGrande, PIQA, ARC-Easy, ARC-Challenge, LAMBADA

**Requires:** `pip install moq[eval]` (installs `lm-eval`)

### `ohr_evaluator.py` — Optimum Hit Rate (OHR) Metric

`OHRMetricEvaluator` implements the paper's OHR metric for evaluating format selection quality.

```python
from moq.evaluation import OHRMetricEvaluator

evaluator = OHRMetricEvaluator(tolerance=0.01)

# From pre-computed accuracy tables
ohr = evaluator.compute_ohr(predicted_formats, layer_accuracies)

# Or run exhaustive search to build the ground-truth table
gt_accs = evaluator.exhaustive_search(
    model, layer_names, candidates, calib_data, eval_fn
)
```

**`compute_ohr(predicted, accuracies)`** — Compute OHR from pre-built accuracy tables

**`exhaustive_search(model, layers, candidates, data, eval_fn)`** — Run every format at every layer independently to build the ground-truth accuracy table. This is expensive (O(layers × formats) forward passes) but necessary for faithful OHR evaluation.
