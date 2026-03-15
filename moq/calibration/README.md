# `moq/calibration/` — MoQ Calibration Engine

This module implements the core MoQ algorithm: **greedy per-layer format selection** using pluggable search strategies.

---

## Files

### `moq_calibrator.py` — Calibration Engine

`MoQCalibrator` orchestrates the per-layer format search:

1. Run calibration data through the model in **full precision** → record reference output
2. For each quantizable layer, invoke the search strategy to evaluate all candidates
3. Return `format_map: {layer_name → best_quantizer}`

```python
from moq.calibration import MoQCalibrator

# Auto-generate MoQ candidate set
candidates = MoQCalibrator.default_candidates(bits=8)
# → [INT8, INT8+ACIQ, FP(e1m6), FP(e1m6)+ACIQ, ..., FP(e6m1), FP(e6m1)+ACIQ]
#    = 14 candidates for 8-bit

calibrator = MoQCalibrator(model, candidates, strategy=MoQEndToEndStrategy())
format_map = calibrator.calibrate(calib_data, layer_names)
```

`default_candidates(bits)` generates the standard MoQ set:
- 2 INT variants (±ACIQ)
- 2×(bits-2) FP variants (all valid exponent splits, ±ACIQ)

### `search_strategies.py` — Pluggable Strategies

All strategies implement `BaseSearchStrategy.select_format()` → `(best_quantizer, score)`.

#### `MoQEndToEndStrategy` ⭐ (Paper's Gold Standard)

For each candidate, inject it at the target layer via a hook, run the calibration data through the **full model**, and measure **output MSE** against the full-precision reference.

This is the paper's core contribution: intermediate-layer MSE does **not** correlate with final accuracy, but end-to-end output MSE does.

```python
from moq.calibration.search_strategies import MoQEndToEndStrategy

strategy = MoQEndToEndStrategy()
```

#### `IntermediateMSEStrategy` (Baseline)

Measures quantization error **at the target layer** only (not propagated through the model). The MoQ paper demonstrates this is suboptimal — included for ablation comparison.

#### `CosineDistanceStrategy` (Baseline)

Selects the format that minimizes `1 - cosine_similarity` between quantized and full-precision activations at the target layer.

#### `StaticFormatStrategy` (Baseline)

Always assigns a fixed format to every layer — measures the cost of not adapting per-layer.

```python
from moq.calibration.search_strategies import StaticFormatStrategy
from moq.quantizers import INTQuantizer

strategy = StaticFormatStrategy(INTQuantizer(bits=8))
```

---

## Strategy Comparison (from MoQ paper)

| Strategy | Per-Layer? | End-to-End? | OHR |
|---|---|---|---|
| Static INT | ✗ | ✗ | Low |
| Static FP | ✗ | ✗ | Low |
| Intermediate MSE | ✓ | ✗ | Medium |
| Cosine Distance | ✓ | ✗ | Medium |
| **MoQ (End-to-End)** | **✓** | **✓** | **High** |
