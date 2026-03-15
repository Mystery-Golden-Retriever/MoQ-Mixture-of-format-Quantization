# `moq/baselines/` — Baseline Comparators

This module provides wrappers for third-party quantization methods so they can be benchmarked against MoQ under the same evaluation pipeline.

---

## Files

### `baseline_evaluator.py`

Defines the `BaselineEvaluator` abstract interface and three concrete integrations.

#### `BaselineEvaluator` (ABC)

```python
class BaselineEvaluator(ABC):
    def get_name(self) -> str: ...
    def quantize_model(self, model, calib_data) -> nn.Module: ...
```

All baselines follow the same API: take a full-precision model and calibration data, return a quantized model ready for evaluation.

#### `GPTQBaseline`

Wraps [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) — weight-only quantization via second-order approximation.

```python
from moq.baselines.baseline_evaluator import GPTQBaseline

baseline = GPTQBaseline(bits=4, group_size=128)
quantized_model = baseline.quantize_model(model, calib_data)
```

**Requires:** `pip install moq[baselines]`

#### `AWQBaseline`

Wraps [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) — activation-aware weight quantization.

```python
baseline = AWQBaseline(bits=4, group_size=128)
```

**Requires:** `pip install moq[baselines]`

#### `SmoothQuantBaseline`

Lightweight re-implementation of the SmoothQuant algorithm — migrates quantization difficulty from activations to weights:

```
W' = W · diag(s),  X' = X · diag(1/s)
s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
```

```python
baseline = SmoothQuantBaseline(bits=8, alpha=0.5)
```

This baseline uses MoQ's own `INTQuantizer` internally, so no extra dependencies are needed.
