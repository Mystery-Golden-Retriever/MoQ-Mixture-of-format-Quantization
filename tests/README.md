# `tests/` — Test Suite

Comprehensive unit and integration tests for the MoQ framework. **89 tests** covering all modules.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single module
pytest tests/test_quantizers.py -v

# With coverage
pytest tests/ --cov=moq --cov-report=term-missing
```

---

## Test Files

### `test_quantizers.py` — Core Quantization Engine (40 tests)

| Test Class | Tests | Coverage |
|---|---|---|
| `TestINTQuantizer` | 10 | INT8/4/2 precision, ACIQ clipping, per-channel, symmetric range, config, determinism, large tensors, zeros |
| `TestFPQuantizer` | 13 | E4M3/E5M2/FP4 formats, precision comparison, sign preservation, zeros, generic FP sweep (all exp_bits combos), per-channel, config |
| `TestACIQ` | 7 | Gaussian/Laplacian priors, monotonic bit-clip relationship, all standard bits (2-8), fallback formula, repr |
| `TestRegistry` | 7 | Factory lookup (int, fp8_e4m3, fp8_e5m2, fp4 variants), list, unknown key error, all-formats forward pass |
| `TestBaseQuantizer` | 3 | Invalid bits validation, callable interface |

### `test_transform.py` — Transformation Backbone (19 tests)

| Test Class | Tests | Coverage |
|---|---|---|
| `TestHookInjector` | 8 | Inject/remove lifecycle, idempotent inject, context manager, exception cleanup, multiple layers, invalid name, input mode, repr |
| `TestModuleReplacer` | 6 | Replace + forward, restore, weight preservation, invalid type error, extra_repr, quantized output differs |
| `TestAdapters` | 5 | Llama/BERT/ViT patterns, pattern count validation, format string generation |

### `test_calibrator.py` — Calibration & Observer (15 tests)

| Test Class | Tests | Coverage |
|---|---|---|
| `TestStatisticsObserver` | 7 | Basic collection, multi-batch, reset, invalid layer, get_layer_stats, Welford accuracy validation, repr |
| `TestMoQCalibrator` | 4 | Format map generation, candidate counts (8-bit and 4-bit), higher-precision preference |
| `TestSearchStrategies` | 4 | Static, intermediate MSE, cosine distance, end-to-end (all 4 strategies) |
| `TestCalibrationIntegration` | 1 | Full pipeline: observe → calibrate → quantize → compare |

### `test_evaluation.py` — Evaluation Pipeline (15 tests)

| Test Class | Tests | Coverage |
|---|---|---|
| `TestOHRMetricEvaluator` | 9 | Perfect/zero/partial OHR, tolerance edge/miss cases, many layers, missing data, custom tolerance, exhaustive search |
| `TestPPLEvaluator` | 3 | Init params, invalid dataset error, dataset map keys |
| `TestZeroShotRunner` | 2 | Init params, default task list |
