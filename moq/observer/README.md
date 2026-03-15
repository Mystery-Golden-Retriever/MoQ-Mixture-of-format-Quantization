# `moq/observer/` — Activation Statistics Observer

This module collects per-layer activation statistics during calibration, using memory-efficient online algorithms.

---

## Files

### `statistics_observer.py` — Welford Online Statistics

`StatisticsObserver` attaches forward hooks to target modules and maintains running statistics using **Welford's online algorithm** — O(1) memory regardless of how many batches are processed.

```python
from moq.observer.statistics_observer import StatisticsObserver

observer = StatisticsObserver(n_bins=256)
observer.attach(model, ["fc1", "fc2", "fc3"])

# Run calibration data through model
for batch in calibration_loader:
    model(batch)

# Retrieve statistics
stats = observer.get_stats()
print(stats["fc1"].mean, stats["fc1"].std)
print(stats["fc1"].min_val, stats["fc1"].max_val)
print(stats["fc1"].kurtosis)       # Distribution shape
print(stats["fc1"].histogram)      # 256-bin histogram

# Clean up
observer.detach()
```

**Collected Statistics (per layer):**

| Stat | Description |
|---|---|
| `min_val`, `max_val` | Global min/max across all batches |
| `mean` | Running mean (Welford) |
| `variance`, `std` | Running variance and standard deviation |
| `kurtosis` | Distribution peakedness (useful for format selection) |
| `histogram` | Fixed-bin histogram of activation values |
| `num_batches` | Number of batches observed |
| `num_elements` | Total number of tensor elements observed |

**Why Welford?** Naive mean/variance computation requires storing all values. Welford's algorithm computes exact statistics incrementally, which is essential when calibrating large models where activation tensors can be hundreds of MB per layer.

**Methods:**
- `attach(model, layer_names)` — Register hooks on named modules
- `detach()` — Remove all hooks
- `get_stats()` → `dict[str, LayerStats]`
- `get_layer_stats(name)` → `LayerStats`
- `reset()` — Clear all accumulated statistics
