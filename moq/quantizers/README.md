# `moq/quantizers/` — Core Quantization Engine

This module implements all quantization formats supported by MoQ. Each quantizer performs **fake quantization** — simulating reduced-precision arithmetic in FP32/BF16 to measure quantization error without actual bit-packing.

---

## Files

### `base.py` — Abstract Base Class

Defines the `BaseQuantizer` interface that all formats must implement:

- **`quantize(x)`** — Apply fake quantization to tensor `x`
- **`get_scale(x)`** — Compute the quantization scale factor
- **`get_config()`** — Serialize quantizer parameters to a dict
- **`forward(x)`** — `nn.Module`-compatible callable (delegates to `quantize`)

All quantizers inherit from `nn.Module` for seamless integration with PyTorch hooks and `torch.compile`.

**Design decision:** Inference-only — no STE gradients, no QAT backward pass.

### `int_quantizer.py` — Uniform Integer Quantizer

`INTQuantizer` implements standard uniform quantization:

```
x_q = round(clamp(x / scale, qmin, qmax)) × scale
```

| Parameter | Options |
|---|---|
| `bits` | 2–16 |
| `symmetric` | `True` (default) / `False` |
| `channel_wise` | Per-tensor (default) / per-channel |
| `use_aciq` | Optional ACIQ clipping |

### `fp_quantizer.py` — Floating-Point Quantizer

`FPQuantizer` simulates arbitrary floating-point formats by decomposing values into sign, exponent, and mantissa, then rounding the mantissa to the target precision:

```
x → (sign, exponent, mantissa) → round mantissa → reconstruct
```

Handles subnormal numbers, exponent overflow, and mantissa overflow correction.

**Convenience subclasses:**

| Class | Format | Exponent | Mantissa |
|---|---|---|---|
| `E4M3Quantizer` | FP8-E4M3 | 4 bits | 3 bits |
| `E5M2Quantizer` | FP8-E5M2 | 5 bits | 2 bits |
| `FP4E2M1Quantizer` | FP4-E2M1 | 2 bits | 1 bit |
| `FP4E3M0Quantizer` | FP4-E3M0 | 3 bits | 0 bits |

### `aciq.py` — Analytical Clipping (ACIQ)

`ACIQClipper` computes the optimal clipping threshold that minimizes MSE for uniform quantization, based on the analytical solution from the ACIQ paper:

```
clip = α(bits) × σ(x)
```

Supports Gaussian and Laplacian distribution priors. Pre-computed α coefficients for 2–8 bits with a fallback formula for higher bit widths.

### `registry.py` — Format Registry

Dynamic registry for quantizer formats:

```python
from moq.quantizers.registry import get_quantizer, list_quantizers

q = get_quantizer("fp8_e4m3")                    # Factory lookup
q = get_quantizer("int", bits=4, use_aciq=True)  # With kwargs
print(list_quantizers())                          # All registered names
```

New formats are registered automatically via the `@register_quantizer("name")` decorator.
