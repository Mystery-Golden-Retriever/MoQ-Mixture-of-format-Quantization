# `moq/transform/` — Transformation Backbone

This module handles injecting quantization into a model's computation graph. It provides two complementary approaches, plus architecture-specific adapters for layer name resolution.

---

## Files

### `hook_injector.py` — Non-Destructive Hook Injection

`HookQuantInjector` attaches PyTorch `forward_hook` / `forward_pre_hook` to target modules. The original model is **unchanged** — hooks can be added and removed freely.

```python
from moq.transform.hook_injector import HookQuantInjector

# Context manager (recommended — auto-cleanup on exit or exception)
with HookQuantInjector(model, format_map) as injector:
    output = model(input_ids)
# Hooks automatically removed here
```

**Features:**
- Context-manager protocol with exception-safe cleanup
- Quantize **output** (default) or **input** activations
- Handles tuple outputs (e.g., attention returns `(attn_out, weights)`)
- Idempotent `inject()` / `remove()`

**Best for:** Calibration, A/B evaluation, temporary quantization.

**Limitation:** `torch.compile(fullgraph=True)` may not inline hooks. Use module replacement for compiled inference.

### `module_replacer.py` — Module Replacement

`ModelQuantizer` replaces `nn.Linear` modules with `QuantizedLinear` wrappers that apply quantization inside the forward pass. This modifies the model's module tree, making it compatible with `torch.compile`.

```python
from moq.transform.module_replacer import ModelQuantizer

replacer = ModelQuantizer()

# Replace: format_map values are (act_quantizer, weight_quantizer) tuples
format_map = {
    "fc1": (INTQuantizer(bits=8), INTQuantizer(bits=8)),  # Both act & weight
    "fc2": (INTQuantizer(bits=4), None),                  # Act only
}
model = replacer.replace(model, format_map)

# Restore original modules
model = replacer.restore(model)
```

**Best for:** Production deployment, `torch.compile`-compatible inference.

---

## `adapters/` — Architecture-Specific Adapters

Adapters enumerate the quantizable `nn.Linear` layers in a specific model architecture by mapping HuggingFace module naming conventions.

### `llama_adapter.py` — Llama / Llama-2 / Llama-3

```python
from moq.transform.adapters import LlamaQuantAdapter

layer_names = LlamaQuantAdapter.get_layer_names(model)
# → ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", ...]
```

**Layers per block (7):** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### `bert_adapter.py` — BERT / RoBERTa

```python
from moq.transform.adapters import BERTQuantAdapter

layer_names = BERTQuantAdapter.get_layer_names(model)
```

**Layers per block (6):** `query`, `key`, `value`, `attention.output.dense`, `intermediate.dense`, `output.dense`

### `vit_adapter.py` — Vision Transformer (ViT)

```python
from moq.transform.adapters import ViTQuantAdapter

layer_names = ViTQuantAdapter.get_layer_names(model)
```

**Layers per block (6):** `attention.attention.{query,key,value}`, `attention.output.dense`, `intermediate.dense`, `output.dense`

All adapters support `include_attn` and `include_mlp` flags for selective quantization, and `get_layer_groups()` for per-block grouping.
