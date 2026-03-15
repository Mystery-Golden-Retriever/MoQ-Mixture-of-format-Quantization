"""Unit tests for the transformation backbone.

Covers:
  * HookQuantInjector (injection, removal, context manager, input/output modes)
  * QuantizedLinear (drop-in replacement, weight + activation quantization)
  * ModelQuantizer (replace, restore, error handling)
  * Model adapters (Llama, BERT, ViT layer name enumeration)
"""

import pytest
import torch
import torch.nn as nn


# =====================================================================
# Shared fixtures
# =====================================================================


class _ToyModel(nn.Module):
    """Simple feed-forward model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class _NestedModel(nn.Module):
    """Model with nested sub-modules for dotted-name testing."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =====================================================================
# HookQuantInjector
# =====================================================================


class TestHookInjector:
    """Tests for HookQuantInjector."""

    def test_inject_and_remove(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.hook_injector import HookQuantInjector

        model = _ToyModel()
        q = INTQuantizer(bits=8)

        injector = HookQuantInjector(model, {"fc1": q})
        assert injector.num_hooks == 0

        injector.inject()
        assert injector.num_hooks == 1

        injector.remove()
        assert injector.num_hooks == 0

    def test_idempotent_inject(self):
        """Calling inject() twice should not double the hooks."""
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.hook_injector import HookQuantInjector

        model = _ToyModel()
        injector = HookQuantInjector(model, {"fc1": INTQuantizer(bits=8)})
        injector.inject()
        injector.inject()  # Second call should be no-op
        assert injector.num_hooks == 1
        injector.remove()

    def test_context_manager(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.hook_injector import HookQuantInjector

        model = _ToyModel()
        q = INTQuantizer(bits=4)
        x = torch.randn(2, 32)

        with torch.no_grad():
            out_fp = model(x)

        with HookQuantInjector(model, {"fc1": q}):
            with torch.no_grad():
                out_q = model(x)

        with torch.no_grad():
            out_fp2 = model(x)

        assert not torch.equal(out_fp, out_q), "Quantization should change output"
        assert torch.equal(out_fp, out_fp2), "Hooks should be removed after context exit"

    def test_context_manager_exception_cleanup(self):
        """Hooks should still be cleaned up if an exception occurs inside the context."""
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.hook_injector import HookQuantInjector

        model = _ToyModel()
        injector = HookQuantInjector(model, {"fc1": INTQuantizer(bits=8)})

        try:
            with injector:
                assert injector.num_hooks == 1
                raise ValueError("Simulated error")
        except ValueError:
            pass

        assert injector.num_hooks == 0, "Hooks should be cleaned up on exception"

    def test_multiple_layers(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.quantizers.fp_quantizer import E4M3Quantizer
        from moq.transform.hook_injector import HookQuantInjector

        model = _ToyModel()
        format_map = {
            "fc1": INTQuantizer(bits=8),
            "fc2": E4M3Quantizer(),
        }

        with HookQuantInjector(model, format_map) as inj:
            assert inj.num_hooks == 2
            x = torch.randn(2, 32)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, 10)

    def test_invalid_module_name(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.hook_injector import HookQuantInjector

        model = _ToyModel()
        injector = HookQuantInjector(model, {"nonexistent": INTQuantizer(bits=8)})

        with pytest.raises(KeyError, match="nonexistent"):
            injector.inject()

    def test_quantize_input_mode(self):
        """Test quantizing input activations instead of output."""
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.hook_injector import HookQuantInjector

        model = _ToyModel()
        q = INTQuantizer(bits=4)
        x = torch.randn(2, 32)

        with torch.no_grad():
            out_fp = model(x)

        with HookQuantInjector(model, {"fc2": q}, quantize_input=True):
            with torch.no_grad():
                out_q = model(x)

        assert not torch.equal(out_fp, out_q)

    def test_repr(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.hook_injector import HookQuantInjector

        model = _ToyModel()
        injector = HookQuantInjector(model, {"fc1": INTQuantizer(bits=8)})
        r = repr(injector)
        assert "HookQuantInjector" in r
        assert "layers=1" in r


# =====================================================================
# QuantizedLinear & ModelQuantizer
# =====================================================================


class TestModuleReplacer:
    """Tests for ModelQuantizer and QuantizedLinear."""

    def test_replace_and_forward(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.module_replacer import ModelQuantizer, QuantizedLinear

        model = _ToyModel()
        replacer = ModelQuantizer()

        format_map = {
            "fc1": (INTQuantizer(bits=8), INTQuantizer(bits=8)),
            "fc2": (INTQuantizer(bits=8), None),
        }
        model = replacer.replace(model, format_map)

        assert isinstance(model.fc1, QuantizedLinear)
        assert isinstance(model.fc2, QuantizedLinear)
        assert model.fc1.act_quantizer is not None
        assert model.fc1.weight_quantizer is not None
        assert model.fc2.weight_quantizer is None

        x = torch.randn(2, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_restore(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.module_replacer import ModelQuantizer

        model = _ToyModel()
        replacer = ModelQuantizer()

        format_map = {"fc1": (INTQuantizer(bits=8), None)}
        model = replacer.replace(model, format_map)
        model = replacer.restore(model)

        assert isinstance(model.fc1, nn.Linear)
        assert model.fc1.in_features == 32
        assert model.fc1.out_features == 64

    def test_weights_preserved(self):
        """QuantizedLinear should preserve the original weight values."""
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.module_replacer import ModelQuantizer, QuantizedLinear

        model = _ToyModel()
        original_weight = model.fc1.weight.data.clone()

        replacer = ModelQuantizer()
        format_map = {"fc1": (None, None)}  # No quantization, just replacement
        model = replacer.replace(model, format_map)

        assert isinstance(model.fc1, QuantizedLinear)
        assert torch.equal(model.fc1.weight.data, original_weight)

    def test_replace_invalid_type(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.module_replacer import ModelQuantizer

        model = _ToyModel()
        replacer = ModelQuantizer()

        # Try to replace ReLU (not a target type)
        format_map = {"relu": (INTQuantizer(bits=8), None)}
        with pytest.raises(TypeError, match="ReLU"):
            replacer.replace(model, format_map)

    def test_quantized_linear_extra_repr(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.module_replacer import QuantizedLinear

        original = nn.Linear(32, 64)
        q = QuantizedLinear(original, weight_quantizer=INTQuantizer(bits=4))
        r = q.extra_repr()
        assert "in_features=32" in r
        assert "weight_q=" in r

    def test_quantized_output_differs(self):
        """Module replacement with quantization should change model output."""
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.module_replacer import ModelQuantizer

        model = _ToyModel()
        model.eval()
        x = torch.randn(2, 32)

        with torch.no_grad():
            out_fp = model(x)

        replacer = ModelQuantizer()
        format_map = {"fc1": (INTQuantizer(bits=4), INTQuantizer(bits=4))}
        model = replacer.replace(model, format_map)

        with torch.no_grad():
            out_q = model(x)

        assert not torch.equal(out_fp, out_q)


# =====================================================================
# Model Adapters
# =====================================================================


class TestAdapters:
    """Tests for model adapter layer-name enumeration.

    These tests verify the naming patterns without loading real models.
    """

    def test_llama_patterns(self):
        from moq.transform.adapters.llama_adapter import LlamaQuantAdapter

        patterns = LlamaQuantAdapter.ALL_PATTERNS
        assert any("q_proj" in p for p in patterns)
        assert any("k_proj" in p for p in patterns)
        assert any("v_proj" in p for p in patterns)
        assert any("o_proj" in p for p in patterns)
        assert any("gate_proj" in p for p in patterns)
        assert any("up_proj" in p for p in patterns)
        assert any("down_proj" in p for p in patterns)
        assert len(patterns) == 7  # 4 attn + 3 mlp

    def test_bert_patterns(self):
        from moq.transform.adapters.bert_adapter import BERTQuantAdapter

        patterns = BERTQuantAdapter.ALL_PATTERNS
        assert any("query" in p for p in patterns)
        assert any("key" in p for p in patterns)
        assert any("value" in p for p in patterns)
        assert any("intermediate.dense" in p for p in patterns)
        assert len(patterns) == 6  # 4 attn + 2 ffn

    def test_vit_patterns(self):
        from moq.transform.adapters.vit_adapter import ViTQuantAdapter

        patterns = ViTQuantAdapter.ALL_PATTERNS
        assert any("attention.attention.query" in p for p in patterns)
        assert any("intermediate.dense" in p for p in patterns)
        assert len(patterns) == 6  # 4 attn + 2 mlp

    def test_llama_format_string(self):
        """Verify that pattern formatting produces correct names."""
        from moq.transform.adapters.llama_adapter import LlamaQuantAdapter

        name = LlamaQuantAdapter.ATTN_PROJ_PATTERNS[0].format(i=5)
        assert name == "model.layers.5.self_attn.q_proj"

    def test_bert_format_string(self):
        from moq.transform.adapters.bert_adapter import BERTQuantAdapter

        name = BERTQuantAdapter.ATTN_PATTERNS[0].format(i=3)
        assert name == "bert.encoder.layer.3.attention.self.query"
