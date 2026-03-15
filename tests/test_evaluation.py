"""Unit tests for the evaluation pipeline.

Covers:
  * OHRMetricEvaluator (perfect, zero, partial OHR, tolerance edge cases)
  * PPLEvaluator (API validates without requiring real models/datasets)
  * ZeroShotRunner (import validation)
"""

import pytest
import torch
import torch.nn as nn


# =====================================================================
# OHR Metric Evaluator
# =====================================================================


class TestOHRMetricEvaluator:
    """Tests for OHRMetricEvaluator."""

    def test_perfect_ohr(self):
        from moq.evaluation.ohr_evaluator import OHRMetricEvaluator

        evaluator = OHRMetricEvaluator(tolerance=0.01)

        predicted = {"layer_0": "int8", "layer_1": "fp8_e4m3"}
        accs = {
            "layer_0": {"int8": 0.95, "fp8_e4m3": 0.90},
            "layer_1": {"int8": 0.80, "fp8_e4m3": 0.93},
        }

        ohr = evaluator.compute_ohr(predicted, accs)
        assert ohr == 1.0

    def test_zero_ohr(self):
        from moq.evaluation.ohr_evaluator import OHRMetricEvaluator

        evaluator = OHRMetricEvaluator(tolerance=0.01)

        predicted = {"layer_0": "int4", "layer_1": "int4"}
        accs = {
            "layer_0": {"int4": 0.50, "int8": 0.95, "fp8": 0.90},
            "layer_1": {"int4": 0.40, "int8": 0.85, "fp8": 0.88},
        }

        ohr = evaluator.compute_ohr(predicted, accs)
        assert ohr == 0.0

    def test_partial_ohr(self):
        from moq.evaluation.ohr_evaluator import OHRMetricEvaluator

        evaluator = OHRMetricEvaluator(tolerance=0.01)

        predicted = {"layer_0": "int8", "layer_1": "int4"}
        accs = {
            "layer_0": {"int8": 0.95, "int4": 0.80},
            "layer_1": {"int8": 0.90, "int4": 0.50},
        }

        ohr = evaluator.compute_ohr(predicted, accs)
        assert ohr == 0.5

    def test_tolerance_edge(self):
        """A prediction within the tolerance margin should count as a hit."""
        from moq.evaluation.ohr_evaluator import OHRMetricEvaluator

        evaluator = OHRMetricEvaluator(tolerance=0.01)

        predicted = {"layer_0": "int8"}
        accs = {
            "layer_0": {
                "int8": 0.94,    # Within 1% of optimal (0.95)
                "fp8": 0.95,     # Optimal
            },
        }

        ohr = evaluator.compute_ohr(predicted, accs)
        assert ohr == 1.0  # 0.94 >= 0.95 - 0.01

    def test_tolerance_miss(self):
        """A prediction just outside the tolerance margin should miss."""
        from moq.evaluation.ohr_evaluator import OHRMetricEvaluator

        evaluator = OHRMetricEvaluator(tolerance=0.01)

        predicted = {"layer_0": "int8"}
        accs = {
            "layer_0": {
                "int8": 0.9399,   # Just outside 1% of optimal
                "fp8": 0.95,
            },
        }

        ohr = evaluator.compute_ohr(predicted, accs)
        assert ohr == 0.0

    def test_many_layers(self):
        """OHR across many layers."""
        from moq.evaluation.ohr_evaluator import OHRMetricEvaluator

        evaluator = OHRMetricEvaluator(tolerance=0.01)

        # 10 layers, 8 correct predictions
        predicted = {}
        accs = {}
        for i in range(10):
            layer = f"layer_{i}"
            predicted[layer] = "int8"
            if i < 8:
                # Hit: int8 is optimal
                accs[layer] = {"int8": 0.90, "fp8": 0.85}
            else:
                # Miss: fp8 is optimal and int8 is far off
                accs[layer] = {"int8": 0.70, "fp8": 0.90}

        ohr = evaluator.compute_ohr(predicted, accs)
        assert abs(ohr - 0.8) < 1e-6

    def test_missing_layer_skip(self):
        """Layers without accuracy data should be skipped gracefully."""
        from moq.evaluation.ohr_evaluator import OHRMetricEvaluator

        evaluator = OHRMetricEvaluator()
        predicted = {"layer_0": "int8", "layer_1": "int8"}
        accs = {
            "layer_0": {"int8": 0.95},
            # layer_1 missing from accs
        }

        ohr = evaluator.compute_ohr(predicted, accs)
        assert ohr == 1.0  # Only layer_0 counted

    def test_custom_tolerance(self):
        """Wider tolerance should produce higher OHR."""
        from moq.evaluation.ohr_evaluator import OHRMetricEvaluator

        predicted = {"layer_0": "int8"}
        accs = {"layer_0": {"int8": 0.85, "fp8": 0.95}}

        strict = OHRMetricEvaluator(tolerance=0.01)
        loose = OHRMetricEvaluator(tolerance=0.15)

        assert strict.compute_ohr(predicted, accs) == 0.0  # 0.85 < 0.95 - 0.01
        assert loose.compute_ohr(predicted, accs) == 1.0   # 0.85 >= 0.95 - 0.15

    def test_exhaustive_search(self):
        """Test the exhaustive_search method on a toy model."""
        from moq.evaluation.ohr_evaluator import OHRMetricEvaluator
        from moq.quantizers.int_quantizer import INTQuantizer

        class _TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 8)

            def forward(self, x):
                return self.fc1(x)

        model = _TinyModel()
        model.eval()
        evaluator = OHRMetricEvaluator()

        candidates = [INTQuantizer(bits=4), INTQuantizer(bits=8)]
        calib_data = [torch.randn(4, 16)]

        def eval_fn(m, data):
            with torch.no_grad():
                return -sum(m(b).pow(2).mean().item() for b in data)

        result = evaluator.exhaustive_search(
            model, ["fc1"], candidates, calib_data, eval_fn, show_progress=False
        )

        assert "fc1" in result
        assert len(result["fc1"]) == 2  # Two candidate formats


# =====================================================================
# PPL Evaluator
# =====================================================================


class TestPPLEvaluator:
    """Tests for PPLEvaluator (API-level only — no real datasets loaded)."""

    def test_init(self):
        from moq.evaluation.ppl_evaluator import PPLEvaluator

        model = nn.Linear(32, 10)  # Dummy
        evaluator = PPLEvaluator(model, tokenizer=None, seq_len=512, stride=256)
        assert evaluator.seq_len == 512
        assert evaluator.stride == 256

    def test_invalid_dataset(self):
        from moq.evaluation.ppl_evaluator import PPLEvaluator

        model = nn.Linear(32, 10)
        evaluator = PPLEvaluator(model, tokenizer=None)
        with pytest.raises(ValueError, match="Unknown dataset"):
            evaluator.evaluate("nonexistent_dataset")

    def test_dataset_map_keys(self):
        from moq.evaluation.ppl_evaluator import PPLEvaluator

        assert "wikitext2" in PPLEvaluator._DATASET_MAP
        assert "c4" in PPLEvaluator._DATASET_MAP
        assert "ptb" in PPLEvaluator._DATASET_MAP


# =====================================================================
# Zero-Shot Runner
# =====================================================================


class TestZeroShotRunner:
    """Tests for ZeroShotRunner (API-level only — no lm-eval required)."""

    def test_init(self):
        from moq.evaluation.zero_shot_runner import ZeroShotRunner

        model = nn.Linear(32, 10)
        runner = ZeroShotRunner(model, tokenizer=None, batch_size=16)
        assert runner.batch_size == 16

    def test_default_tasks(self):
        from moq.evaluation.zero_shot_runner import ZeroShotRunner

        assert "hellaswag" in ZeroShotRunner.DEFAULT_TASKS
        assert "piqa" in ZeroShotRunner.DEFAULT_TASKS
        assert len(ZeroShotRunner.DEFAULT_TASKS) == 6
