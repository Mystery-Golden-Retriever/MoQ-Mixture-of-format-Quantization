"""Unit tests for the calibration engine and search strategies.

Covers:
  * StatisticsObserver (Welford online stats, histogram, reset)
  * MoQCalibrator (format selection, default candidates)
  * MoQEndToEndStrategy (validate output MSE minimization)
  * IntermediateMSEStrategy (validate local-layer metric)
  * StaticFormatStrategy (always returns fixed format)
  * End-to-end calibration pipeline on a toy model
"""

import pytest
import torch
import torch.nn as nn


# =====================================================================
# Shared fixtures
# =====================================================================


class _ToyModel(nn.Module):
    """Simple model for calibration testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# =====================================================================
# Statistics Observer
# =====================================================================


class TestStatisticsObserver:
    """Tests for StatisticsObserver."""

    def test_basic_collection(self):
        from moq.observer.statistics_observer import StatisticsObserver

        model = _ToyModel()
        observer = StatisticsObserver()
        observer.attach(model, ["fc1", "fc2"])

        x = torch.randn(8, 32)
        with torch.no_grad():
            model(x)

        stats = observer.get_stats()
        assert "fc1" in stats
        assert "fc2" in stats
        assert stats["fc1"].num_batches == 1
        assert stats["fc1"].num_elements > 0
        assert stats["fc1"].min_val < stats["fc1"].max_val
        assert stats["fc1"].std > 0

        observer.detach()

    def test_multiple_batches(self):
        from moq.observer.statistics_observer import StatisticsObserver

        model = _ToyModel()
        observer = StatisticsObserver()
        observer.attach(model, ["fc1"])

        for _ in range(5):
            x = torch.randn(4, 32)
            with torch.no_grad():
                model(x)

        stats = observer.get_stats()
        assert stats["fc1"].num_batches == 5
        assert stats["fc1"].histogram is not None
        assert stats["fc1"].histogram.sum() > 0

        observer.detach()

    def test_reset(self):
        from moq.observer.statistics_observer import StatisticsObserver

        model = _ToyModel()
        observer = StatisticsObserver()
        observer.attach(model, ["fc1"])

        with torch.no_grad():
            model(torch.randn(4, 32))

        observer.reset()
        stats = observer.get_stats()
        assert stats["fc1"].num_batches == 0
        assert stats["fc1"].num_elements == 0

        observer.detach()

    def test_invalid_layer(self):
        from moq.observer.statistics_observer import StatisticsObserver

        model = _ToyModel()
        observer = StatisticsObserver()
        with pytest.raises(KeyError, match="nonexistent"):
            observer.attach(model, ["nonexistent"])

    def test_get_layer_stats(self):
        from moq.observer.statistics_observer import StatisticsObserver

        model = _ToyModel()
        observer = StatisticsObserver()
        observer.attach(model, ["fc1"])

        with torch.no_grad():
            model(torch.randn(4, 32))

        s = observer.get_layer_stats("fc1")
        assert s.num_batches == 1
        assert s.mean != 0.0 or s.std != 0.0  # Not all zeros

        with pytest.raises(KeyError):
            observer.get_layer_stats("nonexistent")

        observer.detach()

    def test_welford_accuracy(self):
        """Welford mean/std should be close to batch statistics."""
        from moq.observer.statistics_observer import StatisticsObserver

        model = _ToyModel()
        observer = StatisticsObserver()
        observer.attach(model, ["fc1"])

        # Run many batches
        all_outputs = []
        hook_holder = []

        def collect(_mod, _inp, out):
            all_outputs.append(out.detach().float())

        h = model.fc1.register_forward_hook(collect)

        for _ in range(20):
            with torch.no_grad():
                model(torch.randn(8, 32))

        h.remove()

        # Compare
        all_out = torch.cat(all_outputs).flatten()
        true_mean = all_out.mean().item()
        true_std = all_out.std().item()

        stats = observer.get_stats()["fc1"]
        assert abs(stats.mean - true_mean) < 0.1, \
            f"Welford mean {stats.mean} vs true {true_mean}"
        assert abs(stats.std - true_std) < 0.1, \
            f"Welford std {stats.std} vs true {true_std}"

        observer.detach()

    def test_repr(self):
        from moq.observer.statistics_observer import StatisticsObserver

        observer = StatisticsObserver(n_bins=128)
        assert "bins=128" in repr(observer)


# =====================================================================
# MoQ Calibrator
# =====================================================================


class TestMoQCalibrator:
    """Tests for MoQCalibrator."""

    def test_calibrate_returns_format_map(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.quantizers.fp_quantizer import FPQuantizer
        from moq.calibration.moq_calibrator import MoQCalibrator

        model = _ToyModel()
        candidates = [
            INTQuantizer(bits=8),
            INTQuantizer(bits=4),
            FPQuantizer(bits=8, exp_bits=4),
        ]

        calibrator = MoQCalibrator(model, candidates)
        calib_data = [torch.randn(4, 32) for _ in range(3)]
        format_map = calibrator.calibrate(calib_data, ["fc1", "fc2"], show_progress=False)

        assert "fc1" in format_map
        assert "fc2" in format_map
        for q in format_map.values():
            assert any(type(q) is type(c) for c in candidates)

    def test_default_candidates_8bit(self):
        from moq.calibration.moq_calibrator import MoQCalibrator

        candidates = MoQCalibrator.default_candidates(bits=8)
        # INT + INT+ACIQ + 6 FP variants (exp 1..6) × 2 (±ACIQ)
        assert len(candidates) == 14  # 2 + 6×2

    def test_default_candidates_4bit(self):
        from moq.calibration.moq_calibrator import MoQCalibrator

        candidates = MoQCalibrator.default_candidates(bits=4)
        # INT + INT+ACIQ + 2 FP variants (exp 1..2) × 2 (±ACIQ)
        assert len(candidates) == 6  # 2 + 2×2

    def test_calibrate_prefers_higher_precision(self):
        """Given INT4 and INT8, calibrator should prefer INT8 (lower MSE)."""
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.calibration.moq_calibrator import MoQCalibrator

        model = _ToyModel()
        model.eval()
        candidates = [INTQuantizer(bits=4), INTQuantizer(bits=8)]

        calibrator = MoQCalibrator(model, candidates)
        calib_data = [torch.randn(8, 32) for _ in range(5)]
        format_map = calibrator.calibrate(calib_data, ["fc1"], show_progress=False)

        # INT8 should generally be preferred (lower quantization error)
        selected = format_map["fc1"]
        assert selected.bits == 8, f"Expected INT8 to be selected, got bits={selected.bits}"


# =====================================================================
# Search Strategies
# =====================================================================


class TestSearchStrategies:
    """Tests for individual search strategies."""

    def test_static_strategy(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.calibration.search_strategies import StaticFormatStrategy

        fixed_q = INTQuantizer(bits=4)
        strategy = StaticFormatStrategy(fixed_q)

        model = _ToyModel()
        best, score = strategy.select_format(
            "fc1",
            [INTQuantizer(bits=8), INTQuantizer(bits=4)],
            model,
            [torch.randn(2, 32)],
            torch.randn(2, 10),
        )

        assert best is fixed_q
        assert score == 0.0

    def test_intermediate_mse_strategy(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.calibration.search_strategies import IntermediateMSEStrategy

        model = _ToyModel()
        model.eval()
        strategy = IntermediateMSEStrategy()

        candidates = [INTQuantizer(bits=4), INTQuantizer(bits=8)]
        calib_data = [torch.randn(4, 32)]
        ref = torch.randn(4, 10)  # Not used by this strategy

        best, score = strategy.select_format(
            "fc1", candidates, model, calib_data, ref
        )
        assert best is not None
        assert score >= 0.0

    def test_cosine_strategy(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.calibration.search_strategies import CosineDistanceStrategy

        model = _ToyModel()
        model.eval()
        strategy = CosineDistanceStrategy()

        candidates = [INTQuantizer(bits=4), INTQuantizer(bits=8)]
        calib_data = [torch.randn(4, 32)]
        ref = torch.randn(4, 10)

        best, score = strategy.select_format(
            "fc1", candidates, model, calib_data, ref
        )
        assert best is not None
        assert 0.0 <= score <= 2.0  # Cosine distance in [0, 2]

    def test_end_to_end_strategy(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.calibration.search_strategies import MoQEndToEndStrategy

        model = _ToyModel()
        model.eval()
        strategy = MoQEndToEndStrategy()

        candidates = [INTQuantizer(bits=4), INTQuantizer(bits=8)]
        calib_data = [torch.randn(4, 32)]

        # Compute proper reference
        with torch.no_grad():
            ref_parts = [model(b).cpu().float() for b in calib_data]
        ref = torch.cat(ref_parts)

        best, score = strategy.select_format(
            "fc1", candidates, model, calib_data, ref
        )
        assert best is not None
        assert score >= 0.0


# =====================================================================
# End-to-end calibration integration
# =====================================================================


class TestCalibrationIntegration:
    """End-to-end integration: observe → calibrate → quantize."""

    def test_full_pipeline(self):
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.quantizers.fp_quantizer import FPQuantizer
        from moq.observer.statistics_observer import StatisticsObserver
        from moq.calibration.moq_calibrator import MoQCalibrator
        from moq.transform.hook_injector import HookQuantInjector

        model = _ToyModel()
        model.eval()
        layer_names = ["fc1", "fc2"]

        # Step 1: Observe
        observer = StatisticsObserver()
        observer.attach(model, layer_names)
        calib_data = [torch.randn(8, 32) for _ in range(5)]
        with torch.no_grad():
            for batch in calib_data:
                model(batch)
        stats = observer.get_stats()
        observer.detach()
        assert all(stats[name].num_batches == 5 for name in layer_names)

        # Step 2: Calibrate
        candidates = [
            INTQuantizer(bits=8),
            INTQuantizer(bits=4),
            FPQuantizer(bits=8, exp_bits=4),
        ]
        calibrator = MoQCalibrator(model, candidates)
        format_map = calibrator.calibrate(calib_data, layer_names, show_progress=False)
        assert len(format_map) == 2

        # Step 3: Quantize and verify
        with HookQuantInjector(model, format_map):
            x_test = torch.randn(16, 32)
            with torch.no_grad():
                out_q = model(x_test)
            assert out_q.shape == (16, 10)
            assert not torch.isnan(out_q).any()

        # Step 4: Full-precision comparison
        with torch.no_grad():
            out_fp = model(x_test)
        assert not torch.equal(out_fp, out_q)
