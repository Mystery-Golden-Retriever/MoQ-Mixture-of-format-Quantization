"""Tests for the WeightCalibrator and weight search strategies."""

import pytest
import torch
import torch.nn as nn

from moq.quantizers.int_quantizer import INTQuantizer
from moq.quantizers.fp_quantizer import FPQuantizer
from moq.calibration.weight_calibrator import WeightCalibrator
from moq.calibration.search_strategies import (
    MoQEndToEndStrategy,
    IntermediateMSEStrategy,
    CosineDistanceStrategy,
    StaticFormatStrategy,
)


# ======================================================================
# Fixtures
# ======================================================================

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 8, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


@pytest.fixture
def model():
    torch.manual_seed(42)
    m = _TinyModel()
    m.eval()
    return m


@pytest.fixture
def calib_data():
    torch.manual_seed(42)
    return [torch.randn(4, 16) for _ in range(4)]


@pytest.fixture
def layer_names():
    return ["fc1", "fc2"]


# ======================================================================
# WeightCalibrator tests
# ======================================================================

class TestWeightCalibrator:

    def test_calibrate_returns_format_map(self, model, calib_data, layer_names):
        candidates = [INTQuantizer(bits=8), INTQuantizer(bits=8, use_aciq=True)]
        strategy = IntermediateMSEStrategy(target="weight")
        calibrator = WeightCalibrator(model, candidates, strategy=strategy)
        fmt_map = calibrator.calibrate(calib_data, layer_names, show_progress=False)
        assert isinstance(fmt_map, dict)
        assert set(fmt_map.keys()) == set(layer_names)
        for q in fmt_map.values():
            assert isinstance(q, INTQuantizer)

    def test_default_candidates_8bit(self):
        candidates = WeightCalibrator.default_weight_candidates(8)
        assert len(candidates) == 6

    def test_default_candidates_6bit(self):
        candidates = WeightCalibrator.default_weight_candidates(6)
        assert len(candidates) == 6

    def test_default_candidates_4bit(self):
        candidates = WeightCalibrator.default_weight_candidates(4)
        assert len(candidates) == 5

    def test_default_candidates_fallback(self):
        candidates = WeightCalibrator.default_weight_candidates(3)
        assert len(candidates) >= 2

    def test_calibrate_with_end_to_end(self, model, calib_data, layer_names):
        candidates = [INTQuantizer(bits=8), FPQuantizer(bits=8, exp_bits=4)]
        strategy = MoQEndToEndStrategy(target="weight")
        calibrator = WeightCalibrator(model, candidates, strategy=strategy)
        fmt_map = calibrator.calibrate(calib_data, layer_names, show_progress=False)
        assert len(fmt_map) == 2

    def test_different_strategies_may_differ(self, model, calib_data, layer_names):
        candidates = [
            INTQuantizer(bits=8),
            INTQuantizer(bits=8, use_aciq=True),
            FPQuantizer(bits=8, exp_bits=4),
        ]
        cal_e2e = WeightCalibrator(
            model, candidates,
            strategy=MoQEndToEndStrategy(target="weight")
        )
        cal_mse = WeightCalibrator(
            model, candidates,
            strategy=IntermediateMSEStrategy(target="weight")
        )
        map_e2e = cal_e2e.calibrate(calib_data, layer_names, show_progress=False)
        map_mse = cal_mse.calibrate(calib_data, layer_names, show_progress=False)
        assert len(map_e2e) == 2
        assert len(map_mse) == 2


# ======================================================================
# Weight search strategy tests
# ======================================================================

class TestWeightSearchStrategies:

    def test_static_strategy_weight(self, model, calib_data):
        fixed_q = INTQuantizer(bits=8)
        strategy = StaticFormatStrategy(fixed_quantizer=fixed_q)
        candidates = [INTQuantizer(bits=4), INTQuantizer(bits=8)]
        ref_output = torch.randn(16, 8)
        best_q, score = strategy.select_format(
            "fc1", candidates, model, calib_data, ref_output
        )
        assert best_q is fixed_q
        assert score == 0.0

    def test_intermediate_mse_weight(self, model, calib_data):
        strategy = IntermediateMSEStrategy(target="weight")
        candidates = [INTQuantizer(bits=4), INTQuantizer(bits=8)]
        ref_output = torch.randn(16, 8)
        best_q, score = strategy.select_format(
            "fc1", candidates, model, calib_data, ref_output
        )
        assert isinstance(best_q, INTQuantizer)
        assert best_q.bits == 8
        assert score >= 0

    def test_cosine_weight(self, model, calib_data):
        strategy = CosineDistanceStrategy(target="weight")
        candidates = [INTQuantizer(bits=4), INTQuantizer(bits=8)]
        ref_output = torch.randn(16, 8)
        best_q, score = strategy.select_format(
            "fc1", candidates, model, calib_data, ref_output
        )
        assert isinstance(best_q, INTQuantizer)
        assert score >= 0

    def test_end_to_end_weight(self, model, calib_data):
        strategy = MoQEndToEndStrategy(target="weight")
        candidates = [INTQuantizer(bits=4), INTQuantizer(bits=8)]
        with torch.no_grad():
            ref_output = torch.cat([model(b).cpu().float() for b in calib_data])
        best_q, score = strategy.select_format(
            "fc1", candidates, model, calib_data, ref_output
        )
        assert isinstance(best_q, INTQuantizer)
        assert best_q.bits == 8

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="target"):
            MoQEndToEndStrategy(target="invalid")

    def test_invalid_target_intermediate(self):
        with pytest.raises(ValueError, match="target"):
            IntermediateMSEStrategy(target="invalid")


# ======================================================================
# Weight hook injector test
# ======================================================================

class TestWeightHookInjector:

    def test_weight_hook_modifies_output(self, model, calib_data):
        from moq.transform.hook_injector import HookQuantInjector
        q = INTQuantizer(bits=4)
        with torch.no_grad():
            out_fp = model(calib_data[0])
        with HookQuantInjector(model, {"fc1": q}, quantize_weight=True):
            with torch.no_grad():
                out_wq = model(calib_data[0])
        assert not torch.allclose(out_fp, out_wq, atol=1e-6)

    def test_weight_hook_restores_weight(self, model, calib_data):
        from moq.transform.hook_injector import HookQuantInjector
        q = INTQuantizer(bits=4)
        orig_weight = model.fc1.weight.data.clone()
        with HookQuantInjector(model, {"fc1": q}, quantize_weight=True):
            with torch.no_grad():
                _ = model(calib_data[0])
        torch.testing.assert_close(model.fc1.weight.data, orig_weight)

    def test_weight_hook_repr(self, model):
        from moq.transform.hook_injector import HookQuantInjector
        injector = HookQuantInjector(model, {"fc1": INTQuantizer(bits=8)}, quantize_weight=True)
        assert "weight" in repr(injector)

    def test_mutually_exclusive(self, model):
        from moq.transform.hook_injector import HookQuantInjector
        with pytest.raises(ValueError, match="mutually exclusive"):
            HookQuantInjector(model, {"fc1": INTQuantizer(bits=8)}, quantize_input=True, quantize_weight=True)


# ======================================================================
# Integration test
# ======================================================================

class TestWeightCalibrationIntegration:

    def test_full_pipeline(self, model, calib_data, layer_names):
        from moq.transform.hook_injector import HookQuantInjector
        with torch.no_grad():
            fp_output = model(calib_data[0])
        candidates = WeightCalibrator.default_weight_candidates(8)
        strategy = IntermediateMSEStrategy(target="weight")
        calibrator = WeightCalibrator(model, candidates, strategy=strategy)
        weight_fmt_map = calibrator.calibrate(
            calib_data, layer_names, show_progress=False
        )
        with HookQuantInjector(model, weight_fmt_map, quantize_weight=True):
            with torch.no_grad():
                wq_output = model(calib_data[0])
        assert not torch.allclose(fp_output, wq_output, atol=1e-6)
        mse = (fp_output - wq_output).pow(2).mean()
        assert mse < fp_output.pow(2).mean()
