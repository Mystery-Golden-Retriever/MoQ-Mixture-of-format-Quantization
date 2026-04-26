"""Tests for weight quantization formats: MXFP, NVFP4, NF4, FP6."""

import pytest
import torch
from moq.quantizers.mxfp_quantizer import (
    MXFPQuantizer,
    MXFP8E4M3Quantizer,
    MXFP8E5M2Quantizer,
    MXFP6E3M2Quantizer,
    MXFP6E2M3Quantizer,
    MXFP4Quantizer,
    _quantize_scale_e8m0,
)
from moq.quantizers.nvfp4_quantizer import NVFP4Quantizer
from moq.quantizers.nf_quantizer import NF4Quantizer, _build_nf4_codebook
from moq.quantizers.fp_quantizer import FP6E3M2Quantizer, FP6E2M3Quantizer
from moq.quantizers.registry import get_quantizer, list_quantizers


# ======================================================================
# E8M0 scale quantization
# ======================================================================

class TestE8M0Scale:
    """Test the E8M0 (power-of-two) scale quantization."""

    def test_power_of_two_roundtrip(self):
        x = torch.tensor([1.0, 2.0, 4.0, 8.0, 0.5, 0.25])
        result = _quantize_scale_e8m0(x)
        torch.testing.assert_close(result, x)

    def test_non_power_rounds(self):
        x = torch.tensor([3.0])
        result = _quantize_scale_e8m0(x)
        assert result.item() in (2.0, 4.0)

    def test_output_is_power_of_two(self):
        x = torch.randn(100).abs() + 1e-6
        result = _quantize_scale_e8m0(x)
        log2_result = torch.log2(result)
        torch.testing.assert_close(log2_result, log2_result.round(), atol=1e-5, rtol=0)

    def test_clamp_range(self):
        x = torch.tensor([1e-200, 1e200])
        result = _quantize_scale_e8m0(x)
        assert result[0] >= 2.0**-127
        assert result[1] <= 2.0**127


# ======================================================================
# MXFP quantizer
# ======================================================================

class TestMXFPQuantizer:

    def test_basic_mxfp8_e4m3(self):
        q = MXFP8E4M3Quantizer()
        x = torch.randn(128)
        x_q = q(x)
        assert x_q.shape == x.shape
        assert not torch.allclose(x, x_q, atol=1e-8)
        assert (x - x_q).abs().max() < x.abs().max() * 0.5

    def test_basic_mxfp8_e5m2(self):
        q = MXFP8E5M2Quantizer()
        x = torch.randn(128)
        assert q(x).shape == x.shape

    def test_basic_mxfp6_e3m2(self):
        q = MXFP6E3M2Quantizer()
        x = torch.randn(64)
        assert q(x).shape == x.shape

    def test_basic_mxfp6_e2m3(self):
        q = MXFP6E2M3Quantizer()
        x = torch.randn(64)
        assert q(x).shape == x.shape

    def test_basic_mxfp4(self):
        q = MXFP4Quantizer()
        x = torch.randn(64)
        assert q(x).shape == x.shape

    def test_group_size_default(self):
        q = MXFPQuantizer(element_bits=8, element_exp_bits=4)
        assert q.group_size == 32

    def test_non_divisible_size(self):
        q = MXFPQuantizer(element_bits=8, element_exp_bits=4, group_size=32)
        x = torch.randn(50)
        assert q(x).shape == x.shape

    def test_2d_weight_shape(self):
        q = MXFP8E4M3Quantizer()
        x = torch.randn(256, 128)
        assert q(x).shape == x.shape

    def test_preserves_zeros(self):
        q = MXFP8E4M3Quantizer()
        x = torch.zeros(64)
        torch.testing.assert_close(q(x), x)

    def test_deterministic(self):
        q = MXFP8E4M3Quantizer()
        x = torch.randn(128)
        torch.testing.assert_close(q(x), q(x))

    def test_mxfp8_higher_precision_than_mxfp4(self):
        x = torch.randn(256)
        err8 = (x - MXFP8E4M3Quantizer()(x)).pow(2).mean()
        err4 = (x - MXFP4Quantizer()(x)).pow(2).mean()
        assert err8 < err4

    def test_config(self):
        cfg = MXFP8E4M3Quantizer().get_config()
        assert cfg["element_bits"] == 8
        assert cfg["element_exp_bits"] == 4
        assert cfg["element_man_bits"] == 3
        assert cfg["group_size"] == 32

    def test_invalid_exp_bits(self):
        with pytest.raises(ValueError):
            MXFPQuantizer(element_bits=4, element_exp_bits=5)

    def test_custom_group_size(self):
        q = MXFPQuantizer(element_bits=8, element_exp_bits=4, group_size=16)
        assert q.group_size == 16
        x = torch.randn(128)
        assert q(x).shape == x.shape


# ======================================================================
# NVFP4 quantizer
# ======================================================================

class TestNVFP4Quantizer:

    def test_basic(self):
        q = NVFP4Quantizer()
        x = torch.randn(128)
        assert q(x).shape == x.shape

    def test_block_size_default(self):
        assert NVFP4Quantizer().block_size == 16

    def test_custom_block_size(self):
        q = NVFP4Quantizer(block_size=32)
        assert q.block_size == 32
        x = torch.randn(128)
        assert q(x).shape == x.shape

    def test_2d_weight(self):
        q = NVFP4Quantizer()
        x = torch.randn(64, 32)
        assert q(x).shape == x.shape

    def test_non_divisible(self):
        q = NVFP4Quantizer()
        x = torch.randn(50)
        assert q(x).shape == x.shape

    def test_preserves_zeros(self):
        q = NVFP4Quantizer()
        x = torch.zeros(32)
        torch.testing.assert_close(q(x), x)

    def test_deterministic(self):
        q = NVFP4Quantizer()
        x = torch.randn(64)
        torch.testing.assert_close(q(x), q(x))

    def test_config(self):
        cfg = NVFP4Quantizer().get_config()
        assert cfg["bits"] == 4
        assert cfg["block_size"] == 16
        assert cfg["element_format"] == "E2M1"
        assert cfg["block_scale_format"] == "FP8-E4M3"
        assert cfg["super_scale_format"] == "FP32"

    def test_4bit_precision(self):
        q = NVFP4Quantizer()
        x = torch.randn(256)
        mse = (x - q(x)).pow(2).mean()
        assert mse > 0


# ======================================================================
# NF4 quantizer
# ======================================================================

class TestNF4Quantizer:

    def test_codebook_values(self):
        codebook = _build_nf4_codebook()
        assert codebook.shape == (16,)
        assert codebook.min() >= -1.0
        assert codebook.max() <= 1.0

    def test_codebook_symmetry(self):
        codebook = _build_nf4_codebook()
        assert 0.0 in codebook.tolist()
        assert codebook[0] < 0
        assert codebook[-1] > 0

    def test_codebook_sorted(self):
        codebook = _build_nf4_codebook()
        assert (codebook[1:] >= codebook[:-1]).all()

    def test_basic(self):
        q = NF4Quantizer()
        x = torch.randn(256)
        assert q(x).shape == x.shape

    def test_group_size_default(self):
        assert NF4Quantizer().group_size == 64

    def test_custom_group_size(self):
        q = NF4Quantizer(group_size=32)
        assert q.group_size == 32
        x = torch.randn(128)
        assert q(x).shape == x.shape

    def test_2d_weight(self):
        q = NF4Quantizer()
        x = torch.randn(128, 64)
        assert q(x).shape == x.shape

    def test_non_divisible(self):
        q = NF4Quantizer(group_size=64)
        x = torch.randn(100)
        assert q(x).shape == x.shape

    def test_preserves_zeros(self):
        q = NF4Quantizer()
        x = torch.zeros(64)
        torch.testing.assert_close(q(x), x)

    def test_deterministic(self):
        q = NF4Quantizer()
        x = torch.randn(128)
        torch.testing.assert_close(q(x), q(x))

    def test_output_from_codebook(self):
        q = NF4Quantizer(group_size=64)
        x = torch.randn(64)
        x_q = q(x)
        block_max = x.abs().max()
        normalised = x_q / block_max
        codebook = _build_nf4_codebook()
        for v in normalised:
            dists = (codebook - v.item()).abs()
            assert dists.min() < 0.01

    def test_double_quant(self):
        q_no = NF4Quantizer(double_quant=False)
        q_dq = NF4Quantizer(double_quant=True)
        x = torch.randn(256)
        assert q_no(x).shape == q_dq(x).shape

    def test_config(self):
        cfg = NF4Quantizer().get_config()
        assert cfg["bits"] == 4
        assert cfg["group_size"] == 64
        assert cfg["codebook_size"] == 16


# ======================================================================
# FP6 convenience subclasses
# ======================================================================

class TestFP6Quantizers:

    def test_fp6_e3m2_basic(self):
        q = FP6E3M2Quantizer()
        x = torch.randn(128)
        assert q(x).shape == x.shape

    def test_fp6_e2m3_basic(self):
        q = FP6E2M3Quantizer()
        x = torch.randn(128)
        assert q(x).shape == x.shape

    def test_e2m3_higher_precision_near_zero(self):
        x = torch.randn(256) * 0.5
        err_e3m2 = (x - FP6E3M2Quantizer()(x)).pow(2).mean()
        err_e2m3 = (x - FP6E2M3Quantizer()(x)).pow(2).mean()
        assert err_e2m3 < err_e3m2 * 2

    def test_fp6_config(self):
        cfg = FP6E3M2Quantizer().get_config()
        assert cfg["bits"] == 6
        assert cfg["exp_bits"] == 3
        assert cfg["man_bits"] == 2


# ======================================================================
# Registry integration
# ======================================================================

class TestWeightQuantizerRegistry:

    def test_all_registered(self):
        names = list_quantizers()
        expected = [
            "mxfp", "mxfp8_e4m3", "mxfp8_e5m2",
            "mxfp6_e3m2", "mxfp6_e2m3", "mxfp4",
            "nvfp4", "nf4",
            "fp6_e3m2", "fp6_e2m3",
        ]
        for name in expected:
            assert name in names, f"{name} not in registry"

    def test_factory_lookup(self):
        assert isinstance(get_quantizer("nvfp4"), NVFP4Quantizer)
        assert isinstance(get_quantizer("nf4"), NF4Quantizer)
        assert isinstance(get_quantizer("mxfp8_e4m3"), MXFP8E4M3Quantizer)

    def test_mxfp_generic_factory(self):
        q = get_quantizer("mxfp", element_bits=6, element_exp_bits=3)
        assert isinstance(q, MXFPQuantizer)
        assert q.element_bits == 6

    def test_all_formats_forward(self):
        x = torch.randn(128)
        weight_formats = [
            "mxfp8_e4m3", "mxfp8_e5m2",
            "mxfp6_e3m2", "mxfp6_e2m3", "mxfp4",
            "nvfp4", "nf4",
            "fp6_e3m2", "fp6_e2m3",
        ]
        for name in weight_formats:
            q = get_quantizer(name)
            assert q(x).shape == x.shape, f"{name} changed shape"
