"""Unit tests for the core quantization engine.

Covers:
  * INTQuantizer (symmetric, asymmetric, per-channel, ACIQ)
  * FPQuantizer (E4M3, E5M2, FP4 variants, generic)
  * ACIQClipper (Gaussian, Laplacian)
  * Format Registry (register, lookup, list)
"""

import pytest
import torch

# =====================================================================
# INT Quantizer
# =====================================================================


class TestINTQuantizer:
    """Tests for INTQuantizer."""

    def test_basic_int8(self):
        from moq.quantizers.int_quantizer import INTQuantizer

        q = INTQuantizer(bits=8)
        x = torch.randn(2, 64)
        x_q = q(x)

        assert x_q.shape == x.shape
        assert x_q.dtype == x.dtype
        assert not torch.equal(x, x_q)
        mse = (x - x_q).pow(2).mean().item()
        assert mse < 0.01, f"INT8 MSE too high: {mse}"

    def test_int4_more_error(self):
        from moq.quantizers.int_quantizer import INTQuantizer

        q4 = INTQuantizer(bits=4)
        q8 = INTQuantizer(bits=8)
        x = torch.randn(2, 64)

        mse4 = (x - q4(x)).pow(2).mean().item()
        mse8 = (x - q8(x)).pow(2).mean().item()
        assert mse4 > mse8, f"INT4 MSE ({mse4}) should be > INT8 MSE ({mse8})"

    def test_int2_coarse(self):
        """INT2 should produce very coarse quantization (only 4 levels)."""
        from moq.quantizers.int_quantizer import INTQuantizer

        q = INTQuantizer(bits=2)
        x = torch.randn(4, 128)
        x_q = q(x)

        # 2-bit symmetric: levels = {-2, -1, 0, 1} × scale
        unique_count = x_q.unique().numel()
        # Should have at most 4 unique values (plus zero)
        assert unique_count <= 5, f"INT2 has {unique_count} unique values, expected ≤ 5"

    def test_aciq_clipping(self):
        from moq.quantizers.int_quantizer import INTQuantizer

        q_no_clip = INTQuantizer(bits=4)
        q_clip = INTQuantizer(bits=4, use_aciq=True)

        x = torch.randn(1, 256)
        x[0, 0] = 100.0
        x[0, 1] = -100.0

        x_q_no_clip = q_no_clip(x)
        x_q_clip = q_clip(x)

        mask = x.abs() < 5.0
        mse_no_clip = (x[mask] - x_q_no_clip[mask]).pow(2).mean().item()
        mse_clip = (x[mask] - x_q_clip[mask]).pow(2).mean().item()
        assert mse_clip < mse_no_clip, "ACIQ should improve non-outlier accuracy"

    def test_per_channel(self):
        from moq.quantizers.int_quantizer import INTQuantizer

        q = INTQuantizer(bits=8, channel_wise=True)
        x = torch.randn(4, 64)
        x[0] *= 10
        x[1] *= 0.01

        x_q = q(x)
        assert x_q.shape == x.shape

        # Per-channel should produce different scales per channel
        scale = q.get_scale(x)
        assert scale.shape == (4,)
        assert scale[0] > scale[1], "Channel 0 (×10) should have larger scale"

    def test_symmetric_range(self):
        from moq.quantizers.int_quantizer import INTQuantizer

        q = INTQuantizer(bits=4, symmetric=True)
        x = torch.randn(1, 100)
        x_q = q(x)

        max_q = x_q.abs().max().item()
        scale = q.get_scale(x).item()
        expected_max = scale * 7  # 2^(4-1) - 1 = 7
        assert abs(max_q - expected_max) < scale + 1e-6

    def test_config_and_repr(self):
        from moq.quantizers.int_quantizer import INTQuantizer

        q = INTQuantizer(bits=4, use_aciq=True)
        cfg = q.get_config()
        assert cfg["bits"] == 4
        assert cfg["use_aciq"] is True
        assert cfg["class"] == "INTQuantizer"

        repr_str = repr(q)
        assert "INTQuantizer" in repr_str
        assert "bits=4" in repr_str

    def test_zero_tensor(self):
        from moq.quantizers.int_quantizer import INTQuantizer

        q = INTQuantizer(bits=8)
        x = torch.zeros(2, 32)
        x_q = q(x)
        assert torch.equal(x_q, x), "Quantizing zeros should return zeros"

    def test_deterministic(self):
        """Same input should always produce the same output."""
        from moq.quantizers.int_quantizer import INTQuantizer

        q = INTQuantizer(bits=4)
        x = torch.randn(2, 64)
        x_q1 = q(x)
        x_q2 = q(x)
        assert torch.equal(x_q1, x_q2), "Quantization should be deterministic"

    def test_large_tensor(self):
        """Should handle larger tensors without issues."""
        from moq.quantizers.int_quantizer import INTQuantizer

        q = INTQuantizer(bits=8)
        x = torch.randn(32, 512, 768)
        x_q = q(x)
        assert x_q.shape == x.shape
        assert not torch.isnan(x_q).any()


# =====================================================================
# FP Quantizer
# =====================================================================


class TestFPQuantizer:
    """Tests for FPQuantizer and its variants."""

    def test_basic_fp8_e4m3(self):
        from moq.quantizers.fp_quantizer import E4M3Quantizer

        q = E4M3Quantizer()
        x = torch.randn(2, 64)
        x_q = q(x)

        assert x_q.shape == x.shape
        mse = (x - x_q).pow(2).mean().item()
        assert mse < 0.01, f"FP8-E4M3 MSE too high: {mse}"

    def test_basic_fp8_e5m2(self):
        from moq.quantizers.fp_quantizer import E5M2Quantizer

        q = E5M2Quantizer()
        x = torch.randn(2, 64)
        x_q = q(x)
        assert x_q.shape == x.shape

    def test_fp4_precision(self):
        from moq.quantizers.fp_quantizer import FP4E2M1Quantizer

        q = FP4E2M1Quantizer()
        x = torch.randn(2, 64)
        x_q = q(x)
        mse = (x - x_q).pow(2).mean().item()
        assert mse < 1.0, f"FP4 MSE unreasonably high: {mse}"

    def test_fp4_e3m0(self):
        """FP4-E3M0: powers-of-two only (0 mantissa bits)."""
        from moq.quantizers.fp_quantizer import FP4E3M0Quantizer

        q = FP4E3M0Quantizer()
        x = torch.randn(2, 64)
        x_q = q(x)
        assert x_q.shape == x.shape
        assert not torch.isnan(x_q).any()

    def test_e4m3_vs_e5m2_precision(self):
        """E4M3 should be more precise than E5M2 on regular-range data."""
        from moq.quantizers.fp_quantizer import E4M3Quantizer, E5M2Quantizer

        x = torch.randn(4, 128)
        mse_e4m3 = (x - E4M3Quantizer()(x)).pow(2).mean().item()
        mse_e5m2 = (x - E5M2Quantizer()(x)).pow(2).mean().item()
        assert mse_e4m3 < mse_e5m2, "E4M3 should be more precise than E5M2"

    def test_e5m2_handles_outliers(self):
        """E5M2 has wider range — should handle large values without NaN."""
        from moq.quantizers.fp_quantizer import E4M3Quantizer, E5M2Quantizer

        x = torch.randn(4, 128) * 100
        x_q_e4m3 = E4M3Quantizer()(x)
        x_q_e5m2 = E5M2Quantizer()(x)
        assert not torch.isnan(x_q_e4m3).any()
        assert not torch.isnan(x_q_e5m2).any()

    def test_preserves_sign(self):
        from moq.quantizers.fp_quantizer import FPQuantizer

        q = FPQuantizer(bits=8, exp_bits=4)
        x = torch.tensor([-3.5, -1.0, 0.0, 1.0, 3.5])
        x_q = q(x)

        assert (x_q[x > 0] > 0).all()
        assert (x_q[x < 0] < 0).all()
        assert x_q[x == 0] == 0

    def test_preserves_zeros(self):
        from moq.quantizers.fp_quantizer import FPQuantizer

        q = FPQuantizer(bits=8, exp_bits=4)
        x = torch.zeros(2, 32)
        x_q = q(x)
        assert torch.equal(x_q, x), "FP quantizing zeros should return zeros"

    def test_invalid_exp_bits(self):
        from moq.quantizers.fp_quantizer import FPQuantizer

        with pytest.raises(ValueError, match="exp_bits"):
            FPQuantizer(bits=4, exp_bits=4)

    def test_config(self):
        from moq.quantizers.fp_quantizer import FPQuantizer

        q = FPQuantizer(bits=8, exp_bits=5)
        cfg = q.get_config()
        assert cfg["exp_bits"] == 5
        assert cfg["man_bits"] == 2
        assert cfg["bits"] == 8

    def test_generic_fp_sweep(self):
        """All valid (exp_bits, man_bits) combos for 8-bit should work."""
        from moq.quantizers.fp_quantizer import FPQuantizer

        x = torch.randn(2, 64)
        for exp_bits in range(1, 7):  # exp_bits 1..6, man_bits 6..1
            q = FPQuantizer(bits=8, exp_bits=exp_bits)
            x_q = q(x)
            assert x_q.shape == x.shape, f"Failed for exp_bits={exp_bits}"
            assert not torch.isnan(x_q).any(), f"NaN for exp_bits={exp_bits}"

    def test_per_channel_fp(self):
        from moq.quantizers.fp_quantizer import FPQuantizer

        q = FPQuantizer(bits=8, exp_bits=4, channel_wise=True)
        x = torch.randn(4, 64)
        x[0] *= 100
        x_q = q(x)
        assert x_q.shape == x.shape

    def test_deterministic(self):
        from moq.quantizers.fp_quantizer import E4M3Quantizer

        q = E4M3Quantizer()
        x = torch.randn(2, 64)
        assert torch.equal(q(x), q(x)), "FP quantization should be deterministic"


# =====================================================================
# ACIQ Clipper
# =====================================================================


class TestACIQ:
    """Tests for ACIQClipper."""

    def test_gaussian_clip(self):
        from moq.quantizers.aciq import ACIQClipper

        clipper = ACIQClipper(bits=4, distribution="gaussian")
        x = torch.randn(1000)
        clip_val = clipper.compute_clip(x)

        assert clip_val < x.abs().max()
        assert clip_val > x.std()

    def test_laplacian_clip(self):
        from moq.quantizers.aciq import ACIQClipper

        clipper = ACIQClipper(bits=4, distribution="laplacian")
        x = torch.randn(1000)
        clip_val = clipper.compute_clip(x)
        assert clip_val > 0

    def test_higher_bits_wider_clip(self):
        from moq.quantizers.aciq import ACIQClipper

        x = torch.randn(1000)
        c4 = ACIQClipper(bits=4).compute_clip(x)
        c8 = ACIQClipper(bits=8).compute_clip(x)
        assert c8 > c4

    def test_all_standard_bits(self):
        """All pre-computed bit widths (2-8) should produce valid clips."""
        from moq.quantizers.aciq import ACIQClipper

        x = torch.randn(1000)
        prev_clip = 0.0
        for bits in [2, 3, 4, 5, 6, 7, 8]:
            clip = ACIQClipper(bits=bits).compute_clip(x).item()
            assert clip > 0, f"Clip for {bits} bits should be positive"
            # Monotonic non-decreasing: higher bits → wider or equal clip.
            # Can tie when α·σ >= abs_max (safety guard caps to abs_max).
            assert clip >= prev_clip, f"Clip should not decrease with more bits"
            prev_clip = clip

    def test_fallback_alpha(self):
        """Non-standard bit widths should use the fallback formula."""
        from moq.quantizers.aciq import ACIQClipper

        clipper = ACIQClipper(bits=10)  # Not in lookup table
        x = torch.randn(1000)
        clip_val = clipper.compute_clip(x)
        assert clip_val > 0

    def test_invalid_distribution(self):
        from moq.quantizers.aciq import ACIQClipper

        with pytest.raises(ValueError, match="distribution"):
            ACIQClipper(bits=4, distribution="uniform")

    def test_repr(self):
        from moq.quantizers.aciq import ACIQClipper

        clipper = ACIQClipper(bits=4, distribution="gaussian")
        r = repr(clipper)
        assert "bits=4" in r
        assert "gaussian" in r


# =====================================================================
# Registry
# =====================================================================


class TestRegistry:
    """Tests for the quantizer registry."""

    def test_get_int(self):
        from moq.quantizers.registry import get_quantizer

        q = get_quantizer("int", bits=4)
        assert q.bits == 4

    def test_get_fp(self):
        from moq.quantizers.registry import get_quantizer

        q = get_quantizer("fp8_e4m3")
        assert q.bits == 8
        assert q.exp_bits == 4
        assert q.man_bits == 3

    def test_get_fp8_e5m2(self):
        from moq.quantizers.registry import get_quantizer

        q = get_quantizer("fp8_e5m2")
        assert q.bits == 8
        assert q.exp_bits == 5

    def test_get_fp4_variants(self):
        from moq.quantizers.registry import get_quantizer

        q1 = get_quantizer("fp4_e2m1")
        assert q1.bits == 4 and q1.exp_bits == 2

        q2 = get_quantizer("fp4_e3m0")
        assert q2.bits == 4 and q2.exp_bits == 3

    def test_list_quantizers(self):
        from moq.quantizers.registry import list_quantizers

        names = list_quantizers()
        expected = {"int", "fp", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "fp4_e3m0"}
        assert expected.issubset(set(names))

    def test_unknown_quantizer(self):
        from moq.quantizers.registry import get_quantizer

        with pytest.raises(KeyError, match="not_a_real_quantizer"):
            get_quantizer("not_a_real_quantizer")

    def test_quantizer_forward(self):
        """All registered quantizers should produce valid output."""
        from moq.quantizers.registry import get_quantizer, list_quantizers

        x = torch.randn(2, 32)
        for name in list_quantizers():
            q = get_quantizer(name)
            x_q = q(x)
            assert x_q.shape == x.shape, f"{name} changed shape"
            assert not torch.isnan(x_q).any(), f"{name} produced NaNs"


# =====================================================================
# BaseQuantizer contract
# =====================================================================


class TestBaseQuantizer:
    """Test the BaseQuantizer abstract interface and validation."""

    def test_invalid_bits_low(self):
        from moq.quantizers.int_quantizer import INTQuantizer

        with pytest.raises(ValueError, match="bits"):
            INTQuantizer(bits=1)

    def test_invalid_bits_high(self):
        from moq.quantizers.int_quantizer import INTQuantizer

        with pytest.raises(ValueError, match="bits"):
            INTQuantizer(bits=17)

    def test_callable(self):
        """Quantizers should be callable like nn.Module."""
        from moq.quantizers.int_quantizer import INTQuantizer

        q = INTQuantizer(bits=8)
        x = torch.randn(2, 32)
        # Both forward() and __call__ should work
        assert torch.equal(q.forward(x), q(x))
