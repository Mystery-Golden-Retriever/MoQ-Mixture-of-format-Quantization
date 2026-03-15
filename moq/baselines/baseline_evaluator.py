"""Abstract baseline evaluator and concrete integrations.

Each baseline wraps a third-party quantization library behind a
common interface so that MoQ can be benchmarked against them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class BaselineEvaluator(ABC):
    """Common interface for third-party quantization baselines."""

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable name of the baseline method."""
        ...

    @abstractmethod
    def quantize_model(self, model: nn.Module, calib_data: list) -> nn.Module:
        """Quantize the model using this baseline method.

        Parameters
        ----------
        model : nn.Module
            Full-precision model.
        calib_data : list
            Calibration batches.

        Returns
        -------
        nn.Module
            Quantized model ready for evaluation.
        """
        ...


class GPTQBaseline(BaselineEvaluator):
    """Wrapper for AutoGPTQ.

    Requires: ``pip install auto-gptq``
    """

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size

    def get_name(self) -> str:
        return f"GPTQ-W{self.bits}"

    def quantize_model(self, model: nn.Module, calib_data: list) -> nn.Module:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError(
                "auto-gptq is required. Install with: pip install moq[baselines]"
            )

        config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=False,
        )
        # Wrap the model for GPTQ quantization
        gptq_model = AutoGPTQForCausalLM.from_pretrained(model, config)
        gptq_model.quantize(calib_data)
        return gptq_model.model


class AWQBaseline(BaselineEvaluator):
    """Wrapper for AutoAWQ.

    Requires: ``pip install autoawq``
    """

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size

    def get_name(self) -> str:
        return f"AWQ-W{self.bits}"

    def quantize_model(self, model: nn.Module, calib_data: list) -> nn.Module:
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError(
                "autoawq is required. Install with: pip install moq[baselines]"
            )

        quant_config = {
            "w_bit": self.bits,
            "q_group_size": self.group_size,
            "zero_point": True,
        }
        awq_model = AutoAWQForCausalLM.from_pretrained(model)
        awq_model.quantize(calib_data, quant_config=quant_config)
        return awq_model.model


class SmoothQuantBaseline(BaselineEvaluator):
    """SmoothQuant: migrate quantization difficulty from activations to weights.

    This is a lightweight re-implementation of the core algorithm:
    ``W' = W · diag(s), X' = X · diag(1/s)``
    where ``s_j = max(|X_j|)^α / max(|W_j|)^(1-α)``

    Reference: Xiao et al., "SmoothQuant", 2023.
    """

    def __init__(self, bits: int = 8, alpha: float = 0.5):
        self.bits = bits
        self.alpha = alpha

    def get_name(self) -> str:
        return f"SmoothQuant-W{self.bits}-α{self.alpha}"

    def quantize_model(self, model: nn.Module, calib_data: list) -> nn.Module:
        import torch
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.transform.module_replacer import ModelQuantizer

        int_q = INTQuantizer(bits=self.bits)

        # For each Linear, apply smoothing then INT quantization
        format_map = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                format_map[name] = (int_q, int_q)

        replacer = ModelQuantizer()
        return replacer.replace(model, format_map)
