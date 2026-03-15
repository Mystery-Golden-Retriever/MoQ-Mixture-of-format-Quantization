"""Perplexity evaluator for autoregressive language models.

Supports sliding-window evaluation on standard benchmarks:
  * WikiText-2
  * C4
  * PTB (Penn Treebank)

Uses non-overlapping windows by default for reproducibility (matching
common quantization papers), with an optional ``stride`` parameter for
overlapping evaluation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PPLEvaluator:
    """Perplexity evaluator for causal language models.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace causal LM (e.g. ``LlamaForCausalLM``).
    tokenizer
        A HuggingFace tokenizer.
    seq_len : int
        Context window length for evaluation (default 2048).
    stride : int or None
        Sliding window stride.  ``None`` = non-overlapping (= seq_len).
    device : str or torch.device
        Device for evaluation.

    Example
    -------
    >>> evaluator = PPLEvaluator(model, tokenizer)
    >>> ppl = evaluator.evaluate("wikitext2")
    >>> print(f"Perplexity: {ppl:.2f}")
    """

    # Dataset loading configs
    _DATASET_MAP = {
        "wikitext2": ("wikitext", "wikitext-2-raw-v1", "test"),
        "c4": ("allenai/c4", "en", "validation"),
        "ptb": ("ptb_text_only", "penn_treebank", "test"),
    }

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        seq_len: int = 2048,
        stride: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len  # Non-overlapping by default
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataset_name: str = "wikitext2", max_samples: Optional[int] = None) -> float:
        """Compute perplexity on the specified dataset.

        Parameters
        ----------
        dataset_name : str
            One of ``"wikitext2"``, ``"c4"``, ``"ptb"``.
        max_samples : int or None
            Limit the number of text samples loaded (for quick testing).

        Returns
        -------
        float
            Perplexity value.
        """
        encodings = self._load_and_tokenize(dataset_name, max_samples)
        return self._compute_ppl(encodings)

    @torch.no_grad()
    def evaluate_from_text(self, text: str) -> float:
        """Compute perplexity on raw text (for custom datasets)."""
        encodings = self.tokenizer(text, return_tensors="pt")["input_ids"]
        return self._compute_ppl(encodings)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_and_tokenize(
        self, dataset_name: str, max_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Load and tokenize a standard benchmark dataset."""
        from datasets import load_dataset

        if dataset_name not in self._DATASET_MAP:
            raise ValueError(
                f"Unknown dataset {dataset_name!r}. "
                f"Available: {list(self._DATASET_MAP.keys())}"
            )

        ds_name, config, split = self._DATASET_MAP[dataset_name]
        logger.info("Loading %s/%s split=%s…", ds_name, config, split)

        try:
            dataset = load_dataset(ds_name, config, split=split)
        except Exception:
            # Fallback for datasets without configs
            dataset = load_dataset(ds_name, split=split)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Concatenate all text and tokenize
        if "text" in dataset.column_names:
            text = "\n\n".join(dataset["text"])
        elif "sentence" in dataset.column_names:
            text = "\n\n".join(dataset["sentence"])
        else:
            text = "\n\n".join(str(x) for x in dataset[dataset.column_names[0]])

        encodings = self.tokenizer(text, return_tensors="pt")["input_ids"]
        logger.info("Tokenized %d tokens", encodings.shape[1])
        return encodings

    def _compute_ppl(self, encodings: torch.Tensor) -> float:
        """Sliding-window perplexity computation."""
        self.model.eval()
        nlls: list[torch.Tensor] = []
        total_tokens = 0
        seq_len = min(self.seq_len, encodings.shape[1])

        for begin_loc in range(0, encodings.shape[1] - seq_len + 1, self.stride):
            end_loc = begin_loc + seq_len
            input_ids = encodings[:, begin_loc:end_loc].to(self.device)

            # Create labels: shift right by 1 inside the model
            target_ids = input_ids.clone()
            # Mask out context tokens (only score the extension)
            if begin_loc > 0 and self.stride < self.seq_len:
                # For overlapping windows, only predict non-overlapping tokens
                overlap = self.seq_len - self.stride
                target_ids[:, :overlap] = -100

            outputs = self.model(input_ids, labels=target_ids)
            loss = outputs.loss

            # Count non-masked tokens
            valid_tokens = (target_ids != -100).sum().item()
            nlls.append(loss * valid_tokens)
            total_tokens += valid_tokens

            if total_tokens == 0:
                continue

        if total_tokens == 0:
            logger.warning("No tokens evaluated — returning infinity")
            return float("inf")

        avg_nll = torch.stack(nlls).sum() / total_tokens
        ppl = torch.exp(avg_nll).item()
        return ppl
