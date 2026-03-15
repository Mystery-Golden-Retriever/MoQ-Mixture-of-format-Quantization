"""Zero-shot task evaluation via ``lm-evaluation-harness``.

Wraps the ``lm_eval`` library for standardised zero-shot benchmarks:
  * HellaSwag, WinoGrande, PIQA, ARC-Easy, ARC-Challenge, LAMBADA

Falls back gracefully if ``lm_eval`` is not installed (eval extra).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


class ZeroShotRunner:
    """Zero-shot evaluation using ``lm-evaluation-harness``.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace causal LM.
    tokenizer
        A HuggingFace tokenizer.
    batch_size : int
        Batch size for evaluation.

    Example
    -------
    >>> runner = ZeroShotRunner(model, tokenizer)
    >>> results = runner.run(["hellaswag", "piqa", "arc_easy"])
    >>> print(results)
    {'hellaswag': 0.78, 'piqa': 0.81, 'arc_easy': 0.72}
    """

    DEFAULT_TASKS = [
        "hellaswag",
        "winogrande",
        "piqa",
        "arc_easy",
        "arc_challenge",
        "lambada_openai",
    ]

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    # ------------------------------------------------------------------

    def run(
        self,
        tasks: Optional[list[str]] = None,
        num_fewshot: int = 0,
    ) -> dict[str, float]:
        """Run zero-shot evaluation.

        Parameters
        ----------
        tasks : list[str] or None
            Task names (see ``DEFAULT_TASKS``). ``None`` = all defaults.
        num_fewshot : int
            Number of few-shot examples (0 = zero-shot).

        Returns
        -------
        dict[str, float]
            ``{task_name: accuracy}``.
        """
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
        except ImportError:
            raise ImportError(
                "lm-evaluation-harness is required for zero-shot evaluation. "
                "Install it with: pip install moq[eval]"
            )

        tasks = tasks or self.DEFAULT_TASKS

        logger.info(
            "Running zero-shot evaluation: tasks=%s, num_fewshot=%d",
            tasks,
            num_fewshot,
        )

        lm = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
        )

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
        )

        # Extract accuracy scores
        parsed: dict[str, float] = {}
        for task_name in tasks:
            task_results = results.get("results", {}).get(task_name, {})
            # lm-eval uses different metric keys depending on the task
            for key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                if key in task_results:
                    parsed[task_name] = task_results[key]
                    break
            else:
                logger.warning("No accuracy metric found for task %s", task_name)
                parsed[task_name] = 0.0

        return parsed
