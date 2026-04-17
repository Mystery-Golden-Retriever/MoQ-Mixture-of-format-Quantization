"""Calibration data loading utilities.

Provides helpers to construct calibration datasets of the right size
from HuggingFace Datasets for both NLP and Vision workloads.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def load_calib_data_text(
    dataset_name: str = "wikitext2",
    tokenizer=None,
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """Load calibration data for language models.

    Returns a list of batch dicts ``{"input_ids": tensor}`` suitable for
    feeding to HuggingFace causal LMs.

    Parameters
    ----------
    dataset_name : str
        ``"wikitext2"`` or ``"c4"``.
    tokenizer
        HuggingFace tokenizer.
    n_samples : int
        Number of calibration samples (sequentially sliced).
    seq_len : int
        Token sequence length per sample.
    seed : int
        Random seed for reproducibility.
    """
    from datasets import load_dataset

    _DATASET_MAP = {
        "wikitext2": ("wikitext", "wikitext-2-raw-v1", "train"),
        "c4": ("allenai/c4", "en", "train"),
    }

    if dataset_name not in _DATASET_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds_name, config, split = _DATASET_MAP[dataset_name]
    dataset = load_dataset(ds_name, config, split=split)

    # Concatenate all text
    text = "\n\n".join(
        t for t in dataset["text"] if t.strip()
    )

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")["input_ids"]
    total_tokens = encodings.shape[1]

    # Slice into n_samples chunks of seq_len
    batches = []
    torch.manual_seed(seed)
    max_start = total_tokens - seq_len
    if max_start <= 0:
        raise ValueError(f"Dataset too short ({total_tokens} tokens) for seq_len={seq_len}")

    starts = torch.randint(0, max_start, (n_samples,))
    for start in starts:
        input_ids = encodings[:, start : start + seq_len]
        batches.append({"input_ids": input_ids})

    logger.info(
        "Loaded %d calibration samples (seq_len=%d) from %s",
        len(batches),
        seq_len,
        dataset_name,
    )
    return batches


def load_calib_data_vision(
    dataset_name: str = "cifar10",
    processor=None,
    n_samples: int = 128,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """Load calibration data for vision models.

    Returns a list of dicts ``{"pixel_values": tensor}`` suitable for
    passing to HuggingFace vision models like ``ViTForImageClassification``.

    Parameters
    ----------
    dataset_name : str
        ``"cifar10"`` (default, no auth required) or ``"imagenet"``
        (requires HuggingFace token).
    processor
        HuggingFace ``AutoImageProcessor``.  If ``None``, uses basic
        torchvision transforms and returns raw tensors instead.
    n_samples : int
        Number of calibration images.
    seed : int
        Random seed.
    """
    from datasets import load_dataset

    if dataset_name == "imagenet":
        dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=True)
    elif dataset_name == "cifar10":
        dataset = load_dataset("cifar10", split="test")
    elif dataset_name == "food101":
        dataset = load_dataset("food101", split="validation", streaming=True)
    else:
        raise ValueError(f"Unknown vision dataset: {dataset_name}")

    batches = []
    torch.manual_seed(seed)

    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break
        img = sample.get("image") or sample.get("img")
        if img is None:
            continue
        if hasattr(img, "convert"):
            img = img.convert("RGB")

        if processor is not None:
            # Use HuggingFace processor → returns dict with "pixel_values"
            inputs = processor(images=img, return_tensors="pt")
            batches.append({"pixel_values": inputs["pixel_values"]})
        else:
            # Fallback: torchvision transforms → raw tensor
            try:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
            except ImportError:
                raise ImportError("torchvision is required when no processor is given.")
            tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
            batches.append({"pixel_values": tensor})

    logger.info("Loaded %d calibration images from %s", len(batches), dataset_name)
    return batches
