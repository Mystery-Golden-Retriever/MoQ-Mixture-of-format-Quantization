# `moq/utils/` — Utility Modules

Shared helper functions for data loading and preprocessing.

---

## Files

### `data_utils.py` — Calibration Data Loaders

Provides two functions for constructing properly-formatted calibration datasets from HuggingFace Datasets.

#### `load_calib_data_text()`

Loads and tokenizes text data for language model calibration.

```python
from moq.utils.data_utils import load_calib_data_text

calib_data = load_calib_data_text(
    dataset_name="wikitext2",   # or "c4"
    tokenizer=tokenizer,
    n_samples=128,              # Number of calibration samples
    seq_len=2048,               # Tokens per sample
    seed=42,
)
# → list of {"input_ids": tensor(1, seq_len)}
```

**Supported datasets:** `"wikitext2"` (WikiText-2 train), `"c4"` (C4 English train)

Samples are randomly sliced from the concatenated tokenized corpus. Each batch is a dict with `"input_ids"` ready for HuggingFace causal LMs.

#### `load_calib_data_vision()`

Loads and preprocesses images for vision model calibration.

```python
from moq.utils.data_utils import load_calib_data_vision
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

calib_data = load_calib_data_vision(
    dataset_name="cifar10",     # or "imagenet", "food101"
    processor=processor,        # HuggingFace image processor
    n_samples=128,
    seed=42,
)
# → list of {"pixel_values": tensor(1, 3, 224, 224)}
```

**Supported datasets:** `"cifar10"` (default, no auth), `"imagenet"` (requires HF token), `"food101"`

When `processor` is provided, images are processed via HuggingFace's `AutoImageProcessor`. Without a processor, falls back to torchvision transforms (Resize → CenterCrop → ToTensor → Normalize).
