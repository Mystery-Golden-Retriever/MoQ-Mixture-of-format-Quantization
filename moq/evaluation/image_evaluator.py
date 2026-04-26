import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)

class ImageClassificationEvaluator:
    def __init__(self, model: nn.Module, processor, batch_size: int = 32):
        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def evaluate(self, dataset_name: str = "imagenet", max_samples: int = None) -> float:
        self.model.eval()
        
        if dataset_name == "imagenet":
            dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=True)
        elif dataset_name == "cifar10":
            dataset = load_dataset("cifar10", split="test")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        correct = 0
        total = 0
        
        batch_images = []
        batch_labels = []
        
        def process_batch():
            nonlocal correct, total, batch_images, batch_labels
            if not batch_images:
                return
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            labels = torch.tensor(batch_labels).to(self.device)
            outputs = self.model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            batch_images.clear()
            batch_labels.clear()

        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            img = sample.get("image") or sample.get("img")
            label = sample.get("label")
            if img is None or label is None:
                continue
            if hasattr(img, "convert"):
                img = img.convert("RGB")
                
            batch_images.append(img)
            batch_labels.append(label)
            
            if len(batch_images) == self.batch_size:
                process_batch()
                if total % (self.batch_size * 5) == 0:
                    logger.info(f"Evaluated {total} samples, current accuracy: {correct/total:.4f}")

        process_batch()  # process remaining
        
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Final {dataset_name} accuracy: {accuracy:.4f} over {total} samples")
        return accuracy
