"""
PyTorch Dataset implementation for RxHand Prescription Word Image Dataset.
Preserves existing dataset files and annotations without any modification.
Handles deterministic train/validation/test splitting and multimodal tensor collation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Transcribe the handwritten medication name accurately."


def build_chat_messages(prompt: str = DEFAULT_PROMPT, response: Optional[str] = None) -> List[Dict]:
    """Format prompt and optional response into Qwen3-VL conversation format."""
    clean_prompt = prompt.replace("<image>", "").strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": clean_prompt},
            ],
        }
    ]
    if response is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": str(response).strip()}],
        })
    return messages


class RxHandDataset(Dataset):
    """
    Dataset wrapper for RxHand prescription images and text labels.
    Does NOT modify the source CSV or images.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        prompt: str = DEFAULT_PROMPT,
        max_pixels: int = 512 * 512,
        is_training: bool = True,
    ):
        """
        Args:
            records: List of dicts with 'image_path', 'text', 'image_id'
            prompt: Text prompt instruction for the model
            max_pixels: Maximum pixel budget for input images
            is_training: Whether dataset is used for training
        """
        self.records = records
        self.prompt = prompt
        self.max_pixels = max_pixels
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        image_path = rec["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Fallback blank image to prevent crashes
            image = Image.new("RGB", (128, 128), (255, 255, 255))

        # Pixel limit check
        w, h = image.size
        if w * h > self.max_pixels:
            scale = (self.max_pixels / (w * h)) ** 0.5
            new_w, new_h = max(28, int(w * scale)), max(28, int(h * scale))
            image = image.resize((new_w, new_h), Image.BICUBIC)

        return {
            "image": image,
            "image_id": rec["image_id"],
            "image_path": image_path,
            "text": rec["text"],
            "prompt": self.prompt,
        }


class RxHandCollator:
    """
    Batches multimodal samples and formats them via Qwen3VLProcessor.
    Correctly applies label masking (-100 on prompt tokens, unmasked on answer tokens).
    """

    def __init__(self, processor, is_training: bool = True):
        self.processor = processor
        self.is_training = is_training
        self.im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        responses = [item["text"] for item in batch] if self.is_training else None

        texts = [
            self.processor.apply_chat_template(
                build_chat_messages(p, r),
                tokenize=False,
                add_generation_prompt=(r is None),
            )
            for p, r in zip(prompts, responses or [None] * len(batch))
        ]

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = None
        if self.is_training:
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100

            # Mask everything up to '<|im_start|>assistant\n'
            for row in range(labels.size(0)):
                starts = (inputs["input_ids"][row] == self.im_start_id).nonzero(as_tuple=True)[0]
                if len(starts) > 0:
                    labels[row, : starts[-1] + 3] = -100

        output = dict(inputs)
        if labels is not None:
            output["labels"] = labels

        # Attach metadata for evaluation
        output["image_ids"] = [item["image_id"] for item in batch]
        output["ground_truths"] = [item["text"] for item in batch]
        output["raw_images"] = images
        output["prompts"] = prompts

        return output


def load_rxhand_records(
    labels_csv: str,
    images_dir: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Parse RxHand CSV and map to existing image files on disk.
    Never alters original files.
    """
    csv_path = Path(labels_csv)
    img_dir = Path(images_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    df = pd.read_csv(csv_path)
    records = []

    for idx, row in df.iterrows():
        img_name = str(row.iloc[0]).strip()
        text_label = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else ""

        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            img_name = f"{img_name}.jpg"

        full_img_path = img_dir / img_name
        if full_img_path.exists():
            records.append({
                "image_id": img_name,
                "image_path": str(full_img_path),
                "text": text_label,
            })
        else:
            logger.warning(f"Image {full_img_path} does not exist, skipping.")

        if max_samples and len(records) >= max_samples:
            break

    logger.info(f"Loaded {len(records)} valid samples from {csv_path.name}")
    return records


def get_rxhand_datasets(
    train_labels: str,
    train_images: str,
    test_labels: str,
    test_images: str,
    train_split: float = 0.80,
    seed: int = 3407,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    max_pixels: int = 512 * 512,
) -> Tuple[RxHandDataset, RxHandDataset, RxHandDataset]:
    """
    Create reproducible train, validation, and held-out test datasets.
    Train and validation splits are carved from Train_Label.csv using seed.
    Test set is carved from Test_Labels.csv.
    """
    train_pool = load_rxhand_records(train_labels, train_images)
    test_pool = load_rxhand_records(test_labels, test_images, max_samples=max_test_samples)

    # Deterministic train/validation split
    rng = torch.Generator().manual_seed(seed)
    num_total = len(train_pool)
    indices = torch.randperm(num_total, generator=rng).tolist()

    split_idx = int(num_total * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_records = [train_pool[i] for i in train_indices]
    val_records = [train_pool[i] for i in val_indices]

    if max_train_samples:
        train_records = train_records[:max_train_samples]
    if max_val_samples:
        val_records = val_records[:max_val_samples]

    logger.info(f"RxHand Dataset Splits -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_pool)}")

    train_dataset = RxHandDataset(train_records, max_pixels=max_pixels, is_training=True)
    val_dataset = RxHandDataset(val_records, max_pixels=max_pixels, is_training=False)
    test_dataset = RxHandDataset(test_pool, max_pixels=max_pixels, is_training=False)

    return train_dataset, val_dataset, test_dataset
