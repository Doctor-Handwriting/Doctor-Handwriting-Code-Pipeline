"""
Unit tests for RxHand Dataset, splitting reproducibility, and Collator tensor shapes.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoProcessor

# Ensure experiments/MS-MGCA-Qwen3-VL is in sys.path
EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from configs.base_config import MSMGCAConfig
from dataset.rxhand_dataset import get_rxhand_datasets, load_rxhand_records, RxHandCollator


def test_dataset_loading_and_splits():
    """Verify loading from CSV, image verification, and reproducible train/val splits."""
    cfg = MSMGCAConfig()

    print("\n[TEST] Verifying dataset loading...")
    train_pool = load_rxhand_records(cfg.dataset.train_labels, cfg.dataset.train_images)
    test_pool = load_rxhand_records(cfg.dataset.test_labels, cfg.dataset.test_images)

    assert len(train_pool) == 4463, f"Expected 4463 train samples, got {len(train_pool)}"
    assert len(test_pool) == 1115, f"Expected 1115 test samples, got {len(test_pool)}"

    # Check first record of train
    first = train_pool[0]
    assert first["image_id"] == "P1116.jpg"
    assert first["text"] == "Dexter"
    assert Path(first["image_path"]).exists()
    print(f"  -> Verified first sample: {first['image_id']} -> '{first['text']}'")

    # Check reproducibility of splitting
    print("[TEST] Testing split reproducibility with seed=3407...")
    t1, v1, _ = get_rxhand_datasets(
        cfg.dataset.train_labels, cfg.dataset.train_images,
        cfg.dataset.test_labels, cfg.dataset.test_images,
        train_split=0.80, seed=3407
    )
    t2, v2, _ = get_rxhand_datasets(
        cfg.dataset.train_labels, cfg.dataset.train_images,
        cfg.dataset.test_labels, cfg.dataset.test_images,
        train_split=0.80, seed=3407
    )

    assert len(t1) == int(4463 * 0.80)
    assert len(v1) == 4463 - len(t1)
    assert t1.records[0]["image_id"] == t2.records[0]["image_id"], "Split must be identical for same seed"
    assert v1.records[0]["image_id"] == v2.records[0]["image_id"], "Split must be identical for same seed"
    print(f"  -> Train count: {len(t1)}, Val count: {len(v1)}, Test count: {len(test_pool)}")

    # Different seed should yield different split
    t3, _, _ = get_rxhand_datasets(
        cfg.dataset.train_labels, cfg.dataset.train_images,
        cfg.dataset.test_labels, cfg.dataset.test_images,
        train_split=0.80, seed=999
    )
    assert t1.records[0]["image_id"] != t3.records[0]["image_id"], "Different seeds must produce different ordering"
    print("  -> Deterministic splitting verified.")


def test_collator_and_label_masking():
    """Verify RxHandCollator output tensors, label masking (-100 for prompt), and shapes."""
    cfg = MSMGCAConfig()
    processor = AutoProcessor.from_pretrained(cfg.model.checkpoint)

    _, val_ds, _ = get_rxhand_datasets(
        cfg.dataset.train_labels, cfg.dataset.train_images,
        cfg.dataset.test_labels, cfg.dataset.test_images,
        train_split=0.80, seed=3407, max_val_samples=2
    )

    collator = RxHandCollator(processor, is_training=True)
    batch = [val_ds[0], val_ds[1]]

    print("[TEST] Collating batch of 2 samples...")
    collated = collator(batch)

    assert "input_ids" in collated
    assert "pixel_values" in collated
    assert "image_grid_thw" in collated
    assert "labels" in collated
    assert "ground_truths" in collated

    input_ids = collated["input_ids"]
    labels = collated["labels"]
    pixel_values = collated["pixel_values"]
    image_grid_thw = collated["image_grid_thw"]

    assert input_ids.shape[0] == 2
    assert labels.shape == input_ids.shape
    assert image_grid_thw.shape[0] == 2

    # Verify that prompt is masked with -100 and answer is unmasked
    for row in range(2):
        row_labels = labels[row]
        unmasked = row_labels[row_labels != -100]
        assert len(unmasked) > 0, "Each sample must have unmasked answer tokens"
        gt = collated["ground_truths"][row]
        decoded_answer = processor.decode(unmasked, skip_special_tokens=True).strip()
        assert decoded_answer == gt, f"Decoded unmasked tokens '{decoded_answer}' != GT '{gt}'"
        print(f"  -> Sample {row}: GT='{gt}' -> Decoded unmasked label tokens: '{decoded_answer}' (len={len(unmasked)})")

    print("[TEST] RxHandCollator and label masking tests PASSED!")


if __name__ == "__main__":
    test_dataset_loading_and_splits()
    test_collator_and_label_masking()
