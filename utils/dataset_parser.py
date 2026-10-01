import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import random

logger = logging.getLogger(__name__)


def normalize_path(path_str: str) -> str:
    """Convert Windows paths to forward-slash format for JSON compatibility."""
    return str(path_str).replace("\\", "/")


def filter_trashed_files(file_list: List[str]) -> List[str]:
    """Remove files starting with .trashed prefix."""
    return [f for f in file_list if not os.path.basename(f).startswith(".trashed")]


def load_dhp_dataset(dhp_labels_path: str, dhp_images_dir: str) -> List[Dict]:
    """
    Parse DHP (Doctor Handwriting Prescription) dataset.

    Args:
        dhp_labels_path: Path to doctor_handwriting_labels.csv
        dhp_images_dir: Path to image directory

    Returns:
        List of dataset records with multimodal instruction format
    """
    logger.info(f"Loading DHP dataset from {dhp_labels_path}")

    records = []

    try:
        df = pd.read_csv(dhp_labels_path)
        logger.info(f"Loaded {len(df)} records from DHP labels")
    except FileNotFoundError:
        logger.warning(f"DHP labels file not found: {dhp_labels_path}")
        return records

    img_files = filter_trashed_files(os.listdir(dhp_images_dir))
    img_map = {os.path.splitext(f)[0]: f for f in img_files}
    img_map_full = {f: f for f in img_files}

    for idx, row in df.iterrows():
        img_name = str(row.iloc[0]).strip() if len(row) > 0 else None
        text_label = str(row.iloc[1]).strip() if len(row) > 1 else ""

        matched_img = None
        if img_name:
            if img_name in img_map_full:
                matched_img = img_name
            elif os.path.splitext(img_name)[0] in img_map:
                matched_img = img_map[os.path.splitext(img_name)[0]]

        if matched_img:
            img_path = os.path.join(dhp_images_dir, matched_img)

            if os.path.exists(img_path):
                records.append({
                    "id": f"dhp_{idx}",
                    "image": normalize_path(img_path),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nRead and transcribe the handwritten text from this prescription accurately."
                        },
                        {
                            "from": "gpt",
                            "value": text_label
                        }
                    ]
                })

    logger.info(f"Parsed {len(records)} valid DHP records")
    return records


def load_rxhand_dataset(
    train_labels_path: str,
    test_labels_path: str,
    train_images_dir: str,
    test_images_dir: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse RxHand (Handwritten Prescription Word Image) dataset.

    Args:
        train_labels_path: Path to Train_Label.csv
        test_labels_path: Path to Test_Labels.csv
        train_images_dir: Path to training images
        test_images_dir: Path to test images

    Returns:
        Tuple of (train_records, test_records) with multimodal instruction format
    """
    logger.info("Loading RxHand dataset...")

    train_records = []
    test_records = []

    try:
        train_df = pd.read_csv(train_labels_path)
        logger.info(f"Loaded {len(train_df)} training records from {train_labels_path}")
    except FileNotFoundError:
        logger.warning(f"RxHand train labels not found: {train_labels_path}")
        train_df = pd.DataFrame()

    try:
        test_df = pd.read_csv(test_labels_path)
        logger.info(f"Loaded {len(test_df)} test records from {test_labels_path}")
    except FileNotFoundError:
        logger.warning(f"RxHand test labels not found: {test_labels_path}")
        test_df = pd.DataFrame()

    train_records = _parse_rxhand_split(
        train_df, train_images_dir, "rxhand_train", offset=0
    )
    test_records = _parse_rxhand_split(
        test_df, test_images_dir, "rxhand_test", offset=len(train_records)
    )

    logger.info(f"Parsed {len(train_records)} train and {len(test_records)} test RxHand records")
    return train_records, test_records


def _parse_rxhand_split(
    df: pd.DataFrame,
    images_dir: str,
    split_prefix: str,
    offset: int = 0
) -> List[Dict]:
    """Helper to parse RxHand train/test splits."""
    records = []

    if df.empty:
        return records

    for idx, row in df.iterrows():
        img_id = str(row.iloc[0]).strip() if len(row) > 0 else None
        text_label = str(row.iloc[1]).strip() if len(row) > 1 else ""

        if img_id:
            img_filename = img_id if img_id.endswith(('.jpg', '.png')) else f"{img_id}.jpg"
            img_path = os.path.join(images_dir, img_filename)

            if os.path.exists(img_path):
                records.append({
                    "id": f"{split_prefix}_{offset + idx}",
                    "image": normalize_path(img_path),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nRead and transcribe the handwritten text from this prescription image accurately."
                        },
                        {
                            "from": "gpt",
                            "value": text_label
                        }
                    ]
                })

    return records


def merge_datasets(dhp_records: List[Dict], rxhand_records: List[Dict]) -> List[Dict]:
    """Merge DHP and RxHand records into single dataset."""
    merged = dhp_records + rxhand_records
    logger.info(f"Merged dataset: {len(merged)} total records")
    return merged


def split_dataset(
    records: List[Dict],
    train_split: float = 0.85,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and validation sets.

    Args:
        records: List of dataset records
        train_split: Fraction for training (default 0.85)
        val_split: Fraction for validation (default 0.15)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_records, val_records)
    """
    random.seed(seed)
    random.shuffle(records)

    split_idx = int(len(records) * train_split)
    train = records[:split_idx]
    val = records[split_idx:]

    logger.info(f"Train/Val split: {len(train)} train, {len(val)} val")
    return train, val


def save_dataset_json(records: List[Dict], output_path: str) -> None:
    """Save dataset records to JSON format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(records)} records to {output_path}")


def parse_all_datasets(cfg) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse all datasets (DHP + RxHand), merge, and split.

    Args:
        cfg: DatasetConfig object with all paths

    Returns:
        Tuple of (train_records, val_records)
    """
    dhp_records = load_dhp_dataset(cfg.dhp_labels, cfg.dhp_images)

    rxhand_train, rxhand_test = load_rxhand_dataset(
        cfg.rxhand_train_labels,
        cfg.rxhand_test_labels,
        cfg.rxhand_train_images,
        cfg.rxhand_test_images,
    )

    all_records = merge_datasets(dhp_records, rxhand_train + rxhand_test)

    if cfg.max_samples:
        all_records = all_records[:cfg.max_samples]
        logger.info(f"Limited dataset to {cfg.max_samples} samples")

    train_records, val_records = split_dataset(
        all_records,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=42
    )

    return train_records, val_records
