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


OFFLINE_MANIFEST = "augmentation_manifest.json"


def get_augmented_dataset_paths(cfg, aug_cfg) -> Dict[str, Dict[str, str]]:
    """
    Offline-augmentation locations of the datasets selected by cfg.mode (DatasetConfig).

    Returns {"DHP": {"source_root", "dir", "manifest"}, "RxHand": {...}}. dir is
    <offline_root>/<dataset> and holds original/, augmented/ and the manifest.
    """
    sources = []
    if cfg.mode in ("combined", "dhp"):
        sources.append(("DHP", cfg.dhp_root))
    if cfg.mode in ("combined", "rxhand"):
        sources.append(("RxHand", cfg.rxhand_root))
    return {
        name: {
            "source_root": root,
            "dir": (Path(aug_cfg.offline_root) / name).as_posix(),
            "manifest": (Path(aug_cfg.offline_root) / name / OFFLINE_MANIFEST).as_posix(),
        }
        for name, root in sources
    }


def missing_offline_manifests(cfg, aug_cfg) -> List[str]:
    """Manifest files of the selected datasets that STAGE = "augment" has not created yet."""
    return [d["manifest"] for d in get_augmented_dataset_paths(cfg, aug_cfg).values()
            if not os.path.exists(d["manifest"])]


def _all_dataset_images(cfg) -> Dict[str, List[str]]:
    """Every usable image of each selected dataset, regardless of the train/val split."""
    images = {}
    if cfg.mode in ("combined", "dhp"):
        images["DHP"] = [r["image"] for r in load_dhp_dataset(cfg.dhp_labels, cfg.dhp_images)]
    if cfg.mode in ("combined", "rxhand"):
        train, test = load_rxhand_dataset(
            cfg.rxhand_train_labels, cfg.rxhand_test_labels, cfg.rxhand_train_images, cfg.rxhand_test_images)
        images["RxHand"] = [r["image"] for r in train + test]
    return images


def _relative_image_path(image: str, source_root: str) -> str:
    """Image path relative to its dataset root, with forward slashes ("Train_Set/P0001.jpg")."""
    return Path(image).relative_to(Path(source_root)).as_posix()


def build_offline_dataset(cfg, aug_cfg) -> None:
    """
    Pre-compute the offline augmented dataset for the datasets selected by cfg.mode:

        <offline_root>/<DHP|RxHand>/original/...    copy of every original image
        <offline_root>/<DHP|RxHand>/augmented/...   <name>_<variant>.png for every variant
        <offline_root>/<DHP|RxHand>/augmentation_manifest.json

    Covers every image of the dataset (not just today's training split), so changing the split or
    max_samples later does not require regenerating.
    """
    from utils.augmentor import generate_offline_augmentations, offline_settings

    if not aug_cfg.enable_augmentation:
        raise ValueError("Offline augmentation needs enable_augmentation = True; nothing to generate.")

    paths = get_augmented_dataset_paths(cfg, aug_cfg)
    all_images = _all_dataset_images(cfg)
    total_files = 0
    total_bytes = 0

    for name, info in paths.items():
        pairs = [(img, _relative_image_path(img, info["source_root"])) for img in all_images[name]]
        manifest = generate_offline_augmentations(aug_cfg, pairs, info["dir"], label=name)
        with open(info["manifest"], "w", encoding="utf-8") as f:
            json.dump({
                "dataset": name,
                "source_root": normalize_path(info["source_root"]),
                "settings": offline_settings(aug_cfg),
                "images": manifest,
            }, f, indent=2, ensure_ascii=False)

        folder_files = [os.path.join(d, f) for d, _, fs in os.walk(info["dir"]) for f in fs]
        folder_bytes = sum(os.path.getsize(p) for p in folder_files)
        total_files += len(folder_files)
        total_bytes += folder_bytes
        logger.info(f"{name}: {len(pairs)} originals + {sum(len(v) for v in manifest.values())} augmented "
                    f"in {info['dir']} ({folder_bytes / 1024**2:.1f} MB)")

    logger.info(f"Offline augmented dataset ready: {total_files} files, {total_bytes / 1024**2:.1f} MB "
                f"in {aug_cfg.offline_root}")


def apply_offline_dataset(cfg, aug_cfg, train_records: List[Dict],
                          val_records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Point train/val records at the offline dataset.

    The split is made on the original images first, so no augmented copy of a validation image can reach
    training. Training gets each original (from original/) plus all its variants; validation gets only
    the originals, so metrics stay comparable with online runs.
    """
    paths = get_augmented_dataset_paths(cfg, aug_cfg)
    missing = missing_offline_manifests(cfg, aug_cfg)
    if missing:
        raise FileNotFoundError(
            "Offline augmentation selected but the augmented dataset was not found:\n  " + "\n  ".join(missing)
            + "\nRun with STAGE = \"augment\" (same DATASET_MODE) first, or set AUGMENTATION_MODE = \"online\".")

    from utils.augmentor import offline_settings
    manifests = {}
    for name, info in paths.items():
        with open(info["manifest"], "r", encoding="utf-8") as f:
            manifests[name] = json.load(f)
        if manifests[name].get("settings") != offline_settings(aug_cfg):
            logger.warning(f"{name}: augmentation settings changed since {info['manifest']} was generated "
                           f"(existing images are used as they are). Delete {info['dir']} and re-run "
                           f"STAGE = \"augment\" to regenerate.")

    def locate(record: Dict) -> Tuple[str, str]:
        for name, info in paths.items():
            try:
                return name, _relative_image_path(record["image"], info["source_root"])
            except ValueError:
                continue
        raise ValueError(f"{record['image']} is not inside a selected dataset")

    def redirected(record: Dict, image_path: str, suffix: str = "") -> Dict:
        return {**record, "id": record["id"] + suffix, "image": normalize_path(image_path)}

    def expand(records: List[Dict], with_variants: bool) -> List[Dict]:
        originals, variants, absent, unlisted = [], [], [], []
        for record in records:
            name, rel = locate(record)
            ds_dir = paths[name]["dir"]
            original = os.path.join(ds_dir, "original", rel)
            if rel not in manifests[name]["images"] or not os.path.exists(original):
                unlisted.append(record["image"])
                continue
            originals.append(redirected(record, original))
            if with_variants:
                for variant_rel in manifests[name]["images"][rel]:
                    variant = os.path.join(ds_dir, variant_rel)
                    if os.path.exists(variant):
                        variant_name = os.path.splitext(variant_rel)[0].rsplit("_", 1)[1]
                        variants.append(redirected(record, variant, f"__{variant_name}"))
                    else:
                        absent.append(variant)
        if unlisted or absent:
            examples = (unlisted + absent)[:3]
            raise FileNotFoundError(
                f"Offline dataset is incomplete: {len(unlisted)} images not in the manifest, "
                f"{len(absent)} variant files missing (e.g. {examples}). Re-run STAGE = \"augment\".")
        return originals + variants

    train_out = expand(train_records, with_variants=True)
    val_out = expand(val_records, with_variants=False)
    logger.info(f"Offline augmentation: train {len(train_records)} originals -> {len(train_out)} images "
                f"(originals + variants), val {len(val_out)} originals (no variants)")
    return train_out, val_out


def parse_all_datasets(cfg) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse the datasets selected by cfg.mode and split them into train/val.

    Args:
        cfg: DatasetConfig object with all paths. cfg.mode:
            "combined" - DHP + RxHand merged into one pool, one split (default)
            "dhp"      - DHP only, split on its own
            "rxhand"   - RxHand only; split on its own, or with cfg.rxhand_official_split
                         train = Train_Set, val = Test_Set

    Returns:
        Tuple of (train_records, val_records)
    """
    mode = cfg.mode
    logger.info(f"Dataset mode: {mode}" + (" (official RxHand train/test split)" if cfg.rxhand_official_split else ""))

    dhp_records, rxhand_train, rxhand_test = [], [], []
    if mode in ("combined", "dhp"):
        dhp_records = load_dhp_dataset(cfg.dhp_labels, cfg.dhp_images)
    if mode in ("combined", "rxhand"):
        rxhand_train, rxhand_test = load_rxhand_dataset(
            cfg.rxhand_train_labels,
            cfg.rxhand_test_labels,
            cfg.rxhand_train_images,
            cfg.rxhand_test_images,
        )

    if cfg.rxhand_official_split:
        if not rxhand_train or not rxhand_test:
            raise ValueError("Official RxHand split needs both Train_Set and Test_Set records, found "
                             f"{len(rxhand_train)} train / {len(rxhand_test)} test")
        train_records, val_records = rxhand_train, rxhand_test
        if cfg.max_samples:
            n_train = max(1, int(cfg.max_samples * cfg.train_split))
            train_records = train_records[:n_train]
            val_records = val_records[:max(1, cfg.max_samples - n_train)]
            logger.info(f"Limited dataset to {cfg.max_samples} samples")
        logger.info(f"Official RxHand split: {len(train_records)} train, {len(val_records)} val")
        return train_records, val_records

    all_records = merge_datasets(dhp_records, rxhand_train + rxhand_test)
    if not all_records:
        raise ValueError(f"No records found for dataset mode '{mode}'; check the dataset paths.")

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
