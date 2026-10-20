import albumentations as A
import cv2
import logging
import os
import shutil
import zlib
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Offline variant names, in the order they are applied by the "mixed" variant.
VARIANT_NAMES = ("motionblur", "elasticdeform", "perspective", "brightnesscontrast", "rotate", "shiftscalerotate")
MIXED_NAME = "mixed"


def _build_transforms(cfg, always: bool = False) -> Dict[str, A.BasicTransform]:
    """The six transforms by variant name. always=True forces p=1 (offline: one variant per transform)."""
    def p(prob: float) -> float:
        return 1.0 if always else prob

    return {
        "motionblur": A.MotionBlur(blur_limit=(3, 7), p=p(cfg.motion_blur_prob)),
        "elasticdeform": A.ElasticTransform(alpha=50, sigma=4, p=p(cfg.elastic_deform_prob)),
        "perspective": A.Perspective(scale=(0.05, 0.1), p=p(cfg.perspective_prob)),
        "brightnesscontrast": A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=p(cfg.brightness_contrast_prob)),
        "rotate": A.Rotate(limit=cfg.rotation_limit, border_mode=cv2.BORDER_CONSTANT, p=p(0.3)),
        "shiftscalerotate": A.ShiftScaleRotate(
            shift_limit=cfg.shift_limit, scale_limit=cfg.scale_limit, rotate_limit=cfg.rotation_limit,
            border_mode=cv2.BORDER_CONSTANT, p=p(0.3)),
    }


class MedicalHandwritingAugmentor:
    """
    CPU-based image augmentation tailored to medical handwriting.
    Simulates real-world variations: lighting, perspective, motion blur.
    """

    def __init__(self, cfg):
        """
        Initialize augmentation pipeline.

        Args:
            cfg: AugmentationConfig object
        """
        self.cfg = cfg
        self.enabled = cfg.enable_augmentation

        if self.enabled:
            self.transform = A.Compose(list(_build_transforms(cfg).values()), bbox_params=None)
        else:
            self.transform = None
            logger.info("Augmentation disabled")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to image.

        Args:
            image: numpy array in (H, W, C) format

        Returns:
            Augmented image as numpy array
        """
        if not self.enabled or self.transform is None:
            return image

        return self.transform(image=image)["image"]

    def augment_batch(self, images: list) -> list:
        """
        Apply augmentation to batch of images.

        Args:
            images: List of numpy arrays

        Returns:
            List of augmented images
        """
        return [self.__call__(img) for img in images]


def get_augmentor(cfg):
    """Factory function to create augmentor instance."""
    return MedicalHandwritingAugmentor(cfg)


def offline_variant_names(cfg) -> List[str]:
    """Variant names written per image in offline mode (six single transforms, plus "mixed" if enabled)."""
    return list(VARIANT_NAMES) + ([MIXED_NAME] if cfg.offline_include_mixed else [])


def offline_settings(cfg) -> Dict:
    """Everything that changes the pixels of the generated variants; stored in the manifest."""
    return {
        "variants": offline_variant_names(cfg),
        "seed": cfg.offline_seed,
        "rotation_limit": cfg.rotation_limit,
        "shift_limit": cfg.shift_limit,
        "scale_limit": cfg.scale_limit,
    }


def generate_offline_augmentations(cfg, images: List[Tuple[str, str]], dataset_dir: str,
                                   label: str = "") -> Dict[str, List[str]]:
    """
    Copy originals and write every augmented variant of them to disk.

    Args:
        cfg: AugmentationConfig
        images: (source image path, path relative to the dataset root) for every image
        dataset_dir: <offline_root>/<dataset>; gets original/<rel> (copy) and augmented/<rel-dir>/<stem>_<variant>.png
        label: dataset name for log messages

    Returns:
        {rel: [variant paths relative to dataset_dir]}. Files that already exist are kept, so an interrupted
        run resumes; delete the folder to regenerate after changing the augmentation settings.
    """
    names = offline_variant_names(cfg)
    manifest: Dict[str, List[str]] = {}
    written = 0

    for src, rel in tqdm(images, desc=f"Augmenting {label}".strip(), dynamic_ncols=True):
        original_path = os.path.join(dataset_dir, "original", rel)
        if not os.path.exists(original_path):
            os.makedirs(os.path.dirname(original_path), exist_ok=True)
            shutil.copy2(src, original_path)

        rel_dir, filename = os.path.split(rel)
        stem = os.path.splitext(filename)[0]
        variant_rels, pending = [], []
        for name in names:
            variant_rel = "/".join(p for p in ("augmented", rel_dir, f"{stem}_{name}.png") if p)
            variant_rels.append(variant_rel)
            if not os.path.exists(os.path.join(dataset_dir, variant_rel)):
                pending.append((name, variant_rel))
        manifest[rel] = variant_rels
        if not pending:
            continue

        image = np.array(Image.open(src).convert("RGB"))
        for name, variant_rel in pending:
            # Seeded per (image, variant): identical output on every run and independent of processing order.
            seed = zlib.crc32(f"{cfg.offline_seed}|{rel}|{name}".encode("utf-8"))
            transforms = _build_transforms(cfg, always=True)
            chain = list(transforms.values()) if name == MIXED_NAME else [transforms[name]]
            augmented = A.Compose(chain, seed=seed)(image=image)["image"]
            out_path = os.path.join(dataset_dir, variant_rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(augmented).save(out_path)
            written += 1

    logger.info(f"{label}: {len(images)} images -> {len(images) * len(names)} variants "
                f"({len(names)} per image, {written} newly written)")
    return manifest
