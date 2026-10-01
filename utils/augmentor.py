import albumentations as A
import cv2
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)


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
            self.transform = A.Compose([
                A.MotionBlur(
                    blur_limit=(3, 7),
                    p=cfg.motion_blur_prob
                ),
                A.ElasticTransform(
                    alpha=50,
                    sigma=4,
                    p=cfg.elastic_deform_prob
                ),
                A.Perspective(
                    scale=(0.05, 0.1),
                    p=cfg.perspective_prob
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=cfg.brightness_contrast_prob
                ),
                A.Rotate(
                    limit=cfg.rotation_limit,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.3
                ),
                A.ShiftScaleRotate(
                    shift_limit=cfg.shift_limit,
                    scale_limit=cfg.scale_limit,
                    rotate_limit=cfg.rotation_limit,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.3
                ),
            ], bbox_params=None)
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
