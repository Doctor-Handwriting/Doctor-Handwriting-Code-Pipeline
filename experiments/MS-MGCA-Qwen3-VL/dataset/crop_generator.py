"""
Deterministic crop generator for Multi-Scale visual feature extraction.
Generates fixed-grid or overlapping local crops from prescription images.
"""

from typing import List, Tuple
from PIL import Image


class DeterministicCropGenerator:
    """
    Generates deterministic spatial crops from an image.
    Supports NxN fixed grids and overlapping regions.
    """

    def __init__(
        self,
        grid_size: int = 2,
        overlap_ratio: float = 0.15,
        include_center_crop: bool = True,
        target_crop_size: int = 224,
    ):
        """
        Args:
            grid_size: Grid division per axis (e.g. 2 -> 2x2 = 4 quadrants).
            overlap_ratio: Overlap fraction between neighboring crops (0.0 to 0.5).
            include_center_crop: Whether to include an additional center crop.
            target_crop_size: Output resolution (width and height) for each crop.
        """
        self.grid_size = grid_size
        self.overlap_ratio = overlap_ratio
        self.include_center_crop = include_center_crop
        self.target_crop_size = target_crop_size

    def get_crop_boxes(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculate deterministic bounding boxes (left, top, right, bottom).
        """
        boxes = []
        step_w = width / self.grid_size
        step_h = height / self.grid_size

        pad_w = step_w * self.overlap_ratio
        pad_h = step_h * self.overlap_ratio

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                left = max(0, int(col * step_w - pad_w))
                top = max(0, int(row * step_h - pad_h))
                right = min(width, int((col + 1) * step_w + pad_w))
                bottom = min(height, int((row + 1) * step_h + pad_h))
                boxes.append((left, top, right, bottom))

        if self.include_center_crop:
            c_left = int(width * 0.25)
            c_top = int(height * 0.25)
            c_right = int(width * 0.75)
            c_bottom = int(height * 0.75)
            boxes.append((c_left, c_top, c_right, c_bottom))

        return boxes

    def generate_crops(self, image: Image.Image) -> Tuple[List[Image.Image], List[Tuple[float, float, float, float]]]:
        """
        Extract crops from a PIL image.

        Returns:
            crops: List of cropped PIL images resized to target_crop_size x target_crop_size.
            norm_boxes: Normalized coordinates (x1, y1, x2, y2) in [0, 1].
        """
        w, h = image.size
        boxes = self.get_crop_boxes(w, h)
        crops = []
        norm_boxes = []

        for left, top, right, bottom in boxes:
            cropped = image.crop((left, top, right, bottom))
            if self.target_crop_size:
                cropped = cropped.resize(
                    (self.target_crop_size, self.target_crop_size),
                    Image.BICUBIC,
                )
            crops.append(cropped)
            norm_boxes.append((left / w, top / h, right / w, bottom / h))

        return crops, norm_boxes
