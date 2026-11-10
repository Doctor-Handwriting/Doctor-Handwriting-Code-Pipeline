"""
Inference runner for MS-MGCA-Qwen3-VL.
Generates medication name transcription, calibrated confidence score,
and clinical requires_review flag matching Section 5.5 specifications.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from configs.base_config import MSMGCAConfig
from models.proposed_model import MSMGCAQwen3VL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def predict_single(
    model: MSMGCAQwen3VL,
    image_path: Union[str, Path],
    threshold: Optional[float] = None,
) -> Dict:
    """
    Run inference on a single prescription image.

    Returns:
        {
            "image": "P1116.jpg",
            "prediction": "Dexter",
            "confidence": 0.94,
            "requires_review": False
        }
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    res = model.generate_with_confidence(image)

    if threshold is not None:
        requires_review = bool(res["confidence"] < threshold)
    else:
        requires_review = res["requires_review"]

    return {
        "image": img_path.name,
        "prediction": res["prediction"],
        "confidence": float(round(res["confidence"], 4)),
        "requires_review": requires_review,
    }


def predict_batch(
    model: MSMGCAQwen3VL,
    image_paths: List[Union[str, Path]],
    threshold: Optional[float] = None,
) -> List[Dict]:
    """Run inference on a list of images."""
    return [predict_single(model, p, threshold=threshold) for p in image_paths]


def main():
    parser = argparse.ArgumentParser(description="Run MS-MGCA-Qwen3-VL Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to prescription image")
    parser.add_argument("--config", type=str, default="configs/e6_full_model.yaml", help="Path to config YAML")
    parser.add_argument("--threshold", type=float, default=None, help="Override confidence threshold")
    args = parser.parse_args()

    cfg = MSMGCAConfig.from_yaml(str(EXP_ROOT / args.config))
    model = MSMGCAQwen3VL(cfg, for_training=False)

    result = predict_single(model, args.image, threshold=args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
