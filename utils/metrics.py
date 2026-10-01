import logging
from typing import List, Dict, Tuple
import jiwer
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class TextMetrics:
    """Calculate Word Error Rate (WER) and Character Error Rate (CER)."""

    @staticmethod
    def compute_wer(predictions: List[str], references: List[str]) -> float:
        """
        Compute Word Error Rate (WER).
        Lower is better. WER = (S + D + I) / N where S=substitution, D=deletion, I=insertion.
        """
        if not predictions or not references:
            return 0.0

        predictions = [str(p).strip() for p in predictions]
        references = [str(r).strip() for r in references]

        wer = jiwer.wer(references, predictions)
        return wer

    @staticmethod
    def compute_cer(predictions: List[str], references: List[str]) -> float:
        """
        Compute Character Error Rate (CER).
        Lower is better. CER = (S + D + I) / N at character level.
        """
        if not predictions or not references:
            return 0.0

        predictions = [str(p).strip() for p in predictions]
        references = [str(r).strip() for r in references]

        cer = jiwer.cer(references, predictions)
        return cer

    @staticmethod
    def compute_accuracy(predictions: List[str], references: List[str]) -> float:
        """
        Compute exact match accuracy.
        """
        if not predictions or not references:
            return 0.0

        predictions = [str(p).strip() for p in predictions]
        references = [str(r).strip() for r in references]

        matches = sum(p == r for p, r in zip(predictions, references))
        accuracy = matches / len(references)
        return accuracy

    @staticmethod
    def compute_batch_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute all text metrics at once."""
        return {
            "wer": TextMetrics.compute_wer(predictions, references),
            "cer": TextMetrics.compute_cer(predictions, references),
            "accuracy": TextMetrics.compute_accuracy(predictions, references),
        }


class DetectionMetrics:
    """Calculate mAP and other detection metrics."""

    @staticmethod
    def compute_iou(box1: Tuple[float, float, float, float],
                    box2: Tuple[float, float, float, float]) -> float:
        """
        Compute Intersection over Union (IoU) for bounding boxes.
        Format: (x1, y1, x2, y2) with top-left and bottom-right corners.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    @staticmethod
    def compute_ap(predictions: List[Dict], references: List[Dict], iou_threshold: float = 0.5) -> float:
        """
        Compute Average Precision (AP) at given IoU threshold.

        Args:
            predictions: List of dicts with 'boxes' and 'scores'
            references: List of dicts with 'boxes'
            iou_threshold: IoU threshold for match

        Returns:
            AP score
        """
        if not predictions or not references:
            return 0.0

        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))

        ref_matched = defaultdict(bool)

        for pred_idx, pred in enumerate(predictions):
            best_iou = 0.0
            best_ref_idx = -1

            for ref_idx, ref in enumerate(references):
                if ref_matched[ref_idx]:
                    continue

                iou = DetectionMetrics.compute_iou(pred["box"], ref["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_ref_idx = ref_idx

            if best_iou >= iou_threshold and best_ref_idx >= 0:
                tp[pred_idx] = 1.0
                ref_matched[best_ref_idx] = True
            else:
                fp[pred_idx] = 1.0

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / len(references) if len(references) > 0 else np.zeros_like(tp_cumsum)

        ap = np.trapz(precision, recall) if len(precision) > 0 else 0.0
        return float(ap)


class TrainingMetrics:
    """Track and log training metrics."""

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.history[key].append(float(value))

    def get_last(self, key: str) -> float:
        """Get last recorded value for a metric."""
        return self.history[key][-1] if key in self.history and self.history[key] else 0.0

    def get_avg(self, key: str, last_n: int = None) -> float:
        """Get average of last N values."""
        if key not in self.history or not self.history[key]:
            return 0.0

        values = self.history[key]
        if last_n:
            values = values[-last_n:]

        return np.mean(values) if values else 0.0

    def log_summary(self, epoch: int) -> Dict[str, float]:
        """Get summary of all metrics for an epoch."""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[f"{key}_last"] = values[-1]
                summary[f"{key}_avg"] = np.mean(values)

        logger.info(f"[Epoch {epoch}] {summary}")
        return summary
