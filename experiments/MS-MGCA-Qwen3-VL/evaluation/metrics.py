"""
Text recognition evaluation metrics for doctor's handwriting transcription.
Implements Character Error Rate (CER), Word Error Rate (WER), Exact Match (EM),
Normalized Exact Match, and Character Accuracy.
"""

from typing import Dict, List, Optional, Tuple, Union
import jiwer
import numpy as np


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and standardizing whitespace."""
    return " ".join(str(text).lower().split())


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate: (S + D + I) / N at character level.
    Lower is better.
    """
    if not predictions or not references:
        return 0.0

    preds = [str(p).strip() for p in predictions]
    refs = [str(r).strip() if str(r).strip() else " " for r in references]

    return float(jiwer.cer(refs, preds))


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate: (S + D + I) / N at word level.
    Lower is better.
    """
    if not predictions or not references:
        return 0.0

    preds = [str(p).strip() for p in predictions]
    refs = [str(r).strip() if str(r).strip() else "<empty>" for r in references]

    return float(jiwer.wer(refs, preds))


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match accuracy: percentage of predictions identical to references.
    Higher is better.
    """
    if not predictions or not references:
        return 0.0

    matches = sum(str(p).strip() == str(r).strip() for p, r in zip(predictions, references))
    return float(matches / len(references))


def compute_normalized_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute case-insensitive, whitespace-normalized exact match accuracy.
    Higher is better.
    """
    if not predictions or not references:
        return 0.0

    matches = sum(normalize_text(p) == normalize_text(r) for p, r in zip(predictions, references))
    return float(matches / len(references))


def compute_character_accuracy(cer: float) -> float:
    """
    Character-level accuracy defined as max(0.0, 1.0 - CER).
    Higher is better.
    """
    return float(max(0.0, 1.0 - cer))


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    image_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Compute full metric suite and per-sample results.

    Returns:
        (metrics_summary_dict, per_sample_results_list)
    """
    if not predictions or not references:
        return {}, []

    num_samples = len(references)
    cer = compute_cer(predictions, references)
    wer = compute_wer(predictions, references)
    exact_match = compute_exact_match(predictions, references)
    norm_match = compute_normalized_match(predictions, references)
    char_acc = compute_character_accuracy(cer)

    num_exact_matches = int(round(exact_match * num_samples))
    num_norm_matches = int(round(norm_match * num_samples))

    summary = {
        "num_samples": num_samples,
        "cer": cer,
        "wer": wer,
        "exact_match": exact_match,
        "normalized_match": norm_match,
        "character_accuracy": char_acc,
        "num_correct_exact": num_exact_matches,
        "num_correct_normalized": num_norm_matches,
    }

    per_sample = []
    ids = image_ids or [f"sample_{i}" for i in range(num_samples)]
    for img_id, pred, ref in zip(ids, predictions, references):
        p_str = str(pred).strip()
        r_str = str(ref).strip()
        is_em = (p_str == r_str)
        is_nem = (normalize_text(p_str) == normalize_text(r_str))
        sample_cer = float(jiwer.cer(r_str if r_str else " ", p_str))

        per_sample.append({
            "image_id": img_id,
            "prediction": p_str,
            "reference": r_str,
            "exact_match": is_em,
            "normalized_match": is_nem,
            "cer": sample_cer,
        })

    return summary, per_sample
