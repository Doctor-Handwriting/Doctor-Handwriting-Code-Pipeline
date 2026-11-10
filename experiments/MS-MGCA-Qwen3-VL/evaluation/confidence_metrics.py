"""
Confidence and Uncertainty Evaluation Metrics.
Pure NumPy implementations of Brier Score, Expected Calibration Error (ECE),
AUROC for error detection, and Risk-Coverage curve.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_brier_score(
    confidences: Union[List[float], np.ndarray],
    targets: Union[List[int], np.ndarray],
) -> float:
    """
    Brier score: Mean squared error of predicted probabilities.
    BS = (1/N) * sum (c_i - z_i)^2
    Lower is better (0.0 = perfect calibration).
    """
    c = np.asarray(confidences, dtype=np.float64)
    z = np.asarray(targets, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    return float(np.mean((c - z) ** 2))


def compute_ece(
    confidences: Union[List[float], np.ndarray],
    targets: Union[List[int], np.ndarray],
    num_bins: int = 10,
) -> Tuple[float, List[Dict]]:
    """
    Expected Calibration Error (ECE):
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Returns:
        (ece_score, bin_details_list)
    """
    c = np.asarray(confidences, dtype=np.float64)
    z = np.asarray(targets, dtype=np.float64)
    n = len(c)
    if n == 0:
        return 0.0, []

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Include upper boundary in the last bin
        if i == num_bins - 1:
            in_bin = (c >= bin_lower) & (c <= bin_upper)
        else:
            in_bin = (c >= bin_lower) & (c < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(z[in_bin]))
            bin_conf = float(np.mean(c[in_bin]))
            bin_gap = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * bin_gap

            bin_details.append({
                "bin_idx": i,
                "range": [float(bin_lower), float(bin_upper)],
                "count": bin_count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": bin_gap,
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "range": [float(bin_lower), float(bin_upper)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "gap": 0.0,
            })

    return float(ece), bin_details


def compute_auroc(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic curve (AUROC)
    using the Wilcoxon-Mann-Whitney rank-sum formula.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined when only one class is present

    # Ranks of scores (handling ties by average rank)
    ranks = np.empty_like(scores)
    order = np.argsort(scores)
    sorted_scores = scores[order]

    unique_vals, unique_indices, unique_counts = np.unique(
        sorted_scores, return_index=True, return_counts=True
    )
    for idx, count in zip(unique_indices, unique_counts):
        # Average rank for ties (1-based ranking)
        avg_rank = idx + 1 + (count - 1) / 2.0
        ranks[order[idx : idx + count]] = avg_rank

    sum_pos_ranks = np.sum(ranks[pos_mask])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)
    return float(np.clip(auroc, 0.0, 1.0))


def compute_error_detection_auroc(
    confidences: Union[List[float], np.ndarray],
    targets: Union[List[int], np.ndarray],
) -> float:
    """
    AUROC for detecting incorrect predictions.
    Error indicator: y_err = 1 if target == 0, else 0.
    Uncertainty score: u = 1 - c.
    Higher is better (1.0 = perfect separation between correct and incorrect).
    """
    c = np.asarray(confidences, dtype=np.float64)
    z = np.asarray(targets, dtype=np.int32)
    y_err = (z == 0).astype(np.int32)
    u_score = 1.0 - c
    return compute_auroc(u_score, y_err)


def compute_risk_coverage(
    confidences: Union[List[float], np.ndarray],
    targets: Union[List[int], np.ndarray],
    num_thresholds: int = 21,
) -> List[Dict]:
    """
    Compute selective classification Risk-Coverage curve.
    Threshold tau varies in [0.0, 1.0].
    - Coverage = |{i: c_i >= tau}| / N
    - Risk (Selective Error Rate) = |{i: c_i >= tau and z_i == 0}| / |{i: c_i >= tau}|
    """
    c = np.asarray(confidences, dtype=np.float64)
    z = np.asarray(targets, dtype=np.int32)
    n = len(c)
    if n == 0:
        return []

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    curve = []

    for tau in thresholds:
        accepted = (c >= tau)
        n_accepted = int(np.sum(accepted))

        if n_accepted > 0:
            coverage = float(n_accepted / n)
            # Risk is error rate on accepted samples
            errors_accepted = int(np.sum(z[accepted] == 0))
            risk = float(errors_accepted / n_accepted)
        else:
            coverage = 0.0
            risk = 0.0

        curve.append({
            "threshold": float(round(tau, 3)),
            "coverage": float(round(coverage, 4)),
            "risk": float(round(risk, 4)),
            "num_accepted": n_accepted,
            "num_errors": int(np.sum(z[accepted] == 0)) if n_accepted > 0 else 0,
        })

    return curve


def compute_all_confidence_metrics(
    confidences: Union[List[float], np.ndarray],
    targets: Union[List[int], np.ndarray],
    threshold: float = 0.70,
) -> Dict:
    """
    Compute comprehensive suite of uncertainty and calibration metrics.
    """
    c = np.asarray(confidences, dtype=np.float64)
    z = np.asarray(targets, dtype=np.int32)
    n = len(c)

    brier = compute_brier_score(c, z)
    ece, ece_bins = compute_ece(c, z)
    auroc = compute_error_detection_auroc(c, z)
    rc_curve = compute_risk_coverage(c, z)

    # Threshold evaluation at operational point tau
    requires_review = (c < threshold)
    accepted = ~requires_review

    n_review = int(np.sum(requires_review))
    n_accepted = int(np.sum(accepted))
    review_rate = float(n_review / n) if n > 0 else 0.0

    accepted_em = float(np.mean(z[accepted])) if n_accepted > 0 else 0.0
    accepted_risk = float(1.0 - accepted_em)

    # Review precision: when review is triggered, how often was the prediction actually wrong?
    if n_review > 0:
        review_precision = float(np.mean(z[requires_review] == 0))
    else:
        review_precision = 0.0

    return {
        "num_samples": n,
        "brier_score": float(round(brier, 4)),
        "ece": float(round(ece, 4)),
        "error_detection_auroc": float(round(auroc, 4)),
        "operational_threshold": float(threshold),
        "review_rate": float(round(review_rate, 4)),
        "accepted_samples": n_accepted,
        "accepted_exact_match": float(round(accepted_em, 4)),
        "accepted_risk": float(round(accepted_risk, 4)),
        "review_precision": float(round(review_precision, 4)),
        "ece_bins": ece_bins,
        "risk_coverage_curve": rc_curve,
    }
