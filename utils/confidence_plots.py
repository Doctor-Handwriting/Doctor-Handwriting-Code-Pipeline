"""Reliability diagram and risk-coverage curve for the MS-MGCA confidence head.

Both take the dict produced by compute_all_confidence_metrics (ece_bins, risk_coverage_curve), so they work on
`inference_confidence_metrics.json` of any run:
    plot_reliability_diagram(report, path, title)
    plot_risk_coverage({"E4": report_e4, "E6": report_e6}, path, title)
"""
from typing import Dict

from matplotlib.figure import Figure

from utils.visualization import GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, TRAIN, VAL, VAL_ALT

SERIES_COLOURS = [VAL, TRAIN, VAL_ALT, "#8a5cd0", "#c23b6b", "#6b6a66"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_reliability_diagram(report: Dict, path: str, title: str) -> bool:
    """Bars = observed accuracy per confidence bin, diagonal = perfect calibration. Bin sizes are written above the
    bars, so a few-sample bin is not mistaken for a reliable one."""
    bins = [b for b in report.get("ece_bins", []) if b["count"] > 0]
    if not bins:
        return False
    fig = Figure(figsize=(6.2, 6.2), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_SECONDARY, linewidth=1, label="Perfect calibration")
    centres = [(b["range"][0] + b["range"][1]) / 2 for b in bins]
    width = bins[0]["range"][1] - bins[0]["range"][0]
    ax.bar(centres, [b["accuracy"] for b in bins], width=width * 0.9, color=VAL, alpha=0.85, label="Observed accuracy")
    for x, b in zip(centres, bins):
        ax.annotate(f"n={b['count']}", (x, b["accuracy"]), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT_PRIMARY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Confidence", color=TEXT_PRIMARY)
    ax.set_ylabel("Accuracy (exact match)", color=TEXT_PRIMARY)
    ax.set_title(f"{title}\nECE {report.get('ece', float('nan')):.3f} | Brier {report.get('brier_score', float('nan')):.3f}"
                 f" | AUROC {report.get('error_detection_auroc', float('nan')):.3f} | n={report.get('num_samples', 0)}",
                 loc="left", fontsize=11, color=TEXT_PRIMARY)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=TEXT_PRIMARY)
    fig.savefig(path, dpi=150)
    return True


def plot_risk_coverage(reports: Dict[str, Dict], path: str, title: str) -> bool:
    """Selective error rate (risk) against the share of predictions accepted (coverage), one line per run.
    Lower is better: a good confidence head keeps risk low until coverage gets high."""
    drawn = {name: r["risk_coverage_curve"] for name, r in reports.items() if r and r.get("risk_coverage_curve")}
    if not drawn:
        return False
    fig = Figure(figsize=(7.2, 5.2), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    for colour, (name, curve) in zip(SERIES_COLOURS, drawn.items()):
        points = sorted((p["coverage"], p["risk"]) for p in curve if p["coverage"] > 0)
        ax.plot([c for c, _ in points], [r for _, r in points], marker="o", markersize=3.5, linewidth=1.8,
                color=colour, label=name)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage (share of predictions accepted)", color=TEXT_PRIMARY)
    ax.set_ylabel("Risk (error rate of accepted predictions)", color=TEXT_PRIMARY)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    fig.savefig(path, dpi=150)
    return True
