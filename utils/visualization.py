"""Training-curve plots (loss, accuracy, error rates, learning rate, every evaluation metric)
and final-result bar charts for a single run.

Used live by main.py (redrawn after every evaluation) and as a CLI to redraw any past run:
    python -m utils.visualization runs/<model_name>/<run_name> [more run dirs...]
"""
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

logger = logging.getLogger(__name__)

HISTORY_JSON = "metrics_history.json"
HISTORY_CSV = "metrics_history.csv"
SUMMARY_CSV = "metrics_summary.csv"

# Chart palette (light surface). Train is always blue and validation always orange,
# so the same split keeps its colour across every chart.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TRAIN = "#2a78d6"
VAL = "#eb6834"
VAL_ALT = "#1baf7a"

Series = Tuple[str, List[float], List[float], str, bool]  # label, steps, values, colour, markers

# Display names for evaluation metrics (TextMetrics.compute_batch_metrics + eval_loss/token_accuracy).
# Metrics not listed here are still plotted, after these, under their raw name.
METRIC_LABELS = {
    "loss": "Loss",
    "token_accuracy": "Token accuracy",
    "accuracy": "Exact match",
    "accuracy_normalized": "Exact match (normalized)",
    "char_accuracy": "Character accuracy",
    "cer": "CER",
    "wer": "WER",
    "mer": "MER",
    "wil": "WIL",
}
LOWER_IS_BETTER = {"loss", "cer", "wer", "mer", "wil"}
NOT_PERCENT = {"loss"}
# Trainer bookkeeping logged with every evaluation; not model quality metrics.
NON_METRICS = {"runtime", "samples_per_second", "steps_per_second", "epoch", "num_samples"}


def _label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _direction(metric: str) -> str:
    return "lower is better" if metric in LOWER_IS_BETTER else "higher is better"


def eval_metric_names(history: List[Dict]) -> List[str]:
    """Names (without the "eval_" prefix) of every numeric evaluation metric in the history."""
    found = set()
    for entry in history:
        if "eval_loss" in entry:
            found.update(k[5:] for k, v in entry.items()
                         if k.startswith("eval_") and isinstance(v, (int, float)))
    found -= NON_METRICS
    known = [m for m in METRIC_LABELS if m in found]
    return known + sorted(found - set(known))


def _series(history: List[Dict], key: str, split: str) -> Tuple[List[float], List[float]]:
    """(steps, values) of `key` from train logs ("loss" present) or eval logs ("eval_loss" present).
    Repeated steps (e.g. final evaluation at the last eval step) keep the latest value."""
    marker = "loss" if split == "train" else "eval_loss"
    points = {}
    for entry in history:
        if marker in entry and isinstance(entry.get(key), (int, float)):
            points[entry["step"]] = entry[key]
    steps = sorted(points)
    return steps, [points[s] for s in steps]


def _draw(ax, title: str, ylabel: str, series: List[Series], percent: bool = False,
          log_scale: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=10)
    ax.set_xlabel("Step", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    drawn = [s for s in series if s[1]]
    if not drawn:
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes, ha="center", va="center",
                color=TEXT_SECONDARY)
        return

    for label, steps, values, colour, markers in drawn:
        ax.plot(steps, values, color=colour, linewidth=2, label=label,
                marker="o" if markers else None, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        # Direct-label only the latest value of each line.
        shown = f"{values[-1]:.1%}" if percent else f"{values[-1]:.3g}"
        ax.annotate(shown, (steps[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT_PRIMARY)

    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if log_scale:
        ax.set_yscale("log")
    ax.margins(x=0.08)
    if len(drawn) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)


def _chart_specs(history: List[Dict]) -> Dict[str, dict]:
    return {
        "loss": dict(title="Loss", ylabel="Cross-entropy loss", series=[
            ("Train", *_series(history, "loss", "train"), TRAIN, False),
            ("Validation", *_series(history, "eval_loss", "eval"), VAL, True),
        ]),
        "accuracy": dict(title="Accuracy", ylabel="Accuracy", percent=True, series=[
            ("Train (token)", *_series(history, "token_accuracy", "train"), TRAIN, False),
            ("Validation (token)", *_series(history, "eval_token_accuracy", "eval"), VAL, True),
            ("Validation (exact match)", *_series(history, "eval_accuracy", "eval"), VAL_ALT, True),
        ]),
        "error_rates": dict(title="Validation error rates (lower is better)", ylabel="Error rate",
                            percent=True, series=[
            ("CER", *_series(history, "eval_cer", "eval"), TRAIN, True),
            ("WER", *_series(history, "eval_wer", "eval"), VAL, True),
        ]),
        "learning_rate": dict(title="Learning rate", ylabel="Learning rate", series=[
            ("Learning rate", *_series(history, "learning_rate", "train"), TRAIN, False),
        ]),
    }


def plot_evaluation_metrics(history: List[Dict], path: Path) -> bool:
    """Small multiples: one panel per evaluation metric over training steps."""
    metrics = eval_metric_names(history)
    if not metrics:
        return False
    cols = min(3, len(metrics))
    rows = math.ceil(len(metrics) / cols)
    fig = Figure(figsize=(5 * cols, 3.6 * rows), facecolor=SURFACE, layout="constrained")
    fig.suptitle("Validation metrics during training", x=0.01, ha="left", fontsize=14,
                 color=TEXT_PRIMARY)
    axes = list(fig.subplots(rows, cols, squeeze=False).flat)
    for ax, metric in zip(axes, metrics):
        _draw(ax, title=f"{_label(metric)} ({_direction(metric)})", ylabel="",
              percent=metric not in NOT_PERCENT,
              series=[(_label(metric), *_series(history, f"eval_{metric}", "eval"), VAL, True)])
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    fig.savefig(path, dpi=150)
    return True


def plot_final_metrics(metrics: Dict, path: str, title: str) -> bool:
    """Horizontal bar charts of one set of results (e.g. the final evaluation or the infer stage),
    split into "higher is better" and "lower is better" panels. Keys may carry an "eval_" prefix.
    Loss is left out: it is not a 0-1 rate, so it would distort the shared axis."""
    # A training-history entry also carries step/epoch/etc.: then only its eval_ keys are metrics.
    if any(k.startswith("eval_") for k in metrics):
        metrics = {k[5:]: v for k, v in metrics.items() if k.startswith("eval_")}
    values = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and name not in NON_METRICS | NOT_PERCENT:
            values[name] = value
    ordered = [m for m in METRIC_LABELS if m in values] + sorted(set(values) - set(METRIC_LABELS))
    groups = [(heading, [m for m in ordered if (m in LOWER_IS_BETTER) == lower])
              for heading, lower in (("Higher is better", False), ("Lower is better", True))]
    groups = [g for g in groups if g[1]]
    if not groups:
        return False

    heights = [len(names) for _, names in groups]
    fig = Figure(figsize=(9, 1.2 + 0.55 * sum(heights) + 0.8 * len(groups)),
                 facecolor=SURFACE, layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, color=TEXT_PRIMARY)
    axes = fig.subplots(len(groups), 1, squeeze=False, gridspec_kw={"height_ratios": heights})
    for ax, (heading, names) in zip(axes.flat, groups):
        labels = [_label(m) for m in names][::-1]
        shown = [values[m] for m in names][::-1]
        ax.set_facecolor(SURFACE)
        ax.barh(labels, shown, height=0.55, color=VAL)
        for y, v in enumerate(shown):
            ax.annotate(f"{v:.1%}", (v, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT_PRIMARY)
        ax.set_title(heading, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=8)
        ax.set_xlim(0, max(1.0, max(shown) * 1.12))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.tick_params(axis="y", colors=TEXT_PRIMARY, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.savefig(path, dpi=150)
    return True


def plot_training_curves(history: List[Dict], plots_dir: str) -> List[str]:
    """Write one PNG per chart plus a 2x2 overview, a panel per evaluation metric
    (evaluation_metrics.png) and the latest evaluation as bars (final_metrics.png)
    into plots_dir; return the saved paths."""
    out = Path(plots_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = _chart_specs(history)
    saved = []

    for name, spec in specs.items():
        fig = Figure(figsize=(8, 4.5), facecolor=SURFACE, layout="constrained")
        _draw(fig.add_subplot(), **spec)
        path = out / f"{name}.png"
        fig.savefig(path, dpi=150)
        saved.append(path.as_posix())

    fig = Figure(figsize=(14, 9), facecolor=SURFACE, layout="constrained")
    for ax, spec in zip(fig.subplots(2, 2).flat, specs.values()):
        _draw(ax, **spec)
    path = out / "training_overview.png"
    fig.savefig(path, dpi=150)
    saved.append(path.as_posix())

    path = out / "evaluation_metrics.png"
    if plot_evaluation_metrics(history, path):
        saved.append(path.as_posix())

    evals = [e for e in history if "eval_loss" in e]
    path = out / "final_metrics.png"
    if evals and plot_final_metrics(evals[-1], path,
                                    f"Validation metrics at step {evals[-1]['step']}"):
        saved.append(path.as_posix())
    return saved


def save_history(history: List[Dict], run_dir: str) -> None:
    """Write the raw log history as JSON and as a flat CSV (one row per log entry)."""
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    with open(run / HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    columns = ["step", "epoch"] + sorted({k for e in history for k in e} - {"step", "epoch"})
    with open(run / HISTORY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def summarize_history(history: List[Dict]) -> List[Dict]:
    """Best / average / last value (and the step they occurred at) for train loss and token
    accuracy and for every evaluation metric. "Best" is the minimum for lower-is-better metrics."""
    # (row name, history key, split, higher_is_better)
    summary_metrics = [
        ("train_loss", "loss", "train", False),
        ("train_token_accuracy", "token_accuracy", "train", True),
    ] + [(f"val_{m}", f"eval_{m}", "eval", m not in LOWER_IS_BETTER)
         for m in eval_metric_names(history)]

    rows = []
    for name, key, split, higher_is_better in summary_metrics:
        steps, values = _series(history, key, split)
        if not values:
            continue
        best = max(range(len(values)), key=values.__getitem__) if higher_is_better \
            else min(range(len(values)), key=values.__getitem__)
        rows.append({
            "metric": name,
            "best": values[best],
            "best_step": steps[best],
            "average": sum(values) / len(values),
            "last": values[-1],
            "last_step": steps[-1],
            "num_points": len(values),
        })
    return rows


def save_summary(history: List[Dict], run_dir: str) -> List[Dict]:
    """Write metrics_summary.csv (one row per metric) into the run folder."""
    rows = summarize_history(history)
    columns = ["metric", "best", "best_step", "average", "last", "last_step", "num_points"]
    with open(Path(run_dir) / SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_history(run_dir: str) -> Optional[List[Dict]]:
    """Read a run's metrics history, falling back to the newest checkpoint's trainer_state.json."""
    run = Path(run_dir)
    if (run / HISTORY_JSON).is_file():
        with open(run / HISTORY_JSON, encoding="utf-8") as f:
            return json.load(f)

    states = sorted((run / "checkpoints").glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[-1]))
    if states:
        with open(states[-1], encoding="utf-8") as f:
            return json.load(f).get("log_history", [])
    return None


def main(run_dirs: List[str]) -> int:
    if not run_dirs:
        print("Usage: python -m utils.visualization <run_dir> [<run_dir> ...]")
        return 1
    for run_dir in run_dirs:
        history = load_history(run_dir)
        if not history:
            print(f"[skip] no metrics history found in {run_dir}")
            continue
        for path in plot_training_curves(history, str(Path(run_dir) / "plots")):
            print(f"[saved] {path}")
        save_summary(history, run_dir)
        print(f"[saved] {(Path(run_dir) / SUMMARY_CSV).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
