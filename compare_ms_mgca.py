"""Ablation report for MS-MGCA-Qwen3-VL (experiments E1-E6, and the K sweep).

Collects the latest finished run (one with inference_metrics.json) of every experiment under
runs/MS-MGCA-Qwen3-VL/ and writes, to runs/MS-MGCA-Qwen3-VL/ablation_report/:
    ablation_table.md / .csv     one row per run: text metrics, latency / VRAM, confidence metrics
    metrics_comparison.png       CER / WER and exact-match bars per experiment
    reliability_<run>.png        reliability diagram of every run that has a confidence head
    risk_coverage_comparison.png risk-coverage curves of those runs in one chart

Run it after the infer stage of the experiments you want to compare:
    python compare_ms_mgca.py                                  # every experiment found
    python compare_ms_mgca.py --dataset rxhand --size 2B       # only runs on one dataset / backbone size
Runs are only comparable when dataset, augmentation, backbone and seed match; the table shows all four.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import OUTPUT_ROOT, qwen_params_billions
from utils.confidence_plots import plot_reliability_diagram, plot_risk_coverage
from utils.visualization import GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, TRAIN, VAL

MODEL_DIR = "MS-MGCA-Qwen3-VL"
LABELS = {"e1_baseline": "E1", "e2_multiscale": "E2", "e3_cross_attention": "E3", "e4_confidence": "E4",
          "e5_multiscale_crossattn": "E5", "e6_full_model": "E6"}


def read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def describe(run: Path) -> Optional[Dict]:
    """One row of facts about a finished run, or None when it has no inference results."""
    metrics = read_json(run / "inference_metrics.json")
    config = read_json(run / "run_config.json")
    if not metrics or not config:
        return None
    model, dataset = config.get("model", {}), config.get("dataset", {})
    efficiency = read_json(run / "inference_efficiency.json") or {}
    confidence = read_json(run / "inference_confidence_metrics.json")
    size = qwen_params_billions(model.get("qwen_checkpoint", ""))
    experiment = model.get("ms_mgca_experiment", "?")
    return {
        "run": run.name, "path": run, "experiment": experiment, "label": LABELS.get(experiment, experiment),
        "K": model.get("ms_mgca_num_queries") or "", "size": f"{size:g}B" if size else "custom",
        "dataset": dataset.get("mode", "?"), "augmentation": config.get("augmentation", {}).get("mode", "?"),
        "seed": config.get("training", {}).get("seed", ""),
        "exact_match": metrics.get("accuracy"), "normalized_match": metrics.get("accuracy_normalized"),
        "cer": metrics.get("cer"), "wer": metrics.get("wer"),
        "latency_ms": efficiency.get("latency_ms_per_sample"), "peak_vram_mb": efficiency.get("peak_vram_mb"),
        "ece": confidence.get("ece") if confidence else None,
        "brier": confidence.get("brier_score") if confidence else None,
        "auroc": confidence.get("error_detection_auroc") if confidence else None,
        "review_rate": confidence.get("review_rate") if confidence else None,
        "accepted_exact_match": confidence.get("accepted_exact_match") if confidence else None,
        "confidence_on": confidence.get("evaluated_on", "") if confidence else "",
        "confidence": confidence,
    }


def latest_runs(root: Path, dataset: Optional[str], size: Optional[str]) -> List[Dict]:
    """Newest finished run per (experiment, K, backbone, dataset, augmentation)."""
    chosen: Dict[tuple, Dict] = {}
    for run in sorted(root.glob("run_*")):  # names start with a timestamp, so later runs overwrite earlier ones
        row = describe(run)
        if not row or (dataset and row["dataset"] != dataset) or (size and row["size"] != size):
            continue
        chosen[(row["experiment"], row["K"], row["size"], row["dataset"], row["augmentation"])] = row
    return sorted(chosen.values(), key=lambda r: (r["size"], r["dataset"], r["augmentation"], r["experiment"], str(r["K"])))


COLUMNS = [("label", "Exp"), ("K", "K"), ("size", "Backbone"), ("dataset", "Data"), ("augmentation", "Aug"),
           ("seed", "Seed"), ("exact_match", "Exact match"), ("normalized_match", "Norm. match"), ("cer", "CER"),
           ("wer", "WER"), ("latency_ms", "ms/sample"), ("peak_vram_mb", "Peak VRAM MB"), ("ece", "ECE"),
           ("brier", "Brier"), ("auroc", "Err. AUROC"), ("review_rate", "Review rate"),
           ("accepted_exact_match", "Acc. exact match")]


def fmt(value) -> str:
    if value is None or value == "":
        return "-"
    return f"{value:.4f}" if isinstance(value, float) and abs(value) < 100 else (f"{value:.0f}" if isinstance(value, float) else str(value))


def write_tables(rows: List[Dict], out: Path) -> None:
    with open(out / "ablation_table.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([title for _, title in COLUMNS] + ["Run", "Confidence scored on"])
        for r in rows:
            writer.writerow([fmt(r[key]) for key, _ in COLUMNS] + [r["run"], r["confidence_on"]])
    lines = ["| " + " | ".join(title for _, title in COLUMNS) + " |",
             "|" + "|".join("---" for _ in COLUMNS) + "|"]
    lines += ["| " + " | ".join(fmt(r[key]) for key, _ in COLUMNS) + " |" for r in rows]
    (out / "ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric_bars(rows: List[Dict], path: Path) -> bool:
    if not rows:
        return False
    names = [f"{r['label']}" + (f" K={r['K']}" if r["K"] else "") + (f"\n{r['size']}" if len({x['size'] for x in rows}) > 1 else "")
             for r in rows]
    fig = Figure(figsize=(max(7.5, 1.15 * len(rows) + 3), 4.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2)
    for ax, (heading, series) in zip(axes, (("Error rates (lower is better)", (("cer", "CER", VAL), ("wer", "WER", TRAIN))),
                                           ("Match rates (higher is better)", (("exact_match", "Exact", VAL),
                                                                              ("normalized_match", "Normalized", TRAIN))))):
        ax.set_facecolor(SURFACE)
        width = 0.38
        for i, (key, label, colour) in enumerate(series):
            xs = [j + (i - 0.5) * width for j in range(len(rows))]
            ax.bar(xs, [r[key] or 0 for r in rows], width=width, color=colour, label=label)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, fontsize=9, color=TEXT_PRIMARY)
        ax.set_title(heading, loc="left", fontsize=11, color=TEXT_PRIMARY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    fig.savefig(path, dpi=150)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Comparison table and plots over the MS-MGCA-Qwen3-VL runs.")
    p.add_argument("--runs-dir", default=str(Path(OUTPUT_ROOT) / MODEL_DIR), help="folder holding the run_* folders")
    p.add_argument("--dataset", choices=["combined", "dhp", "rxhand"], help="only runs on this dataset")
    p.add_argument("--size", help="only runs with this backbone size, e.g. 2B or 4B")
    args = p.parse_args()

    root = Path(args.runs_dir)
    rows = latest_runs(root, args.dataset, args.size)
    if not rows:
        print(f"No finished MS-MGCA runs (with inference_metrics.json) under {root}. "
              f"Train and infer first: python main.py --model ms-mgca --experiment e1_baseline --stage train / infer")
        return 1

    out = root / "ablation_report"
    out.mkdir(parents=True, exist_ok=True)
    write_tables(rows, out)
    plot_metric_bars(rows, out / "metrics_comparison.png")
    curves = {}
    for r in rows:
        if r["confidence"]:
            name = r["label"] + (f" K={r['K']}" if r["K"] else "") + f" {r['size']}"
            plot_reliability_diagram(r["confidence"], str(out / f"reliability_{r['run']}.png"), f"Reliability diagram - {name}")
            curves[name] = r["confidence"]
    plot_risk_coverage(curves, str(out / "risk_coverage_comparison.png"), "Risk-coverage: confidence-aware models")

    print((out / "ablation_table.md").read_text(encoding="utf-8"))
    print(f"Report written to {out}")
    mixed = {(r["size"], r["dataset"], r["augmentation"], r["seed"]) for r in rows}
    if len(mixed) > 1:
        print("NOTE: the rows differ in backbone / dataset / augmentation / seed, so they are not all directly comparable "
              "(use --dataset / --size to pick one setting).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
