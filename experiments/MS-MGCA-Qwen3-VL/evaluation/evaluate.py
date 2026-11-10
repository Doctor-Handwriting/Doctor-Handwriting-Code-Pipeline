"""
Standalone evaluation runner for MS-MGCA-Qwen3-VL.
Measures CER, WER, Exact Match, Character Accuracy, Latency, and Peak GPU Memory.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from configs.base_config import MSMGCAConfig
from dataset.rxhand_dataset import get_rxhand_datasets
from models.baseline_qwen3vl import load_qwen3vl_baseline, Qwen3VLBaselineRunner
from evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    processor,
    dataset,
    output_dir: Optional[str] = None,
    max_new_tokens: int = 32,
    batch_desc: str = "Evaluating",
) -> Dict:
    """
    Run evaluation loop over a dataset.
    Measures recognition metrics, latency, throughput, and peak VRAM.
    """
    runner = Qwen3VLBaselineRunner(model, processor)
    device = next(model.parameters()).device

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.time()
    predictions = []
    references = []
    image_ids = []

    logger.info(f"Evaluating {len(dataset)} samples...")

    for i in tqdm(range(len(dataset)), desc=batch_desc):
        item = dataset[i]
        image = item["image"]
        ref = item["text"]
        img_id = item["image_id"]

        pred = runner.transcribe(image, max_new_tokens=max_new_tokens)

        predictions.append(pred)
        references.append(ref)
        image_ids.append(img_id)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_mb = peak_vram_bytes / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    total_time = time.time() - start_time
    num_samples = len(dataset)
    latency_ms = (total_time / num_samples) * 1000 if num_samples > 0 else 0.0
    throughput = num_samples / total_time if total_time > 0 else 0.0

    summary, per_sample = compute_all_metrics(predictions, references, image_ids)

    # Attach efficiency metrics
    summary["latency_per_sample_ms"] = float(round(latency_ms, 2))
    summary["throughput_samples_per_sec"] = float(round(throughput, 2))
    summary["peak_gpu_memory_mb"] = float(round(peak_vram_mb, 2))
    summary["total_eval_time_sec"] = float(round(total_time, 2))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    summary["trainable_parameters"] = trainable_params
    summary["total_parameters"] = total_params

    # Pretty print summary
    print("\n" + "=" * 50)
    print(" EVALUATION SUMMARY")
    print("=" * 50)
    print(f" Samples Evaluated     : {summary['num_samples']}")
    print(f" Character Error Rate  : {summary['cer']:.4f}")
    print(f" Word Error Rate       : {summary['wer']:.4f}")
    print(f" Exact Match Accuracy  : {summary['exact_match']:.4f} ({summary['num_correct_exact']}/{summary['num_samples']})")
    print(f" Normalized Match Acc  : {summary['normalized_match']:.4f} ({summary['num_correct_normalized']}/{summary['num_samples']})")
    print(f" Character Accuracy    : {summary['character_accuracy']:.4f}")
    print(f" Latency per sample    : {summary['latency_per_sample_ms']:.2f} ms")
    print(f" Throughput            : {summary['throughput_samples_per_sec']:.2f} samples/sec")
    print(f" Peak GPU Memory       : {summary['peak_gpu_memory_mb']:.2f} MB")
    print("=" * 50 + "\n")

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        summary_file = out_p / "eval_summary.json"
        preds_file = out_p / "predictions.json"

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(preds_file, "w", encoding="utf-8") as f:
            json.dump(per_sample, f, indent=2)

        logger.info(f"Saved evaluation results to {out_p}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate MS-MGCA-Qwen3-VL Baseline")
    parser.add_argument("--config", type=str, default="configs/e1_baseline.yaml", help="Path to config YAML")
    parser.add_argument("--split", type=str, choices=["val", "test", "train"], default="val", help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate (None for full)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    cfg = MSMGCAConfig.from_yaml(str(EXP_ROOT / args.config))

    logger.info(f"Loading datasets with seed {cfg.dataset.seed}...")
    train_ds, val_ds, test_ds = get_rxhand_datasets(
        train_labels=cfg.dataset.train_labels,
        train_images=cfg.dataset.train_images,
        test_labels=cfg.dataset.test_labels,
        test_images=cfg.dataset.test_images,
        train_split=cfg.dataset.train_split,
        seed=cfg.dataset.seed,
        max_val_samples=args.max_samples if args.split == "val" else None,
        max_test_samples=args.max_samples if args.split == "test" else None,
    )

    dataset_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    eval_dataset = dataset_map[args.split]

    logger.info("Loading baseline model...")
    model, processor = load_qwen3vl_baseline(cfg, for_training=False)

    out_dir = args.output_dir or str(EXP_ROOT / "results" / cfg.training.run_name / args.split)
    evaluate_model(model, processor, eval_dataset, output_dir=out_dir)


if __name__ == "__main__":
    main()
