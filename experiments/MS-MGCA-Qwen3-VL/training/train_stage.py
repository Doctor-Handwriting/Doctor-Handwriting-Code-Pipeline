"""
Execution driver for training stages 1 through 5.
Usage:
    python train_stage.py --config configs/e6_full_model.yaml --stage 5 --epochs 3
"""

import argparse
import logging
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from configs.base_config import MSMGCAConfig
from dataset.rxhand_dataset import get_rxhand_datasets
from models.proposed_model import MSMGCAQwen3VL
from training.trainer import MSMGCATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_stage():
    parser = argparse.ArgumentParser(description="Run MS-MGCA-Qwen3-VL Training Stage")
    parser.add_argument("--config", type=str, default="configs/e6_full_model.yaml", help="Path to config YAML")
    parser.add_argument("--stage", type=int, default=5, choices=[2, 3, 4, 5], help="Progressive Training Stage (2 to 5)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Debug subset limit")
    parser.add_argument("--max_val_samples", type=int, default=50, help="Validation subset limit")
    args = parser.parse_args()

    cfg_path = EXP_ROOT / args.config
    cfg = MSMGCAConfig.from_yaml(str(cfg_path))

    if args.epochs:
        cfg.training.num_epochs = args.epochs

    logger.info(f"Loading datasets for Stage {args.stage}...")
    train_ds, val_ds, test_ds = get_rxhand_datasets(
        train_labels=cfg.dataset.train_labels,
        train_images=cfg.dataset.train_images,
        test_labels=cfg.dataset.test_labels,
        test_images=cfg.dataset.test_images,
        train_split=cfg.dataset.train_split,
        seed=cfg.dataset.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    logger.info("Initializing unified MS-MGCA-Qwen3-VL model...")
    model = MSMGCAQwen3VL(cfg, for_training=True)

    trainer = MSMGCATrainer(
        model=model,
        cfg=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        stage=args.stage,
    )

    best_cer = float("inf")
    for epoch in range(cfg.training.num_epochs):
        train_loss = trainer.train_epoch(epoch)
        val_summary = trainer.evaluate(max_samples=args.max_val_samples)

        logger.info(
            f"Epoch {epoch+1}/{cfg.training.num_epochs} -> "
            f"Train Loss: {train_loss:.4f} | "
            f"Val CER: {val_summary['cer']:.4f} | "
            f"Val Exact Match: {val_summary['exact_match']:.4f}"
        )

        if val_summary["cer"] < best_cer:
            best_cer = val_summary["cer"]
            trainer.save_checkpoint("best")

    trainer.save_checkpoint("final")
    logger.info("Training stage completed successfully.")


if __name__ == "__main__":
    run_training_stage()
