import os
import sys
import torch
import logging
import tracemalloc
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments
import json

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import PipelineConfig
from models.models import load_qwen_vl, load_yolo_model
from utils.dataset_parser import parse_all_datasets, save_dataset_json
from utils.augmentor import get_augmentor
from utils.metrics import TextMetrics, TrainingMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


STAGE = "train"  # Options: "augment", "train", "infer"
MODEL_CHOICE = "qwen-vl" #Options: "qwen-vl" or "yolo"


class VLMDataset(Dataset):
    """PyTorch Dataset for Vision-Language Model instruction tuning."""

    def __init__(self, records: List[Dict], processor, augmentor=None, max_samples: int = None):
        self.records = records[:max_samples] if max_samples else records
        self.processor = processor
        self.augmentor = augmentor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]

        try:
            from PIL import Image
            image = Image.open(record["image"]).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {record['image']}: {e}")
            return self.__getitem__((idx + 1) % len(self.records))

        import numpy as np
        image_array = np.array(image)
        if self.augmentor:
            image_array = self.augmentor(image_array)
            image = Image.fromarray(image_array)

        conversations = record["conversations"]
        prompt = conversations[0]["value"]
        response = conversations[1]["value"] if len(conversations) > 1 else ""

        text = f"{prompt}\n{response}"

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            max_pixels=512 * 512,
            min_pixels=256 * 256,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "pixel_values": inputs.get("pixel_values", torch.tensor([])).squeeze(),
            "attention_mask": inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).squeeze(),
        }


class MemoryProfiler:
    """Monitor GPU and CPU memory usage during training."""

    def __init__(self):
        self.trace_on = False

    def start(self):
        tracemalloc.start()
        self.trace_on = True

    def end(self) -> Tuple[float, float]:
        """Return (allocated_mb, peak_mb)."""
        if not self.trace_on:
            return 0.0, 0.0

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return current / 1024 / 1024, peak / 1024 / 1024

    @staticmethod
    def get_gpu_memory_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0


def setup_directories(cfg: PipelineConfig) -> None:
    """Create output directories."""
    for directory in [cfg.output_dir, cfg.checkpoint_dir, cfg.log_dir]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ready: {directory}")


def augment_stage(cfg: PipelineConfig) -> None:
    """Stage 1: Generate augmented dataset."""
    logger.info("=" * 60)
    logger.info("STAGE: AUGMENTATION")
    logger.info("=" * 60)

    setup_directories(cfg)

    logger.info("Parsing datasets...")
    train_records, val_records = parse_all_datasets(cfg.dataset)

    augmentor = get_augmentor(cfg.augmentation)

    output_train = os.path.join(cfg.output_dir, "train_augmented.json")
    output_val = os.path.join(cfg.output_dir, "val_augmented.json")

    save_dataset_json(train_records, output_train)
    save_dataset_json(val_records, output_val)

    logger.info(f"Augmentation stage complete. Saved to {cfg.output_dir}")


def train_stage(cfg: PipelineConfig) -> None:
    """Stage 2: Train Vision-Language Model with LoRA."""
    logger.info("=" * 60)
    logger.info("STAGE: TRAINING")
    logger.info("=" * 60)

    setup_directories(cfg)
    mem_profiler = MemoryProfiler()
    mem_profiler.start()

    logger.info(f"Device: {cfg.device}")
    logger.info(f"Model: {cfg.model_type}")

    logger.info("Parsing datasets...")
    train_records, val_records = parse_all_datasets(cfg.dataset)

    logger.info("Loading model and processor...")
    if cfg.model_type == "qwen-vl":
        model, processor = load_qwen_vl(cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")

    logger.info(f"GPU Memory: {mem_profiler.get_gpu_memory_mb():.2f} MB")

    augmentor = get_augmentor(cfg.augmentation)

    train_dataset = VLMDataset(train_records, processor, augmentor)
    val_dataset = VLMDataset(val_records, processor)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=cfg.checkpoint_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        num_train_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        logging_steps=cfg.log_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        fp16=cfg.training.use_fp16,
        bf16=cfg.training.use_bf16,
        dataloader_num_workers=cfg.training.num_workers,
        dataloader_pin_memory=cfg.training.pin_memory,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        max_grad_norm=cfg.training.max_grad_norm,
        seed=cfg.training.seed,
        logging_dir=cfg.log_dir,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting training...")
    trainer.train()

    allocated_mb, peak_mb = mem_profiler.end()
    logger.info(f"Memory used - Allocated: {allocated_mb:.2f} MB, Peak: {peak_mb:.2f} MB")
    logger.info(f"Final GPU Memory: {mem_profiler.get_gpu_memory_mb():.2f} MB")

    model.save_pretrained(os.path.join(cfg.checkpoint_dir, "final"))
    logger.info(f"Model saved to {os.path.join(cfg.checkpoint_dir, 'final')}")


def infer_stage(cfg: PipelineConfig) -> None:
    """Stage 3: Run inference on sample images."""
    logger.info("=" * 60)
    logger.info("STAGE: INFERENCE")
    logger.info("=" * 60)

    setup_directories(cfg)

    logger.info("Loading model and processor...")
    if cfg.model_type == "qwen-vl":
        model, processor = load_qwen_vl(cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")

    model.eval()

    logger.info("Parsing datasets...")
    train_records, _ = parse_all_datasets(cfg.dataset)

    sample_size = min(5, len(train_records))
    sample_records = train_records[:sample_size]

    results = []

    with torch.no_grad():
        for record in tqdm(sample_records, desc="Running inference"):
            try:
                from PIL import Image
                image = Image.open(record["image"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {record['image']}: {e}")
                continue

            prompt = record["conversations"][0]["value"]

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(cfg.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
            )

            generated_text = processor.decode(outputs[0], skip_special_tokens=True)

            results.append({
                "image_id": record["id"],
                "prompt": prompt,
                "generated_text": generated_text,
                "ground_truth": record["conversations"][1]["value"],
            })

    output_file = os.path.join(cfg.output_dir, "inference_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Inference complete. Results saved to {output_file}")

    if results:
        predictions = [r["generated_text"] for r in results]
        references = [r["ground_truth"] for r in results]
        metrics = TextMetrics.compute_batch_metrics(predictions, references)

        logger.info("\n" + "=" * 40)
        logger.info("INFERENCE METRICS")
        logger.info("=" * 40)
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")


def main():
    """Main pipeline orchestrator."""
    cfg = PipelineConfig(
        stage=STAGE,
        model_type=MODEL_CHOICE,
    )

    logger.info("\n" + "=" * 60)
    logger.info("VLM TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Stage: {cfg.stage}")
    logger.info(f"Model: {cfg.model_type}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info("=" * 60 + "\n")

    if cfg.stage == "augment":
        augment_stage(cfg)
    elif cfg.stage == "train":
        train_stage(cfg)
    elif cfg.stage == "infer":
        infer_stage(cfg)
    else:
        logger.error(f"Unknown stage: {cfg.stage}")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
