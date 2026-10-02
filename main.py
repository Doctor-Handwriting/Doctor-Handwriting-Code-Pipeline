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
from transformers.trainer_callback import ProgressCallback
from transformers.integrations import TensorBoardCallback
import json
import numpy as np
from PIL import Image

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


def build_messages(prompt: str, response: str = None) -> List[Dict]:
    """Qwen-VL chat format; the image placeholder is expanded by the processor."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt.replace("<image>", "").strip()},
        ],
    }]
    if response is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return messages


def limit_image_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.BICUBIC)


@torch.no_grad()
def generate_text(model, processor, image: Image.Image, prompt: str,
                  max_pixels: int, max_new_tokens: int) -> str:
    """Greedy-decode the model's transcription for a single image."""
    text = processor.apply_chat_template(
        build_messages(prompt), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[limit_image_pixels(image, max_pixels)],
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def run_generation_eval(model, processor, records: List[Dict], max_pixels: int,
                        max_new_tokens: int, desc: str = "Generating") -> Tuple[Dict[str, float], List[Dict]]:
    """Generate transcriptions for records and score them with all text metrics."""
    was_training = model.training
    model.eval()

    results = []
    for record in tqdm(records, desc=desc, leave=False, dynamic_ncols=True):
        try:
            image = Image.open(record["image"]).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {record['image']}: {e}")
            continue

        prompt = record["conversations"][0]["value"]
        generated_text = generate_text(model, processor, image, prompt, max_pixels, max_new_tokens)
        ground_truth = record["conversations"][1]["value"]
        results.append({
            "image_id": record["id"],
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "cer": TextMetrics.compute_cer([generated_text], [ground_truth]) if ground_truth else None,
        })

    if was_training:
        model.train()

    metrics = TextMetrics.compute_batch_metrics(
        [r["generated_text"] for r in results], [r["ground_truth"] for r in results]
    )
    return metrics, results


def format_metrics_table(metrics: Dict, title: str) -> str:
    """Render a metrics dict as an aligned text table."""
    rows = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
    width = max([len(k) for k, _ in rows] + [len(title)]) + 2
    lines = ["", "=" * (width + 14), f" {title}", "-" * (width + 14)]
    for key, value in rows:
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f" {key:<{width}}{shown:>12}")
    lines.append("=" * (width + 14))
    return "\n".join(lines)


class MetricsProgressCallback(ProgressCallback):
    """Training progress bar with live loss/LR/eval metrics in the postfix and a
    formatted table printed after every evaluation."""

    POSTFIX_KEYS = ("loss", "learning_rate", "grad_norm", "epoch")
    EVAL_POSTFIX_KEYS = ("eval_loss", "eval_cer", "eval_wer", "eval_accuracy")

    def __init__(self):
        super().__init__()
        self.postfix = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return

        bar = getattr(self, "training_bar", None)
        write = bar.write if bar is not None else tqdm.write

        if any(k.startswith("eval_") for k in logs):
            write(format_metrics_table(logs, f"EVALUATION @ step {state.global_step}/{state.max_steps}"))
            keys = self.EVAL_POSTFIX_KEYS
        elif "train_runtime" in logs:
            write(format_metrics_table(logs, "TRAINING SUMMARY"))
            return
        else:
            keys = self.POSTFIX_KEYS

        for key in keys:
            if key in logs:
                value = logs[key]
                self.postfix[key.replace("learning_rate", "lr")] = f"{value:.3g}" if isinstance(value, float) else value
        if bar is not None:
            bar.set_postfix(self.postfix, refresh=True)


class GenerationEvalTrainer(Trainer):
    """Trainer whose evaluation also decodes a val subset with generate() and reports
    WER/CER/accuracy alongside eval_loss, so all metrics are logged together."""

    def __init__(self, *args, processor=None, gen_eval_records=None,
                 max_pixels: int = 512 * 512, max_new_tokens: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.gen_eval_records = gen_eval_records or []
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        prefix = kwargs.get("metric_key_prefix", "eval")

        if self.gen_eval_records and self.processor is not None:
            with self.autocast_smart_context_manager():
                gen_metrics, _ = run_generation_eval(
                    self.model, self.processor, self.gen_eval_records,
                    self.max_pixels, self.max_new_tokens, desc="Eval generation",
                )
            output.metrics.update({f"{prefix}_{k}": v for k, v in gen_metrics.items()})

        return output


class QwenVLCollator:
    """Runs the processor on the whole batch so every model-specific key
    (pixel_values, image_grid_thw, mm_token_type_ids, ...) is passed through untouched."""

    def __init__(self, processor, max_pixels: int):
        self.processor = processor
        self.max_pixels = max_pixels
        self.im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

    def __call__(self, batch: List[Dict]) -> Dict:
        texts = [
            self.processor.apply_chat_template(
                build_messages(item["prompt"], item["response"]),
                tokenize=False,
                add_generation_prompt=False,
            )
            for item in batch
        ]
        images = [limit_image_pixels(item["image"], self.max_pixels) for item in batch]

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        # Train only on the answer: mask everything up to and including "<|im_start|>assistant\n".
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        for row in range(labels.size(0)):
            starts = (inputs["input_ids"][row] == self.im_start_id).nonzero(as_tuple=True)[0]
            labels[row, : starts[-1] + 3] = -100
        inputs["labels"] = labels

        return dict(inputs)


class VLMDataset(Dataset):
    """Returns raw image + text; tokenization happens in QwenVLCollator."""

    def __init__(self, records: List[Dict], augmentor=None, max_samples: int = None):
        self.records = records[:max_samples] if max_samples else records
        self.augmentor = augmentor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]

        try:
            image = Image.open(record["image"]).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {record['image']}: {e}")
            return self.__getitem__((idx + 1) % len(self.records))

        if self.augmentor:
            image = Image.fromarray(self.augmentor(np.array(image)))

        conversations = record["conversations"]
        return {
            "image": image,
            "prompt": conversations[0]["value"],
            "response": conversations[1]["value"] if len(conversations) > 1 else "",
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

    train_dataset = VLMDataset(train_records, augmentor)
    val_dataset = VLMDataset(val_records)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    n_gen = cfg.evaluation.train_eval_samples
    gen_eval_records = val_records[:n_gen] if n_gen else val_records
    logger.info(f"Generation metrics (WER/CER/accuracy) computed on {len(gen_eval_records)} val samples per eval")

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
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        fp16=cfg.training.use_fp16,
        bf16=cfg.training.use_bf16,
        dataloader_num_workers=cfg.training.num_workers,
        dataloader_pin_memory=cfg.training.pin_memory,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=cfg.training.max_grad_norm,
        seed=cfg.training.seed,
        remove_unused_columns=False,
        report_to="none",  # TensorBoard is attached explicitly below
        disable_tqdm=False,
    )

    trainer = GenerationEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=QwenVLCollator(processor, cfg.model.max_pixels),
        processor=processor,
        gen_eval_records=gen_eval_records,
        max_pixels=cfg.model.max_pixels,
        max_new_tokens=cfg.evaluation.max_new_tokens,
    )
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(MetricsProgressCallback())
    trainer.add_callback(TensorBoardCallback(tb_writer=SummaryWriter(log_dir=cfg.log_dir)))

    logger.info("Starting training...")
    trainer.train()

    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    metrics_file = os.path.join(cfg.output_dir, "eval_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Final evaluation metrics saved to {metrics_file}")

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

    adapter_path = cfg.evaluation.adapter_path
    if not os.path.isdir(adapter_path):
        logger.warning(f"No trained adapter at {adapter_path}; evaluating the base model")
        adapter_path = None

    logger.info("Loading model and processor...")
    if cfg.model_type == "qwen-vl":
        model, processor = load_qwen_vl(cfg, adapter_path=adapter_path)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")

    model.eval()

    logger.info("Parsing datasets...")
    _, val_records = parse_all_datasets(cfg.dataset)

    n_samples = cfg.evaluation.infer_samples
    sample_records = val_records[:n_samples] if n_samples else val_records
    logger.info(f"Evaluating on {len(sample_records)} validation samples")

    use_amp = cfg.training.use_fp16 and torch.cuda.is_available()
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        metrics, results = run_generation_eval(
            model, processor, sample_records, cfg.model.max_pixels,
            cfg.evaluation.max_new_tokens, desc="Running inference",
        )

    output_file = os.path.join(cfg.output_dir, "inference_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Inference complete. Results saved to {output_file}")

    if metrics:
        metrics["num_samples"] = len(results)
        metrics_file = os.path.join(cfg.output_dir, "inference_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(format_metrics_table(metrics, "INFERENCE METRICS"))
        logger.info(f"Metrics saved to {metrics_file}")


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
    logger.info(f"Dataset root: {cfg.dataset.dataset_root}")
    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info("=" * 60 + "\n")

    missing = cfg.dataset.missing_paths()
    if missing:
        logger.error("Dataset files not found:\n  " + "\n  ".join(missing))
        logger.error("Place the dataset at <repo>/Dataset (DHP/ and RxHand/) "
                     "or set the VLM_DATASET_ROOT environment variable.")
        sys.exit(1)

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
