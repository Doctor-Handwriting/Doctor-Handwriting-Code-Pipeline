import os
import sys
import logging

# Triton is optional (not shipped on Windows); silence the warning repeated by every dataloader worker.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import torch
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import ProgressCallback, TrainerCallback
from transformers.integrations import TensorBoardCallback
import json
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import PipelineConfig, HARDWARE_PROFILES
from models.models import load_qwen_vl, load_yolo_model
from utils.dataset_parser import parse_all_datasets, save_dataset_json
from utils.augmentor import get_augmentor
from utils.metrics import TextMetrics, TrainingMetrics
from utils.visualization import plot_final_metrics, plot_training_curves, save_history, save_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


STAGE = "train"  # Options: "augment", "train", "infer"
MODEL_CHOICE = "qwen-vl" #Options: "qwen-vl" or "yolo"
# Options: "auto"          -> pick by detected GPU VRAM
#          "laptop_3050ti" -> RTX 3050 Ti 4GB / 16GB RAM (4-bit + LoRA, fp16, batch 1x8)
#          "pc_5070"       -> RTX 5070 12GB / 32GB RAM (Qwen3-VL-8B QLoRA, bf16 compute, batch 2x4)
# Profiles are defined in configs/config.py (HARDWARE_PROFILES).
HARDWARE_PROFILE = "pc_5070"


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
            "correct": TextMetrics.is_match(generated_text, ground_truth),
            "correct_normalized": TextMetrics.is_match(generated_text, ground_truth, normalized=True),
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
    # Ground truth / correctly predicted, e.g. 100/97 when 3 of 100 samples are wrong.
    for label, normalized in (("gt/pred (exact)", False), ("gt/pred (normalized)", True)):
        pair = TextMetrics.format_gt_pred(metrics, normalized)
        if pair:
            lines.append(f" {label:<{width}}{pair:>12}")
    lines.append("=" * (width + 14))
    return "\n".join(lines)


class MetricsProgressCallback(ProgressCallback):
    """Training progress bar with live loss/LR/eval metrics in the postfix and a
    formatted table printed after every evaluation."""

    POSTFIX_KEYS = ("loss", "token_accuracy", "learning_rate", "grad_norm", "epoch")
    EVAL_POSTFIX_KEYS = ("eval_loss", "eval_token_accuracy", "eval_cer", "eval_wer", "eval_accuracy")

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
                short = key.replace("learning_rate", "lr").replace("token_accuracy", "tok_acc")
                self.postfix[short] = f"{value:.3g}" if isinstance(value, float) else value
        gt_pred = TextMetrics.format_gt_pred(logs)
        if gt_pred:
            self.postfix["gt/pred"] = gt_pred
        if bar is not None:
            bar.set_postfix(self.postfix, refresh=True)


class TrainingPlotCallback(TrainerCallback):
    """Keeps the full metric history of the run and redraws the loss/accuracy plots
    (plus metrics_history.json/.csv) after every evaluation and at the end of training."""

    def __init__(self, run_dir: str, plots_dir: str):
        self.run_dir = run_dir
        self.plots_dir = plots_dir
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            self.history.append({"step": state.global_step, "epoch": state.epoch, **logs})

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.save()

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.save()

    def save(self) -> None:
        # A plotting failure must never stop training.
        try:
            save_history(self.history, self.run_dir)
            save_summary(self.history, self.run_dir)
            plot_training_curves(self.history, self.plots_dir)
        except Exception as e:
            logger.warning(f"Could not save training plots: {e}")


class GenerationEvalTrainer(Trainer):
    """Trainer whose evaluation also decodes a val subset with generate() and reports
    WER/CER/accuracy alongside eval_loss, so all metrics are logged together.
    Also tracks next-token accuracy on answer tokens: logged as token_accuracy (train,
    averaged over each logging window) and eval_token_accuracy (validation)."""

    def __init__(self, *args, processor=None, gen_eval_records=None,
                 max_pixels: int = 512 * 512, max_new_tokens: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.gen_eval_records = gen_eval_records or []
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        # split -> [correct, total] answer tokens, kept as tensors to avoid a GPU sync per step
        self._token_acc = {"train": [0, 0], "eval": [0, 0]}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        logits = getattr(outputs, "logits", None)
        if labels is not None and logits is not None:
            with torch.no_grad():
                preds = logits[:, :-1].argmax(dim=-1)
                targets = labels[:, 1:].to(preds.device)
                mask = targets != -100
                acc = self._token_acc["train" if model.training else "eval"]
                acc[0] += (preds.eq(targets) & mask).sum()
                acc[1] += mask.sum()

        return (loss, outputs) if return_outputs else loss

    def _pop_token_accuracy(self, split: str):
        correct, total = self._token_acc[split]
        self._token_acc[split] = [0, 0]
        total = int(total)
        return int(correct) / total if total else None

    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            token_accuracy = self._pop_token_accuracy("train")
            if token_accuracy is not None:
                logs["token_accuracy"] = token_accuracy
        super().log(logs, *args, **kwargs)

    def evaluation_loop(self, *args, **kwargs):
        self._token_acc["eval"] = [0, 0]
        output = super().evaluation_loop(*args, **kwargs)
        prefix = kwargs.get("metric_key_prefix", "eval")

        token_accuracy = self._pop_token_accuracy("eval")
        if token_accuracy is not None:
            output.metrics[f"{prefix}_token_accuracy"] = token_accuracy

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


def setup_directories(*directories: str) -> None:
    """Create output directories."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ready: {directory}")


def augment_stage(cfg: PipelineConfig) -> None:
    """Stage 1: Generate augmented dataset."""
    logger.info("=" * 60)
    logger.info("STAGE: AUGMENTATION")
    logger.info("=" * 60)

    setup_directories(cfg.output_dir)

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

    setup_directories(cfg.run_dir, cfg.checkpoint_dir, cfg.log_dir, cfg.plots_dir)
    logger.info(f"Run folder: {cfg.run_dir}")
    with open(os.path.join(cfg.run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_name": cfg.model_name, **asdict(cfg)}, f, indent=2, default=str)
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
        optim=cfg.training.optim,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        lr_scheduler_kwargs={"num_cycles": cfg.training.lr_scheduler_num_cycles}
        if cfg.training.lr_scheduler_type == "cosine_with_restarts" else {},
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        logging_steps=cfg.log_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.training.use_fp16,
        bf16=cfg.training.use_bf16,
        dataloader_num_workers=cfg.training.num_workers,
        dataloader_pin_memory=cfg.training.pin_memory,
        # Windows spawns workers slowly (~6s each); keep them alive across epochs/evals.
        dataloader_persistent_workers=cfg.training.num_workers > 0,
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
    trainer.add_callback(TrainingPlotCallback(cfg.run_dir, cfg.plots_dir))

    logger.info("Starting training...")
    trainer.train()

    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    final_metrics["eval_gt_pred"] = TextMetrics.format_gt_pred(final_metrics)
    final_metrics["eval_gt_pred_normalized"] = TextMetrics.format_gt_pred(final_metrics, normalized=True)
    metrics_file = os.path.join(cfg.run_dir, "eval_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Final evaluation metrics saved to {metrics_file}")
    logger.info(f"Training plots saved to {cfg.plots_dir}")

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

    setup_directories(cfg.run_dir)
    logger.info(f"Run folder: {cfg.run_dir}")

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

    use_amp = torch.cuda.is_available()
    with torch.autocast("cuda", dtype=cfg.compute_dtype, enabled=use_amp):
        metrics, results = run_generation_eval(
            model, processor, sample_records, cfg.model.max_pixels,
            cfg.evaluation.max_new_tokens, desc="Running inference",
        )

    output_file = os.path.join(cfg.run_dir, "inference_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Inference complete. Results saved to {output_file}")

    if metrics:
        metrics_file = os.path.join(cfg.run_dir, "inference_metrics.json")
        logger.info(format_metrics_table(metrics, "INFERENCE METRICS"))
        metrics["gt_pred"] = TextMetrics.format_gt_pred(metrics)
        metrics["gt_pred_normalized"] = TextMetrics.format_gt_pred(metrics, normalized=True)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

        os.makedirs(cfg.plots_dir, exist_ok=True)
        plot_file = os.path.join(cfg.plots_dir, "inference_metrics.png")
        if plot_final_metrics(metrics, plot_file, f"Inference metrics ({len(results)} samples)"):
            logger.info(f"Metrics chart saved to {plot_file}")


def main():
    """Main pipeline orchestrator."""
    cfg = PipelineConfig(
        stage=STAGE,
        model_type=MODEL_CHOICE,
        hardware_profile=HARDWARE_PROFILE,
    )

    logger.info("\n" + "=" * 60)
    logger.info("VLM TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Stage: {cfg.stage}")
    logger.info(f"Model: {cfg.model_type}")
    logger.info(f"Hardware profile: {cfg.hardware_profile} "
                f"({HARDWARE_PROFILES[cfg.hardware_profile].description})")
    logger.info(f"Precision: {cfg.mixed_precision}, 4-bit: {cfg.quantization.load_in_4bit}, "
                f"batch: {cfg.training.per_device_train_batch_size} x "
                f"{cfg.training.gradient_accumulation_steps} accum, workers: {cfg.training.num_workers}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"PyTorch: {torch.__version__} (CUDA build: {torch.version.cuda or 'none - CPU only'})")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"Dataset root: {cfg.dataset.dataset_root}")
    logger.info(f"Output dir: {cfg.output_dir}")
    if cfg.stage != "augment":
        logger.info(f"Run dir: {cfg.run_dir}")
    logger.info("=" * 60 + "\n")

    # Fail fast (before downloading the model) when the GPU is not usable.
    if cfg.stage in ("train", "infer") and cfg.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA GPU not available to PyTorch - the model cannot be loaded.")
        if torch.version.cuda is None:
            logger.error("This PyTorch is a CPU-only build. Reinstall a CUDA build, e.g. for RTX 50xx:\n"
                         "  pip uninstall -y torch torchvision\n"
                         "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        else:
            logger.error(f"PyTorch was built for CUDA {torch.version.cuda} but cannot reach the GPU. "
                         "Check `nvidia-smi` works and the NVIDIA driver is recent enough "
                         "(CUDA 12.8 builds need driver >= 570).")
        sys.exit(1)

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
