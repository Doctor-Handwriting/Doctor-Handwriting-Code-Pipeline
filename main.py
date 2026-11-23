import os
import sys
import logging

# Triton is optional (not shipped on Windows); silence the warning repeated by every dataloader worker.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import argparse
import time
import torch
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.trainer_callback import ProgressCallback, TrainerCallback
from transformers.integrations import TensorBoardCallback
import json
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import AugmentationConfig, PipelineConfig, ModelConfig, DatasetConfig, HARDWARE_PROFILES
from models.models import (
    CRNNProcessor, YOLOProcessor, build_charset, build_class_list, load_crnn, load_qwen_vl,
    load_yolo_classifier, save_crnn, save_yolo,
)
from models.ms_mgca import (
    CALIBRATED_DIR, calibration_split, capture_pooled_features, confidence_report, fit_confidence_head, head_scores,
    heldout_ids, load_experiment, load_ms_mgca, load_ms_mgca_weights, ms_mgca_param_groups, save_calibrated,
    save_ms_mgca, square_pad,
)
from utils.dataset_parser import (
    apply_offline_dataset, build_offline_dataset, missing_offline_manifests, parse_all_datasets, parse_test_records,
    save_dataset_json,
)
from utils.augmentor import get_augmentor
from utils.confidence_plots import plot_reliability_diagram, plot_risk_coverage
from utils.metrics import TextMetrics, TrainingMetrics
from utils.visualization import load_history, plot_final_metrics, plot_training_curves, save_history, save_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


#   "augment"   -> pre-compute the offline augmented dataset (only when AUGMENTATION_MODE = "offline")
#   "train"     -> fine-tune / train the model
#   "calibrate" -> MS-MGCA E4/E6 only: fit the confidence head on the model's real right/wrong transcriptions of held-out
#                  validation images (run after "train", before "infer")
#   "infer"     -> evaluate the latest trained run on the validation set
# Every constant below can also be given on the command line (python main.py --help), which overrides it.
STAGE = "train"  # Options: "augment", "train", "calibrate", "infer"
# Model to train / run inference with (no need to touch models.py or config.py):
#   "qwen-vl" -> Qwen-VL fine-tuned with LoRA/QLoRA; the checkpoint is QWEN_CHECKPOINT below
#   "crnn"    -> CRNN from the paper (13 VGG16 conv + 3 BiLSTM + CTC), trained from scratch with the
#                paper's recipe (lr 5e-4, batch 10, early stopping); architecture settings: CRNNConfig
#   "yolo"    -> Ultralytics YOLO *classification* net (YOLO_CHECKPOINT, a "-cls" model) used as a
#                closed-vocabulary word recognizer: each distinct training label is a class and the
#                predicted class is the transcription. It can never output a word absent from the
#                training labels (~21% of validation words on the current split, so exact match tops
#                out near 79%); settings: YOLOConfig. For free-form text use "qwen-vl" or "crnn".
#   "yolo26"  -> the same word classifier on the newer YOLO26 classification backbones (YOLO26_CHECKPOINT,
#                sizes n/s/m/l/x); per-size batch/accumulation for each hardware profile and per-size
#                learning rate: YOLO26Config + HardwareProfile.yolo26_micro_batch. Same closed-vocabulary
#                ceiling as "yolo".
#   "ms-mgca" -> MS-MGCA-Qwen3-VL (experiments/MS-MGCA-Qwen3-VL): Qwen3-VL + LoRA with multi-scale gated fusion,
#                medication-guided cross-attention and an uncertainty-aware confidence head. The ablation (E1-E6) is
#                MSMGCA_EXPERIMENT and the backbone is QWEN_CHECKPOINT (a Qwen3-VL model). Trains with batch 1 x
#                gradient accumulation (its multi-scale path handles one image at a time); the infer stage also
#                reports the confidence head's calibration and the share of predictions flagged for review.
MODEL_CHOICE = "ms-mgca"  # Options: "qwen-vl", "crnn", "yolo", "yolo26", "ms-mgca"
# Used when MODEL_CHOICE == "qwen-vl". Any Qwen2.5-VL / Qwen3-VL checkpoint, e.g.
#   "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"
# (8B needs a 12GB+ profile and trains as 4-bit QLoRA; 2B/4B train in plain bf16 on pc_5070ti;
#  use a 2B/3B model on the 4GB laptop).
QWEN_CHECKPOINT = "Qwen/Qwen3-VL-8B-Instruct"
# Used when MODEL_CHOICE == "ms-mgca": which ablation of experiments/MS-MGCA-Qwen3-VL/configs/ to run
#   "e1_baseline"             Qwen3-VL + LoRA only (reference)
#   "e2_multiscale"           + multi-scale gated fusion (Component A)
#   "e3_cross_attention"      + medication-guided cross-attention (Component B)
#   "e4_confidence"           + uncertainty-aware confidence head (Component D)
#   "e5_multiscale_crossattn" A + B
#   "e6_full_model"           A + B + D = the full proposed model
# All components train jointly with LoRA. Each experiment/backbone gets its own run folder
# (run_<time>_<profile>_<experiment>_<size>...), and STAGE = "infer" evaluates the latest run of the chosen one.
MSMGCA_EXPERIMENT = "e6_full_model"
# Number of learnable medication queries K of the cross-attention (E3, E5, E6). None = the experiment YAML's value (4);
# set 1, 4 or 8 for the K sweep (the run folder then ends in _k<K>).
MSMGCA_NUM_QUERIES = None
# Used when MODEL_CHOICE == "yolo": must be a classification model, one of
#   yolov8n-cls.pt | yolov8s-cls.pt | yolov8m-cls.pt | yolov8l-cls.pt | yolov8x-cls.pt
YOLO_CHECKPOINT = "yolov8m-cls.pt"
# Used when MODEL_CHOICE == "yolo26": must be a YOLO26 classification model, one of
#   yolo26n-cls.pt (3.6M params) | yolo26s-cls.pt (7.5M) | yolo26m-cls.pt (12.4M) | yolo26l-cls.pt (14.9M) | yolo26x-cls.pt (30.4M)
# (all five fit every hardware profile; weights are downloaded on first use).
YOLO26_CHECKPOINT = "yolo26m-cls.pt"
# Options: "auto"         -> pick by detected GPU VRAM
#          "laptop_3050ti" -> RTX 3050 Ti 4GB / 16GB RAM (4-bit + LoRA, fp16, batch 1x8)
#          "pc_3070ti"     -> RTX 3070 Ti 8GB / 32GB RAM (Qwen 2B/4B QLoRA, bf16 compute, batch 1x8)
#          "pc_5070"       -> RTX 5070 12GB / 32GB RAM (Qwen3-VL-8B QLoRA, bf16 compute, batch 2x4)
#          "pc_5070ti"     -> RTX 5070 Ti 16GB / 32GB RAM (8B QLoRA batch 4x2; Qwen 2B/4B in plain bf16)
# Profiles are defined in configs/config.py (HARDWARE_PROFILES).
HARDWARE_PROFILE = "pc_5070"
# Which data to train and validate on:
#   "combined" -> DHP + RxHand merged into one pool, then a single train/val split (default)
#   "dhp"      -> DHP only, split into its own train/val
#   "rxhand"   -> RxHand only, split into its own train/val
# For "rxhand", RXHAND_OFFICIAL_SPLIT = True trains on RxHand's Train_Set and validates on its Test_Set
# instead of re-splitting them together. Non-combined runs get the mode in their run folder name
# (run_<timestamp>_<profile>_dhp), so separate experiments never overwrite each other.
DATASET_MODE = "rxhand"  # Options: "combined", "dhp", "rxhand"
RXHAND_OFFICIAL_SPLIT = False
# For "rxhand", RXHAND_HELDOUT_TEST = True splits Train_Set 80/20 into train/val and keeps the official Test_Set
# for the infer stage only. The split lives in configs/splits/rxhand/ so every machine uses the same images.
RXHAND_HELDOUT_TEST = True
# How training images are augmented:
#   "online"  -> random transforms applied to each training image on the fly (nothing stored on disk, a new
#                variation every epoch, same behaviour as before)
#   "offline" -> STAGE = "augment" first pre-computes the images into Augmented_Dataset/<DHP|RxHand>/
#                (original/ copies + augmented/ <name>_<variant>.png: motionblur, elasticdeform, perspective,
#                brightnesscontrast, rotate, shiftscalerotate, mixed + augmentation_manifest.json);
#                STAGE = "train"/"infer" then read from there with no augmentation at train time. Only training
#                images get variants (the split is made on the originals), validation uses originals.
#                Training epochs get ~8x larger, so lower num_epochs accordingly. Offline runs get "_offline"
#                in their run folder name.
AUGMENTATION_MODE = "offline"


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


def prepare_ms_mgca_image(image: Image.Image, max_pixels: int, multiscale: bool) -> Image.Image:
    """Image as MS-MGCA sees it: square-padded when the multi-scale module is on, then pixel-limited."""
    image = image.convert("RGB")
    if multiscale:
        image = square_pad(image)
    return limit_image_pixels(image, max_pixels)


@torch.no_grad()
def generate_text(model, processor, image: Image.Image, prompt: str,
                  max_pixels: int, max_new_tokens: int) -> Tuple[str, Dict]:
    """Greedy-decode the model's transcription for a single image.
    Returns (text, extra): extra holds model-specific fields for the results (MS-MGCA: confidence, requires_review)."""
    if isinstance(processor, (CRNNProcessor, YOLOProcessor)):
        return processor.transcribe(model, image), {}

    if hasattr(model, "generate_with_confidence"):  # MS-MGCA-Qwen3-VL
        out = model.generate_with_confidence(
            prepare_ms_mgca_image(image, max_pixels, model.multiscale_enabled),
            prompt=prompt, max_new_tokens=max_new_tokens)
        extra = {"confidence": out["confidence"], "requires_review": out["requires_review"]} \
            if model.uncertainty_enabled else {}
        return out["prediction"], extra

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
    return processor.decode(new_tokens, skip_special_tokens=True).strip(), {}


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
        generated_text, extra = generate_text(model, processor, image, prompt, max_pixels, max_new_tokens)
        ground_truth = record["conversations"][1]["value"]
        results.append({
            "image_id": record["id"],
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "correct": TextMetrics.is_match(generated_text, ground_truth),
            "correct_normalized": TextMetrics.is_match(generated_text, ground_truth, normalized=True),
            "cer": TextMetrics.compute_cer([generated_text], [ground_truth]) if ground_truth else None,
            **extra,
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
                 max_pixels: int = 512 * 512, max_new_tokens: int = 64,
                 track_token_accuracy: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        # Next-token accuracy only makes sense for the autoregressive Qwen-VL (not CTC-based CRNN).
        self.track_token_accuracy = track_token_accuracy
        self.gen_eval_records = gen_eval_records or []
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        # split -> [correct, total] answer tokens, kept as tensors to avoid a GPU sync per step
        self._token_acc = {"train": [0, 0], "eval": [0, 0]}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # HF model outputs expose .logits; MS-MGCA returns a dict.
        logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
        if self.track_token_accuracy and labels is not None and logits is not None:
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


class MSMGCATrainer(GenerationEvalTrainer):
    """GenerationEvalTrainer for MS-MGCA-Qwen3-VL, whose model is a plain nn.Module around a PEFT backbone:
      - checkpoints hold only the LoRA adapter + MS-MGCA modules (not the whole backbone)
      - the best checkpoint is reloaded with the same loader (load_best_model_at_end)
      - one AdamW group per component, each with its own learning rate (the experiment's recipe)
      - with the confidence head: joint objective L_total = L_text + lambda_c * L_conf. The correctness label z of a
        sample is 1 when the model's teacher-forced greedy prediction reproduces every answer token, else 0 (derived
        from the logits of the same forward pass, so it needs no second pass). This is only a training-time proxy on
        data the model is fitting; STAGE = "calibrate" re-fits the head on real held-out transcriptions.
        eval_loss (checkpoint selection) stays the text loss alone."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conf_parts = {"loss_text": 0, "loss_conf": 0, "conf_target_rate": 0, "steps": 0}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        conf_logits = outputs.get("conf_logits") if isinstance(outputs, dict) else None
        if model.training and conf_logits is not None:
            with torch.no_grad():
                preds = outputs["logits"][:, :-1].argmax(dim=-1)
                targets = inputs["labels"][:, 1:].to(preds.device)
                z = (preds.eq(targets) | (targets == -100)).all(dim=1).float()
            loss_conf = model.uncertainty_decoder.confidence_head.compute_loss(conf_logits, z.to(conf_logits.device))
            parts = self._conf_parts
            parts["loss_text"] += loss.detach()
            parts["loss_conf"] += loss_conf.detach()
            parts["conf_target_rate"] += z.mean()
            parts["steps"] += 1
            loss = loss + model.cfg.uncertainty.loss_weight * loss_conf
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        parts = self._conf_parts
        if "loss" in logs and parts["steps"]:
            n = parts["steps"]
            logs["loss_text"] = float(parts["loss_text"]) / n
            logs["loss_conf"] = float(parts["loss_conf"]) / n
            logs["conf_target_rate"] = float(parts["conf_target_rate"]) / n
            self._conf_parts = {"loss_text": 0, "loss_conf": 0, "conf_target_rate": 0, "steps": 0}
        super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        save_ms_mgca(self.model, output_dir)

    def _load_best_model(self):
        logger.info(f"Loading best MS-MGCA checkpoint from {self.state.best_model_checkpoint} "
                    f"(score: {self.state.best_metric})")
        load_ms_mgca_weights(self.model, self.state.best_model_checkpoint)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # --resume: the Trainer's own loader looks for a whole-model weights file, which these checkpoints do not
        # have (adapter + module weights only). Optimizer, scheduler, RNG and step counter are restored by the Trainer.
        logger.info(f"Restoring MS-MGCA weights from {resume_from_checkpoint}")
        load_ms_mgca_weights(model if model is not None else self.model, resume_from_checkpoint)

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                ms_mgca_param_groups(self.model, self.args.learning_rate),
                weight_decay=self.args.weight_decay)
        return self.optimizer


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


class MSMGCACollator(QwenVLCollator):
    """QwenVLCollator plus what the MS-MGCA model's forward needs: the prepared PIL images (`raw_images`, source of
    the multi-scale crops). The confidence targets are not set here: they depend on the model's own prediction, so
    MSMGCATrainer.compute_loss derives them from the logits of the same forward pass."""

    def __init__(self, processor, max_pixels: int, multiscale: bool):
        # Plain flag, not the model: DataLoader workers pickle the collator.
        super().__init__(processor, max_pixels)
        self.multiscale = multiscale

    def __call__(self, batch: List[Dict]) -> Dict:
        images = [prepare_ms_mgca_image(item["image"], self.max_pixels, self.multiscale) for item in batch]
        out = super().__call__([{**item, "image": image} for item, image in zip(batch, images)])
        out["raw_images"] = images
        return out


class CRNNCollator:
    """Image -> normalized grayscale tensor; text -> character ids padded with -100 (CTC targets)."""

    def __init__(self, processor: CRNNProcessor):
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Dict:
        pixel_values = torch.stack([self.processor.preprocess(item["image"]) for item in batch])
        ids = [self.processor.encode(item["response"]) for item in batch]
        labels = torch.full((len(ids), max(1, max(len(i) for i in ids))), -100, dtype=torch.long)
        for row, row_ids in enumerate(ids):
            labels[row, : len(row_ids)] = torch.tensor(row_ids, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


class YOLOCollator:
    """Image -> normalized RGB tensor; text -> class id (-100 when the word is not in the vocabulary)."""

    def __init__(self, processor: YOLOProcessor):
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Dict:
        return {
            "pixel_values": torch.stack([self.processor.preprocess(item["image"]) for item in batch]),
            "labels": torch.tensor([self.processor.class_index(item["response"]) for item in batch],
                                   dtype=torch.long),
        }


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


def load_records(cfg: PipelineConfig) -> Tuple[List[Dict], List[Dict]]:
    """Train/val records; in offline augmentation mode they point at the pre-computed Augmented_Dataset."""
    train_records, val_records = parse_all_datasets(cfg.dataset)
    if cfg.augmentation.mode == "offline":
        train_records, val_records = apply_offline_dataset(
            cfg.dataset, cfg.augmentation, train_records, val_records)
    return train_records, val_records


def load_eval_records(cfg: PipelineConfig) -> Tuple[List[Dict], str]:
    """Records the infer stage scores: the held-out test set when RXHAND_HELDOUT_TEST is on, otherwise val."""
    test_records = parse_test_records(cfg.dataset)
    if not test_records:
        return load_records(cfg)[1], "validation"
    if cfg.augmentation.mode == "offline":
        _, test_records = apply_offline_dataset(cfg.dataset, cfg.augmentation, [], test_records)
    return test_records, "test"


def augment_stage(cfg: PipelineConfig) -> None:
    """Stage 1: Generate augmented dataset."""
    logger.info("=" * 60)
    logger.info(f"STAGE: AUGMENTATION ({cfg.augmentation.mode})")
    logger.info("=" * 60)

    setup_directories(cfg.output_dir)

    if cfg.augmentation.mode == "offline":
        logger.info(f"Generating offline augmented dataset in {cfg.augmentation.offline_root}...")
        build_offline_dataset(cfg.dataset, cfg.augmentation)

    logger.info("Parsing datasets...")
    train_records, val_records = load_records(cfg)

    output_train = os.path.join(cfg.output_dir, "train_augmented.json")
    output_val = os.path.join(cfg.output_dir, "val_augmented.json")

    save_dataset_json(train_records, output_train)
    save_dataset_json(val_records, output_val)

    logger.info(f"Augmentation stage complete. Saved to {cfg.output_dir}")


def latest_checkpoint(checkpoint_dir: str) -> Tuple[str, int] | None:
    """(folder, step) of the newest complete checkpoint-<step> in checkpoint_dir, or None. A checkpoint is complete
    once its trainer_state.json exists (the Trainer writes it last, so a run killed mid-save is skipped)."""
    found = []
    for path in Path(checkpoint_dir).glob("checkpoint-*"):
        step = path.name.split("-", 1)[1]
        if step.isdigit() and (path / "trainer_state.json").is_file():
            found.append((int(step), path))
    if not found:
        return None
    step, path = max(found)
    return path.as_posix(), step


def stage_output_exists(cfg: PipelineConfig) -> bool:
    """True when the output this stage would produce is already in the run folder (--resume skips such stages)."""
    if cfg.stage == "train":
        return os.path.isdir(os.path.join(cfg.checkpoint_dir, "final"))
    if cfg.stage == "calibrate":
        return os.path.isdir(os.path.join(cfg.checkpoint_dir, CALIBRATED_DIR))
    if cfg.stage == "infer":
        return os.path.isfile(os.path.join(cfg.run_dir, "inference_metrics.json"))
    return False


def train_stage(cfg: PipelineConfig) -> None:
    """Stage 2: Train Vision-Language Model with LoRA."""
    logger.info("=" * 60)
    logger.info("STAGE: TRAINING")
    logger.info("=" * 60)

    setup_directories(cfg.run_dir, cfg.checkpoint_dir, cfg.log_dir, cfg.plots_dir)
    logger.info(f"Run folder: {cfg.run_dir}")
    config_path = os.path.join(cfg.run_dir, "run_config.json")
    resume_from = latest_checkpoint(cfg.checkpoint_dir) if cfg.resume else None
    if resume_from and os.path.isfile(config_path):
        # The learning-rate schedule is laid out for the original number of steps; a different length would
        # silently bend it, so a resumed run must use the original settings. The original run_config.json is kept.
        with open(config_path, encoding="utf-8") as f:
            original = json.load(f)
        was = original.get("training", {}).get("num_epochs")
        if was is not None and was != cfg.training.num_epochs:
            raise ValueError(f"This run was started with {was} epochs but --resume is using {cfg.training.num_epochs}. "
                             f"Resume with the original settings (--epochs {was}).")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"model_name": cfg.model_name, **asdict(cfg)}, f, indent=2, default=str)
    mem_profiler = MemoryProfiler()
    mem_profiler.start()

    logger.info(f"Device: {cfg.device}")
    logger.info(f"Model: {cfg.model_type}")

    logger.info("Parsing datasets...")
    train_records, val_records = load_records(cfg)

    logger.info("Loading model and processor...")
    if cfg.model_type == "qwen-vl":
        model, processor = load_qwen_vl(cfg)
        collator = QwenVLCollator(processor, cfg.model.max_pixels)
    elif cfg.is_ms_mgca:
        # The new modules (queries, gates, confidence head) are initialised randomly while the model is built, i.e.
        # before the Trainer seeds anything: seed here so E1-E6 start from the same random state.
        set_seed(cfg.training.seed)
        model, processor = load_ms_mgca(cfg)
        collator = MSMGCACollator(processor, cfg.model.max_pixels, multiscale=model.multiscale_enabled)
    elif cfg.model_type == "crnn":
        # CTC needs a fixed character set: every character that appears in any label.
        model, processor = load_crnn(cfg, charset=build_charset(train_records + val_records))
        collator = CRNNCollator(processor)
    elif cfg.is_yolo:
        # Vocabulary = training labels only, so validation honestly counts unseen words as errors.
        classes = build_class_list(train_records)
        seen = set(classes)
        unseen = sum(r["conversations"][1]["value"] not in seen for r in val_records)
        logger.warning(f"YOLO classifier vocabulary: {len(classes)} words. {unseen} of {len(val_records)} "
                       f"validation samples ({unseen / len(val_records):.1%}) have a word it has never seen "
                       f"and will always count as wrong (exact match tops out at {1 - unseen / len(val_records):.1%}).")
        model, processor = load_yolo_classifier(cfg, classes=classes)
        collator = YOLOCollator(processor)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")

    logger.info(f"GPU Memory: {mem_profiler.get_gpu_memory_mb():.2f} MB")

    # Offline: the training images are already augmented on disk, so nothing is applied here.
    augmentor = get_augmentor(cfg.augmentation) if cfg.augmentation.mode == "online" else None

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
        eval_strategy="epoch" if cfg.eval_by_epoch else "steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_strategy="epoch" if cfg.eval_by_epoch else "steps",
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        # Val CER from the generation eval, not eval_loss (see TrainingConfig.metric_for_best_model).
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        fp16=cfg.training.use_fp16,
        bf16=cfg.training.use_bf16,
        dataloader_num_workers=cfg.training.num_workers,
        dataloader_pin_memory=cfg.training.pin_memory,
        # Windows spawns workers slowly (~6s each); keep them alive across epochs/evals.
        dataloader_persistent_workers=cfg.training.num_workers > 0,
        # MS-MGCA enables checkpointing on its backbone inside the loader (its wrapper has no such method for the Trainer).
        gradient_checkpointing=cfg.training.gradient_checkpointing and not cfg.is_ms_mgca,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=cfg.training.max_grad_norm,
        seed=cfg.training.seed,
        remove_unused_columns=False,
        report_to="none",  # TensorBoard is attached explicitly below
        disable_tqdm=False,
    )

    trainer_class = MSMGCATrainer if cfg.is_ms_mgca else GenerationEvalTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processor=processor,
        gen_eval_records=gen_eval_records,
        max_pixels=cfg.model.max_pixels,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        track_token_accuracy=cfg.model_type in ("qwen-vl", "ms-mgca"),
    )
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(MetricsProgressCallback())
    trainer.add_callback(TensorBoardCallback(tb_writer=SummaryWriter(log_dir=cfg.log_dir)))
    plot_callback = TrainingPlotCallback(cfg.run_dir, cfg.plots_dir)
    if resume_from:
        # Keep the logged history up to the checkpoint; the steps after it are about to be trained again.
        plot_callback.history = [e for e in (load_history(cfg.run_dir) or []) if e.get("step", 0) <= resume_from[1]]
    trainer.add_callback(plot_callback)
    # Patience counts evaluations without a better metric_for_best_model (val CER). CRNN / YOLO evaluate once per
    # epoch, so theirs is in epochs (CRNN: 20 as in the paper); Qwen-VL / MS-MGCA evaluate every eval_steps.
    patience = (cfg.crnn.early_stopping_patience if cfg.model_type == "crnn"
                else cfg.yolo_settings.early_stopping_patience if cfg.is_yolo
                else cfg.training.early_stopping_patience)
    if patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    if resume_from:
        logger.info(f"RESUMING from {resume_from[0]} (step {resume_from[1]}) in {cfg.run_dir}")
    elif cfg.resume:
        logger.info("--resume: no checkpoint in this run folder yet; training from the start.")
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from[0] if resume_from else None)

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

    final_dir = os.path.join(cfg.checkpoint_dir, "final")
    if cfg.model_type == "crnn":
        save_crnn(model, processor, final_dir)
    elif cfg.is_yolo:
        save_yolo(model, processor, Path(cfg.yolo_checkpoint).stem, final_dir)
    elif cfg.is_ms_mgca:
        save_ms_mgca(model, final_dir)
    else:
        model.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")


def calibrate_stage(cfg: PipelineConfig) -> None:
    """MS-MGCA confidence calibration (E4 / E6). Needs a trained run (STAGE = "train").

    1. The model transcribes the validation images, which it never trained on, so every transcription is an
       out-of-sample prediction and gets an empirical label z = 1 (exact match) or 0 (wrong).
    2. The val set is split once (EvaluationConfig.calibration_fraction, fixed seed) into a calibration part and a
       held-out evaluation part.
    3. The confidence head, warm-started from the jointly trained one, is fitted with BCE on the calibration part.
    4. ECE / Brier / AUROC are measured on the held-out part before and after, and the calibrated checkpoint is saved
       to checkpoints/calibrated (STAGE = "infer" uses it and scores confidence on the held-out part only)."""
    logger.info("=" * 60)
    logger.info("STAGE: CONFIDENCE CALIBRATION")
    logger.info("=" * 60)
    if not cfg.is_ms_mgca:
        raise ValueError('STAGE = "calibrate" is only for MODEL_CHOICE = "ms-mgca".')

    final_dir = cfg.evaluation.adapter_path
    if not os.path.isdir(final_dir):
        raise FileNotFoundError(f"No trained ms-mgca model at {final_dir}. Run STAGE = \"train\" first.")
    logger.info(f"Run folder: {cfg.run_dir}")

    model, processor = load_ms_mgca(cfg, checkpoint_dir=final_dir)
    if not model.uncertainty_enabled:
        raise ValueError(f"'{cfg.model.ms_mgca_experiment}' has no confidence head; calibrate E4 (e4_confidence) "
                         f"or E6 (e6_full_model).")
    model.eval()

    _, val_records = load_records(cfg)
    n_samples = cfg.evaluation.infer_samples
    records = val_records[:n_samples] if n_samples else val_records
    ev = cfg.evaluation
    cal_ids, held_ids = calibration_split([r["id"] for r in records], ev.calibration_fraction, ev.calibration_seed)
    logger.info(f"{len(records)} validation images: {len(cal_ids)} calibration, {len(held_ids)} held-out evaluation")

    use_amp = torch.cuda.is_available()
    with torch.autocast("cuda", dtype=cfg.compute_dtype, enabled=use_amp), capture_pooled_features(model) as feats:
        metrics, results = run_generation_eval(model, processor, records, cfg.model.max_pixels,
                                               ev.max_new_tokens, desc="Calibration predictions")
    if len(feats) != len(results):
        raise RuntimeError(f"Captured {len(feats)} feature vectors for {len(results)} predictions.")

    by_id = {r["image_id"]: (f, int(r["correct"])) for r, f in zip(results, feats)}
    def part(ids):
        rows = [by_id[i] for i in ids if i in by_id]
        return torch.stack([f for f, _ in rows]), torch.tensor([float(z) for _, z in rows])
    x_cal, z_cal = part(cal_ids)
    x_held, z_held = part(held_ids)
    logger.info(f"Exact match of the model: calibration part {z_cal.mean():.3f} ({int(z_cal.sum())}/{len(z_cal)}), "
                f"held-out part {z_held.mean():.3f} ({int(z_held.sum())}/{len(z_held)})")

    exp = load_experiment()
    tau = model.cfg.uncertainty.confidence_threshold
    def evaluate():
        return exp.compute_all_confidence_metrics(head_scores(model, x_held).tolist(), z_held.int().tolist(), threshold=tau)

    before = evaluate()
    fit = fit_confidence_head(model, x_cal, z_cal, steps=ev.calibration_steps, lr=ev.calibration_lr,
                              seed=ev.calibration_seed)
    after = evaluate() if fit else before

    for name, rep in (("BEFORE calibration (jointly trained head)", before), ("AFTER calibration", after)):
        logger.info(format_metrics_table(rep, f"CONFIDENCE ON HELD-OUT PART - {name}"))

    os.makedirs(cfg.plots_dir, exist_ok=True)
    plot_reliability_diagram(before, os.path.join(cfg.plots_dir, "reliability_before_calibration.png"),
                             "Reliability (held-out, before calibration)")
    plot_reliability_diagram(after, os.path.join(cfg.plots_dir, "reliability_after_calibration.png"),
                             "Reliability (held-out, after calibration)")
    plot_risk_coverage({"before calibration": before, "after calibration": after},
                       os.path.join(cfg.plots_dir, "risk_coverage_calibration.png"),
                       "Risk-coverage on the held-out part")

    out_dir = os.path.join(cfg.checkpoint_dir, CALIBRATED_DIR)
    save_calibrated(model, out_dir, {"calibration_ids": cal_ids, "heldout_ids": held_ids,
                                     "fraction": ev.calibration_fraction, "seed": ev.calibration_seed})
    report_file = os.path.join(cfg.run_dir, "calibration_metrics.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({"calibration_samples": len(z_cal), "heldout_samples": len(z_held),
                   "calibration_exact_match": float(z_cal.mean()), "heldout_exact_match": float(z_held.mean()),
                   "fit": fit, "before": before, "after": after}, f, indent=2)
    logger.info(f"Calibrated checkpoint saved to {out_dir}")
    logger.info(f"Calibration report saved to {report_file}")


def infer_stage(cfg: PipelineConfig) -> None:
    """Stage 3: Run inference on sample images."""
    logger.info("=" * 60)
    logger.info("STAGE: INFERENCE")
    logger.info("=" * 60)

    setup_directories(cfg.run_dir)
    logger.info(f"Run folder: {cfg.run_dir}")

    adapter_path = cfg.evaluation.adapter_path
    calibrated_path = os.path.join(cfg.checkpoint_dir, CALIBRATED_DIR)
    if cfg.is_ms_mgca and os.path.isdir(calibrated_path):
        logger.info(f"Using the calibrated checkpoint {calibrated_path}")
        adapter_path = calibrated_path
    if not os.path.isdir(adapter_path):
        if cfg.model_type == "crnn" or cfg.is_yolo or cfg.is_ms_mgca:
            # Unlike Qwen's adapter, these have no usable base model to fall back to (the CRNN is random,
            # the YOLO classifier has no head for our vocabulary, MS-MGCA's modules are untrained).
            raise FileNotFoundError(f"No trained {cfg.model_type} model at {adapter_path}. "
                                    f"Run STAGE = \"train\" first.")
        logger.warning(f"No trained adapter at {adapter_path}; evaluating the base model")
        adapter_path = None

    logger.info("Loading model and processor...")
    if cfg.model_type == "qwen-vl":
        model, processor = load_qwen_vl(cfg, adapter_path=adapter_path)
    elif cfg.is_ms_mgca:
        model, processor = load_ms_mgca(cfg, checkpoint_dir=adapter_path)
    elif cfg.model_type == "crnn":
        model, processor = load_crnn(cfg, checkpoint_dir=adapter_path)
    elif cfg.is_yolo:
        model, processor = load_yolo_classifier(cfg, checkpoint_dir=adapter_path)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")

    model.eval()

    logger.info("Parsing datasets...")
    eval_records, eval_split = load_eval_records(cfg)

    n_samples = cfg.evaluation.infer_samples
    sample_records = eval_records[:n_samples] if n_samples else eval_records
    logger.info(f"Evaluating on {len(sample_records)} {eval_split} samples")

    use_amp = torch.cuda.is_available() and cfg.model_type in ("qwen-vl", "ms-mgca")  # CRNN runs in fp32
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=cfg.compute_dtype, enabled=use_amp):
        metrics, results = run_generation_eval(
            model, processor, sample_records, cfg.model.max_pixels,
            cfg.evaluation.max_new_tokens, desc="Running inference",
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

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

    if cfg.is_ms_mgca:
        efficiency = {"samples": len(results), "latency_ms_per_sample": 1000 * elapsed / max(len(results), 1),
                      "throughput_samples_per_sec": len(results) / elapsed if elapsed else 0.0,
                      "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else None}
        with open(os.path.join(cfg.run_dir, "inference_efficiency.json"), "w", encoding="utf-8") as f:
            json.dump(efficiency, f, indent=2)
        logger.info(f"Latency {efficiency['latency_ms_per_sample']:.0f} ms/sample, "
                    f"peak VRAM {efficiency['peak_vram_mb'] or 0:.0f} MB")

    # MS-MGCA with the confidence head: how well confidence separates right from wrong transcriptions.
    # With a calibrated checkpoint the head was fitted on the calibration part, so only the held-out part counts.
    # On the held-out test set every image is unseen by calibration, so no filtering is needed.
    held = heldout_ids(adapter_path) if cfg.is_ms_mgca and adapter_path and eval_split == "validation" else None
    report = confidence_report(model, results, only_ids=held) if cfg.is_ms_mgca else None
    if report:
        scope = f"held-out part ({report['num_samples']} images)" if held else f"all {report['num_samples']} images"
        report["evaluated_on"] = scope
        logger.info(format_metrics_table(report, f"CONFIDENCE METRICS - {scope}"))
        confidence_file = os.path.join(cfg.run_dir, "inference_confidence_metrics.json")
        with open(confidence_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Confidence metrics saved to {confidence_file}")
        os.makedirs(cfg.plots_dir, exist_ok=True)
        plot_reliability_diagram(report, os.path.join(cfg.plots_dir, "reliability_diagram.png"),
                                 f"Reliability diagram ({cfg.model.ms_mgca_experiment})")
        plot_risk_coverage({cfg.model.ms_mgca_experiment: report},
                           os.path.join(cfg.plots_dir, "risk_coverage.png"), "Risk-coverage")


def parse_args(argv=None) -> argparse.Namespace:
    """Optional command-line overrides of the constants at the top of this file (nothing given = the constants)."""
    p = argparse.ArgumentParser(
        description="Doctor-handwriting VLM pipeline. Every option overrides the constant of the same name "
                    "at the top of main.py.")
    p.add_argument("--stage", choices=["augment", "train", "calibrate", "infer"], help=f"STAGE (default {STAGE})")
    p.add_argument("--model", choices=["qwen-vl", "crnn", "yolo", "yolo26", "ms-mgca"],
                   help=f"MODEL_CHOICE (default {MODEL_CHOICE})")
    p.add_argument("--checkpoint", help=f"QWEN_CHECKPOINT (default {QWEN_CHECKPOINT})")
    p.add_argument("--experiment", help=f"MSMGCA_EXPERIMENT, e1_baseline ... e6_full_model (default {MSMGCA_EXPERIMENT})")
    p.add_argument("--queries", type=int, help=f"MSMGCA_NUM_QUERIES, K = 1, 4 or 8 (default {MSMGCA_NUM_QUERIES})")
    p.add_argument("--profile", help=f"HARDWARE_PROFILE (default {HARDWARE_PROFILE})")
    p.add_argument("--dataset", choices=["combined", "dhp", "rxhand"], help=f"DATASET_MODE (default {DATASET_MODE})")
    p.add_argument("--augmentation", choices=["online", "offline"], help=f"AUGMENTATION_MODE (default {AUGMENTATION_MODE})")
    p.add_argument("--official-split", action="store_true", default=None,
                   help=f"RXHAND_OFFICIAL_SPLIT = True (default {RXHAND_OFFICIAL_SPLIT})")
    p.add_argument("--heldout-test", action="store_true", default=None,
                   help=f"RXHAND_HELDOUT_TEST = True (default {RXHAND_HELDOUT_TEST})")
    p.add_argument("--resume", action="store_true",
                   help="MS-MGCA: continue an interrupted run. train picks up the newest run of this exact setting "
                        "(model, experiment, backbone, K, dataset, augmentation) from its latest checkpoint, with the "
                        "original --epochs; a stage whose output already exists (finished training, calibrated "
                        "checkpoint, inference results) is skipped")
    p.add_argument("--epochs", type=int, help="number of training epochs (default: the hardware profile's)")
    p.add_argument("--samples", type=int, help="limit the validation images used by calibrate/infer (default: all)")
    return p.parse_args(argv)


def main(argv=None):
    """Main pipeline orchestrator."""
    args = parse_args(argv)
    pick = lambda cli, constant: constant if cli is None else cli
    cfg = PipelineConfig(
        stage=pick(args.stage, STAGE),
        resume=args.resume,
        model_type=pick(args.model, MODEL_CHOICE),
        hardware_profile=pick(args.profile, HARDWARE_PROFILE),
        # Passed at construction: the run folder name is derived from the checkpoint and dataset mode.
        dataset=DatasetConfig(mode=pick(args.dataset, DATASET_MODE),
                              rxhand_official_split=pick(args.official_split, RXHAND_OFFICIAL_SPLIT),
                              rxhand_heldout_test=pick(args.heldout_test, RXHAND_HELDOUT_TEST)),
        augmentation=AugmentationConfig(mode=pick(args.augmentation, AUGMENTATION_MODE)),
        model=ModelConfig(qwen_checkpoint=pick(args.checkpoint, QWEN_CHECKPOINT), yolo_checkpoint=YOLO_CHECKPOINT,
                          yolo26_checkpoint=YOLO26_CHECKPOINT,
                          ms_mgca_experiment=pick(args.experiment, MSMGCA_EXPERIMENT),
                          ms_mgca_num_queries=pick(args.queries, MSMGCA_NUM_QUERIES)),
    )
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.samples:
        cfg.evaluation.infer_samples = args.samples

    logger.info("\n" + "=" * 60)
    logger.info("VLM TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Stage: {cfg.stage}")
    logger.info(f"Model: {cfg.model_type} ({cfg.model_name})")
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
    logger.info(f"Dataset mode: {cfg.dataset.mode}, augmentation: {cfg.augmentation.mode}"
                + (f" ({cfg.augmentation.offline_root})" if cfg.augmentation.mode == "offline" else ""))
    logger.info(f"Output dir: {cfg.output_dir}")
    if cfg.stage != "augment":
        logger.info(f"Run dir: {cfg.run_dir}")
    logger.info("=" * 60 + "\n")

    # Fail fast (before downloading the model) when the GPU is not usable.
    if cfg.stage in ("train", "calibrate", "infer") and cfg.device == "cuda" and not torch.cuda.is_available():
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

    if cfg.augmentation.mode == "offline" and cfg.stage in ("train", "calibrate", "infer"):
        absent = missing_offline_manifests(cfg.dataset, cfg.augmentation)
        if absent:
            logger.error("AUGMENTATION_MODE is \"offline\" but the augmented dataset is missing:\n  "
                         + "\n  ".join(absent))
            logger.error("Run once with STAGE = \"augment\" (same DATASET_MODE) to create it, "
                         "or set AUGMENTATION_MODE = \"online\".")
            sys.exit(1)

    if cfg.resume:
        if not cfg.is_ms_mgca:
            logger.error('--resume is implemented for MODEL_CHOICE = "ms-mgca" only.')
            sys.exit(1)
        if cfg.stage != "augment" and stage_output_exists(cfg):
            logger.info(f"--resume: the {cfg.stage} stage is already done in {cfg.run_dir}; nothing to do.")
            return

    if cfg.stage == "augment":
        augment_stage(cfg)
    elif cfg.stage == "train":
        train_stage(cfg)
    elif cfg.stage == "calibrate":
        calibrate_stage(cfg)
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
