import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType, PeftModel
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


def _detect_qwen_version(checkpoint: str) -> str:
    """Detect Qwen model version from checkpoint name."""
    if "qwen3" in checkpoint.lower():
        return "qwen3"
    elif "qwen2.5" in checkpoint.lower() or "qwen-2.5" in checkpoint.lower():
        return "qwen2.5"
    else:
        return "qwen2.5"


def _get_qwen_model_class(version: str):
    """Get appropriate Qwen model class for the version."""
    if version == "qwen3":
        try:
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration
        except ImportError:
            logger.warning("Qwen3VLForConditionalGeneration not available, falling back to Qwen2.5")
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration


# Restrict LoRA to the language model; vision tower layer names differ between Qwen versions.
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_qwen_vl(cfg, adapter_path: str = None):
    """
    Load Qwen-VL model (4-bit quantized or full bf16/fp16, per hardware profile) with LoRA.
    Supports both Qwen 2.5-VL and Qwen 3-VL models.

    Args:
        cfg: PipelineConfig object with quantization and LoRA settings
        adapter_path: If given, load this trained LoRA adapter for inference
            instead of attaching a fresh trainable one.

    Returns:
        model, processor: Quantized and LoRA-adapted model and processor
    """
    checkpoint = cfg.model.qwen_checkpoint
    version = _detect_qwen_version(checkpoint)
    model_class = _get_qwen_model_class(version)

    quantize = cfg.quantization.load_in_4bit
    compute_dtype = cfg.quantization.bnb_4bit_compute_dtype

    logger.info(f"Detected Qwen version: {version}")
    logger.info(f"Loading {checkpoint} "
                f"{'with 4-bit quantization' if quantize else f'in {compute_dtype} (no quantization)'}...")

    # Naming "visual" here keeps the vision tower in compute_dtype. Passing a list replaces
    # transformers' automatic lm_head exclusion, so lm_head is listed explicitly.
    skip_modules = None if cfg.quantization.quantize_vision else ["lm_head", "visual"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        llm_int8_skip_modules=skip_modules,
    ) if quantize else None

    # Pin everything to GPU 0: "auto" may offload to CPU, which breaks training.
    model = model_class.from_pretrained(
        checkpoint,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )

    processor = AutoProcessor.from_pretrained(checkpoint)
    processor.tokenizer.padding_side = "right"

    if adapter_path:
        logger.info(f"Loading trained LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        return model, processor

    if quantize and cfg.quantization.upcast_fp32:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.model.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif cfg.model.gradient_checkpointing:
        # Also the quantized path without fp32 upcast (8B on 12GB): same steps as prepare_model_for_kbit_training
        # minus casting the embeddings/lm_head to fp32.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Frozen base weights: inputs must require grad for checkpointed LoRA layers to backprop.
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_REGEX,
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0

    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")

    return model, processor


def load_yolo_model(cfg):
    """
    Load YOLO model for detection tasks.

    Args:
        cfg: PipelineConfig object

    Returns:
        model: YOLO model instance
    """
    logger.info(f"Loading {cfg.model.yolo_checkpoint}...")

    model = YOLO(cfg.model.yolo_checkpoint)

    if cfg.device == "cuda":
        model = model.to("cuda")

    logger.info(f"YOLO model loaded on {model.device}")

    return model


CRNN_CONFIG_FILE = "crnn_config.json"
CRNN_WEIGHTS_FILE = "model.safetensors"  # same name the HF Trainer uses for plain nn.Module checkpoints

# VGG16 convolution stack: 13 conv layers (ints = output channels) and 5 poolings, as in the paper.
# "M1" halves height and width; "M2" halves only the height so the width (= CTC time steps) is kept.
_VGG16_CFG = [64, 64, "M1", 128, 128, "M1", 256, 256, 256, "M2",
              512, 512, 512, "M2", 512, 512, 512, "M2"]


class BidirectionalLSTM(nn.Module):
    """BiLSTM whose forward and backward outputs are combined by a linear layer at the end of the layer."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(2 * n_hidden, n_out)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.embedding(out)


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network (Achkar et al., 2019): 13 VGG16 conv layers (ReLU after
    each) -> 3 stacked BiLSTM layers -> per-time-step class scores, trained with CTC loss.

    Input  pixel_values: [B, 1, H, W] grayscale. Output logits: [B, T, num_classes] with T = W / 4.
    Class 0 is the CTC blank; classes 1..N are the charset characters.
    Called with `labels` ([B, L] character ids, padded with -100) it also returns the CTC loss, so it
    plugs straight into the HF Trainer."""

    def __init__(self, num_classes: int, lstm_hidden: int = 256, lstm_layers: int = 3,
                 batch_norm: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.batch_norm = batch_norm

        layers, in_ch = [], 1
        for item in _VGG16_CFG:
            if isinstance(item, str):
                layers.append(nn.MaxPool2d((2, 2) if item == "M1" else (2, 1)))
                continue
            layers.append(nn.Conv2d(in_ch, item, kernel_size=3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(item))
            layers.append(nn.ReLU(inplace=True))
            in_ch = item
        self.cnn = nn.Sequential(*layers)
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse the remaining height

        sizes = [in_ch] + [lstm_hidden] * (lstm_layers - 1) + [num_classes]
        self.rnn = nn.Sequential(*[
            BidirectionalLSTM(sizes[i], lstm_hidden, sizes[i + 1]) for i in range(lstm_layers)
        ])

    def forward(self, pixel_values, labels=None):
        features = self.height_pool(self.cnn(pixel_values)).squeeze(2)  # [B, C, T]
        logits = self.rnn(features.permute(0, 2, 1))                      # [B, T, classes]
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self.ctc_loss(logits, labels)
        return out

    @staticmethod
    def ctc_loss(logits, labels):
        # CTC is numerically unsafe in fp16/bf16, so score in fp32.
        log_probs = logits.float().log_softmax(-1).transpose(0, 1)  # [T, B, classes]
        labels = labels.to(logits.device)
        mask = labels != -100
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device)
        return F.ctc_loss(log_probs, labels[mask], input_lengths, mask.sum(1),
                          blank=0, reduction="mean", zero_infinity=True)


class CRNNProcessor:
    """CRNN counterpart of the Qwen processor: image preprocessing, character <-> id mapping
    and greedy CTC decoding. Holds the charset, which is stored next to the weights."""

    def __init__(self, charset: str, img_height: int = 64, img_width: int = 340):
        self.charset = charset
        self.img_height = img_height
        self.img_width = img_width
        self._index = {ch: i + 1 for i, ch in enumerate(charset)}  # 0 is the CTC blank

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Grayscale, aspect-preserving resize into height x width, white padding, scaled to [-1, 1]."""
        image = image.convert("L")
        w, h = image.size
        scale = min(self.img_height / h, self.img_width / w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = Image.new("L", (self.img_width, self.img_height), 255)
        canvas.paste(image.resize((new_w, new_h), Image.BICUBIC), (0, (self.img_height - new_h) // 2))
        pixels = np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy((pixels - 0.5) / 0.5).unsqueeze(0)

    def encode(self, text: str) -> List[int]:
        """Characters outside the charset are dropped (the charset is built from all labels)."""
        return [self._index[ch] for ch in text if ch in self._index]

    def decode(self, logits: torch.Tensor) -> List[str]:
        """Greedy CTC decoding of [B, T, classes] scores: argmax, collapse repeats, drop blanks."""
        texts = []
        for sequence in logits.argmax(-1).cpu().tolist():
            chars, previous = [], 0
            for idx in sequence:
                if idx != previous and idx != 0:
                    chars.append(self.charset[idx - 1])
                previous = idx
            texts.append("".join(chars))
        return texts

    @torch.no_grad()
    def transcribe(self, model: CRNN, image: Image.Image) -> str:
        device = next(model.parameters()).device
        pixel_values = self.preprocess(image).unsqueeze(0).to(device)
        return self.decode(model(pixel_values=pixel_values)["logits"])[0]


def build_charset(records: List[Dict]) -> str:
    """Sorted unique characters over every label (conversations[1]) of the given records."""
    return "".join(sorted({ch for r in records for ch in r["conversations"][1]["value"]}))


def save_crnn(model: CRNN, processor: CRNNProcessor, directory: str) -> None:
    """Write weights + architecture/charset so the model can be rebuilt without the training run."""
    os.makedirs(directory, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              os.path.join(directory, CRNN_WEIGHTS_FILE))
    spec = {"charset": processor.charset, "img_height": processor.img_height,
            "img_width": processor.img_width, "lstm_hidden": model.lstm_hidden,
            "lstm_layers": model.lstm_layers, "batch_norm": model.batch_norm}
    with open(os.path.join(directory, CRNN_CONFIG_FILE), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)


def load_crnn(cfg, charset: str = None, checkpoint_dir: str = None):
    """
    Build the CRNN (from scratch, or from a directory written by save_crnn).

    Args:
        cfg: PipelineConfig (cfg.crnn holds the architecture when building from scratch)
        charset: characters to recognize; required when training from scratch
        checkpoint_dir: if given, restore architecture, charset and weights from it (inference)

    Returns:
        model, processor
    """
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, CRNN_CONFIG_FILE), encoding="utf-8") as f:
            spec = json.load(f)
    else:
        if not charset:
            raise ValueError("A charset is required to build a new CRNN")
        c = cfg.crnn
        spec = {"charset": charset, "img_height": c.img_height, "img_width": c.img_width,
                "lstm_hidden": c.lstm_hidden, "lstm_layers": c.lstm_layers, "batch_norm": c.batch_norm}

    processor = CRNNProcessor(spec["charset"], spec["img_height"], spec["img_width"])
    model = CRNN(len(spec["charset"]) + 1, spec["lstm_hidden"], spec["lstm_layers"], spec["batch_norm"])
    if checkpoint_dir:
        model.load_state_dict(load_file(os.path.join(checkpoint_dir, CRNN_WEIGHTS_FILE)))
        logger.info(f"Loaded CRNN weights from {checkpoint_dir}")
    if cfg.device == "cuda":
        model = model.to("cuda")

    trainable, total = count_parameters(model)
    logger.info(f"CRNN: {len(spec['charset'])} characters + blank, "
                f"{trainable:,} trainable parameters ({total:,} total)")
    return model, processor


def count_parameters(model):
    """Return count of trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
