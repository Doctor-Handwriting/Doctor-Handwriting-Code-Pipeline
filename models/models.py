import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration
)
from peft import get_peft_model, LoraConfig, TaskType
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


def load_qwen_vl(cfg):
    """
    Load Qwen-VL model with 4-bit quantization and LoRA adaptation.

    Args:
        cfg: PipelineConfig object with quantization and LoRA settings

    Returns:
        model, processor: Quantized and LoRA-adapted model and processor
    """
    logger.info(f"Loading {cfg.model.qwen_checkpoint} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=cfg.quantization.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model.qwen_checkpoint,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        cfg.model.qwen_checkpoint,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.lora.target_modules,
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

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


def count_parameters(model):
    """Return count of trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
