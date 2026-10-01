import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
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


def load_qwen_vl(cfg):
    """
    Load Qwen-VL model with 4-bit quantization and LoRA adaptation.
    Supports both Qwen 2.5-VL and Qwen 3-VL models.

    Args:
        cfg: PipelineConfig object with quantization and LoRA settings

    Returns:
        model, processor: Quantized and LoRA-adapted model and processor
    """
    checkpoint = cfg.model.qwen_checkpoint
    version = _detect_qwen_version(checkpoint)
    model_class = _get_qwen_model_class(version)

    logger.info(f"Detected Qwen version: {version}")
    logger.info(f"Loading {checkpoint} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=cfg.quantization.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
    )

    # Pin everything to GPU 0: "auto" may offload to CPU, which breaks training.
    model = model_class.from_pretrained(
        checkpoint,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=cfg.quantization.bnb_4bit_compute_dtype,
    )

    processor = AutoProcessor.from_pretrained(checkpoint)
    processor.tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

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


def count_parameters(model):
    """Return count of trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
