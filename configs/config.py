from dataclasses import dataclass, field
from typing import Literal, List
import torch


@dataclass
class QuantizationConfig:
    """4-bit quantization configuration for memory-constrained training."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation configuration for parameter-efficient fine-tuning."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """Training hyperparameters optimized for 4GB VRAM."""
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_workers: int = 2
    pin_memory: bool = False
    use_bf16: bool = False
    use_fp16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42


@dataclass
class DatasetConfig:
    """Dataset paths and processing parameters."""
    dataset_root: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\Dataset"
    dhp_root: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\Dataset\DHP"
    dhp_labels: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\Dataset\DHP\doctor_handwriting_labels.csv"
    dhp_images: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\Dataset\DHP\img\img"

    rxhand_root: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\Dataset\RxHand"
    rxhand_train_labels: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\Dataset\RxHand\Train_Label.csv"
    rxhand_test_labels: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\Dataset\RxHand\Test_Labels.csv"
    rxhand_train_images: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\Dataset\RxHand\Train_Set"
    rxhand_test_images: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\Dataset\RxHand\Test_Set"

    train_split: float = 0.85
    val_split: float = 0.15
    max_samples: int = None


@dataclass
class AugmentationConfig:
    """Data augmentation parameters."""
    enable_augmentation: bool = True
    motion_blur_prob: float = 0.3
    elastic_deform_prob: float = 0.3
    perspective_prob: float = 0.2
    brightness_contrast_prob: float = 0.5
    rotation_limit: int = 15
    shift_limit: float = 0.1
    scale_limit: float = 0.2


@dataclass
class ModelConfig:
    """Model architecture and checkpoint parameters."""
    model_type: Literal["qwen-vl", "yolo"] = "qwen-vl"
    qwen_checkpoint: str = "Qwen/Qwen3-VL-2B-Instruct"
    yolo_checkpoint: str = "yolov8n.pt"
    max_pixels: int = 512 * 512
    min_pixels: int = 256 * 256
    max_seq_length: int = 512
    gradient_checkpointing: bool = True


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    stage: Literal["augment", "train", "infer"] = "train"
    model_type: Literal["qwen-vl", "yolo"] = "qwen-vl"
    device: str = "cuda"
    mixed_precision: str = "fp16"

    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    output_dir: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\runs"
    checkpoint_dir: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\runs\checkpoints"
    log_dir: str = r"D:\Data_D\All Programming Language\2_Doctor Prescription\runs\logs"
    save_steps: int = 100
    eval_steps: int = 100
    log_steps: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_type != self.model.model_type:
            self.model.model_type = self.model_type
        if self.stage != "infer" and self.training.per_device_train_batch_size > 1:
            print(f"[WARNING] Reducing batch size to 1 for 4GB VRAM safety")
            self.training.per_device_train_batch_size = 1
