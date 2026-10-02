import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, List, Optional
import torch


# All paths are resolved relative to the repository root, so the project runs on any machine
# as long as the folder structure is kept:
#   <repo>/Dataset/DHP/...  and  <repo>/Dataset/RxHand/...
# Override locations without editing code via environment variables:
#   VLM_DATASET_ROOT=/mnt/data/Dataset   VLM_OUTPUT_DIR=/mnt/runs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(os.environ.get("VLM_DATASET_ROOT", PROJECT_ROOT / "Dataset")).resolve()
OUTPUT_ROOT = Path(os.environ.get("VLM_OUTPUT_DIR", PROJECT_ROOT / "runs")).resolve()


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
    dataset_root: str = DATASET_ROOT.as_posix()
    # Sub-paths left empty are derived from dataset_root in __post_init__.
    dhp_root: str = ""
    dhp_labels: str = ""
    dhp_images: str = ""

    rxhand_root: str = ""
    rxhand_train_labels: str = ""
    rxhand_test_labels: str = ""
    rxhand_train_images: str = ""
    rxhand_test_images: str = ""

    train_split: float = 0.80
    val_split: float = 0.20 
    max_samples: int = None

    def __post_init__(self):
        root = Path(self.dataset_root)
        self.dhp_root = self.dhp_root or (root / "DHP").as_posix()
        self.dhp_labels = self.dhp_labels or (Path(self.dhp_root) / "doctor_handwriting_labels.csv").as_posix()
        self.dhp_images = self.dhp_images or (Path(self.dhp_root) / "img" / "img").as_posix()

        self.rxhand_root = self.rxhand_root or (root / "RxHand").as_posix()
        rx = Path(self.rxhand_root)
        self.rxhand_train_labels = self.rxhand_train_labels or (rx / "Train_Label.csv").as_posix()
        self.rxhand_test_labels = self.rxhand_test_labels or (rx / "Test_Labels.csv").as_posix()
        self.rxhand_train_images = self.rxhand_train_images or (rx / "Train_Set").as_posix()
        self.rxhand_test_images = self.rxhand_test_images or (rx / "Test_Set").as_posix()

    def missing_paths(self) -> List[str]:
        """Return expected dataset files/folders that do not exist."""
        paths = [
            self.dhp_labels, self.dhp_images,
            self.rxhand_train_labels, self.rxhand_test_labels,
            self.rxhand_train_images, self.rxhand_test_images,
        ]
        return [p for p in paths if not os.path.exists(p)]


@dataclass
class EvaluationConfig:
    """Generation-based evaluation (WER/CER/accuracy) settings."""
    # Val samples decoded with model.generate at every eval step during training.
    # Generation is slow, so keep this small on a 4GB GPU; None = full val set.
    train_eval_samples: Optional[int] = 50
    # Val samples used by the "infer" stage; None = full val set.
    infer_samples: Optional[int] = None
    max_new_tokens: int = 64
    # Adapter loaded by the "infer" stage (defaults to <checkpoint_dir>/final).
    adapter_path: str = ""


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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    output_dir: str = OUTPUT_ROOT.as_posix()
    # Left empty -> derived from output_dir in __post_init__.
    checkpoint_dir: str = ""
    log_dir: str = ""
    save_steps: int = 100
    eval_steps: int = 100
    log_steps: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.checkpoint_dir = self.checkpoint_dir or (Path(self.output_dir) / "checkpoints").as_posix()
        self.log_dir = self.log_dir or (Path(self.output_dir) / "logs").as_posix()
        self.evaluation.adapter_path = self.evaluation.adapter_path or (Path(self.checkpoint_dir) / "final").as_posix()

        if self.model_type != self.model.model_type:
            self.model.model_type = self.model_type
        if self.stage != "infer" and self.training.per_device_train_batch_size > 1:
            print(f"[WARNING] Reducing batch size to 1 for 4GB VRAM safety")
            self.training.per_device_train_batch_size = 1
