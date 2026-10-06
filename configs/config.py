import os
from datetime import datetime
from dataclasses import dataclass, field, replace
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
class HardwareProfile:
    """Machine-specific settings applied on top of the defaults below."""
    description: str
    load_in_4bit: bool
    compute_dtype: torch.dtype
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    num_workers: int
    pin_memory: bool
    max_pixels: int
    train_eval_samples: Optional[int]
    allow_tf32: bool
    # kbit_upcast_fp32: run peft's prepare_model_for_kbit_training, which upcasts every non-quantized
    #   weight (embeddings, lm_head) to fp32 - costs ~5GB extra on an 8B model, so off for the 12GB GPU.
    # quantize_vision: also 4-bit quantize the vision tower; False keeps it in bf16 (better
    #   handwriting features for ~0.6GB more VRAM).
    kbit_upcast_fp32: bool = True
    quantize_vision: bool = True


# Select one in main.py via HARDWARE_PROFILE (or "auto" to pick by detected VRAM).
# Both keep an effective batch size of 8 so results stay comparable across machines.
HARDWARE_PROFILES = {
    # Laptop: Ryzen 7, 16GB RAM, RTX 3050 Ti 4GB
    "laptop_3050ti": HardwareProfile(
        description="RTX 3050 Ti 4GB / 16GB RAM - 4-bit NF4 + LoRA, fp16",
        load_in_4bit=True,
        compute_dtype=torch.float16,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_workers=2,
        pin_memory=False,
        max_pixels=512 * 512,
        train_eval_samples=50,
        allow_tf32=False,
    ),
    # PC: Ryzen 5 7500F (6C/12T), 32GB RAM, RTX 5070 12GB (Blackwell, sm_120)
    # Sized for Qwen3-VL-8B: bf16 weights alone are ~16GB, so it runs as QLoRA (NF4 weights ~6-7GB,
    # bf16 compute, vision tower kept in bf16). Batch 2 x accum 4 = the paper's effective batch of 8;
    # if CUDA OOM, drop to 1 x 8 (same effective batch).
    "pc_5070": HardwareProfile(
        description="RTX 5070 12GB / 32GB RAM - Qwen3-VL-8B QLoRA (NF4 + bf16 compute)",
        load_in_4bit=True,
        compute_dtype=torch.bfloat16,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_workers=4,
        pin_memory=True,
        max_pixels=768 * 768,
        train_eval_samples=100,
        allow_tf32=True,
        kbit_upcast_fp32=False,
        quantize_vision=False,
    ),
}


def find_latest_run(model_dir: Path) -> Optional[Path]:
    """Most recently trained run under <output_dir>/<model_name> (one with checkpoints/final)."""
    finals = [p for p in model_dir.glob("*/checkpoints/final") if p.is_dir()]
    return max(finals, key=lambda p: p.stat().st_mtime).parent.parent if finals else None


def detect_hardware_profile() -> str:
    """Pick a profile from the detected GPU's VRAM (>=10GB -> pc_5070)."""
    if not torch.cuda.is_available():
        return "laptop_3050ti"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return "pc_5070" if vram_gb >= 10 else "laptop_3050ti"


@dataclass
class QuantizationConfig:
    """4-bit quantization configuration for memory-constrained training."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True
    # Set from the hardware profile (see HardwareProfile for what they do).
    upcast_fp32: bool = True
    quantize_vision: bool = True


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation configuration for parameter-efficient fine-tuning.
    r / alpha / dropout follow Table 1 of the VLM handwriting fine-tuning paper (rank 32, alpha 64, dropout 0.05)."""
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        # "q_proj", "k_proj", "v_proj", "o_proj",
        # "gate_proj", "up_proj", "down_proj"
        "q_proj", "v_proj", "o_proj"
    ])


@dataclass
class TrainingConfig:
    """Training hyperparameters. Batch/accumulation/precision are overwritten by the hardware profile;
    the rest follows Table 1 of the paper (7B/8B tier: 10 epochs, lr 2e-4, 8-bit AdamW,
    cosine with restarts, weight decay 0.01, seed 3407)."""
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 10
    learning_rate: float = 2e-4
    # Paper uses 2000 warmup steps on a far larger synthetic set; here an epoch is only ~560 optimizer
    # steps (~4.5k train images / batch 8), so warm up over ~5% of the 10-epoch run instead.
    warmup_steps: int = 300
    weight_decay: float = 0.01
    optim: str = "adamw_bnb_8bit"  # 8-bit AdamW (bitsandbytes); "adamw_torch" for plain AdamW
    lr_scheduler_type: str = "cosine_with_restarts"
    lr_scheduler_num_cycles: int = 2  # one warm restart halfway; paper does not state the cycle count
    # Keep the checkpoint with the lowest val loss (small dataset + 10 epochs can overfit late),
    # and keep only the newest few so a long run does not fill the disk.
    load_best_model_at_end: bool = True
    save_total_limit: int = 3
    max_grad_norm: float = 1.0
    num_workers: int = 2
    pin_memory: bool = False
    use_bf16: bool = True         # Set to True since your 5070 natively rocks bf16
    use_fp16: bool = False        # Turn off fp16 when using bf16
    gradient_checkpointing: bool = True
    seed: int = 3407


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
    qwen_checkpoint: str = "Qwen/Qwen3-VL-8B-Instruct"
    yolo_checkpoint: str = "yolov8n.pt"
    max_pixels: int = 512 * 512
    min_pixels: int = 256 * 256
    max_seq_length: int = 512
    gradient_checkpointing: bool = True


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    stage: Literal["augment", "train", "infer"] = "train"
    hardware_profile: str = "auto"  # "auto", or a key of HARDWARE_PROFILES
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
    # Every run gets its own folder: <output_dir>/<model_name>/<run_name>/
    #   checkpoints/  logs/  plots/  metrics_history.{json,csv}  eval_metrics.json  run_config.json
    # Left empty -> "run_<timestamp>_<hardware_profile>" for train; for infer, the latest trained run.
    run_name: str = ""
    # Left empty -> derived from output_dir/run_name in __post_init__.
    run_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    plots_dir: str = ""
    save_steps: int = 100
    eval_steps: int = 100
    log_steps: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_type != self.model.model_type:
            self.model.model_type = self.model_type
        self.apply_hardware_profile()
        self.resolve_run_paths()

    @property
    def model_name(self) -> str:
        """Folder name for the model, e.g. "Qwen3-VL-2B-Instruct" or "yolov8n"."""
        if self.model_type == "yolo":
            name = Path(self.model.yolo_checkpoint).name
            return name[:-3] if name.endswith(".pt") else name
        return self.model.qwen_checkpoint.rstrip("/").split("/")[-1]

    def resolve_run_paths(self) -> None:
        """Derive this run's folder and its sub-folders (directories are created by the stages)."""
        model_dir = Path(self.output_dir) / self.model_name
        if not self.run_name:
            latest = find_latest_run(model_dir) if self.stage == "infer" else None
            self.run_name = latest.name if latest else (
                f"run_{datetime.now():%Y%m%d-%H%M%S}_{self.hardware_profile}")
        run = Path(self.run_dir or model_dir / self.run_name)
        self.run_dir = run.as_posix()
        self.checkpoint_dir = self.checkpoint_dir or (run / "checkpoints").as_posix()
        self.log_dir = self.log_dir or (run / "logs").as_posix()
        self.plots_dir = self.plots_dir or (run / "plots").as_posix()
        self.evaluation.adapter_path = self.evaluation.adapter_path or (Path(self.checkpoint_dir) / "final").as_posix()

    def apply_hardware_profile(self) -> None:
        """Overwrite machine-dependent settings with the selected hardware profile."""
        if self.hardware_profile == "auto":
            self.hardware_profile = detect_hardware_profile()
        if self.hardware_profile not in HARDWARE_PROFILES:
            raise ValueError(f"Unknown hardware profile '{self.hardware_profile}'. "
                             f"Options: auto, {', '.join(HARDWARE_PROFILES)}")
        hw = HARDWARE_PROFILES[self.hardware_profile]

        if hw.compute_dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            print("[WARNING] GPU does not support bf16, falling back to fp16")
            hw = replace(hw, compute_dtype=torch.float16)
        use_bf16 = hw.compute_dtype == torch.bfloat16

        self.quantization.load_in_4bit = hw.load_in_4bit
        self.quantization.bnb_4bit_compute_dtype = hw.compute_dtype
        self.quantization.upcast_fp32 = hw.kbit_upcast_fp32
        self.quantization.quantize_vision = hw.quantize_vision
        self.mixed_precision = "bf16" if use_bf16 else "fp16"

        t = self.training
        t.per_device_train_batch_size = hw.per_device_train_batch_size
        t.per_device_eval_batch_size = hw.per_device_eval_batch_size
        t.gradient_accumulation_steps = hw.gradient_accumulation_steps
        t.gradient_checkpointing = hw.gradient_checkpointing
        t.num_workers = hw.num_workers
        t.pin_memory = hw.pin_memory
        t.use_bf16 = use_bf16
        t.use_fp16 = not use_bf16

        self.model.gradient_checkpointing = hw.gradient_checkpointing
        self.model.max_pixels = hw.max_pixels
        self.evaluation.train_eval_samples = hw.train_eval_samples

        if hw.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.quantization.bnb_4bit_compute_dtype
