"""Configuration types and machine plumbing for the pipeline.

Every value that changes a result (hyperparameters, LoRA, quantization, MS-MGCA architecture, split, evaluation,
augmentation) is set in the EXPERIMENT SETUP block of main.py and passed in explicitly: the dataclasses below have no
defaults for such values, so nothing is set silently here. This file only holds
  - the dataclass types the pipeline passes around,
  - the hardware profiles, which decide *memory* settings only: how the fixed effective batch is split into
    micro-batch x gradient accumulation, data-loader workers and gradient checkpointing (same experiment on every GPU),
  - path resolution and run-folder naming.
"""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, List, Optional, Tuple, Union
import torch


# All paths are resolved relative to the repository root, so the project runs on any machine
# as long as the folder structure is kept:
#   <repo>/Dataset/DHP/...  and  <repo>/Dataset/RxHand/...
# Override locations without editing code via environment variables:
#   VLM_DATASET_ROOT=/mnt/data/Dataset   VLM_OUTPUT_DIR=/mnt/runs   VLM_AUGMENTED_DIR=/mnt/data/Augmented_Dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(os.environ.get("VLM_DATASET_ROOT", PROJECT_ROOT / "Dataset")).resolve()
OUTPUT_ROOT = Path(os.environ.get("VLM_OUTPUT_DIR", PROJECT_ROOT / "runs")).resolve()
AUGMENTED_ROOT = Path(os.environ.get("VLM_AUGMENTED_DIR", PROJECT_ROOT / "Augmented_Dataset")).resolve()

MODEL_TYPES = ("qwen-vl", "crnn", "yolo", "yolo26", "ms-mgca")


# ---------------------------------------------------------------------------------------------------------------------
# Hardware profiles: memory only
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareProfile:
    """What a machine may change: how the effective batch (fixed in main.py) is split into micro-batch x gradient
    accumulation, data-loader workers, and gradient checkpointing. None of these change what is learned (gradient
    checkpointing recomputes activations, it does not change them), so a run gives the same experiment on every GPU.
    Precision, quantization, image resolution and all hyperparameters are fixed in main.py instead."""
    description: str
    # Qwen-VL per-device micro-batch for training / evaluation. MS-MGCA always uses 1 (its multi-scale path handles
    # one image at a time), so on every machine it reaches the effective batch through gradient accumulation only.
    vlm_micro_batch: int
    vlm_eval_batch: int
    gradient_checkpointing: bool
    num_workers: int
    pin_memory: bool
    # MODEL_CHOICE = "yolo": largest per-device batch that fits. MODEL_CHOICE = "yolo26": the same per model size
    # (n/s/m/l/x), measured on the 3050 Ti at 224px with bf16 autocast and extrapolated for the larger GPUs.
    yolo_micro_batch: int
    yolo26_micro_batch: Dict[str, int] = field(default_factory=dict)


# Select one in main.py via HARDWARE_PROFILE (or "auto" to pick by detected VRAM).
HARDWARE_PROFILES = {
    # Laptop: Ryzen 7, 16GB RAM, RTX 3050 Ti 4GB (Ampere, sm_86, bf16 supported). Qwen 2B at most.
    # YOLO26x: batch 16 x accumulation (peak ~1.7GB); anything past ~4GB spills into shared system memory and
    # trains ~10x slower.
    "laptop_3050ti": HardwareProfile(
        description="RTX 3050 Ti 4GB / 16GB RAM",
        vlm_micro_batch=1, vlm_eval_batch=2, gradient_checkpointing=True, num_workers=2, pin_memory=False,
        yolo_micro_batch=32, yolo26_micro_batch={"n": 32, "s": 32, "m": 32, "l": 32, "x": 16},
    ),
    # PC: Ryzen 5 3600 (6C/12T), 32GB RAM, RTX 3070 Ti 8GB (Ampere, sm_86). Sized for Qwen3-VL-2B/4B QLoRA;
    # Qwen3-VL-8B is not recommended here.
    "pc_3070ti": HardwareProfile(
        description="RTX 3070 Ti 8GB / 32GB RAM",
        vlm_micro_batch=1, vlm_eval_batch=2, gradient_checkpointing=True, num_workers=4, pin_memory=True,
        yolo_micro_batch=64, yolo26_micro_batch={"n": 64, "s": 64, "m": 64, "l": 64, "x": 32},
    ),
    # PC: Ryzen 5 7500F (6C/12T), 32GB RAM, RTX 5070 12GB (Blackwell, sm_120). Sized for Qwen3-VL-8B QLoRA.
    # If CUDA OOM on qwen-vl 8B, set vlm_micro_batch=1 (the effective batch stays the same).
    "pc_5070": HardwareProfile(
        description="RTX 5070 12GB / 32GB RAM",
        vlm_micro_batch=2, vlm_eval_batch=2, gradient_checkpointing=True, num_workers=4, pin_memory=True,
        yolo_micro_batch=64, yolo26_micro_batch={"n": 64, "s": 64, "m": 64, "l": 64, "x": 64},
    ),
    # PC: Ryzen 5 7500F (6C/12T), 32GB RAM, RTX 5070 Ti 16GB (Blackwell, sm_120).
    # If CUDA OOM on qwen-vl 8B: vlm_micro_batch 2, then 1 (the effective batch stays the same).
    "pc_5070ti": HardwareProfile(
        description="RTX 5070 Ti 16GB / 32GB RAM",
        vlm_micro_batch=4, vlm_eval_batch=4, gradient_checkpointing=True, num_workers=4, pin_memory=True,
        yolo_micro_batch=64, yolo26_micro_batch={"n": 64, "s": 64, "m": 64, "l": 64, "x": 64},
    ),
}


def detect_hardware_profile() -> str:
    """Pick a profile from the detected GPU's VRAM (>=14GB -> pc_5070ti, >=10GB -> pc_5070, >=7GB -> pc_3070ti).
    A "16GB" card reports ~15.9GiB, a "12GB" one ~11.9GiB and an "8GB" one ~8.0GiB, hence the thresholds."""
    if not torch.cuda.is_available():
        return "laptop_3050ti"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if vram_gb >= 14:
        return "pc_5070ti"
    if vram_gb >= 10:
        return "pc_5070"
    return "pc_3070ti" if vram_gb >= 7 else "laptop_3050ti"


# ---------------------------------------------------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------------------------------------------------
# YOLO26 classification checkpoints (ultralytics >= 8.4): yolo26{n,s,m,l,x}-cls.pt, ImageNet-pretrained.
YOLO26_SIZES = ("n", "s", "m", "l", "x")


def yolo26_size(checkpoint: str) -> str:
    """Size letter of a YOLO26 classification checkpoint ("yolo26m-cls.pt" -> "m")."""
    match = re.fullmatch(r"yolo26([nslmx])-cls(?:\.pt)?", Path(checkpoint).name)
    if not match:
        raise ValueError(
            f"'{checkpoint}' is not a YOLO26 classification checkpoint. Use one of "
            f"{', '.join(f'yolo26{s}-cls.pt' for s in YOLO26_SIZES)} (set YOLO26_CHECKPOINT in main.py). "
            f"Detection/segmentation/pose checkpoints (yolo26n.pt, yolo26n-seg.pt, ...) and the open-vocabulary "
            f"YOLOE-26 models cannot transcribe text without bounding-box labels.")
    return match.group(1)


def qwen_params_billions(checkpoint: str) -> Optional[float]:
    """Parameter count in billions parsed from a checkpoint name ("Qwen3-VL-8B-Instruct" -> 8.0),
    or None when the name has no size (e.g. a local folder)."""
    match = re.search(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", checkpoint.rstrip("/").split("/")[-1])
    return float(match.group(1)) if match else None


def find_latest_run(model_dir: Path, tag: str = "") -> Optional[Path]:
    """Most recently trained run under <output_dir>/<model_name> (one with checkpoints/final).
    `tag` restricts the search to runs whose folder name contains it (MS-MGCA: experiment + backbone size)."""
    finals = [p for p in model_dir.glob("*/checkpoints/final") if p.is_dir() and tag in p.parent.parent.name]
    return max(finals, key=lambda p: p.stat().st_mtime).parent.parent if finals else None


def find_run_by_suffix(model_dir: Path, suffix: str) -> Optional[Path]:
    """Newest run folder (finished or not) whose name ends with `suffix`, e.g. "_pc_5070_e1_baseline_2B_rxhand_offline".
    Run names start with a timestamp, so the newest sorts last. Used by --resume to find the run to continue."""
    runs = [p for p in model_dir.glob("run_*") if p.is_dir() and p.name.endswith(suffix)]
    return max(runs, key=lambda p: p.name) if runs else None


# ---------------------------------------------------------------------------------------------------------------------
# Configuration types (values come from the EXPERIMENT SETUP block of main.py)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class QuantizationConfig:
    """4-bit NF4 quantization of the Qwen backbone (QLoRA). Used by qwen-vl and ms-mgca only."""
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: torch.dtype
    bnb_4bit_use_double_quant: bool
    # quantize_vision: also 4-bit quantize the vision tower (False keeps it in compute dtype).
    # upcast_fp32: run peft's prepare_model_for_kbit_training, which upcasts every non-quantized weight
    #   (embeddings, lm_head) to fp32 (qwen-vl only).
    quantize_vision: bool
    upcast_fp32: bool


@dataclass
class LoRAConfig:
    """LoRA adapter on the language model of the Qwen backbone (qwen-vl and ms-mgca alike)."""
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: Tuple[str, ...]


@dataclass
class TrainingConfig:
    """Training recipe of the selected model (main.py picks the recipe for MODEL_CHOICE).
    The hardware profile fills in the micro-batch / accumulation split of effective_batch_size and the loader
    settings; nothing else is changed after main.py builds it."""
    effective_batch_size: int
    num_epochs: int
    learning_rate: float
    # int = number of warmup steps; float in [0, 1) = fraction of the total training steps.
    warmup_steps: Union[int, float]
    lr_scheduler_type: str
    weight_decay: float
    optim: str
    max_grad_norm: float
    precision: Literal["bf16", "fp32"]
    allow_tf32: bool
    eval_strategy: Literal["steps", "epoch"]
    eval_steps: Optional[int]           # used when eval_strategy == "steps"; save_steps must be a multiple of it
    save_steps: Optional[int]
    logging_steps: int
    metric_for_best_model: str
    greater_is_better: bool
    early_stopping_patience: int        # evaluations without a better metric_for_best_model; 0 disables
    load_best_model_at_end: bool
    save_total_limit: int
    seed: int
    # Set from the hardware profile (memory only):
    per_device_train_batch_size: int = 0
    per_device_eval_batch_size: int = 0
    gradient_accumulation_steps: int = 0
    num_workers: int = 0
    pin_memory: bool = False

    @property
    def use_bf16(self) -> bool:
        return self.precision == "bf16"

    @property
    def use_fp16(self) -> bool:
        return False  # every supported GPU (Ampere and newer) runs bf16

    @property
    def eval_by_epoch(self) -> bool:
        return self.eval_strategy == "epoch"


@dataclass
class DatasetConfig:
    """Which data is used and how it is split. Paths default to <dataset_root>/DHP and <dataset_root>/RxHand."""
    # "combined" -> DHP + RxHand merged into one pool, then one train/val split
    # "dhp"      -> DHP only: its own train/val split, RxHand is ignored
    # "rxhand"   -> RxHand only: its own train/val split, DHP is ignored
    mode: Literal["combined", "dhp", "rxhand"]
    # mode == "rxhand" only: train on RxHand's official Train_Set and validate on its Test_Set.
    rxhand_official_split: bool
    # mode == "rxhand" only: Train_Set is split into train/val and the official Test_Set is kept as a held-out test
    # set that only the infer stage reads. The split is written once to rxhand_split_dir and read back from there.
    rxhand_heldout_test: bool
    train_split: float
    val_split: float
    split_seed: int
    max_samples: Optional[int]

    dataset_root: str = DATASET_ROOT.as_posix()
    rxhand_split_dir: str = (PROJECT_ROOT / "configs" / "splits" / "rxhand").as_posix()
    # Sub-paths left empty are derived from dataset_root in __post_init__.
    dhp_root: str = ""
    dhp_labels: str = ""
    dhp_images: str = ""
    rxhand_root: str = ""
    rxhand_train_labels: str = ""
    rxhand_test_labels: str = ""
    rxhand_train_images: str = ""
    rxhand_test_images: str = ""

    def __post_init__(self):
        if self.mode not in ("combined", "dhp", "rxhand"):
            raise ValueError(f"Unknown dataset mode '{self.mode}'. Options: combined, dhp, rxhand")
        if self.rxhand_official_split and self.mode != "rxhand":
            raise ValueError("rxhand_official_split only applies to dataset mode 'rxhand'")
        if self.rxhand_heldout_test and self.mode != "rxhand":
            raise ValueError("rxhand_heldout_test only applies to dataset mode 'rxhand'")
        if self.rxhand_heldout_test and self.rxhand_official_split:
            raise ValueError("Choose either rxhand_official_split or rxhand_heldout_test, not both")
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

    @property
    def mode_tag(self) -> str:
        """Suffix for run folder names; empty for the combined setup."""
        if self.mode == "combined":
            return ""
        if self.rxhand_official_split:
            return f"_{self.mode}-official"
        if self.rxhand_heldout_test:
            return f"_{self.mode}-heldout"
        return f"_{self.mode}"

    def missing_paths(self) -> List[str]:
        """Return expected dataset files/folders (of the datasets the mode uses) that do not exist."""
        paths = []
        if self.mode in ("combined", "dhp"):
            paths += [self.dhp_labels, self.dhp_images]
        if self.mode in ("combined", "rxhand"):
            paths += [self.rxhand_train_labels, self.rxhand_test_labels,
                      self.rxhand_train_images, self.rxhand_test_images]
        return [p for p in paths if not os.path.exists(p)]


@dataclass
class EvaluationConfig:
    """Generation-based evaluation (WER/CER/accuracy) and MS-MGCA confidence calibration."""
    train_eval_samples: Optional[int]   # val images decoded at every eval during training (None = all)
    infer_samples: Optional[int]        # images scored by calibrate / infer (None = all)
    max_new_tokens: int
    # STAGE = "calibrate": the val set is split once into a calibration part (fits the confidence head on the model's
    # real right/wrong transcriptions) and an evaluation part (reports ECE / Brier / AUROC).
    calibration_fraction: float
    calibration_steps: int
    calibration_lr: float
    calibration_weight_decay: float
    calibration_batch_size: int
    calibration_val_fraction: float     # share of the calibration part kept aside to pick the best fitting step
    calibration_seed: int
    # Adapter loaded by the "infer" stage (empty = <checkpoint_dir>/final).
    adapter_path: str = ""


@dataclass
class AugmentationConfig:
    """Data augmentation. mode "online": random transforms per training image while training; "offline": variants
    pre-computed once by STAGE = "augment" into offline_root and read from there."""
    mode: Literal["online", "offline"]
    enable_augmentation: bool
    offline_include_mixed: bool   # offline: also write one variant with all six transforms applied together
    offline_seed: int             # offline: variants are deterministic for a given seed and settings
    motion_blur_prob: float
    elastic_deform_prob: float
    perspective_prob: float
    brightness_contrast_prob: float
    rotation_limit: int
    shift_limit: float
    scale_limit: float
    offline_root: str = AUGMENTED_ROOT.as_posix()

    def __post_init__(self):
        if self.mode not in ("online", "offline"):
            raise ValueError(f"Unknown augmentation mode '{self.mode}'. Options: online, offline")

    @property
    def mode_tag(self) -> str:
        """Suffix for run folder names; empty for online augmentation."""
        return "_offline" if self.mode == "offline" else ""


@dataclass
class CRNNConfig:
    """CRNN architecture (13 VGG16 conv layers -> BiLSTM -> CTC). Its training recipe is a TrainingConfig."""
    img_height: int
    img_width: int
    lstm_hidden: int
    lstm_layers: int
    batch_norm: bool


@dataclass
class YOLOConfig:
    """YOLO classification backbone used as a closed-vocabulary word recognizer (yolo and yolo26 alike)."""
    imgsz: int
    label_smoothing: float


@dataclass
class MSMGCASetup:
    """MS-MGCA-Qwen3-VL architecture and its extra learning rates. The pipeline builds the experiment's config from
    this (the YAML files in experiments/MS-MGCA-Qwen3-VL/configs are only read by that package's own scripts)."""
    # Ablation name -> (multi-scale fusion A, medication cross-attention B, confidence head D)
    ablations: Dict[str, Tuple[bool, bool, bool]]
    module_learning_rate: float        # multi-scale and cross-attention modules
    confidence_learning_rate: float    # confidence head
    multiscale: Dict
    cross_attention: Dict
    uncertainty: Dict


@dataclass
class ModelConfig:
    """Which model is trained and its input size."""
    model_type: Literal["qwen-vl", "crnn", "yolo", "yolo26", "ms-mgca"]
    qwen_checkpoint: str
    # MODEL_CHOICE = "ms-mgca": which ablation (a key of MSMGCASetup.ablations); the backbone is qwen_checkpoint.
    ms_mgca_experiment: str
    # Number of medication queries K (None = MSMGCASetup.cross_attention["num_queries"]). In the run name when set.
    ms_mgca_num_queries: Optional[int]
    yolo_checkpoint: str
    yolo26_checkpoint: str
    max_pixels: int
    gradient_checkpointing: bool = False  # set from the hardware profile for qwen-vl / ms-mgca


@dataclass
class PipelineConfig:
    """Master configuration, built by main.py (build_config)."""
    stage: Literal["augment", "train", "calibrate", "infer"]
    # --resume (MS-MGCA): train continues the newest run of this exact setting from its latest checkpoint (a new run
    # when there is none); a stage whose output already exists is skipped (see main.py).
    resume: bool
    hardware_profile: str  # "auto", or a key of HARDWARE_PROFILES
    model_type: Literal["qwen-vl", "crnn", "yolo", "yolo26", "ms-mgca"]

    quantization: QuantizationConfig
    lora: LoRAConfig
    training: TrainingConfig
    dataset: DatasetConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    ms_mgca: MSMGCASetup
    crnn: CRNNConfig
    yolo: YOLOConfig
    evaluation: EvaluationConfig

    device: str = "cuda"
    output_dir: str = OUTPUT_ROOT.as_posix()
    # Every run gets its own folder: <output_dir>/<model_name>/<run_name>/
    #   checkpoints/  logs/  plots/  metrics_history.{json,csv}  eval_metrics.json  run_config.json
    # Left empty -> "run_<timestamp>_<hardware_profile>..." for train; for calibrate/infer, the latest trained run.
    run_name: str = ""
    run_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    plots_dir: str = ""

    def __post_init__(self):
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model '{self.model_type}'. Options: {', '.join(MODEL_TYPES)}")
        self.model.model_type = self.model_type
        self.validate()
        self.apply_hardware_profile()
        self.resolve_run_paths()

    # --- model family ---------------------------------------------------------------------------------------------
    @property
    def is_ms_mgca(self) -> bool:
        return self.model_type == "ms-mgca"

    @property
    def is_vlm(self) -> bool:
        """Qwen-based models (LoRA / QLoRA on the Qwen backbone)."""
        return self.model_type in ("qwen-vl", "ms-mgca")

    @property
    def is_yolo(self) -> bool:
        """True for the YOLO word classifiers ("yolo" = YOLOv8 etc., "yolo26" = YOLO26)."""
        return self.model_type in ("yolo", "yolo26")

    @property
    def yolo_checkpoint(self) -> str:
        """Classification checkpoint of the selected YOLO family."""
        return self.model.yolo26_checkpoint if self.model_type == "yolo26" else self.model.yolo_checkpoint

    @property
    def yolo_settings(self) -> YOLOConfig:
        return self.yolo

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.quantization.bnb_4bit_compute_dtype if self.training.use_bf16 else torch.float32

    @property
    def model_name(self) -> str:
        """Folder name for the model, e.g. "Qwen3-VL-8B-Instruct", "CRNN", "yolov8n" or "yolo26m-cls"."""
        if self.model_type == "crnn":
            return "CRNN"
        if self.is_ms_mgca:
            return "MS-MGCA-Qwen3-VL"
        if self.is_yolo:
            name = Path(self.yolo_checkpoint).name
            return name[:-3] if name.endswith(".pt") else name
        return self.model.qwen_checkpoint.rstrip("/").split("/")[-1]

    @property
    def run_tag(self) -> str:
        """Run-folder fragment that keeps MS-MGCA ablations and backbones apart, e.g. "_e6_full_model_2B"
        ("_e6_full_model_2B_k8" when the number of medication queries is overridden)."""
        if not self.is_ms_mgca:
            return ""
        size_b = qwen_params_billions(self.model.qwen_checkpoint)
        k = f"_k{self.model.ms_mgca_num_queries}" if self.model.ms_mgca_num_queries else ""
        return f"_{self.model.ms_mgca_experiment}_{f'{size_b:g}B' if size_b else 'custom'}{k}"

    # --- setup ----------------------------------------------------------------------------------------------------
    def validate(self) -> None:
        """Reject combinations that cannot run, instead of silently changing a setting."""
        if self.training.use_bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bf16, which the fixed setup uses (precision 'bf16'). "
                             "Every supported profile is Ampere or newer.")
        if self.training.eval_strategy == "steps" and self.training.save_steps % self.training.eval_steps:
            raise ValueError("save_steps must be a multiple of eval_steps (load_best_model_at_end needs a checkpoint "
                             "at every evaluation).")
        if self.is_vlm and "qwen" not in self.model.qwen_checkpoint.lower():
            raise ValueError(f"'{self.model.qwen_checkpoint}' is not a Qwen-VL checkpoint (set QWEN_CHECKPOINT).")
        if self.is_ms_mgca:
            if self.model.ms_mgca_experiment not in self.ms_mgca.ablations:
                raise ValueError(f"Unknown MS-MGCA experiment '{self.model.ms_mgca_experiment}'. "
                                 f"Options: {', '.join(self.ms_mgca.ablations)}")
            k = self.model.ms_mgca_num_queries
            if k is not None and k < 1:
                raise ValueError(f"MSMGCA_NUM_QUERIES must be a positive integer (1, 4 or 8 in the paper), got {k}.")
            if "qwen3" not in self.model.qwen_checkpoint.lower():
                raise ValueError(f"MS-MGCA needs a Qwen3-VL backbone, got '{self.model.qwen_checkpoint}' "
                                 f"(set QWEN_CHECKPOINT in main.py, e.g. Qwen/Qwen3-VL-2B-Instruct).")
        if self.model_type == "yolo26":
            yolo26_size(self.model.yolo26_checkpoint)

    def apply_hardware_profile(self) -> None:
        """Split the fixed effective batch into micro-batch x gradient accumulation for this machine and set the
        loader settings. Nothing that changes what is learned is touched here."""
        if self.hardware_profile == "auto":
            self.hardware_profile = detect_hardware_profile()
        if self.hardware_profile not in HARDWARE_PROFILES:
            raise ValueError(f"Unknown hardware profile '{self.hardware_profile}'. "
                             f"Options: auto, {', '.join(HARDWARE_PROFILES)}")
        hw = HARDWARE_PROFILES[self.hardware_profile]
        t = self.training
        effective = t.effective_batch_size

        if self.is_ms_mgca:
            micro, eval_batch = 1, 1
        elif self.model_type == "qwen-vl":
            micro, eval_batch = hw.vlm_micro_batch, hw.vlm_eval_batch
        elif self.model_type == "yolo26":
            micro = hw.yolo26_micro_batch.get(yolo26_size(self.model.yolo26_checkpoint), hw.yolo_micro_batch)
            eval_batch = micro
        elif self.model_type == "yolo":
            micro = eval_batch = hw.yolo_micro_batch
        else:  # crnn: tiny, the whole batch fits on every machine
            micro = eval_batch = effective
        micro = min(micro, effective)
        if effective % micro:
            raise ValueError(f"Effective batch {effective} is not a multiple of the {self.hardware_profile} "
                             f"micro-batch {micro}; adjust the profile so the effective batch stays exact.")

        t.per_device_train_batch_size = micro
        t.per_device_eval_batch_size = eval_batch
        t.gradient_accumulation_steps = effective // micro
        t.num_workers = hw.num_workers
        t.pin_memory = hw.pin_memory
        self.model.gradient_checkpointing = hw.gradient_checkpointing and self.is_vlm

        if t.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def resolve_run_paths(self) -> None:
        """Derive this run's folder and its sub-folders (directories are created by the stages)."""
        model_dir = Path(self.output_dir) / self.model_name
        if not self.run_name:
            # Held-out runs only evaluate models trained on the held-out split: older runs saw Test_Set images.
            tag = self.run_tag + (self.dataset.mode_tag if self.dataset.rxhand_heldout_test else "")
            latest = find_latest_run(model_dir, tag) if self.stage in ("calibrate", "infer") else None
            if self.resume and self.stage == "train":
                latest = find_run_by_suffix(
                    model_dir, f"_{self.hardware_profile}{self.run_tag}{self.dataset.mode_tag}{self.augmentation.mode_tag}")
                if latest is None:
                    print("[resume] No earlier run of this setting found; starting a new run.")
            self.run_name = latest.name if latest else (
                f"run_{datetime.now():%Y%m%d-%H%M%S}_{self.hardware_profile}{self.run_tag}"
                f"{self.dataset.mode_tag}{self.augmentation.mode_tag}")
        run = Path(self.run_dir or model_dir / self.run_name)
        self.run_dir = run.as_posix()
        self.checkpoint_dir = self.checkpoint_dir or (run / "checkpoints").as_posix()
        self.log_dir = self.log_dir or (run / "logs").as_posix()
        self.plots_dir = self.plots_dir or (run / "plots").as_posix()
        self.evaluation.adapter_path = self.evaluation.adapter_path or (Path(self.checkpoint_dir) / "final").as_posix()
