"""
Configuration dataclasses and YAML serialization for MS-MGCA-Qwen3-VL.
Supports modular toggling for all ablation experiments (E1 to E6).
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional
import yaml
import torch

# Root of the entire Doctor-Handwriting-Code-Pipeline repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset" / "RxHand"


@dataclass
class ModelConfig:
    """Qwen3-VL backbone configuration."""
    # Default to locally cached checkpoint for immediate execution; supports 4B/8B seamlessly
    checkpoint: str = "Qwen/Qwen3-VL-2B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    quantize_vision: bool = False  # Keep vision encoder in bf16 for maximum handwriting detail
    max_pixels: int = 512 * 512
    min_pixels: int = 128 * 128
    device_map: str = "cuda:0"


@dataclass
class LoRAConfig:
    """LoRA parameter-efficient fine-tuning configuration."""
    enabled: bool = True
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "o_proj", "k_proj"
    ])
    bias: str = "none"


@dataclass
class MultiScaleConfig:
    """Component A: Multi-Scale visual feature extraction and gated fusion."""
    enabled: bool = False
    crop_size: int = 224             # Local crop resolution
    grid_size: int = 2              # 2x2 grid = 4 patches
    overlap_ratio: float = 0.15     # Overlap fraction between neighboring crops
    reduction_method: str = "spatial_pool"  # "spatial_pool", "token_select", "attention"
    fusion_method: str = "gated"    # "gated", "concat", "attention"
    hidden_dim: Optional[int] = None # Will match backbone vision output dim (e.g. 2048 or 2560)


@dataclass
class CrossAttentionConfig:
    """Component B: Medication-Guided Cross-Attention."""
    enabled: bool = False
    num_queries: int = 4            # K learnable medication queries (1, 4, 8)
    num_heads: int = 8
    dropout: float = 0.1
    integration_mode: str = "adapter" # "adapter" (residual visual modulation) or "prefix"


@dataclass
class UncertaintyConfig:
    """Component D: Uncertainty-Aware Confidence Estimation."""
    enabled: bool = False
    head_type: str = "mlp"          # "linear", "mlp"
    pooling_method: str = "mean"    # "mean", "last", "attention"
    confidence_threshold: float = 0.70 # Tau for inference decision (flag review)
    loss_weight: float = 0.5        # Lambda_c for total loss: L_text + lambda_c * L_conf


@dataclass
class DatasetConfig:
    """RxHand dataset paths and parameters."""
    dataset_root: str = str(DATASET_ROOT)
    train_labels: str = str(DATASET_ROOT / "Train_Label.csv")
    test_labels: str = str(DATASET_ROOT / "Test_Labels.csv")
    train_images: str = str(DATASET_ROOT / "Train_Set")
    test_images: str = str(DATASET_ROOT / "Test_Set")
    train_split: float = 0.80
    val_split: float = 0.20
    seed: int = 3407
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: str = str(PROJECT_ROOT / "runs" / "MS-MGCA-Qwen3-VL")
    run_name: str = "E1_baseline"
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 10
    learning_rate: float = 2e-4
    adapter_learning_rate: float = 5e-4
    confidence_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    seed: int = 3407
    save_steps: int = 200
    eval_steps: int = 100
    logging_steps: int = 10


@dataclass
class MSMGCAConfig:
    """Master configuration encompassing all components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    multiscale: MultiScaleConfig = field(default_factory=MultiScaleConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "MSMGCAConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            lora=LoRAConfig(**raw.get("lora", {})),
            multiscale=MultiScaleConfig(**raw.get("multiscale", {})),
            cross_attention=CrossAttentionConfig(**raw.get("cross_attention", {})),
            uncertainty=UncertaintyConfig(**raw.get("uncertainty", {})),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )
