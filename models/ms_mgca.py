"""Bridge between this pipeline and experiments/MS-MGCA-Qwen3-VL (Multi-Scale, Medication-Guided Cross-Attention,
uncertainty-aware Qwen3-VL).

The experiment is a self-contained research package whose modules use top-level imports (`configs`, `models`,
`dataset`, ...). Two of those names (`configs`, `models`) are also this pipeline's packages, so the experiment is
imported in isolation (load_experiment) instead of being put on sys.path permanently, and its files stay untouched.

Used by main.py when MODEL_CHOICE = "ms-mgca":
    load_ms_mgca()        build the model (fresh for training, or from a checkpoint folder for inference)
    save_ms_mgca()        write a checkpoint: LoRA adapter + the MS-MGCA modules + the experiment config
    ms_mgca_param_groups  optimizer groups with the per-component learning rates of the experiment
    confidence_report()   calibration / selective-prediction metrics of the confidence head
"""
import importlib
import logging
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from configs.config import MSMGCA_EXPERIMENTS, PROJECT_ROOT

logger = logging.getLogger(__name__)

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "MS-MGCA-Qwen3-VL"
MODULES_FILE = "ms_mgca_modules.safetensors"
CONFIG_FILE = "ms_mgca_config.yaml"
ADAPTER_DIR = "lora_adapter"

# Top-level package names used inside the experiment. Pipeline packages with the same name are parked while
# the experiment is imported and put back afterwards.
_EXPERIMENT_PACKAGES = ("configs", "models", "dataset", "evaluation", "training", "inference")


def _in_experiment_namespace(module_name: str) -> bool:
    return module_name.split(".")[0] in _EXPERIMENT_PACKAGES


@lru_cache(maxsize=1)
def load_experiment() -> SimpleNamespace:
    """Import the experiment package once and return the pieces the pipeline needs
    (MSMGCAConfig, MSMGCAQwen3VL, compute_all_confidence_metrics)."""
    if not EXPERIMENT_DIR.is_dir():
        raise FileNotFoundError(f"MS-MGCA experiment not found at {EXPERIMENT_DIR}")

    parked = {n: m for n, m in sys.modules.items() if _in_experiment_namespace(n)}
    for name in parked:
        del sys.modules[name]
    sys.path.insert(0, str(EXPERIMENT_DIR))
    try:
        config_module = importlib.import_module("configs.base_config")
        model_module = importlib.import_module("models.proposed_model")
        confidence_module = importlib.import_module("evaluation.confidence_metrics")
        imported = {n: m for n, m in sys.modules.items() if _in_experiment_namespace(n)}
    finally:
        sys.path.remove(str(EXPERIMENT_DIR))
        for name in [n for n in sys.modules if _in_experiment_namespace(n)]:
            del sys.modules[name]
        sys.modules.update(parked)

    # Keep the experiment modules importable by their real (collision-free) names, e.g. for pickling.
    sys.modules.update({f"ms_mgca_experiment.{n}": m for n, m in imported.items()})
    return SimpleNamespace(
        MSMGCAConfig=config_module.MSMGCAConfig,
        MSMGCAQwen3VL=model_module.MSMGCAQwen3VL,
        compute_all_confidence_metrics=confidence_module.compute_all_confidence_metrics,
    )


def experiment_yaml(name: str) -> Path:
    if name not in MSMGCA_EXPERIMENTS:
        raise ValueError(f"Unknown MS-MGCA experiment '{name}'. Options: {', '.join(MSMGCA_EXPERIMENTS)}")
    return EXPERIMENT_DIR / "configs" / f"{name}.yaml"


def build_ms_mgca_config(cfg, checkpoint_dir: Optional[str] = None):
    """Experiment config for this run. Architecture switches (multiscale / cross_attention / uncertainty) come from
    the experiment YAML, or from the config saved next to a trained checkpoint so inference always rebuilds the
    architecture that was trained. Machine / backbone settings come from the pipeline's PipelineConfig."""
    exp = load_experiment()
    saved = Path(checkpoint_dir) / CONFIG_FILE if checkpoint_dir else None
    msc = exp.MSMGCAConfig.from_yaml(str(saved if saved and saved.is_file() else experiment_yaml(cfg.model.ms_mgca_experiment)))

    if cfg.model.ms_mgca_num_queries and not checkpoint_dir:  # a trained checkpoint keeps the K it was trained with
        msc.cross_attention.num_queries = cfg.model.ms_mgca_num_queries
    msc.model.checkpoint = cfg.model.qwen_checkpoint
    msc.model.load_in_4bit = cfg.quantization.load_in_4bit
    msc.model.bnb_4bit_compute_dtype = "bfloat16"
    msc.model.max_pixels = cfg.model.max_pixels
    msc.lora.r, msc.lora.lora_alpha, msc.lora.lora_dropout = cfg.lora.r, cfg.lora.lora_alpha, cfg.lora.lora_dropout
    msc.training.gradient_checkpointing = cfg.model.gradient_checkpointing
    msc.training.output_dir = cfg.output_dir
    msc.training.run_name = cfg.run_name
    return msc


def _components(model) -> Iterator[Tuple[str, torch.nn.Module]]:
    """The MS-MGCA modules present in this ablation (E1 has none), keyed by their checkpoint prefix."""
    for key, attr in (("multiscale", "multiscale_extractor"), ("cross_attention", "med_cross_attention"),
                      ("uncertainty", "uncertainty_decoder")):
        module = getattr(model, attr, None)
        if module is not None:
            yield key, module


def _finish_setup(model) -> None:
    # Attributes the HF Trainer inspects on a plain nn.Module:
    #  - quantization_method: skip Trainer's model.to(device) on a bitsandbytes model
    #  - accepts_loss_kwargs: the model's loss is a plain mean per micro-batch, so the Trainer must divide by the
    #    gradient-accumulation steps itself instead of passing num_items_in_batch down
    model.quantization_method = getattr(model.model, "quantization_method", None)
    model.accepts_loss_kwargs = False


def load_ms_mgca(cfg, checkpoint_dir: Optional[str] = None):
    """Build MS-MGCA-Qwen3-VL.

    checkpoint_dir=None  -> fresh model for training: LoRA attached, backbone frozen, MS-MGCA modules trainable
    checkpoint_dir=<dir> -> trained model for inference, restored from a folder written by save_ms_mgca
    Returns (model, processor); `model` is the experiment's MSMGCAQwen3VL (an nn.Module).
    """
    exp = load_experiment()
    msc = build_ms_mgca_config(cfg, checkpoint_dir)
    components = [name for name, on in (("multi-scale", msc.multiscale.enabled),
                                        ("cross-attention", msc.cross_attention.enabled),
                                        ("confidence", msc.uncertainty.enabled)) if on]
    logger.info(f"MS-MGCA experiment '{cfg.model.ms_mgca_experiment}' on {msc.model.checkpoint}: "
                f"{', '.join(components) or 'baseline (no extra components)'}")

    model = exp.MSMGCAQwen3VL(msc, for_training=checkpoint_dir is None)
    if checkpoint_dir:
        load_ms_mgca_weights(model, checkpoint_dir, attach_adapter=True)
        model.eval()
    _finish_setup(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"MS-MGCA trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model, model.processor


def save_ms_mgca(model, directory: str) -> None:
    """Checkpoint = LoRA adapter + MS-MGCA module weights + the experiment config (small: no backbone weights)."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    model.cfg.save_yaml(str(out / CONFIG_FILE))
    if hasattr(model.model, "save_pretrained"):
        model.model.save_pretrained(str(out / ADAPTER_DIR))
    tensors = {f"{key}.{name}": value.detach().cpu().contiguous()
               for key, module in _components(model) for name, value in module.state_dict().items()}
    if tensors:
        save_file(tensors, str(out / MODULES_FILE))


def load_ms_mgca_weights(model, directory: str, attach_adapter: bool = False) -> None:
    """Restore a save_ms_mgca checkpoint into `model`. attach_adapter=True wraps the (adapter-less) backbone in the
    saved LoRA adapter (inference); otherwise the adapter weights are copied into the existing trainable adapter
    (reloading the best checkpoint at the end of training)."""
    from peft import PeftModel, set_peft_model_state_dict

    root = Path(directory)
    adapter_path = root / ADAPTER_DIR
    if adapter_path.is_dir():
        if attach_adapter:
            model.model = PeftModel.from_pretrained(model.model, str(adapter_path), is_trainable=False)
        else:
            set_peft_model_state_dict(model.model, load_file(str(adapter_path / "adapter_model.safetensors")))

    modules_path = root / MODULES_FILE
    components = list(_components(model))
    if components and not modules_path.is_file():
        raise FileNotFoundError(f"{modules_path} is missing; this checkpoint was not trained with these MS-MGCA modules.")
    if components:
        tensors = load_file(str(modules_path))
        for key, module in components:
            prefix = f"{key}."
            module.load_state_dict({n[len(prefix):]: v for n, v in tensors.items() if n.startswith(prefix)})
    logger.info(f"Loaded MS-MGCA checkpoint from {root}")


def ms_mgca_param_groups(model, lora_lr: float) -> List[Dict]:
    """Optimizer groups with the experiment's per-component learning rates: LoRA at the pipeline learning rate,
    multi-scale / cross-attention adapters at training.adapter_learning_rate, the confidence head at
    training.confidence_learning_rate."""
    t = model.cfg.training
    specs = [("lora", [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad], lora_lr)]
    for key, module in _components(model):
        lr = t.confidence_learning_rate if key == "uncertainty" else t.adapter_learning_rate
        specs.append((key, [p for p in module.parameters() if p.requires_grad], lr))

    groups = [{"params": params, "lr": lr, "name": name} for name, params, lr in specs if params]
    covered = {id(p) for g in groups for p in g["params"]}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in covered]
    if rest:
        groups.append({"params": rest, "lr": lora_lr, "name": "other"})
    return groups


def square_pad(image: Image.Image) -> Image.Image:
    """Pad to a square on white. The multi-scale spatial pooling maps local crops onto a square global token grid,
    which a non-square image would not give. Square images (the RxHand crops) are returned unchanged."""
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


# --------------------------------------------------------------------------------------------------------------
# Confidence calibration (STAGE = "calibrate")
#
# Component D must learn whether a *generated* transcription is right. That needs real right/wrong labels z_i from
# predictions the model did not train on (training images are mostly transcribed correctly, so their labels would
# say "always right"). The val set is that held-out data: it is split once into a calibration part (fits the head)
# and an evaluation part (reports ECE / Brier / AUROC, never seen by the head).
# --------------------------------------------------------------------------------------------------------------
CALIBRATED_DIR = "calibrated"
CALIBRATION_SPLIT_FILE = "calibration_split.json"


def calibration_split(ids: List[str], fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    """Deterministic split of record ids into (calibration ids, held-out evaluation ids)."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"calibration_fraction must be between 0 and 1, got {fraction}")
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)).tolist()
    n_cal = min(max(int(round(len(ids) * fraction)), 1), len(ids) - 1)
    return [ids[i] for i in order[:n_cal]], [ids[i] for i in order[n_cal:]]


class capture_pooled_features:
    """Context manager collecting the pooled decoder representation h that the confidence head receives at every
    generate_with_confidence() call (hook on the head, so the experiment's code is untouched)."""

    def __init__(self, model):
        self.head = model.uncertainty_decoder.confidence_head
        self.features: List[torch.Tensor] = []

    def __enter__(self):
        self._handle = self.head.register_forward_hook(
            lambda _m, inputs, _out: self.features.append(inputs[0].detach().float().cpu().reshape(-1)))
        return self.features

    def __exit__(self, *exc):
        self._handle.remove()


def head_scores(model, features: torch.Tensor) -> torch.Tensor:
    """Confidence c = sigmoid(head(h)) for cached features [N, D], computed in fp32."""
    import copy
    head = copy.deepcopy(model.uncertainty_decoder.confidence_head).float().eval()
    device = next(head.parameters()).device
    with torch.no_grad():
        return head(features.float().to(device))[1].reshape(-1).cpu()


def fit_confidence_head(model, features: torch.Tensor, targets: torch.Tensor,
                        steps: int = 300, lr: float = 1e-3, seed: int = 3407) -> Optional[Dict]:
    """Fine-tune the confidence head (warm start = the jointly trained head) with BCE on cached (h, z) pairs, in fp32.
    20% of the pairs are kept aside to pick the best step, so the head does not memorise a few hundred samples.
    Returns the fit log, or None (head untouched) when the labels contain a single class."""
    import copy
    n = len(targets)
    n_pos = int(targets.sum().item())
    if n_pos == 0 or n_pos == n:
        logger.warning(f"All {n} calibration labels are {'correct' if n_pos else 'wrong'}; a confidence head cannot "
                       f"be fitted on one class. Head left unchanged.")
        return None

    live = model.uncertainty_decoder.confidence_head
    head = copy.deepcopy(live).float()
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=gen)
    n_val = max(int(0.2 * n), 1) if n >= 10 else 0
    val_idx, fit_idx = order[:n_val], order[n_val:]
    device = next(live.parameters()).device
    x, z = features.float().to(device), targets.float().to(device)
    head.to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    best = (float("inf"), copy.deepcopy(head.state_dict()), 0)
    log = []
    for step in range(1, steps + 1):
        head.train()
        batch = fit_idx[torch.randint(len(fit_idx), (min(64, len(fit_idx)),), generator=gen)]
        loss = head.compute_loss(head(x[batch])[0], z[batch])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if n_val and (step % 10 == 0 or step == steps):
            head.eval()
            with torch.no_grad():
                val_loss = head.compute_loss(head(x[val_idx])[0], z[val_idx]).item()
            log.append({"step": step, "train_bce": float(loss.item()), "val_bce": val_loss})
            if val_loss < best[0]:
                best = (val_loss, copy.deepcopy(head.state_dict()), step)
    if n_val:
        head.load_state_dict(best[1])
    live.load_state_dict(head.state_dict())  # casts back to the head's bf16
    return {"samples": n, "positives": n_pos, "best_step": best[2] if n_val else steps,
            "best_val_bce": best[0] if n_val else None, "log": log}


def save_calibrated(model, directory: str, split: Dict) -> None:
    """Write the calibrated checkpoint (same layout as save_ms_mgca) plus the calibration/evaluation id split."""
    import json
    save_ms_mgca(model, directory)
    (Path(directory) / CALIBRATION_SPLIT_FILE).write_text(json.dumps(split, indent=2), encoding="utf-8")


def heldout_ids(checkpoint_dir: str) -> Optional[set]:
    """Ids the calibrated head never saw (None for an uncalibrated checkpoint)."""
    import json
    path = Path(checkpoint_dir) / CALIBRATION_SPLIT_FILE
    return set(json.loads(path.read_text(encoding="utf-8"))["heldout_ids"]) if path.is_file() else None


def confidence_report(experiment_model, results: List[Dict], only_ids: Optional[set] = None) -> Optional[Dict]:
    """Calibration / selective-prediction metrics of the confidence head over infer results
    (each carrying `confidence` and `correct`), optionally restricted to `only_ids` (the held-out evaluation part).
    None when this ablation has no confidence head."""
    if not getattr(experiment_model, "uncertainty_enabled", False):
        return None
    scored = [r for r in results if "confidence" in r and (only_ids is None or r["image_id"] in only_ids)]
    if not scored:
        return None
    return load_experiment().compute_all_confidence_metrics(
        [r["confidence"] for r in scored],
        [int(r["correct"]) for r in scored],
        threshold=experiment_model.cfg.uncertainty.confidence_threshold,
    )
