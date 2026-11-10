"""
Progressive Training Engine for MS-MGCA-Qwen3-VL.
Supports Stage 2 (Multi-Scale), Stage 3 (Medication Cross-Attention),
Stage 4 (Confidence Head), and Stage 5 (Joint Fine-Tuning) with separate parameter group learning rates.
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from configs.base_config import MSMGCAConfig
from dataset.rxhand_dataset import RxHandCollator
from models.proposed_model import MSMGCAQwen3VL
from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class MSMGCATrainer:
    """
    Modular Trainer for progressive training stages.
    """

    def __init__(
        self,
        model: MSMGCAQwen3VL,
        cfg: MSMGCAConfig,
        train_dataset,
        val_dataset,
        stage: int = 5,
    ):
        self.model = model
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.stage = stage
        self.device = model.device

        # Setup collator
        self.collator = RxHandCollator(model.processor, is_training=True)
        self.val_collator = RxHandCollator(model.processor, is_training=False)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.val_collator,
        )

        # Setup parameter groups and freeze according to stage
        self.setup_stage_freeze(stage)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        self.output_dir = Path(cfg.training.output_dir) / cfg.training.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_stage_freeze(self, stage: int):
        """Configure which modules are trainable per progressive stage."""
        logger.info(f"Configuring parameter freeze for Training Stage {stage}...")

        # Initially set all parameters to not require grad except LoRA
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = (stage in [2, 3, 5])
            else:
                param.requires_grad = False

        if stage in [2, 3, 5] and self.model.multiscale_enabled:
            for param in self.model.multiscale_extractor.parameters():
                param.requires_grad = True

        if stage in [3, 5] and self.model.cross_attention_enabled:
            for param in self.model.med_cross_attention.parameters():
                param.requires_grad = True

        if stage in [4, 5] and self.model.uncertainty_enabled:
            for param in self.model.uncertainty_decoder.parameters():
                param.requires_grad = True

        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        logger.info(f"Stage {stage}: {len(trainable)} parameter tensors marked trainable.")

    def configure_optimizers(self):
        """Create separate parameter groups with independent learning rates."""
        param_groups = []

        # 1. Backbone LoRA
        lora_params = [p for n, p in self.model.named_parameters() if "lora" in n and p.requires_grad]
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": self.cfg.training.learning_rate,
                "name": "lora",
            })

        # 2. Multi-Scale module
        if self.model.multiscale_enabled:
            ms_params = [p for p in self.model.multiscale_extractor.parameters() if p.requires_grad]
            if ms_params:
                param_groups.append({
                    "params": ms_params,
                    "lr": self.cfg.training.adapter_learning_rate,
                    "name": "multiscale",
                })

        # 3. Medication Cross-Attention
        if self.model.cross_attention_enabled:
            ca_params = [p for p in self.model.med_cross_attention.parameters() if p.requires_grad]
            if ca_params:
                param_groups.append({
                    "params": ca_params,
                    "lr": self.cfg.training.adapter_learning_rate,
                    "name": "cross_attention",
                })

        # 4. Uncertainty Decoder
        if self.model.uncertainty_enabled:
            unc_params = [p for p in self.model.uncertainty_decoder.parameters() if p.requires_grad]
            if unc_params:
                param_groups.append({
                    "params": unc_params,
                    "lr": self.cfg.training.confidence_learning_rate,
                    "name": "uncertainty",
                })

        if not param_groups:
            # Fallback
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": 1e-4}]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.training.weight_decay,
        )

        total_steps = (len(self.train_loader) // self.cfg.training.gradient_accumulation_steps) * self.cfg.training.num_epochs
        warmup_steps = int(total_steps * self.cfg.training.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        accum_steps = self.cfg.training.gradient_accumulation_steps
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.training.num_epochs}")
        step = 0

        for batch in pbar:
            step += 1
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            image_grid_thw = batch["image_grid_thw"].to(self.device)
            mm_token_type_ids = batch.get("mm_token_type_ids")
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.to(self.device)
            labels = batch["labels"].to(self.device)
            raw_images = batch["raw_images"]

            # Compute automatic correctness targets if uncertainty is enabled
            # In teacher forcing, ground truth targets are 1.0 (exact match by definition)
            correctness_targets = torch.ones(input_ids.shape[0], device=self.device) if self.model.uncertainty_enabled else None

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                labels=labels,
                correctness_targets=correctness_targets,
                raw_images=raw_images,
            )

            loss = outputs["loss"] / accum_steps
            loss.backward()

            total_loss += loss.item() * accum_steps

            if step % accum_steps == 0 or step == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})

        return total_loss / len(self.train_loader)

    def evaluate(self, max_samples: Optional[int] = 50) -> Dict:
        """Run validation evaluation."""
        self.model.eval()
        predictions = []
        references = []
        image_ids = []

        count = 0
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            img = batch["raw_images"][0]
            ref = batch["ground_truths"][0]
            img_id = batch["image_ids"][0]

            with torch.no_grad():
                res = self.model.generate_with_confidence(img)

            predictions.append(res["prediction"])
            references.append(ref)
            image_ids.append(img_id)

            count += 1
            if max_samples and count >= max_samples:
                break

        summary, _ = compute_all_metrics(predictions, references, image_ids)
        return summary

    def save_checkpoint(self, name: str = "best"):
        """Save model checkpoint."""
        save_path = self.output_dir / f"checkpoint_{name}"
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.cfg.save_yaml(str(save_path / "config.yaml"))

        # Save modules
        torch_state = {}
        if self.model.multiscale_enabled:
            torch_state["multiscale"] = self.model.multiscale_extractor.state_dict()
        if self.model.cross_attention_enabled:
            torch_state["cross_attention"] = self.model.med_cross_attention.state_dict()
        if self.model.uncertainty_enabled:
            torch_state["uncertainty"] = self.model.uncertainty_decoder.state_dict()

        torch.save(torch_state, str(save_path / "ms_mgca_modules.pt"))

        # Save LoRA adapter if present
        if hasattr(self.model.model, "save_pretrained"):
            self.model.model.save_pretrained(str(save_path / "lora_adapter"))

        logger.info(f"Saved checkpoint to {save_path}")
