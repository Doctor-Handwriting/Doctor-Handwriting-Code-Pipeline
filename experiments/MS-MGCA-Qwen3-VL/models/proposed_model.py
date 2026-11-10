"""
Unified MS-MGCA-Qwen3-VL Architecture.
Combines:
1. Qwen3-VL Backbone (with 4-bit NF4 and LoRA).
2. Component A: Multi-Scale Visual Extractor & Feature Fusion.
3. Component B: Medication-Guided Cross-Attention with Visual Adapter.
4. Component D: Uncertainty-Aware Confidence Decoding.
Supports independent ablation switches for all experiments (E1 to E6).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLCausalLMOutputWithPast,
)

from configs.base_config import MSMGCAConfig
from dataset.crop_generator import DeterministicCropGenerator
from models.modules.multiscale_extractor import MultiScaleVisualExtractor
from models.modules.medication_cross_attention import MedicationCrossAttention
from models.modules.uncertainty_decoder import UncertaintyDecoder
from models.baseline_qwen3vl import load_qwen3vl_baseline, build_chat_messages


class MSMGCAQwen3VL(nn.Module):
    """
    Multi-Scale and Uncertainty-Aware Qwen3-VL Architecture with Medication-Guided Cross-Attention.
    """

    def __init__(self, cfg: MSMGCAConfig, for_training: bool = False):
        super().__init__()
        self.cfg = cfg
        self.for_training = for_training

        # 1. Backbone Qwen3-VL Model and Processor
        self.model, self.processor = load_qwen3vl_baseline(cfg, for_training=for_training)
        self.device = next(self.model.parameters()).device

        # Determine visual feature dimension and text hidden dimension
        # In Qwen3-VL: vision merger out_hidden_size matches or projects to text hidden_size
        vision_cfg = self.model.config.vision_config
        text_cfg = self.model.config.text_config
        self.vision_out_dim = vision_cfg.out_hidden_size
        self.text_hidden_dim = text_cfg.hidden_size

        # 2. Component A: Multi-Scale Visual Feature Extraction & Fusion
        self.multiscale_enabled = cfg.multiscale.enabled
        if self.multiscale_enabled:
            self.crop_generator = DeterministicCropGenerator(
                grid_size=cfg.multiscale.grid_size,
                overlap_ratio=cfg.multiscale.overlap_ratio,
                include_center_crop=False,
                target_crop_size=cfg.multiscale.crop_size,
            )
            self.multiscale_extractor = MultiScaleVisualExtractor(
                in_dim=self.vision_out_dim,
                hidden_dim=cfg.multiscale.hidden_dim or self.vision_out_dim,
                reduction_method=cfg.multiscale.reduction_method,
                fusion_method=cfg.multiscale.fusion_method,
                grid_size=cfg.multiscale.grid_size,
            ).to(device=self.device, dtype=torch.bfloat16)

        # 3. Component B: Medication-Guided Cross-Attention
        self.cross_attention_enabled = cfg.cross_attention.enabled
        if self.cross_attention_enabled:
            self.med_cross_attention = MedicationCrossAttention(
                hidden_dim=self.vision_out_dim,
                num_queries=cfg.cross_attention.num_queries,
                num_heads=cfg.cross_attention.num_heads,
                dropout=cfg.cross_attention.dropout,
                integration_mode=cfg.cross_attention.integration_mode,
            ).to(device=self.device, dtype=torch.bfloat16)

        # 4. Component D: Uncertainty-Aware Confidence Estimation
        self.uncertainty_enabled = cfg.uncertainty.enabled
        if self.uncertainty_enabled:
            self.uncertainty_decoder = UncertaintyDecoder(
                hidden_dim=self.text_hidden_dim,
                head_type=cfg.uncertainty.head_type,
                pooling_method=cfg.uncertainty.pooling_method,
                confidence_threshold=cfg.uncertainty.confidence_threshold,
            ).to(device=self.device, dtype=torch.bfloat16)

    def extract_and_fuse_vision(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        raw_images: Optional[List[Image.Image]] = None,
    ) -> BaseModelOutputWithDeepstackFeatures:
        """
        Extract global visual features, optionally generate local crops,
        and apply multi-scale fusion (Comp A) and medication cross-attention (Comp B).
        """
        # Global vision forward
        vision_output = self.model.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )

        f_global = vision_output.pooler_output[0].unsqueeze(0)  # [1, T_g, D]

        # Apply Component A (Multi-Scale) if enabled and raw images are provided
        if self.multiscale_enabled and raw_images is not None:
            crop_embeds_list = []
            for img in raw_images:
                crops, _ = self.crop_generator.generate_crops(img)
                for crop in crops:
                    c_inputs = self.processor(
                        text=["<|image_pad|>"],
                        images=[crop],
                        return_tensors="pt",
                    ).to(self.device)
                    c_out = self.model.model.get_image_features(
                        pixel_values=c_inputs["pixel_values"],
                        image_grid_thw=c_inputs["image_grid_thw"],
                        return_dict=True,
                    )
                    crop_embeds_list.append(c_out.pooler_output[0].unsqueeze(0))

            f_fused = self.multiscale_extractor(f_global, crop_embeds_list)
        else:
            f_fused = f_global

        # Apply Component B (Medication-Guided Cross-Attention) if enabled
        if self.cross_attention_enabled:
            f_refined = self.med_cross_attention(f_fused)
        else:
            f_refined = f_fused

        # Return updated vision output packaging the modified visual tokens
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=vision_output.last_hidden_state,
            pooler_output=(f_refined[0],),
            deepstack_features=vision_output.deepstack_features,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        correctness_targets: Optional[torch.Tensor] = None,
        raw_images: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with text generation loss and optional confidence estimation loss.
        """
        # If Component A or B is enabled, preprocess visual features
        if (self.multiscale_enabled or self.cross_attention_enabled) and pixel_values is not None:
            custom_vision = self.extract_and_fuse_vision(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                raw_images=raw_images,
            )
            mm_encoder_outputs = {"image": custom_vision}
            # Clear pixel_values so Qwen3VLModel uses mm_encoder_outputs directly
            pixel_values_forward = None
        else:
            mm_encoder_outputs = None
            pixel_values_forward = pixel_values

        outputs: Qwen3VLCausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values_forward,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            mm_encoder_outputs=mm_encoder_outputs,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        loss_text = outputs.loss
        logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1] if outputs.hidden_states is not None else None

        loss_conf = None
        conf_logits = None
        conf_probs = None

        # Component D: Uncertainty loss
        if self.uncertainty_enabled and last_hidden_state is not None:
            answer_mask = (labels != -100) if labels is not None else None
            conf_logits, conf_probs = self.uncertainty_decoder(last_hidden_state, answer_mask)

            if correctness_targets is not None:
                loss_conf = self.uncertainty_decoder.confidence_head.compute_loss(
                    conf_logits, correctness_targets
                )

        # Combined Loss: L_total = L_text + lambda_c * L_conf
        loss_total = loss_text
        if loss_text is not None and loss_conf is not None:
            lambda_c = self.cfg.uncertainty.loss_weight
            loss_total = loss_text + lambda_c * loss_conf

        return {
            "loss": loss_total,
            "loss_text": loss_text,
            "loss_conf": loss_conf,
            "logits": logits,
            "conf_logits": conf_logits,
            "conf_probs": conf_probs,
            "last_hidden_state": last_hidden_state,
        }

    @torch.no_grad()
    def generate_with_confidence(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 32,
    ) -> Dict:
        """
        Autoregressive generation with clinical uncertainty estimation.
        Returns:
            {
                "prediction": "Dexter",
                "confidence": 0.94,
                "requires_review": False,
                "threshold": 0.70
            }
        """
        prompt = prompt or "Transcribe the handwritten medication name accurately."
        messages = build_chat_messages(prompt=prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=[image.convert("RGB")],
            return_tensors="pt",
        ).to(self.device)

        # Multi-scale and Cross-attention visual preparation
        if self.multiscale_enabled or self.cross_attention_enabled:
            custom_vision = self.extract_and_fuse_vision(
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                raw_images=[image],
            )
            mm_encoder_outputs = {"image": custom_vision}
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "image_grid_thw": inputs["image_grid_thw"],
                "mm_token_type_ids": inputs.get("mm_token_type_ids"),
                "mm_encoder_outputs": mm_encoder_outputs,
            }
        else:
            gen_kwargs = dict(inputs)

        gen_tokens = self.model.generate(
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        input_len = inputs["input_ids"].shape[1]
        new_token_ids = gen_tokens.sequences[0][input_len:]
        prediction_text = self.processor.decode(new_token_ids, skip_special_tokens=True).strip()

        # Compute confidence from generated hidden states
        if self.uncertainty_enabled and gen_tokens.hidden_states is not None:
            # gen_tokens.hidden_states is a tuple of length num_generated_tokens
            # Extract last layer hidden state at each generated step
            step_hidden = [step[-1][:, -1:, :] for step in gen_tokens.hidden_states]
            gen_hidden = torch.cat(step_hidden, dim=1)  # [1, num_gen_tokens, D]

            _, conf_probs = self.uncertainty_decoder(gen_hidden)
            confidence_val = float(conf_probs.item())
            decision = self.uncertainty_decoder.evaluate_decision(confidence_val)
        else:
            confidence_val = 1.0
            decision = {
                "confidence": 1.0,
                "threshold": self.cfg.uncertainty.confidence_threshold,
                "requires_review": False,
                "accepted": True,
            }

        return {
            "prediction": prediction_text,
            "confidence": decision["confidence"],
            "requires_review": decision["requires_review"],
            "accepted": decision["accepted"],
            "threshold": decision["threshold"],
        }
