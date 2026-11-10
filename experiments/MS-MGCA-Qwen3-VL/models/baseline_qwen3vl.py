"""
Qwen3-VL Baseline Module.
Loads the pretrained Qwen3-VL backbone (with optional 4-bit NF4 quantization and LoRA)
and provides standard inference and hidden-state extraction utilities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from configs.base_config import MSMGCAConfig

logger = logging.getLogger(__name__)

# Standard target modules for Qwen3-VL language model
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"

DEFAULT_PROMPT = "Transcribe the handwritten medication name accurately."


def build_chat_messages(prompt: str = DEFAULT_PROMPT, response: Optional[str] = None) -> List[Dict]:
    """Build Qwen3-VL chat template structure."""
    clean_prompt = prompt.replace("<image>", "").strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": clean_prompt},
            ],
        }
    ]
    if response is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
        })
    return messages


def load_qwen3vl_baseline(
    cfg: MSMGCAConfig,
    adapter_path: Optional[str] = None,
    for_training: bool = False,
) -> Tuple[Union[Qwen3VLForConditionalGeneration, PeftModel], Qwen3VLProcessor]:
    """
    Load Qwen3-VL baseline model and processor.

    Args:
        cfg: MSMGCAConfig configuration instance.
        adapter_path: Path to trained LoRA adapter to load for inference.
        for_training: Whether the model will be trained (attaches LoRA adapter if cfg.lora.enabled).

    Returns:
        (model, processor)
    """
    checkpoint = cfg.model.checkpoint
    quantize = cfg.model.load_in_4bit
    compute_dtype = getattr(torch, cfg.model.bnb_4bit_compute_dtype, torch.bfloat16)

    logger.info(f"Loading Qwen3-VL from {checkpoint}")
    logger.info(f"Quantization: load_in_4bit={quantize}, compute_dtype={compute_dtype}")

    # Keep vision tower in bf16 if quantize_vision is False
    skip_modules = None if cfg.model.quantize_vision else ["lm_head", "visual"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
        llm_int8_skip_modules=skip_modules,
    ) if quantize else None

    device_map = {"": 0} if torch.cuda.is_available() else "cpu"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
    )

    processor = AutoProcessor.from_pretrained(checkpoint)
    processor.tokenizer.padding_side = "right"

    if adapter_path:
        logger.info(f"Loading trained LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        return model, processor

    if for_training and cfg.lora.enabled:
        if cfg.training.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()

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
        logger.info(f"Trainable parameters (LoRA): {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, processor


class Qwen3VLBaselineRunner:
    """Convenience runner for evaluation, generation, and forward inspection."""

    def __init__(self, model, processor, default_prompt: str = DEFAULT_PROMPT):
        self.model = model
        self.processor = processor
        self.default_prompt = default_prompt
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def transcribe(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Transcribe an image using greedy decoding."""
        prompt = prompt or self.default_prompt
        messages = build_chat_messages(prompt=prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=[image.convert("RGB")],
            return_tensors="pt",
        ).to(self.device)

        gen_tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = gen_tokens[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def forward_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        responses: Optional[List[str]] = None,
        output_hidden_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass for a batch of images and texts.
        Masks prompt tokens so loss is computed exclusively on response medication names.
        """
        texts = [
            self.processor.apply_chat_template(
                build_chat_messages(p, r),
                tokenize=False,
                add_generation_prompt=(r is None),
            )
            for p, r in zip(prompts, responses or [None] * len(prompts))
        ]
        rgb_images = [img.convert("RGB") for img in images]

        inputs = self.processor(
            text=texts,
            images=rgb_images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        labels = None
        if responses is not None:
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100
            im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            for row in range(labels.size(0)):
                starts = (inputs["input_ids"][row] == im_start_id).nonzero(as_tuple=True)[0]
                if len(starts) > 0:
                    labels[row, : starts[-1] + 3] = -100

        outputs = self.model(
            **inputs,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "last_hidden_state": outputs.hidden_states[-1] if outputs.hidden_states is not None else None,
            "inputs": inputs,
            "labels": labels,
        }
