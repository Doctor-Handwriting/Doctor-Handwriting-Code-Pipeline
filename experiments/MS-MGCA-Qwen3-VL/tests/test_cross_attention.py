"""
Unit tests for Component B: Medication-Guided Cross-Attention.
Tests query configurations (K=1, 4, 8), adapter modulation, gradient flow,
and combined Multi-Scale + Medication Cross-Attention (Experiment E5 forward).
"""

import sys
from pathlib import Path
import torch
from PIL import Image

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from configs.base_config import MSMGCAConfig
from dataset.crop_generator import DeterministicCropGenerator
from models.modules.multiscale_extractor import MultiScaleVisualExtractor
from models.modules.medication_cross_attention import MedicationCrossAttention
from models.baseline_qwen3vl import load_qwen3vl_baseline, build_chat_messages
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures


def test_medication_cross_attention_k_variants():
    """Verify MedicationCrossAttention with K=1, 4, 8 queries and gradient backpropagation."""
    print("\n[TEST] Testing MedicationCrossAttention with K in {1, 4, 8}...")
    batch_size = 2
    num_tokens = 64
    dim = 2048

    for k in [1, 4, 8]:
        med_module = MedicationCrossAttention(
            hidden_dim=dim,
            num_queries=k,
            integration_mode="adapter",
        )
        f_in = torch.randn(batch_size, num_tokens, dim, requires_grad=True)

        f_out = med_module(f_in)
        assert f_out.shape == (batch_size, num_tokens, dim), f"K={k} shape mismatch: {f_out.shape}"

        loss = f_out.sum()
        loss.backward()

        assert f_in.grad is not None, f"Gradient did not flow to input for K={k}"
        assert med_module.med_queries.grad is not None, f"Gradient did not flow to queries for K={k}"
        assert med_module.gamma.grad is not None, f"Gradient did not flow to gamma for K={k}"

        print(f"  -> K={k} queries passed: shape={f_out.shape}, med_queries.grad={med_module.med_queries.grad.shape}")


def test_prefix_mode():
    """Verify prefix integration mode returns (f_visual, h_prime) with shape (B, K, D)."""
    print("[TEST] Testing MedicationCrossAttention in prefix mode...")
    dim = 2048
    k = 4
    med_module = MedicationCrossAttention(hidden_dim=dim, num_queries=k, integration_mode="prefix")
    f_in = torch.randn(2, 64, dim)

    f_vis, h_med = med_module(f_in)
    assert f_vis.shape == (2, 64, dim)
    assert h_med.shape == (2, k, dim)
    print(f"  -> Prefix mode passed: f_vis={f_vis.shape}, h_med={h_med.shape}")


def test_combined_multiscale_cross_attention_forward():
    """Verify combined pipeline (E5): Multi-Scale Fusion + Medication Cross-Attention on real sample."""
    print("[TEST] Testing combined Multi-Scale + Medication Cross-Attention (E5) forward on real image...")
    cfg = MSMGCAConfig()
    cfg.model.load_in_4bit = True

    model, processor = load_qwen3vl_baseline(cfg, for_training=False)
    device = next(model.parameters()).device

    project_root = EXP_ROOT.parent.parent
    img_path = project_root / "Dataset" / "RxHand" / "Train_Set" / "P1116.jpg"
    image = Image.open(img_path)
    prompt = "Transcribe the handwritten medication name accurately."
    ground_truth = "Dexter"

    text = processor.apply_chat_template(
        build_chat_messages(prompt, ground_truth),
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    # 1. Vision extraction for global image
    with torch.no_grad():
        global_vision_out = model.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            return_dict=True,
        )

    global_embeds = global_vision_out.pooler_output[0]  # [64, D]
    num_tokens, dim = global_embeds.shape

    # 2. Local crops extraction
    crop_gen = DeterministicCropGenerator(grid_size=2, overlap_ratio=0.15, include_center_crop=False, target_crop_size=128)
    crops, _ = crop_gen.generate_crops(image)

    crop_embeds_list = []
    with torch.no_grad():
        for crop in crops:
            c_inputs = processor(text=["<|image_pad|>"], images=[crop], return_tensors="pt").to(device)
            c_out = model.model.get_image_features(
                pixel_values=c_inputs["pixel_values"],
                image_grid_thw=c_inputs["image_grid_thw"],
                return_dict=True,
            )
            crop_embeds_list.append(c_out.pooler_output[0].unsqueeze(0))

    # 3. Component A: Multi-Scale Extractor
    ms_extractor = MultiScaleVisualExtractor(
        in_dim=dim,
        hidden_dim=dim,
        reduction_method="attention",
        fusion_method="gated",
    ).to(device=device, dtype=global_embeds.dtype)

    f_fused = ms_extractor(global_embeds.unsqueeze(0), crop_embeds_list)  # [1, 64, D]

    # 4. Component B: Medication-Guided Cross-Attention
    med_attn = MedicationCrossAttention(
        hidden_dim=dim,
        num_queries=4,
        integration_mode="adapter",
    ).to(device=device, dtype=global_embeds.dtype)

    f_med = med_attn(f_fused)  # [1, 64, D]
    assert f_med.shape == (1, 64, dim), f"Refined features shape mismatch: {f_med.shape}"

    # 5. Pack into custom vision output
    custom_vision_output = BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=global_vision_out.last_hidden_state,
        pooler_output=(f_med[0],),
        deepstack_features=global_vision_out.deepstack_features,
    )

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    starts = (inputs["input_ids"][0] == im_start_id).nonzero(as_tuple=True)[0]
    labels[0, : starts[-1] + 3] = -100

    # 6. Model forward pass
    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
        mm_encoder_outputs={"image": custom_vision_output},
        labels=labels,
        output_hidden_states=True,
    )

    assert out.loss is not None
    assert torch.isfinite(out.loss)
    print(f"  -> Combined MS + MGCA forward passed! Loss: {out.loss.item():.4f}, Logits: {out.logits.shape}")


if __name__ == "__main__":
    test_medication_cross_attention_k_variants()
    test_prefix_mode()
    test_combined_multiscale_cross_attention_forward()
