"""
Unit tests for Component A: Deterministic Crop Generator, Multi-Scale Visual Extractor,
Feature Fusion strategies, and end-to-end Qwen3-VL forward pass.
"""

import sys
from pathlib import Path
import torch
from PIL import Image

# Ensure experiments/MS-MGCA-Qwen3-VL is in sys.path
EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from configs.base_config import MSMGCAConfig
from dataset.crop_generator import DeterministicCropGenerator
from models.modules.feature_fusion import MultiScaleFeatureFusion
from models.modules.multiscale_extractor import MultiScaleVisualExtractor
from models.baseline_qwen3vl import load_qwen3vl_baseline, build_chat_messages
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures


def test_crop_generator():
    """Verify deterministic crop generation and bounding boxes."""
    print("\n[TEST] Testing DeterministicCropGenerator...")
    gen = DeterministicCropGenerator(grid_size=2, overlap_ratio=0.15, include_center_crop=True, target_crop_size=224)
    img = Image.new("RGB", (300, 200), color=(100, 150, 200))

    crops, norm_boxes = gen.generate_crops(img)
    # 2x2 grid + 1 center crop = 5 crops
    assert len(crops) == 5, f"Expected 5 crops, got {len(crops)}"
    assert len(norm_boxes) == 5, f"Expected 5 boxes, got {len(norm_boxes)}"

    for i, (c, b) in enumerate(zip(crops, norm_boxes)):
        assert c.size == (224, 224), f"Crop {i} size mismatch: {c.size}"
        assert 0.0 <= b[0] < b[2] <= 1.0, f"Invalid x bounds: {b}"
        assert 0.0 <= b[1] < b[3] <= 1.0, f"Invalid y bounds: {b}"

    print("  -> Verified 5 deterministic crops with normalized bounds.")


def test_feature_fusion_strategies():
    """Verify all 3 fusion strategies (gated, concat, attention) preserve shapes and backward pass."""
    print("[TEST] Testing MultiScaleFeatureFusion strategies...")
    batch_size = 2
    num_tokens = 64
    dim = 2048

    strategies = ["gated", "concat", "attention"]
    for strat in strategies:
        fusion_mod = MultiScaleFeatureFusion(hidden_dim=dim, fusion_method=strat)
        f_g = torch.randn(batch_size, num_tokens, dim, requires_grad=True)
        f_l = torch.randn(batch_size, num_tokens, dim, requires_grad=True)

        out = fusion_mod(f_g, f_l)
        assert out.shape == (batch_size, num_tokens, dim), f"Strategy {strat} output shape mismatch: {out.shape}"

        # Test backward pass
        loss = out.sum()
        loss.backward()
        assert f_g.grad is not None, f"Gradient did not flow to f_g in {strat}"
        assert f_l.grad is not None, f"Gradient did not flow to f_l in {strat}"
        print(f"  -> Fusion '{strat}' passed: shape={out.shape}, gradients verified.")


def test_multiscale_extractor_reductions():
    """Verify both reduction methods (attention and spatial_pool) in MultiScaleVisualExtractor."""
    print("[TEST] Testing MultiScaleVisualExtractor reduction methods...")
    batch_size = 2
    num_global_tokens = 64  # 8x8 grid
    num_crops = 4          # 2x2 grid
    crop_tokens = 16       # 4x4 patches per crop
    dim = 2048

    for method in ["attention", "spatial_pool"]:
        extractor = MultiScaleVisualExtractor(
            in_dim=dim,
            hidden_dim=dim,
            reduction_method=method,
            fusion_method="gated",
            grid_size=2,
        )

        f_g = torch.randn(batch_size, num_global_tokens, dim, requires_grad=True)
        # List of crop tensors
        f_l_list = [torch.randn(batch_size, crop_tokens, dim, requires_grad=True) for _ in range(num_crops)]

        fused = extractor(f_g, f_l_list)
        assert fused.shape == (batch_size, num_global_tokens, dim), f"Extractor {method} shape mismatch: {fused.shape}"

        loss = fused.sum()
        loss.backward()
        assert f_g.grad is not None, f"No gradient for f_g in {method}"
        for c_idx, c_feat in enumerate(f_l_list):
            assert c_feat.grad is not None, f"No gradient for crop {c_idx} in {method}"

        print(f"  -> Extractor '{method}' reduction passed: output shape={fused.shape}, gradients verified.")


def test_end_to_end_multiscale_forward():
    """Verify end-to-end integration: Qwen3-VL vision extraction + multi-scale fusion + LM forward."""
    print("[TEST] Testing end-to-end Qwen3-VL multi-scale forward pass with real image...")
    cfg = MSMGCAConfig()
    cfg.model.load_in_4bit = True

    model, processor = load_qwen3vl_baseline(cfg, for_training=False)
    device = next(model.parameters()).device

    # Real sample
    project_root = EXP_ROOT.parent.parent
    img_path = project_root / "Dataset" / "RxHand" / "Train_Set" / "P1116.jpg"
    image = Image.open(img_path)
    prompt = "Transcribe the handwritten medication name accurately."
    ground_truth = "Dexter"

    # 1. Global image input
    text = processor.apply_chat_template(
        build_chat_messages(prompt, ground_truth),
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    # 2. Extract global features via visual encoder
    with torch.no_grad():
        global_vision_out = model.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            return_dict=True,
        )

    # global_vision_out.pooler_output is a tuple of chunks, one per image
    global_embeds = global_vision_out.pooler_output[0]  # [64, D]
    num_tokens, dim = global_embeds.shape
    assert num_tokens == 64, f"Expected 64 visual tokens, got {num_tokens}"

    # 3. Generate 4 local crops and extract their features
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
            crop_embeds_list.append(c_out.pooler_output[0].unsqueeze(0))  # [1, T_c, D]

    # 4. Initialize MultiScaleVisualExtractor
    extractor = MultiScaleVisualExtractor(
        in_dim=dim,
        hidden_dim=dim,
        reduction_method="attention",
        fusion_method="gated",
    ).to(device=device, dtype=global_embeds.dtype)

    # 5. Fuse features
    f_g = global_embeds.unsqueeze(0)  # [1, 64, D]
    fused_embeds = extractor(f_g, crop_embeds_list)  # [1, 64, D]
    assert fused_embeds.shape == (1, 64, dim), f"Fused embeds shape mismatch: {fused_embeds.shape}"

    # 6. Inject fused features via mm_encoder_outputs
    custom_vision_output = BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=global_vision_out.last_hidden_state,
        pooler_output=(fused_embeds[0],),  # tuple of split chunks
        deepstack_features=global_vision_out.deepstack_features,
    )

    # Prepare labels
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    starts = (inputs["input_ids"][0] == im_start_id).nonzero(as_tuple=True)[0]
    labels[0, : starts[-1] + 3] = -100

    # 7. Model forward with fused multimodal encoder outputs
    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
        mm_encoder_outputs={"image": custom_vision_output},
        labels=labels,
        output_hidden_states=True,
    )

    assert out.loss is not None, "Loss should be computed"
    assert torch.isfinite(out.loss), f"Loss is not finite: {out.loss.item()}"
    print(f"  -> End-to-end multi-scale forward passed! Loss: {out.loss.item():.4f}, Logits: {out.logits.shape}")


if __name__ == "__main__":
    test_crop_generator()
    test_feature_fusion_strategies()
    test_multiscale_extractor_reductions()
    test_end_to_end_multiscale_forward()
