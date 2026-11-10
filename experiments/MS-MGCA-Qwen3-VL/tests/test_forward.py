"""
Unit tests for Qwen3-VL Baseline forward pass and transcription.
Tests tensor shapes, prompt masking, loss computation, and hidden state retrieval.
"""

import os
import sys
from pathlib import Path
import torch
from PIL import Image

# Ensure experiments/MS-MGCA-Qwen3-VL is in sys.path
EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from configs.base_config import MSMGCAConfig
from models.baseline_qwen3vl import load_qwen3vl_baseline, Qwen3VLBaselineRunner


def test_baseline_forward_and_generate():
    """Verify loading, forward pass, loss computation, and generation on a real RxHand image."""
    config_path = EXP_ROOT / "configs" / "e1_baseline.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"

    cfg = MSMGCAConfig.from_yaml(str(config_path))
    # Use 4-bit quantization on GPU for testing
    cfg.model.load_in_4bit = True

    print("\n[TEST] Loading Qwen3-VL baseline model...")
    model, processor = load_qwen3vl_baseline(cfg, for_training=False)
    runner = Qwen3VLBaselineRunner(model, processor)

    # Verify model presence
    assert model is not None
    assert processor is not None
    assert hasattr(model, "model")
    assert hasattr(model.model, "visual")

    # Locate test image P1116.jpg
    project_root = EXP_ROOT.parent.parent
    img_path = project_root / "Dataset" / "RxHand" / "Train_Set" / "P1116.jpg"
    assert img_path.exists(), f"Sample image {img_path} not found"

    image = Image.open(img_path)
    prompt = "Transcribe the handwritten medication name accurately."
    ground_truth = "Dexter"

    print(f"[TEST] Testing forward_batch with image={img_path.name}, ground_truth='{ground_truth}'")
    out = runner.forward_batch(
        images=[image],
        prompts=[prompt],
        responses=[ground_truth],
        output_hidden_states=True,
    )

    loss = out["loss"]
    logits = out["logits"]
    last_hidden_state = out["last_hidden_state"]
    labels = out["labels"]
    inputs = out["inputs"]

    # 1. Assertions on loss
    assert loss is not None, "Loss should be computed when responses are provided"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print(f"  -> Loss: {loss.item():.4f}")

    # 2. Assertions on logits and hidden states
    batch_size, seq_len, vocab_size = logits.shape
    assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
    assert vocab_size == model.config.text_config.vocab_size, f"Vocab size mismatch: {vocab_size} vs {model.config.text_config.vocab_size}"
    assert last_hidden_state is not None, "Expected last_hidden_state"
    assert last_hidden_state.shape[:2] == (batch_size, seq_len), "Hidden state seq_len mismatch"
    print(f"  -> Logits shape: {logits.shape}, Last hidden state shape: {last_hidden_state.shape}")

    # 3. Assertions on prompt label masking
    answer_mask = (labels != -100)
    assert answer_mask.sum() > 0, "At least some tokens must be unmasked for loss computation"
    print(f"  -> Total tokens: {seq_len}, Answer tokens unmasked: {answer_mask.sum().item()}")

    # 4. Assertions on generation
    print("[TEST] Testing greedy generation...")
    pred_text = runner.transcribe(image, prompt=prompt, max_new_tokens=16)
    assert isinstance(pred_text, str), "Generated text must be a string"
    print(f"  -> Baseline generation on P1116.jpg: '{pred_text}' (Ground Truth: '{ground_truth}')")

    print("[TEST] Baseline forward test PASSED successfully!")


if __name__ == "__main__":
    test_baseline_forward_and_generate()
