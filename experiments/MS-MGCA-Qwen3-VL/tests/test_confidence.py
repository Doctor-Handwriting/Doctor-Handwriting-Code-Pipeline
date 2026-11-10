"""
Unit tests for Component D: Confidence Head, Uncertainty Decoder,
Calibration Metrics (ECE, Brier Score, AUROC, Risk-Coverage),
and end-to-end Qwen3-VL hidden-state integration.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from configs.base_config import MSMGCAConfig
from models.modules.confidence_head import ConfidenceHead
from models.modules.uncertainty_decoder import UncertaintyDecoder
from models.baseline_qwen3vl import load_qwen3vl_baseline, build_chat_messages
from evaluation.confidence_metrics import (
    compute_brier_score,
    compute_ece,
    compute_auroc,
    compute_error_detection_auroc,
    compute_risk_coverage,
    compute_all_confidence_metrics,
)


def test_confidence_head_forward_and_loss():
    """Verify ConfidenceHead logits, probabilities in [0, 1], and BCE loss."""
    print("\n[TEST] Testing ConfidenceHead forward and BCE loss...")
    batch_size = 4
    dim = 2048

    head = ConfidenceHead(hidden_dim=dim, head_type="mlp")
    h = torch.randn(batch_size, dim, requires_grad=True)
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32)

    logits, probs = head(h)
    assert logits.shape == (batch_size, 1)
    assert probs.shape == (batch_size, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities must be in [0, 1]"

    loss = head.compute_loss(logits, targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    assert h.grad is not None, "Gradient did not flow to input h"
    print(f"  -> ConfidenceHead passed: loss={loss.item():.4f}, probs={probs.squeeze().tolist()}")


def test_uncertainty_decoder_pooling_and_thresholding():
    """Verify token pooling methods (mean, last, attention) and decision rule."""
    print("[TEST] Testing UncertaintyDecoder pooling and thresholding...")
    batch_size = 2
    seq_len = 10
    dim = 2048

    for method in ["mean", "last", "attention"]:
        decoder = UncertaintyDecoder(
            hidden_dim=dim,
            head_type="mlp",
            pooling_method=method,
            confidence_threshold=0.70,
        )

        h_seq = torch.randn(batch_size, seq_len, dim)
        # Mask where only tokens 4 to 7 are valid (answer tokens)
        token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        token_mask[:, 4:8] = True

        logits, probs = decoder(h_seq, token_mask)
        assert logits.shape == (batch_size, 1)
        assert probs.shape == (batch_size, 1)

        decision_high = decoder.evaluate_decision(0.85)
        assert decision_high["requires_review"] is False
        assert decision_high["accepted"] is True

        decision_low = decoder.evaluate_decision(0.42)
        assert decision_low["requires_review"] is True
        assert decision_low["accepted"] is False

        print(f"  -> Pooling '{method}' passed: prob={probs[0].item():.4f}, decision rules verified.")


def test_confidence_metrics_computation():
    """Verify Brier Score, ECE, AUROC, and Risk-Coverage curve."""
    print("[TEST] Testing confidence and calibration metrics...")
    # Synthetic ground truths: 5 correct (1), 5 wrong (0)
    targets = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # Synthetic well-calibrated confidences
    confs = [0.95, 0.90, 0.88, 0.85, 0.80, 0.20, 0.15, 0.25, 0.30, 0.10]

    brier = compute_brier_score(confs, targets)
    assert 0.0 <= brier <= 1.0
    print(f"  -> Brier score: {brier:.4f}")

    ece, bins = compute_ece(confs, targets, num_bins=5)
    assert 0.0 <= ece <= 1.0
    print(f"  -> ECE: {ece:.4f}")

    # Perfect error detection ordering -> AUROC should be 1.0
    auroc = compute_error_detection_auroc(confs, targets)
    assert np.isclose(auroc, 1.0), f"Expected perfect AUROC 1.0, got {auroc}"
    print(f"  -> Error detection AUROC: {auroc:.4f}")

    # Risk-coverage curve
    rc_curve = compute_risk_coverage(confs, targets)
    assert len(rc_curve) > 0
    # At high threshold (e.g. 0.8), coverage should be 0.5 (only the 5 correct ones) and risk should be 0.0
    high_tau_point = [pt for pt in rc_curve if pt["threshold"] >= 0.75][0]
    assert high_tau_point["risk"] == 0.0, f"Expected 0 risk at high threshold, got {high_tau_point['risk']}"
    print(f"  -> Risk-Coverage verified: at tau={high_tau_point['threshold']}, coverage={high_tau_point['coverage']}, risk={high_tau_point['risk']}")

    # Comprehensive summary
    full_metrics = compute_all_confidence_metrics(confs, targets, threshold=0.70)
    assert full_metrics["review_rate"] == 0.5
    assert full_metrics["accepted_exact_match"] == 1.0
    print("  -> Full confidence metrics suite passed.")


def test_end_to_end_uncertainty_forward():
    """Verify joint forward pass: Qwen3-VL hidden state extraction + Uncertainty Decoder + combined loss."""
    print("[TEST] Testing end-to-end Qwen3-VL hidden states + UncertaintyDecoder forward...")
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

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    starts = (inputs["input_ids"][0] == im_start_id).nonzero(as_tuple=True)[0]
    labels[0, : starts[-1] + 3] = -100

    # 1. Forward with output_hidden_states=True
    outputs = model(
        **inputs,
        labels=labels,
        output_hidden_states=True,
    )

    last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, 2048]
    dim = last_hidden_state.shape[-1]

    # 2. UncertaintyDecoder
    unc_decoder = UncertaintyDecoder(
        hidden_dim=dim,
        head_type="mlp",
        pooling_method="mean",
        confidence_threshold=0.70,
    ).to(device=device, dtype=last_hidden_state.dtype)

    # 3. Answer token mask (only target tokens)
    answer_mask = (labels != -100)  # [1, seq_len]

    # 4. Predict confidence
    conf_logits, conf_probs = unc_decoder(last_hidden_state, answer_mask)
    assert conf_probs.shape == (1, 1)

    # 5. Combined objective: L_total = L_text + lambda_c * L_conf
    # Ground truth is correct (z = 1) for this example
    z_target = torch.tensor([1.0], device=device)
    loss_conf = unc_decoder.confidence_head.compute_loss(conf_logits, z_target)

    lambda_c = 0.5
    loss_total = outputs.loss + lambda_c * loss_conf

    assert torch.isfinite(loss_total)
    decision = unc_decoder.evaluate_decision(conf_probs.item())

    print(f"  -> Joint forward passed! L_text: {outputs.loss.item():.4f}, L_conf: {loss_conf.item():.4f}, L_total: {loss_total.item():.4f}")
    print(f"  -> Predicted confidence: {decision['confidence']:.4f}, Requires Review: {decision['requires_review']}")


if __name__ == "__main__":
    test_confidence_head_forward_and_loss()
    test_uncertainty_decoder_pooling_and_thresholding()
    test_confidence_metrics_computation()
    test_end_to_end_uncertainty_forward()
