# MS-MGCA-Qwen3-VL: Multi-Scale and Uncertainty-Aware Qwen3-VL Architecture with Medication-Guided Cross-Attention

Research implementation of a modified **Qwen3-VL** multimodal architecture designed for handwritten doctor's prescription medication recognition.

---

## 1. Architecture Overview

```text
Prescription Image (I)
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
Global Branch                            Local Crops (2x2 Grid)
       │                                        │
Qwen3-VL Vision Encoder                  Qwen3-VL Vision Encoder
       │                                        │
Global Features: F_g in R^(B x 64 x d)   Local Features: F_l in R^(B x M x T_c x d)
       │                                        │
       └───────────────────┬────────────────────┘
                           ▼
          [Component A: Multi-Scale Feature Fusion]
          • Cross-Attention / Spatial Pooling Reduction
          • Learnable Gated Fusion: F_fused = G * F_g + (1 - G) * F_l
                           │
                           ▼
          [Component B: Medication-Guided Cross-Attention]
          • K Learnable Medication Queries: Q_m in R^(K x d)
          • Query Cross-Attention: H_m = MHA(Q_m, F_fused, F_fused)
          • Visual Adapter Modulation: F_med = F_fused + gamma * Adapter(H'_m)
                           │
                           ▼
          Visual-Language Integration (mm_encoder_outputs)
          • Token Invariant Preserved (T_v = 64 tokens)
          • 3D M-RoPE & DeepStack Residual Compatibility
                           │
                           ▼
          Qwen3-VL Language Decoder (4-bit NF4 + LoRA)
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
       Text Decoder                  Last Hidden State (h)
            │                             │
     Medication Text               [Component D: Uncertainty Decoder]
            │                      • Mean/Last/Attention Token Pooling
            │                      • Confidence Head: c = sigmoid(W_c h + b_c)
            │                      • Objective: L_total = L_text + lambda_c * L_conf
            │                             │
            └──────────────┬──────────────┘
                           ▼
                Clinical Decision Rule
                • c < tau  => Requires Review (Flagged)
                • c >= tau => Accepted Automated Prediction
```

---

## 2. Experimental Ablation Matrix

The architecture provides independent configuration switches for the research ablation studies:

| Experiment | Config File | Multi-Scale (Comp A) | Medication Cross-Attn (Comp B) | Confidence Head (Comp D) | Description |
|---|---|:---:|:---:|:---:|---|
| **E1** | `configs/e1_baseline.yaml` | ❌ | ❌ | ❌ | Original Qwen3-VL baseline |
| **E2** | `configs/e2_multiscale.yaml` | ✅ | ❌ | ❌ | Qwen3-VL + Multi-Scale Gated Fusion |
| **E3** | `configs/e3_cross_attention.yaml` | ❌ | ✅ | ❌ | Qwen3-VL + Medication Queries |
| **E4** | `configs/e4_confidence.yaml` | ❌ | ❌ | ✅ | Qwen3-VL + Uncertainty-Aware Decoding |
| **E5** | `configs/e5_multiscale_crossattn.yaml`| ✅ | ✅ | ❌ | Qwen3-VL + Multi-Scale + Cross-Attention |
| **E6** | `configs/e6_full_model.yaml` | ✅ | ✅ | ✅ | **Full Proposed MS-MGCA-Qwen3-VL** |

---

## 3. Directory Structure

```text
experiments/MS-MGCA-Qwen3-VL/
├── configs/
│   ├── base_config.py              # Central MSMGCAConfig dataclasses & YAML serialization
│   ├── e1_baseline.yaml            # E1 configuration
│   ├── e2_multiscale.yaml          # E2 configuration
│   ├── e3_cross_attention.yaml     # E3 configuration
│   ├── e4_confidence.yaml          # E4 configuration
│   ├── e5_multiscale_crossattn.yaml# E5 configuration
│   └── e6_full_model.yaml          # E6 full proposed model configuration
├── dataset/
│   ├── rxhand_dataset.py           # PyTorch Dataset, deterministic 80/20 split, Collator
│   └── crop_generator.py           # DeterministicCropGenerator (2x2 grid, overlapping bounds)
├── models/
│   ├── baseline_qwen3vl.py         # 4-bit NF4 quantized loader, LoRA adapter, runner
│   ├── proposed_model.py           # Unified MSMGCAQwen3VL model
│   └── modules/
│       ├── multiscale_extractor.py # Local-to-global attention & spatial pooling reduction
│       ├── feature_fusion.py       # Gated, concatenation, and attention fusion
│       ├── medication_cross_attention.py # K learnable queries + visual adapter
│       ├── confidence_head.py      # MLP/Linear correctness predictor
│       └── uncertainty_decoder.py  # Token pooling & clinical decision evaluation
├── training/
│   ├── trainer.py                  # Multi-loss progressive trainer with independent LR groups
│   └── train_stage.py              # CLI training driver for stages 2 to 5
├── evaluation/
│   ├── metrics.py                  # CER, WER, Exact Match, Normalized Match, Character Accuracy
│   ├── confidence_metrics.py       # Pure NumPy Brier Score, ECE, Error AUROC, Risk-Coverage
│   └── evaluate.py                 # Standalone benchmark evaluation CLI
├── inference/
│   └── predict.py                  # Single-image & batch inference with requires_review flag
├── results/
│   └── E1_baseline/                # Quantitative benchmark results & predictions
└── tests/
    ├── test_dataset.py             # Dataset loading & split reproducibility tests
    ├── test_forward.py             # Baseline forward & token masking tests
    ├── test_multiscale.py          # Component A unit tests
    ├── test_cross_attention.py     # Component B unit tests
    ├── test_confidence.py          # Component D unit tests
    └── test_full_model.py          # Full integrated E6 model tests
```

---

## 4. Quick Start & Execution Guide

Activate the Conda environment:
```powershell
conda activate doctor_vlm
```

### 4.1 Running All Unit Tests
Verify every component independently:
```powershell
python experiments/MS-MGCA-Qwen3-VL/tests/test_dataset.py
python experiments/MS-MGCA-Qwen3-VL/tests/test_forward.py
python experiments/MS-MGCA-Qwen3-VL/tests/test_multiscale.py
python experiments/MS-MGCA-Qwen3-VL/tests/test_cross_attention.py
python experiments/MS-MGCA-Qwen3-VL/tests/test_confidence.py
python experiments/MS-MGCA-Qwen3-VL/tests/test_full_model.py
```

### 4.2 Evaluating a Model (Benchmark Evaluation)
Run evaluation on the validation split:
```powershell
python experiments/MS-MGCA-Qwen3-VL/evaluation/evaluate.py --config configs/e1_baseline.yaml --split val --max_samples 100
```
Outputs are automatically saved to `results/<run_name>/<split>/eval_summary.json` and `predictions.json`.

### 4.3 Training Stages
The architecture supports progressive training (Section 7):

* **Stage 2 (Multi-Scale Module):**
  ```powershell
  python experiments/MS-MGCA-Qwen3-VL/training/train_stage.py --config configs/e2_multiscale.yaml --stage 2 --epochs 5
  ```
* **Stage 3 (Medication Cross-Attention):**
  ```powershell
  python experiments/MS-MGCA-Qwen3-VL/training/train_stage.py --config configs/e5_multiscale_crossattn.yaml --stage 3 --epochs 5
  ```
* **Stage 4 (Confidence Head Calibration):**
  ```powershell
  python experiments/MS-MGCA-Qwen3-VL/training/train_stage.py --config configs/e4_confidence.yaml --stage 4 --epochs 3
  ```
* **Stage 5 (Joint Fine-Tuning - Full Model):**
  ```powershell
  python experiments/MS-MGCA-Qwen3-VL/training/train_stage.py --config configs/e6_full_model.yaml --stage 5 --epochs 10
  ```

### 4.4 Inference with Uncertainty Decisions
Transcribe a prescription image and obtain a calibrated confidence score and review flag:
```powershell
python experiments/MS-MGCA-Qwen3-VL/inference/predict.py --image Dataset/RxHand/Train_Set/P1116.jpg --config configs/e6_full_model.yaml --threshold 0.70
```

Sample output:
```json
{
  "image": "P1116.jpg",
  "prediction": "Dexter",
  "confidence": 0.9421,
  "requires_review": false
}
```

When confidence is low ($c < \tau$):
```json
{
  "image": "P1117.jpg",
  "prediction": "Clavusef sw",
  "confidence": 0.4128,
  "requires_review": true
}
```

---

## 5. Hardware Profile & Memory Efficiency

Benchmarked on **NVIDIA GeForce RTX 5070 12GB**:
* **Quantization:** 4-bit NF4 quantized language decoder; unquantized native BF16 vision encoder.
* **Peak GPU Memory:**
  * Baseline (E1): ~1.55 GB
  * Multi-Scale Fusion (E2): ~1.73 GB
  * Full Architecture (E6): ~1.85 GB
* **Throughput:** ~3.58 samples / sec (batch size = 1 greedy decoding).
* **Headroom:** Leaves >9.0 GB VRAM available for training batches and gradient accumulation.
