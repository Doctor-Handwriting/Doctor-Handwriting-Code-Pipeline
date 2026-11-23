# Vision-Language Model Training Pipeline

Production-ready handwriting-recognition training pipeline optimized for **RTX 3050 Ti (4GB VRAM)** and larger RTX 5070/5070 Ti systems. It supports Qwen-VL, the proposed MS-MGCA-Qwen3-VL, the paper-based CRNN, and YOLO/YOLO26 classification models.

## Docs

- [docs/MODEL_STATUS.md](docs/MODEL_STATUS.md): models already run and not yet run, with results and datasets
- [docs/how_to_run.md](docs/how_to_run.md): commands for every stage and model
- [docs/NOTEBOOK_GUIDE.md](docs/NOTEBOOK_GUIDE.md): running training from Jupyter / Colab / Kaggle, hardware profiles

## Features

- **Memory-Optimized**: 4-bit quantization (NF4), LoRA adaptation, gradient checkpointing
- **Dataset Parsing**: Automated parsing of DHP + RxHand datasets with multimodal instruction format
- **Augmentation**: Medical handwriting-specific augmentations (motion blur, elastic deformation, perspective)
- **Training**: Full Hugging Face Trainer integration with mixed precision support
- **Inference**: Batch inference with WER/CER/accuracy metrics
- **Monitoring**: TensorBoard logging, memory profiling, checkpoint management

## Project Structure

```
Doctor-Handwriting-Code-Pipeline/
├── configs/
│   ├── __init__.py
│   ├── config.py          # Dataclass configs (hardware, LoRA, training, paths)
│   └── splits/rxhand/     # train.csv / val.csv / test.csv of the held-out RxHand split (committed)
├── docs/                  # MODEL_STATUS.md, how_to_run.md, NOTEBOOK_GUIDE.md
├── paper/                 # LaTeX manuscript (manuscript/) and reference PDFs (references/)
├── models/
│   ├── __init__.py
│   ├── models.py          # Qwen-VL, CRNN, and YOLO/YOLO26 model loaders
│   └── ms_mgca.py         # Adapter that plugs experiments/MS-MGCA-Qwen3-VL into the pipeline (+ confidence calibration)
├── experiments/
│   └── MS-MGCA-Qwen3-VL/  # Proposed model: code, E1-E6 ablation configs, unit tests (own README)
├── utils/
│   ├── __init__.py
│   ├── dataset_parser.py  # DHP + RxHand parsing to instruction format
│   ├── augmentor.py       # Medical handwriting augmentations
│   ├── metrics.py         # WER, CER, and accuracy computation
│   ├── visualization.py   # Loss/accuracy/CER/WER/LR training plots
│   └── confidence_plots.py  # Reliability diagram + risk-coverage curve (MS-MGCA confidence head)
├── run_ms_mgca_ablations.py  # Runs E1-E6 (or the K sweep) end to end: train -> calibrate -> infer
├── compare_ms_mgca.py     # Ablation table + comparison plots over the finished MS-MGCA runs
├── Augmented_Dataset/     # AUGMENTATION_MODE = "offline" only (created by STAGE = "augment", git-ignored)
│   ├── DHP/    {original/, augmented/, augmentation_manifest.json}
│   └── RxHand/ {original/, augmented/, augmentation_manifest.json}
├── Dataset/               # DHP/ and RxHand/ (not committed, see "Dataset Location")
├── runs/                  # Outputs, one folder per model per run (committed except model weights)
│   ├── train_augmented.json, val_augmented.json   # augment stage
│   └── <model_name>/                               # e.g. Qwen3-VL-8B-Instruct
│       └── run_<YYYYmmdd-HHMMSS>_<profile>[_<dataset>[-heldout]][_offline]/
│           ├── checkpoints/        # checkpoint-*/ and final/ (LoRA adapter; weight files are git-ignored)
│           ├── logs/               # TensorBoard events
│           ├── plots/              # loss, accuracy, error_rates, learning_rate, training_overview,
│           │                       # evaluation_metrics (every metric), final_metrics, inference_metrics .png
│           ├── metrics_history.json / .csv   # every logged train/eval value
│           ├── metrics_summary.csv # best / average / last loss & accuracy
│           ├── eval_metrics.json   # final evaluation
│           ├── run_config.json     # full config snapshot of the run
│           └── inference_results.json, inference_metrics.json   # infer stage
├── main.py                # Pipeline orchestrator (augment/train/calibrate/infer)
└── requirements.txt       # All dependencies
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# or
source venv/bin/activate      # Linux/macOS

# For CUDA support, install torch matching your GPU driver first.
# RTX 5070/5070 Ti require the cu128 build and an NVIDIA driver >= 570:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

## Dataset Location (any machine)

`Dataset/` is not committed to git. After cloning, copy it into the repo root with the same structure:

```
<repo>/Dataset/
├── DHP/
│   ├── doctor_handwriting_labels.csv
│   └── img/img/
└── RxHand/
    ├── Train_Label.csv
    ├── Test_Labels.csv
    ├── Train_Set/
    └── Test_Set/
```

All paths are resolved relative to the repo, so no code edits are needed. To keep the data or outputs elsewhere, set environment variables instead:

```bash
export VLM_DATASET_ROOT=/mnt/data/Dataset   # Linux/Mac
export VLM_OUTPUT_DIR=/mnt/data/runs        # optional, defaults to <repo>/runs
$env:VLM_DATASET_ROOT = "E:\Dataset"        # Windows PowerShell
```

`main.py` checks these paths on startup and lists any missing files.

## Quick Start

### 1. Data Preparation (Automatic)

The pipeline automatically:
- Parses DHP (Doctor Handwriting) dataset with labels
- Parses RxHand (Handwritten Prescription) dataset splits
- Filters corrupted `.trashed-*` files
- Creates multimodal instruction JSON format
- Splits the data into train / validation (and test, with the held-out split below)

**Team standard from now on: RxHand with the held-out split** (`--dataset rxhand --heldout-test`, or
`DATASET_MODE = "rxhand"` + `RXHAND_HELDOUT_TEST = True` in `main.py`):

| Part | Content | Used by |
|---|---|---|
| train | 80% of `Train_Set` | `train` |
| val | 20% of `Train_Set` | `train` (checkpoint selection), `calibrate` |
| test | official `Test_Set`, untouched | `infer` only |

The image lists are stored in `configs/splits/rxhand/{train,val,test}.csv` and committed, so every machine uses
the same images. They are written on the first held-out run (files sorted by name, shuffled with seed 42) and read
back afterwards; a listed image missing on disk stops the run, which means that machine has a different RxHand copy.
Held-out runs are named `..._rxhand-heldout`, and `infer` only picks held-out runs, never older ones that saw
`Test_Set` during training.

Other setups (used by the runs before the held-out split, no separate test set):

| Setting | Split |
|---|---|
| `DATASET_MODE = "combined"` | DHP + RxHand merged, then 80% train / 20% val (seed 42) |
| `DATASET_MODE = "dhp"` / `"rxhand"` | one dataset only, `Train_Set` + `Test_Set` merged, then 80 / 20 (seed 42) |
| `"rxhand"` + `RXHAND_OFFICIAL_SPLIT = True` | train = `Train_Set`, val = `Test_Set` (test images also pick the checkpoint) |

Results from these setups are not directly comparable with held-out runs. The standalone scripts in
`experiments/MS-MGCA-Qwen3-VL/` (`evaluation/evaluate.py`, ...) use their own split (seed 3407); report paper
numbers from `main.py --stage infer --heldout-test`. Details: [docs/how_to_run.md](docs/how_to_run.md).

### 2. Run Pipeline

**Augmentation Stage:**
```bash
python main.py
# Modify STAGE = "augment" in main.py first
```

**Training Stage:**
```bash
# Edit main.py:
STAGE = "train"
MODEL_CHOICE = "qwen-vl"  # or "ms-mgca", "crnn", "yolo", "yolo26"
QWEN_CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"  # backbone for "qwen-vl" and "ms-mgca" (Qwen3-VL only)
MSMGCA_EXPERIMENT = "e6_full_model"   # used when MODEL_CHOICE = "ms-mgca": e1_baseline ... e6_full_model
YOLO26_CHECKPOINT = "yolo26m-cls.pt"  # n / s / m / l / x, used when MODEL_CHOICE = "yolo26"
HARDWARE_PROFILE = "pc_5070ti"  # or "laptop_3050ti", "pc_3070ti", "pc_5070", "auto"
DATASET_MODE = "combined"  # or "dhp" / "rxhand" to train and validate on one dataset only
RXHAND_OFFICIAL_SPLIT = False  # "rxhand" only: True = train on Train_Set, validate on Test_Set
AUGMENTATION_MODE = "online"  # or "offline": pre-computed Augmented_Dataset/ (run STAGE = "augment" first)

python main.py
```

**Augmentation mode** (`AUGMENTATION_MODE` in `main.py`):

| | `"online"` (default) | `"offline"` |
|---|---|---|
| What happens | random transforms applied to each training image while training | variants pre-computed once, training reads them from disk |
| Extra disk | none | ~8x the images (measured 0.7 GB for DHP + RxHand: 0.2 GB DHP, 0.5 GB RxHand) |
| Variety | new variation every epoch | fixed set of variants (reproducible, seeded) |
| Inspect the augmented images | no | yes, browse `Augmented_Dataset/` |
| Train epoch size | 1x | ~8x (original + 7 variants), so use fewer epochs |

```python
AUGMENTATION_MODE = "offline"
STAGE = "augment"   # python main.py -> builds Augmented_Dataset/ (once; re-runs only fill in what is missing)
STAGE = "train"     # python main.py -> trains on the pre-computed images; "infer" validates on them too
```

Offline layout (per dataset; paths below `original/` and `augmented/` mirror the dataset's own folders):
```
Augmented_Dataset/DHP/original/img/img/<name>.jpg                 copy of every original
Augmented_Dataset/DHP/augmented/img/img/<name>_motionblur.png     and _elasticdeform, _perspective,
                                                                  _brightnesscontrast, _rotate,
                                                                  _shiftscalerotate, _mixed (all six together)
Augmented_Dataset/DHP/augmentation_manifest.json                  original -> its variants, plus the settings used
Augmented_Dataset/RxHand/{original,augmented}/{Train_Set,Test_Set}/...
```
Only training images get variants: the train/val split is made on the originals first, so no augmented copy of a
validation image can leak into training, and validation scores stay comparable with online runs. Variants are
generated for every image of the datasets selected by `DATASET_MODE`, so changing the split later needs no
regeneration; a different `DATASET_MODE` needs `STAGE = "augment"` run once for it. Labels still come from
`Dataset/`. To regenerate after changing the augmentation settings, delete `Augmented_Dataset/` and re-run
the augment stage (a settings mismatch is warned about at train time). Offline runs get `_offline` in their run
folder name. `VLM_AUGMENTED_DIR` moves the folder; `AugmentationConfig.offline_include_mixed = False` drops
the `_mixed` variant.

**YOLO26 (`MODEL_CHOICE = "yolo26"`)** fine-tunes the YOLO26 classification backbones
(`yolo26{n,s,m,l,x}-cls.pt`, 3.6M-30.4M parameters) as a closed-vocabulary word recognizer: each distinct
training label is a class. Ultralytics has no text-generating YOLO26 variant (the open-vocabulary YOLOE-26 models
are box detectors), so classification is the only YOLO26 task that fits box-less word labels. Every size runs on
every profile at the same effective batch per profile (32 laptop / 64 `pc_3070ti` / 64 `pc_5070` / 128 `pc_5070ti`);
`yolo26x` uses gradient accumulation on the 4GB laptop (16 x 2) and on `pc_3070ti` (32 x 2), everything else fits
in one pass.

**Inference Stage:**
```bash
STAGE = "infer"
python main.py
```

## MS-MGCA-Qwen3-VL (`MODEL_CHOICE = "ms-mgca"`)

Multi-Scale and Uncertainty-Aware Qwen3-VL with Medication-Guided Cross-Attention
(`experiments/MS-MGCA-Qwen3-VL/`, see its README for the architecture). It runs through the same
`main.py` stages, datasets (`DATASET_MODE`, `AUGMENTATION_MODE`), metrics, plots and `runs/` layout as the
other models, so its validation scores are directly comparable.

Pick the ablation with `MSMGCA_EXPERIMENT` and the backbone with `QWEN_CHECKPOINT`:

| `MSMGCA_EXPERIMENT` | Multi-scale fusion (A) | Medication cross-attention (B) | Confidence head (D) |
|---|:---:|:---:|:---:|
| `e1_baseline` | | | |
| `e2_multiscale` | yes | | |
| `e3_cross_attention` | | yes | |
| `e4_confidence` | | | yes |
| `e5_multiscale_crossattn` | yes | yes | |
| `e6_full_model` (default) | yes | yes | yes |

```python
MODEL_CHOICE = "ms-mgca"
QWEN_CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"   # any Qwen3-VL checkpoint; 2B trains in plain bf16 on pc_5070
MSMGCA_EXPERIMENT = "e6_full_model"
MSMGCA_NUM_QUERIES = None   # medication queries K (E3/E5/E6): None = 4, or 1 / 4 / 8 for the K sweep
STAGE = "train"   # then "calibrate" (E4/E6 only), then "infer"
```

Every one of these constants can also be passed on the command line instead of editing `main.py`
(`python main.py --help`):

```bash
python main.py --model ms-mgca --experiment e6_full_model --checkpoint Qwen/Qwen3-VL-2B-Instruct --stage train
python main.py --model ms-mgca --experiment e6_full_model --checkpoint Qwen/Qwen3-VL-2B-Instruct --stage calibrate
python main.py --model ms-mgca --experiment e6_full_model --checkpoint Qwen/Qwen3-VL-2B-Instruct --stage infer
```

### Full ablation (E1-E6) and report

```bash
python run_ms_mgca_ablations.py                       # E1..E6: train -> calibrate (E4/E6 only) -> infer, 2B backbone
python run_ms_mgca_ablations.py --sweep-k 1 4 8       # E6 with K = 1, 4, 8
python run_ms_mgca_ablations.py --checkpoint Qwen/Qwen3-VL-4B-Instruct   # the same with the 4B backbone
python compare_ms_mgca.py                             # table + plots in runs/MS-MGCA-Qwen3-VL/ablation_report/
```

**Resuming after an interruption** (MS-MGCA only): add `--resume` to the same command (and the same `--epochs`).

```bash
python run_ms_mgca_ablations.py --resume              # or: python main.py ... --stage train --resume
```

`train` finds the newest run of that exact setting (model, experiment, backbone, K, dataset, augmentation), restores
the LoRA/module weights, optimizer, learning-rate schedule and step counter from its latest `checkpoint-<step>`, and
continues in the same run folder (the plots and `metrics_history` keep the earlier steps; only the steps after the
checkpoint are repeated). Stages whose output already exists are skipped: a finished training, a `calibrated/`
checkpoint, an `inference_metrics.json`. With no earlier run it simply starts a new one. `--epochs` must match the
original run, otherwise the learning-rate schedule would change, so the run is refused with a message.
Checkpoints are written every `save_steps` (100) optimizer steps, so at most that much work is lost.

All runs of one ablation share backbone, dataset, augmentation and seed (3407; the random initialisation of the new
modules is seeded too), so the rows of the table differ only in the architecture. `compare_ms_mgca.py` writes
`ablation_table.md/.csv` (exact match, normalized match, CER, WER, latency, peak VRAM, ECE, Brier, error-detection
AUROC, review rate), `metrics_comparison.png`, one `reliability_<run>.png` per confidence-aware run and
`risk_coverage_comparison.png`. Use `--dataset` / `--size` to compare only runs of one setting.

- **Training** fine-tunes LoRA and all enabled MS-MGCA components jointly. Each component has its own learning
  rate from the experiment config (LoRA 2e-4, adapters 5e-4, confidence head 1e-3). With the confidence head the loss
  is `L_text + lambda_c * L_conf`; during training a sample counts as correct (z = 1) when the model's teacher-forced
  prediction reproduces every answer token, which is only a proxy on data the model is fitting (see calibration).
  Checkpoints are selected on the text loss alone.
- **Calibration (`STAGE = "calibrate"`, E4/E6):** the confidence head must learn whether a *generated* transcription
  is right, which needs labels from images the model did not train on. The stage transcribes the validation images
  (never trained on), labels each transcription right/wrong (exact match), splits them once into a calibration part
  (`EvaluationConfig.calibration_fraction`, default 50%) and a held-out part, fits the head on the calibration part
  with BCE, and reports ECE / Brier / AUROC on the held-out part before and after
  (`calibration_metrics.json`, `plots/reliability_*_calibration.png`). The result is saved as
  `checkpoints/calibrated/`, which the infer stage then uses, scoring confidence on the held-out part only.
  This is a one-fold hold-out version of out-of-fold calibration (a true K-fold would retrain the model K times).
- **Batch size:** the multi-scale / cross-attention path handles one image at a time, so MS-MGCA always uses
  micro-batch 1 with gradient accumulation (effective batch 8 on every profile, same as the other models).
  It needs a GPU with bf16 support.
- **Outputs:** `runs/MS-MGCA-Qwen3-VL/run_<time>_<profile>_<experiment>_<size>[_dataset][_offline]/`, e.g.
  `run_20261111-101500_pc_5070_e6_full_model_2B_rxhand`. `checkpoints/final/` holds only the LoRA adapter
  (`lora_adapter/`), the module weights (`ms_mgca_modules.safetensors`) and the experiment config
  (`ms_mgca_config.yaml`). `STAGE = "infer"` evaluates the latest run of the selected experiment and backbone and
  rebuilds the architecture from that saved config.
- **Confidence head (E4, E6):** `inference_results.json` gets `confidence` and `requires_review` per sample
  (flagged when confidence < the experiment's threshold, default 0.70), and the infer stage writes
  `inference_confidence_metrics.json` (Brier score, ECE, error-detection AUROC, review rate, risk-coverage curve) plus
  `plots/reliability_diagram.png` and `plots/risk_coverage.png`. Without a calibrated checkpoint the head is the
  jointly trained one and is scored on all validation images, which the file records under `evaluated_on`.
- **Every MS-MGCA infer run** also writes `inference_efficiency.json` (latency per sample, peak GPU memory).
- **K sweep:** `MSMGCA_NUM_QUERIES` (or `--queries`) sets the number of medication queries; the run folder ends in
  `_k<K>`, so the three runs do not overwrite each other.
- **Images:** with multi-scale fusion on, non-square images are padded to a square on white (the RxHand crops are
  already square, so they are unchanged), because the fusion maps local crops onto a square token grid.
- The experiment's own scripts (`training/train_stage.py` progressive stages 2-5, `evaluation/evaluate.py`,
  `inference/predict.py`, `tests/`) still work standalone and use their own RxHand split; use `main.py` for results
  that must match the other models.

## Configuration

Edit `configs/config.py` to customize:

### Hardware Profiles

Set `HARDWARE_PROFILE` in `main.py` to `auto`, `laptop_3050ti`, `pc_3070ti`, `pc_5070`, or `pc_5070ti`.
The selected profile supplies batch size, accumulation, workers, image resolution, and precision:

| Profile | Precision | Train batch x accumulation | Max image pixels |
|---|---|---:|---:|
| `laptop_3050ti` | 4-bit NF4 + fp16 | 1 x 8 | 512 x 512 |
| `pc_3070ti` | NF4 QLoRA + bf16 for Qwen3-VL-2B/4B (8B not recommended) | 1 x 8 | 640 x 640 |
| `pc_5070` | NF4 QLoRA + bf16 for Qwen3-VL-8B | 2 x 4 | 768 x 768 |
| `pc_5070ti` | NF4 QLoRA + bf16 for Qwen3-VL-8B | 4 x 2 | 768 x 768 |

Qwen checkpoints that fit the profile's bf16 parameter limit skip 4-bit quantization. The `crnn` recipe
always uses fp32 and batch 10; YOLO/YOLO26 batch sizes come from the selected profile. If the 8B Qwen model
causes CUDA OOM, reduce the profile batch to 1 x 8 in `configs/config.py` or use a smaller checkpoint.

The laptop-specific defaults are:
```python
per_device_train_batch_size = 1           # Don't increase!
gradient_accumulation_steps = 8           # Effective batch = 8
num_workers = 2                           # Prevent RAM pressure
```

### Quantization
```python
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = torch.float16
bnb_4bit_use_double_quant = True
```

### LoRA Adaptation
```python
r = 32, lora_alpha = 64, lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
```

### Augmentation
```python
enable_augmentation = True  # used by online augmentation
motion_blur_prob = 0.3
elastic_deform_prob = 0.3
perspective_prob = 0.2
brightness_contrast_prob = 0.5
```

`AUGMENTATION_MODE = "offline"` is a separate workflow: run `STAGE = "augment"` first to create
`Augmented_Dataset/`, then run `train` or `infer`. Offline runs include `_offline` in their run name and
use fixed, reproducible variants; validation images are never augmented.

## Dataset Format

Datasets are automatically converted to multimodal instruction format:

```json
[
  {
    "id": "rxhand_train_1116",
    "image": "D:/path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nRead and transcribe the handwritten text from this prescription accurately."
      },
      {
        "from": "gpt",
        "value": "Extracted prescription text here"
      }
    ]
  }
]
```

## Memory Management

**4GB VRAM Optimization:**
- Batch size: 1 (per device)
- Gradient accumulation: 8 steps (effective batch = 8)
- Mixed precision: fp16
- Gradient checkpointing: enabled
- Dynamic resolution: 256×256 to 512×512

**Expected Memory Usage:**
- Model weights (4-bit): ~1.5 GB
- Activations + optimizer: ~2.0 GB
- Reserve: ~0.5 GB for overhead

## Monitoring

### TensorBoard
```bash
tensorboard --logdir runs   # every run of every model, side by side
```

### Training Plots
After every evaluation and at the end of training, PNG charts are (re)written to
`runs/<model_name>/<run>/plots/`: loss (train vs val), accuracy (train/val token accuracy and
val exact match), val CER/WER, learning rate, a 2x2 `training_overview.png`, `evaluation_metrics.png` (one panel
per evaluation metric: loss, token accuracy, exact match, normalized match, char accuracy, CER, WER,
MER, WIL) and `final_metrics.png` (bar chart of the latest evaluation). The infer stage adds
`inference_metrics.png`. `metrics_summary.csv` holds best/average/last of every metric.
Redraw them for any past run (falls back to the newest `trainer_state.json` for older runs):
```bash
python -m utils.visualization runs/Qwen3-VL-2B-Instruct/run_20261002-120000_laptop_3050ti
```

### Checkpoints
- Auto-saved every `save_steps` iterations
- Final adapter in `runs/<model_name>/<run>/checkpoints/final`
- Resume training from checkpoint (automatic)

### Progress & Metrics
Training shows a progress bar with live `loss`, `tok_acc`, `lr`, `grad_norm` and the latest eval metrics.
Every `eval_steps` a table of all evaluation metrics is printed and logged to TensorBoard;
the final evaluation is saved to `<run>/eval_metrics.json`. The infer stage evaluates the
latest trained run of the model (`<run>/checkpoints/final`; set `cfg.run_name` to pick another)
on the validation set and writes `inference_results.json` (per sample) and
`inference_metrics.json` into that run's folder.

- **token_accuracy / eval_token_accuracy**: Next-token accuracy on answer tokens (teacher-forced), train and val

- **eval_loss**: Validation loss (teacher-forced)
- **WER / MER / WIL**: Word error / match error / word information lost rates — lower is better
- **CER** (Character Error Rate): Lower is better
- **char_accuracy**: 1 − CER — higher is better
- **accuracy**: Exact match %; **accuracy_normalized**: case/whitespace-insensitive match %
- **YOLO/YOLO26 accuracy**: these models are closed-vocabulary classifiers. Each distinct training label is
  a class, so a word absent from the training vocabulary can never be predicted. Use Qwen-VL or CRNN for
  free-form transcription. YOLO detection checkpoints such as `yolov8n.pt` are not supported; use a
  `-cls` checkpoint.

WER/CER/accuracy require `model.generate`, which is slow, so during training they are computed on
`cfg.evaluation.train_eval_samples` (default 50) val samples. On a bigger GPU set it higher or to
`None` for the whole val set; `cfg.evaluation.infer_samples` controls the infer stage.

## Training Tips

1. **First Run**: Monitor GPU memory on first step
   ```python
   # In main.py, reduce max_samples temporarily
   cfg.dataset.max_samples = 100
   ```

2. **Checkpointing**: Resume interrupted training
   ```python
   # Trainer automatically resumes from latest checkpoint
   ```

3. **Validation**: Check `runs/<model_name>/<run>/plots/` or TensorBoard

4. **Fine-tune Learning Rate**: Start at 1e-4, adjust if loss plateaus
   ```python
   cfg.training.learning_rate = 5e-5  # More conservative
   ```

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_eval_batch_size` to 1
- Disable augmentation: `cfg.augmentation.enable_augmentation = False`
- Reduce `max_seq_length` in ModelConfig

### Slow Data Loading
- Increase `num_workers` cautiously (check RAM usage)
- Pre-process images to consistent size
- Use SSD/NVMe for dataset (avoid network drives)

### Missing Dataset Files
- Make sure `Dataset/` sits in the repo root, or set `VLM_DATASET_ROOT` (see "Dataset Location")
- Check `.trashed-*` files are being filtered
- Run with `cfg.dataset.max_samples = 10` to debug

## Performance Benchmarks

On RTX 3050 Ti (4GB):
- **Training Speed**: ~2-3 samples/sec with gradient accumulation
- **Inference Speed**: ~1 sample/sec
- **Epoch Time**: ~40-60 minutes (varies by dataset size)
- **Memory Peak**: ~3.8 GB (safe margin from limit)

## References

- Qwen-VL: https://github.com/QwenLM/Qwen-VL
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- PEFT/LoRA: https://github.com/huggingface/peft
- Bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- Hugging Face Trainer: https://huggingface.co/docs/transformers/training

## License

Educational and research use only.
