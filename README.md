# Vision-Language Model Training Pipeline

Handwriting-recognition training pipeline for the RTX 3050 Ti (4GB), RTX 3070 Ti (8GB) and RTX 5070 / 5070 Ti
machines. It supports Qwen-VL, the proposed MS-MGCA-Qwen3-VL, the paper-based CRNN, and YOLO/YOLO26 classification
models. Every hyperparameter is fixed in one place (the `EXPERIMENT SETUP` block of `main.py`, see
[Configuration](#configuration)), and the GPU only changes memory settings, so a run is the same experiment on
every machine.

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
│   ├── config.py          # Config types, memory-only hardware profiles, paths (no hyperparameter values)
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
│           ├── run_config.json     # the full resolved setup of the run (every hyperparameter)
│           └── inference_results.json, inference_metrics.json   # infer stage
├── main.py                # Run selection + EXPERIMENT SETUP (all hyperparameters) + pipeline stages
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
DATASET_MODE = "rxhand"       # or "combined" / "dhp"
RXHAND_HELDOUT_TEST = True    # "rxhand" only: train/val from Train_Set, official Test_Set for infer only
RXHAND_OFFICIAL_SPLIT = False # "rxhand" only: True = train on Train_Set, validate on Test_Set
AUGMENTATION_MODE = "offline" # or "online"; "offline" needs STAGE = "augment" once first

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
folder name. `VLM_AUGMENTED_DIR` moves the folder; `offline_include_mixed=False` in `AUGMENTATION` (`main.py`) drops
the `_mixed` variant.

**YOLO26 (`MODEL_CHOICE = "yolo26"`)** fine-tunes the YOLO26 classification backbones
(`yolo26{n,s,m,l,x}-cls.pt`, 3.6M-30.4M parameters) as a closed-vocabulary word recognizer: each distinct
training label is a class. Ultralytics has no text-generating YOLO26 variant (the open-vocabulary YOLOE-26 models
are box detectors), so classification is the only YOLO26 task that fits box-less word labels. Every size runs on
every profile with the same effective batch of 64 (`YOLO_TRAINING`); where a size does not fit in one pass the
profile splits it with gradient accumulation (laptop: 32 x 2, `yolo26x` 16 x 4; `pc_3070ti` `yolo26x`: 32 x 2).

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
QWEN_CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"   # any Qwen3-VL checkpoint; every size trains as 4-bit QLoRA
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
checkpoint, an `inference_metrics.json`. With no earlier run it simply starts a new one. The run must continue with
the setup it started with: if any value of the `EXPERIMENT SETUP` (or `--epochs`) differs from its `run_config.json`,
the run is refused and the differing values are listed. Runs trained before the current setup therefore cannot be
resumed; start them again. Checkpoints are written every `save_steps` (500) optimizer steps, so at most that much
work is lost.

All runs of one ablation share backbone, dataset, augmentation and seed (3407; the random initialisation of the new
modules is seeded too), so the rows of the table differ only in the architecture. `compare_ms_mgca.py` writes
`ablation_table.md/.csv` (exact match, normalized match, CER, WER, latency, peak VRAM, ECE, Brier, error-detection
AUROC, review rate), `metrics_comparison.png`, one `reliability_<run>.png` per confidence-aware run and
`risk_coverage_comparison.png`. Use `--dataset` / `--size` to compare only runs of one setting.

- **Training** fine-tunes LoRA and all enabled MS-MGCA components jointly, with the same QLoRA recipe as `qwen-vl`
  (see [Configuration](#configuration)). The randomly initialised modules get their own learning rate
  (`MSMGCA_MODULE_LR` / `MSMGCA_CONFIDENCE_LR`, 1.8e-4; LoRA 1.4e-4). With the confidence head the loss is
  `L_text + lambda_c * L_conf`; during training a sample counts as correct (z = 1) when the model's teacher-forced
  prediction reproduces every answer token, which is only a proxy on data the model is fitting (see calibration).
  Checkpoints are selected on validation CER.
- **Architecture settings** (the E1-E6 switches, crop grid, K, heads, confidence threshold, loss weight) come from
  `MSMGCA_ABLATIONS`, `MSMGCA_MULTISCALE`, `MSMGCA_CROSS_ATTENTION` and `MSMGCA_UNCERTAINTY` in `main.py`. The YAML
  files in `experiments/MS-MGCA-Qwen3-VL/configs/` are only read by that package's standalone scripts.
- **Calibration (`STAGE = "calibrate"`, E4/E6):** the confidence head must learn whether a *generated* transcription
  is right, which needs labels from images the model did not train on. The stage transcribes the validation images
  (never trained on), labels each transcription right/wrong (exact match), splits them once into a calibration part
  (`calibration_fraction` in `EVALUATION`, 50%) and a held-out part, fits the head on the calibration part
  with BCE, and reports ECE / Brier / AUROC on the held-out part before and after
  (`calibration_metrics.json`, `plots/reliability_*_calibration.png`). The result is saved as
  `checkpoints/calibrated/`, which the infer stage then uses, scoring confidence on the held-out part only.
  This is a one-fold hold-out version of out-of-fold calibration (a true K-fold would retrain the model K times).
- **Batch size:** the multi-scale / cross-attention path handles one image at a time, so MS-MGCA always uses
  micro-batch 1 with gradient accumulation (effective batch 8 on every profile, same as `qwen-vl`).
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

`main.py` has two blocks at the top:

- **RUN SELECTION**: what to run (`STAGE`, `MODEL_CHOICE`, `QWEN_CHECKPOINT`, `MSMGCA_EXPERIMENT`,
  `HARDWARE_PROFILE`, `DATASET_MODE`, ...). Each constant can also be given on the command line (`python main.py --help`).
- **EXPERIMENT SETUP**: every value that changes a result: training recipes, QLoRA / LoRA, the MS-MGCA architecture
  and ablation switches, split, evaluation, calibration and augmentation. Each value cites the paper it comes from.
  `configs/config.py` holds no values of its own: a setting missing from `main.py` is an error, never a silent
  default. The resolved setup of every run is written to `run_config.json`.

### Setup for Qwen-VL and MS-MGCA (all ablations, every backbone size)

| Setting | Value | Source |
|---|---|---|
| Quantization | 4-bit NF4 + double quantization, bf16 compute, vision tower in bf16 | QLoRA (Dettmers et al., NeurIPS 2023), App. B.2 |
| LoRA | r = 64, alpha = 16, dropout 0.1, on all 7 linear layers of the language model (q/k/v/o/gate/up/down) | QLoRA Sec. 4 and App. B.2; rank 64 also in Huttner et al. (handwriting) and the Manchu VLM-OCR paper (arXiv:2507.06761) |
| Learning rate | LoRA 1.4e-4, MS-MGCA modules 1.8e-4 | QLoRA 2e-4 at batch 16 and the LLaVA-1.5 connector 1e-3 at batch 256 (CVPR 2024), scaled to batch 8 with the square-root rule for Adam (Malladi et al., NeurIPS 2022) |
| Schedule | 10% linear warmup, then cosine decay | longer warmup lowers LR sensitivity (Wortsman et al., ICLR 2024); cosine as in LLaVA-1.5 |
| Optimizer | 8-bit AdamW, weight decay 0.01, gradient clipping 0.3 | Dettmers et al., ICLR 2022; clipping from QLoRA App. B.2 |
| Batch | effective 8 | |
| Length | at most 3 epochs, early stopping after 4 evaluations (2000 steps) without a better val CER | Prechelt 1998; with offline augmentation an epoch shows every original ~8 times |
| Selection | best validation CER (full val set every 500 steps) | val loss rose while CER still improved in earlier runs |
| Input | at most 512 x 512 px (the RxHand crops are 128 x 128; only DHP images are downscaled), 64 new tokens | |
| Seeds | training 3407, split 42 (keeps `configs/splits/rxhand/` valid) | |

CRNN (`CRNN_TRAINING`) keeps the recipe of its paper (lr 5e-4, batch 10, fp32, early stopping after 20 epochs), and
YOLO / YOLO26 (`YOLO_TRAINING`, `YOLO26_LEARNING_RATE`) use batch 64, AdamW + cosine and early stopping after 10
epochs.

### Hardware profiles (memory only)

Set `HARDWARE_PROFILE` in `main.py` to `auto`, `laptop_3050ti`, `pc_3070ti`, `pc_5070`, or `pc_5070ti`. A profile only
decides how the fixed effective batch is split into micro-batch x gradient accumulation, the data-loader workers and
gradient checkpointing; precision, quantization, resolution and every hyperparameter are the same on every machine.

| Profile | Qwen-VL micro-batch x accumulation | MS-MGCA | YOLO | Intended backbone |
|---|---:|---:|---:|---|
| `laptop_3050ti` | 1 x 8 | 1 x 8 | 32 x 2 | Qwen3-VL-2B |
| `pc_3070ti` | 1 x 8 | 1 x 8 | 64 x 1 | Qwen3-VL-2B / 4B |
| `pc_5070` | 2 x 4 | 1 x 8 | 64 x 1 | Qwen3-VL-8B |
| `pc_5070ti` | 4 x 2 | 1 x 8 | 64 x 1 | Qwen3-VL-8B |

All profiles need a GPU with bf16 (Ampere or newer). On CUDA OOM, lower `vlm_micro_batch` of the profile in
`configs/config.py` (the effective batch, and so the experiment, stays the same) or use a smaller backbone.

### Augmentation

`AUGMENTATION` in `main.py` holds the transform probabilities and limits (online mode) and the offline seed.
`AUGMENTATION_MODE = "offline"` is a separate workflow: run `STAGE = "augment"` first to create
`Augmented_Dataset/`, then run `train` or `infer`. Offline runs include `_offline` in their run name and
use fixed, reproducible variants; validation images are never augmented. After changing a value, delete
`Augmented_Dataset/` and run the augment stage again.

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

- Qwen backbones are 4-bit NF4; gradient checkpointing is on for every profile.
- LoRA r = 64 on all 7 linear layers adds ~175M trainable parameters on Qwen3-VL-8B (~70M on 2B). With 8-bit AdamW
  that is roughly 1.5-2 GB for the adapter weights, gradients and optimizer state on 8B.
- On Windows, a run that exceeds VRAM does not crash: it spills into shared system memory and trains several times
  slower (GPU utilisation drops). Watch the first steps in Task Manager / `nvidia-smi`.

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
- Saved every `save_steps` (500) optimizer steps for Qwen-VL / MS-MGCA, every epoch for CRNN / YOLO
- `checkpoints/final` holds the best checkpoint (lowest val CER), restored at the end of training
- MS-MGCA runs can be resumed with `--resume` (see above)

### Progress & Metrics
Training shows a progress bar with live `loss`, `tok_acc`, `lr`, `grad_norm` and the latest eval metrics.
Every `eval_steps` a table of all evaluation metrics is printed and logged to TensorBoard;
the final evaluation is saved to `<run>/eval_metrics.json`. The infer stage evaluates the
latest trained run of the model (`<run>/checkpoints/final`) on the held-out test set (the validation set when the
dataset mode has none) and writes `inference_results.json` (per sample) and `inference_metrics.json` into that
run's folder.

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

WER/CER/accuracy require `model.generate`, which is slow on a VLM, but checkpoint selection and early stopping use
val CER, so it is computed on the whole val set at every evaluation (`train_eval_samples=None` in `EVALUATION`).
`--samples N` limits the images scored by calibrate / infer.

## Training Tips

1. **First run**: watch GPU memory and `grad_norm` in the first steps. For a quick smoke test set
   `MAX_SAMPLES = 100` in `main.py` (do not report such runs).
2. **Validation**: check `runs/<model_name>/<run>/plots/` or TensorBoard. A healthy run's train loss keeps falling
   through the end of warmup (10% of the steps); a sudden jump to ~2 with val CER above 1 is divergence, not
   overfitting.
3. **Changing a hyperparameter**: edit it in the `EXPERIMENT SETUP` block, and use the same value for every run you
   compare. Runs with different setups are told apart by their `run_config.json`, not by the folder name.

## Troubleshooting

### CUDA Out of Memory
- Lower `vlm_micro_batch` / `vlm_eval_batch` of your profile in `configs/config.py` (the effective batch stays 8)
- Use a smaller backbone (`QWEN_CHECKPOINT`), e.g. Qwen3-VL-4B or 2B

### Slow Data Loading
- Increase `num_workers` cautiously (check RAM usage)
- Pre-process images to consistent size
- Use SSD/NVMe for dataset (avoid network drives)

### Missing Dataset Files
- Make sure `Dataset/` sits in the repo root, or set `VLM_DATASET_ROOT` (see "Dataset Location")
- Check `.trashed-*` files are being filtered
- Run with `MAX_SAMPLES = 10` in `main.py` to debug

## References

Setup sources (cited in the `EXPERIMENT SETUP` block of `main.py`):

- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023. https://arxiv.org/abs/2305.14314
- Malladi et al., "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms", NeurIPS 2022. https://arxiv.org/abs/2205.10287
- Liu et al., "Improved Baselines with Visual Instruction Tuning" (LLaVA-1.5), CVPR 2024. https://arxiv.org/abs/2310.03744
- Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities", ICLR 2024. https://arxiv.org/abs/2309.14322
- Huttner et al., "Low-Rank Adaptation vs. Fine-Tuning for Handwritten Text Recognition" (`paper/references/`)
- Chung & Choi, "Finetuning Vision-Language Models as OCR Systems for Low-Resource Languages: A Case Study of Manchu", arXiv:2507.06761. https://arxiv.org/abs/2507.06761
- Dettmers et al., "8-bit Optimizers via Block-wise Quantization", ICLR 2022. https://arxiv.org/abs/2110.02861
- Prechelt, "Early Stopping - But When?", Neural Networks: Tricks of the Trade, Springer 1998
- Achkar et al., "Medical Handwritten Prescription Recognition Using CRNN", 2019 (CRNN baseline)

Libraries:

- Qwen-VL: https://github.com/QwenLM/Qwen-VL
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- PEFT/LoRA: https://github.com/huggingface/peft
- Bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- Hugging Face Trainer: https://huggingface.co/docs/transformers/training

## License

Educational and research use only.
