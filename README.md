# Vision-Language Model Training Pipeline

Production-ready multimodal VLM training pipeline optimized for **RTX 3050 Ti (4GB VRAM)** using Qwen-VL and lightweight YOLO models.

## Features

- **Memory-Optimized**: 4-bit quantization (NF4), LoRA adaptation, gradient checkpointing
- **Dataset Parsing**: Automated parsing of DHP + RxHand datasets with multimodal instruction format
- **Augmentation**: Medical handwriting-specific augmentations (motion blur, elastic deformation, perspective)
- **Training**: Full Hugging Face Trainer integration with mixed precision support
- **Inference**: Batch inference with WER/CER/accuracy metrics
- **Monitoring**: TensorBoard logging, memory profiling, checkpoint management

## Project Structure

```
vlm_pipeline/
├── configs/
│   ├── __init__.py
│   └── config.py          # Dataclass configs (hardware, LoRA, training, paths)
├── models/
│   ├── __init__.py
│   └── models.py          # Qwen-VL & YOLO loaders with 4-bit quantization
├── utils/
│   ├── __init__.py
│   ├── dataset_parser.py  # DHP + RxHand parsing to instruction format
│   ├── augmentor.py       # Medical handwriting augmentations
│   ├── metrics.py         # WER, CER, accuracy, mAP computation
│   └── visualization.py   # Loss/accuracy/CER/WER/LR training plots
├── Augmented_Dataset/     # AUGMENTATION_MODE = "offline" only (created by STAGE = "augment", git-ignored)
│   ├── DHP/    {original/, augmented/, augmentation_manifest.json}
│   └── RxHand/ {original/, augmented/, augmentation_manifest.json}
├── runs/                  # Outputs, one folder per model per run
│   ├── train_augmented.json, val_augmented.json   # augment stage
│   └── <model_name>/                               # e.g. Qwen3-VL-8B-Instruct
│       └── run_<YYYYmmdd-HHMMSS>_<profile>/
│           ├── checkpoints/        # checkpoint-*/ and final/ (LoRA adapter)
│           ├── logs/               # TensorBoard events
│           ├── plots/              # loss, accuracy, error_rates, learning_rate, training_overview,
│           │                       # evaluation_metrics (every metric), final_metrics, inference_metrics .png
│           ├── metrics_history.json / .csv   # every logged train/eval value
│           ├── metrics_summary.csv # best / average / last loss & accuracy
│           ├── eval_metrics.json   # final evaluation
│           ├── run_config.json     # full config snapshot of the run
│           └── inference_results.json, inference_metrics.json   # infer stage
├── main.py                # Pipeline orchestrator (augment/train/infer)
└── requirements.txt       # All dependencies
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac

# For CUDA support, install torch matching your GPU driver first, e.g.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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
- Splits into 85% train / 15% validation

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
MODEL_CHOICE = "qwen-vl"  # or "crnn", "yolo", "yolo26"
YOLO26_CHECKPOINT = "yolo26m-cls.pt"  # n / s / m / l / x, used when MODEL_CHOICE = "yolo26"
HARDWARE_PROFILE = "pc_5070ti"  # or "laptop_3050ti", "pc_5070", "auto"
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
all three profiles at the same effective batch (32 laptop / 64 `pc_5070` / 128 `pc_5070ti`); on the 4GB laptop
`yolo26x` uses 16 x 2 gradient accumulation, everything else fits in one pass.

**Inference Stage:**
```bash
STAGE = "infer"
python main.py
```

## Configuration

Edit `configs/config.py` to customize:

### Hardware (Fixed for 4GB VRAM)
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
r = 8, lora_alpha = 16, lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
```

### Augmentation
```python
enable_augmentation = True
motion_blur_prob = 0.3
elastic_deform_prob = 0.3
perspective_prob = 0.2
brightness_contrast_prob = 0.5
```

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
- **mAP** (for YOLO): Mean Average Precision @IoU=0.5

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
- PEFT/LoRA: https://github.com/huggingface/peft
- Bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- Hugging Face Trainer: https://huggingface.co/docs/transformers/training

## License

Educational and research use only.
