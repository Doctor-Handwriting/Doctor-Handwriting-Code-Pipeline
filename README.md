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
│   └── metrics.py         # WER, CER, accuracy, mAP computation
├── runs/                  # Outputs: logs, checkpoints, metrics
│   ├── logs/
│   └── checkpoints/
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
MODEL_CHOICE = "qwen-vl"  # or "yolo"

python main.py
```

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
tensorboard --logdir runs/logs
```

### Checkpoints
- Auto-saved every `save_steps` iterations
- Best model in `runs/checkpoints/final`
- Resume training from checkpoint (automatic)

### Progress & Metrics
Training shows a progress bar with live `loss`, `lr`, `grad_norm` and the latest eval metrics.
Every `eval_steps` a table of all evaluation metrics is printed and logged to TensorBoard;
the final evaluation is saved to `runs/eval_metrics.json`. The infer stage evaluates the
trained adapter (`runs/checkpoints/final`) on the validation set and writes
`runs/inference_results.json` (per sample) and `runs/inference_metrics.json`.

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

3. **Validation**: Check `runs/logs/events.out.tfevents.*` in TensorBoard

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
