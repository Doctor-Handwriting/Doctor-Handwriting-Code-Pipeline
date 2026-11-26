# Panduan Membuat Notebook Training

Notebook hanya menjadi "pembungkus" untuk `main.py`. Semua logika training tetap di repo, jadi hasil dari notebook dan terminal sama persis.

Buat notebook di root repo (sejajar dengan `main.py`), satu file per keluarga model, tanpa membuat folder baru:

```
00_setup_and_smoke_test.ipynb
01_qwen_vl.ipynb
02_ms_mgca.ipynb
03_crnn.ipynb
04_yolo.ipynb
```

Sebelum commit, kosongkan output cell (`Kernel > Restart & Clear Output`) supaya diff tetap kecil.

## 1. Hardware profile

Profil hanya mengatur memori: pembagian effective batch menjadi micro-batch × gradient accumulation, jumlah worker, dan gradient checkpointing. Presisi (bf16), kuantisasi (NF4 untuk semua ukuran Qwen), resolusi gambar (maks 512×512) dan semua hyperparameter sama di setiap mesin, dan diatur di blok `EXPERIMENT SETUP` di `main.py`.

| Profil | Mesin | GPU | Qwen-VL | MS-MGCA | YOLO |
|---|---|---|---|---|---|
| `laptop_3050ti` | Ryzen 7, 16GB RAM | RTX 3050 Ti 4GB | 1 × 8 | 1 × 8 | 32 × 2 |
| `pc_3070ti` | Ryzen 5 3600, 32GB RAM | RTX 3070 Ti 8GB | 1 × 8 | 1 × 8 | 64 × 1 |
| `pc_5070` | Ryzen 5 7500F, 32GB RAM | RTX 5070 12GB | 2 × 4 | 1 × 8 | 64 × 1 |
| `pc_5070ti` | Ryzen 5 7500F, 32GB RAM | RTX 5070 Ti 16GB | 4 × 2 | 1 × 8 | 64 × 1 |

Effective batch: 8 untuk Qwen-VL / MS-MGCA, 64 untuk YOLO, 10 untuk CRNN, di semua profil.

Profil didefinisikan di `configs/config.py` (`HARDWARE_PROFILES`). Dengan `--profile auto`, profil dipilih dari VRAM GPU: ≥14GB `pc_5070ti`, ≥10GB `pc_5070`, ≥7GB `pc_3070ti`, selain itu `laptop_3050ti`.

Model yang realistis di 8GB: Qwen3-VL-2B/4B, MS-MGCA dengan backbone 2B, CRNN, semua YOLO. Qwen3-VL-8B tidak disarankan.

## 2. Kernel Jupyter lokal (sekali per PC)

Jalankan di terminal dari root repo. Folder `venv/` sudah diabaikan git.

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt ipykernel
python -m ipykernel install --user --name doctor-hw --display-name "Python (doctor-hw)"
```

Lalu pilih kernel **Python (doctor-hw)** di Jupyter atau VS Code. Sudah dicoba dengan Python 3.13 dan torch 2.11 + CUDA 12.8 di `pc_3070ti`. Di Colab/Kaggle langkah ini tidak perlu; pakai Cell 3 di bawah.

## 3. Notebook setup (`00_setup_and_smoke_test.ipynb`)

### Cell 1: cek GPU
```python
!nvidia-smi
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### Cell 2: pindah ke root repo
```python
import os
from pathlib import Path

REPO = Path.cwd()
assert (REPO / "main.py").exists(), "Buka notebook dari root repo"
print(REPO)
```

Di Colab/Kaggle, clone dulu:
```python
!git clone https://github.com/Doctor-Handwriting/Doctor-Handwriting-Code-Pipeline.git
%cd Doctor-Handwriting-Code-Pipeline
```

### Cell 3: install dependensi
```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
!pip install -r requirements.txt
```

### Cell 4: lokasi dataset
Dataset tidak ada di repo. Atur lokasinya lewat environment variable:
```python
os.environ["VLM_DATASET_ROOT"] = r"D:\Medical Handwritten\Dataset"
# Colab:  "/content/drive/MyDrive/Dataset"
# Kaggle: "/kaggle/input/<nama-dataset>"
```

Struktur yang diharapkan:
```
Dataset/
├── DHP/
│   ├── doctor_handwriting_labels.csv
│   └── img/img/
└── RxHand/
    ├── Train_Label.csv
    ├── Test_Labels.csv
    ├── Train_Set/
    └── Test_Set/
```

### Cell 5: cek dataset
```python
import main
cfg = main.build_config(main.parse_args(["--dataset", "combined", "--stage", "augment"]))
missing = cfg.dataset.missing_paths()
print("OK" if not missing else missing)
```

### Cell 6: smoke test
Jalankan CRNN 1 epoch untuk memastikan pipeline jalan:
```python
!python main.py --stage train --model crnn --dataset rxhand --heldout-test --augmentation online --profile auto --epochs 1
```

## 4. Notebook per model

Setiap notebook mengikuti urutan yang sama: atur parameter, (opsional) augment, train, infer, lihat hasil.

### Cell parameter
```python
PROFILE      = "auto"         # atau laptop_3050ti | pc_3070ti | pc_5070 | pc_5070ti
DATASET      = "rxhand"      # combined | dhp | rxhand
SPLIT        = "--heldout-test"     # split standar tim (hanya untuk DATASET = "rxhand")
AUGMENTATION = "online"      # online | offline
```

Jumlah epoch tidak perlu diberikan: setiap model memakai resep di `EXPERIMENT SETUP` (`main.py`), misalnya Qwen-VL / MS-MGCA maks 3 epoch dengan early stopping. `--epochs N` mengganti nilai itu, tetapi run yang dibandingkan di paper harus memakai nilai yang sama.

### Qwen3-VL (`01_qwen_vl.ipynb`)
```python
CKPT = "Qwen/Qwen3-VL-2B-Instruct"   # atau Qwen3-VL-4B-Instruct
!python main.py --stage train --model qwen-vl --checkpoint {CKPT} --profile {PROFILE} --dataset {DATASET} {SPLIT} --augmentation {AUGMENTATION}
!python main.py --stage infer --model qwen-vl --checkpoint {CKPT} --profile {PROFILE} --dataset {DATASET} {SPLIT} --augmentation {AUGMENTATION}
```

### MS-MGCA (`02_ms_mgca.ipynb`)
Satu eksperimen:
```python
EXP = "e2_multiscale"   # e1_baseline ... e6_full_model
CKPT = "Qwen/Qwen3-VL-2B-Instruct"
!python main.py --stage train --model ms-mgca --experiment {EXP} --checkpoint {CKPT} --profile {PROFILE} --dataset {DATASET} {SPLIT} --augmentation {AUGMENTATION}
!python main.py --stage calibrate --model ms-mgca --experiment {EXP} --checkpoint {CKPT} --profile {PROFILE} --dataset {DATASET} {SPLIT} --augmentation {AUGMENTATION}
!python main.py --stage infer --model ms-mgca --experiment {EXP} --checkpoint {CKPT} --profile {PROFILE} --dataset {DATASET} {SPLIT} --augmentation {AUGMENTATION}
```
`calibrate` hanya berpengaruh untuk E4 dan E6.

Semua eksperimen sekaligus:
```python
!python run_ms_mgca_ablations.py --checkpoint {CKPT} --profile {PROFILE} --dataset {DATASET} --augmentation {AUGMENTATION} --keep-going --resume
```

### CRNN (`03_crnn.ipynb`)
```python
!python main.py --stage train --model crnn --profile {PROFILE} --dataset {DATASET} {SPLIT} --augmentation {AUGMENTATION}
!python main.py --stage infer --model crnn --profile {PROFILE} --dataset {DATASET} {SPLIT} --augmentation {AUGMENTATION}
```
CRNN memakai resep paper (300 epoch maks, early stopping 20 epoch), jadi `--epochs` tidak perlu.

### YOLO (`04_yolo.ipynb`)
```python
!python main.py --stage train --model yolo26 --profile {PROFILE} --dataset {DATASET} {SPLIT} --augmentation {AUGMENTATION}
!python main.py --stage infer --model yolo26 --profile {PROFILE} --dataset {DATASET} {SPLIT} --augmentation {AUGMENTATION}
```
Ukuran model (n/s/m/l/x) diatur lewat `YOLO26_CHECKPOINT` di `main.py`. Belum ada argumen CLI untuk ini, jadi ubah konstanta itu sebelum menjalankan ukuran berikutnya.

### Augmentasi offline
Kalau `AUGMENTATION = "offline"`, jalankan sekali sebelum training:
```python
!python main.py --stage augment --dataset {DATASET} --augmentation offline
```

## 5. Melihat hasil

```python
import json, pandas as pd
from pathlib import Path

run = sorted(Path("runs/Qwen3-VL-2B-Instruct").glob("run_*"))[-1]
print(run.name)
display(pd.read_csv(run / "metrics_summary.csv"))
print(json.load(open(run / "inference_metrics.json")))
```

Plot:
```python
from IPython.display import Image, display
for png in sorted((run / "plots").glob("*.png")):
    display(Image(str(png)))
```

TensorBoard:
```python
%load_ext tensorboard
%tensorboard --logdir runs
```

## 6. Setelah run selesai

1. Commit file kecil di `runs/<model>/<run>/` (CSV, JSON, plot). Bobot model otomatis diabaikan git.
2. Tambahkan satu baris dan catatan penting di [MODEL_STATUS.md](MODEL_STATUS.md).
3. Commit dengan pesan yang jelas, misalnya `Qwen3-VL-4B rxhand-heldout test: acc 0.68, CER 0.12`.

## 7. Yang tidak boleh di-commit

- `Dataset/`, `Augmented_Dataset/`
- Bobot model (`*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `*.ckpt`) dan `checkpoints/` berukuran besar. File kecil di `runs/` (CSV, JSON, plot) memang di-commit tim.
- Output cell notebook yang berisi gambar besar atau path pribadi
