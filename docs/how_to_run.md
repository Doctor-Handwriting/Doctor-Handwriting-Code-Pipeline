# Cara Menjalankan

## 1. Install

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Driver NVIDIA di bawah 570: pakai `cu126` sebagai ganti `cu128`.

## 2. Dataset

Dataset tidak ada di repo. Letakkan seperti ini, atau atur `VLM_DATASET_ROOT`:

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

### Aturan split (mulai sekarang, RxHand saja)

Pakai `--dataset rxhand --heldout-test`:

| Bagian | Isi | Dipakai di |
|---|---|---|
| train | 80% dari `Train_Set` | `train` |
| val | 20% dari `Train_Set` | `train` (pilih checkpoint), `calibrate` |
| test | `Test_Set` resmi, utuh | `infer` saja |

- Daftar gambar tiap bagian disimpan di `configs/splits/rxhand/` (`train.csv`, `val.csv`, `test.csv`) dan ikut di-commit. Semua PC membaca file yang sama, jadi split-nya identik.
- File itu dibuat otomatis saat pertama kali dijalankan (urut nama file, diacak seed 42). Setelah itu jangan dibuat ulang.
- Kalau ada gambar di file split yang tidak ditemukan, pipeline berhenti dengan pesan error. Artinya salinan RxHand di PC itu berbeda dengan milik tim.
- Nama folder run berakhiran `_rxhand-heldout`.

Run lama (sebelum aturan ini) memakai cara lain: Train_Set + Test_Set digabung lalu dibagi 80/20 (seed 42), tanpa test set. Hasilnya tidak bisa dibandingkan langsung dengan run `-heldout`.

`evaluate.py` di `experiments/MS-MGCA-Qwen3-VL/evaluation/` masih memakai split sendiri (seed 3407). Untuk angka paper, pakai `python main.py --stage infer ... --heldout-test`.

## 3. Menjalankan

Semua pengaturan bisa diubah di bagian atas `main.py` atau lewat argumen. Argumen menimpa konstanta di `main.py`.

| Argumen | Pilihan |
|---|---|
| `--stage` | `augment`, `train`, `calibrate`, `infer` |
| `--model` | `qwen-vl`, `crnn`, `yolo`, `yolo26`, `ms-mgca` |
| `--checkpoint` | checkpoint Qwen, mis. `Qwen/Qwen3-VL-2B-Instruct` |
| `--experiment` | MS-MGCA: `e1_baseline` ... `e6_full_model` |
| `--profile` | `auto`, `laptop_3050ti`, `pc_3070ti`, `pc_5070`, `pc_5070ti` |
| `--dataset` | `combined`, `dhp`, `rxhand` |
| `--heldout-test` | split standar tim: Train_Set → train/val 80/20, Test_Set → test |
| `--official-split` | train = Train_Set, val = Test_Set (tanpa test terpisah) |
| `--augmentation` | `online`, `offline` |
| `--epochs` | jumlah epoch |
| `--resume` | MS-MGCA: lanjutkan run yang terputus |
| `--samples` | batasi jumlah sampel untuk calibrate/infer |

Ukuran YOLO (`YOLO_CHECKPOINT`, `YOLO26_CHECKPOINT`) hanya bisa diubah di `main.py`.

### Augmentasi offline (sekali saja)
```bash
python main.py --stage augment --dataset rxhand --augmentation offline
```

### Qwen3-VL
```bash
python main.py --stage train --model qwen-vl --checkpoint Qwen/Qwen3-VL-2B-Instruct --dataset rxhand --heldout-test
python main.py --stage infer --model qwen-vl --checkpoint Qwen/Qwen3-VL-2B-Instruct --dataset rxhand --heldout-test
```

### MS-MGCA
```bash
python main.py --stage train     --model ms-mgca --experiment e6_full_model --checkpoint Qwen/Qwen3-VL-2B-Instruct --dataset rxhand
python main.py --stage calibrate --model ms-mgca --experiment e6_full_model --checkpoint Qwen/Qwen3-VL-2B-Instruct --dataset rxhand
python main.py --stage infer     --model ms-mgca --experiment e6_full_model --checkpoint Qwen/Qwen3-VL-2B-Instruct --dataset rxhand
```

Semua eksperimen E1–E6 sekaligus, lalu bandingkan:
```bash
python run_ms_mgca_ablations.py --checkpoint Qwen/Qwen3-VL-2B-Instruct --dataset rxhand --heldout-test --keep-going --resume
python compare_ms_mgca.py --dataset rxhand --split heldout
```

### CRNN
```bash
python main.py --stage train --model crnn --dataset rxhand
python main.py --stage infer --model crnn --dataset rxhand
```

### YOLO / YOLO26
```bash
python main.py --stage train --model yolo26 --dataset rxhand
python main.py --stage infer --model yolo26 --dataset rxhand
```

## 4. Output

Setiap run disimpan di `runs/<model>/run_<waktu>_<profil>[_dataset][_offline]/`:

```
checkpoints/          bobot (tidak di-commit)
logs/                 TensorBoard
plots/                grafik loss, akurasi, CER/WER
metrics_history.csv   semua nilai per step
metrics_summary.csv   best / rata-rata / terakhir
eval_metrics.json     evaluasi akhir
run_config.json       snapshot config
inference_metrics.json, inference_results.json   dari tahap infer
```

TensorBoard:
```bash
tensorboard --logdir runs
```

Setelah run selesai, catat hasilnya di [MODEL_STATUS.md](MODEL_STATUS.md). Untuk menjalankan lewat Jupyter/Colab/Kaggle, lihat [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md).
