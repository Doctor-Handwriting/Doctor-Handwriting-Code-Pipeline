# Status Model dan Hasil

Terakhir diperbarui: 28 Nov 2026. Sumber: folder `runs/` (`metrics_summary.csv`, `eval_metrics.json`, `trainer_state.json`).

Semua angka adalah nilai terbaik di validation set selama training. CER dan WER: makin kecil makin baik.

## Dataset

| Kode | Isi | Split |
|---|---|---|
| `combined` | DHP + RxHand digabung | 80% train / 20% val |
| `dhp` | DHP saja | 80 / 20 |
| `rxhand` | RxHand (RxHandBD) saja | 80 / 20 (split resmi belum dipakai) |

Dataset tidak disimpan di repo. Letakkan di `Dataset/DHP/` dan `Dataset/RxHand/`, atau atur `VLM_DATASET_ROOT`.

## Sudah dijalankan

### Qwen3-VL (LoRA)

| Run | Model | Dataset | Aug | Epoch | Akurasi | Akurasi norm. | CER | WER | Eval |
|---|---|---|---|---|---|---|---|---|---|
| `run_20261126-115345_pc_3070ti_rxhand-heldout_offline` | Qwen3-VL-4B | rxhand held-out | offline | 2.8 | 66.5% | 72.0% | 0.122 | 0.328 | full val |
| ↳ test set (1115 gambar) | Qwen3-VL-4B | rxhand held-out | offline | | **40.8%** | **48.2%** | **0.307** | 0.631 | Test_Set |
| `run_20261007-160021_pc_5070` | Qwen3-VL-8B | combined | online | 10 | 64% | 73% | 0.121 | 0.360 | 100 sampel |
| `run_20261003-062700_pc_5070` | Qwen3-VL-2B | combined | online | 25 | 66% | 71% | 0.133 | 0.316 | 100 sampel |
| `run_20261004-111109_pc_5070` | Qwen3-VL-2B | combined | online | 50 | 66% | 71% | 0.131 | 0.333 | 100 sampel |
| `run_20261005-190331_pc_5070` | Qwen3-VL-2B | combined | online | 15 | 63% | 70% | 0.154 | 0.360 | 100 sampel |
| `run_20261002-214530_pc_5070` | Qwen3-VL-2B | combined | online | 10 | 61% | 71% | 0.135 | 0.377 | 100 sampel |
| `runs/checkpoints/` (tanpa nama run) | Qwen3-VL-2B | combined | online | 3 | 59% | 65% | 0.159 | 0.395 | 100 sampel |
| `run_20261004-100819_pc_5070` | Qwen3-VL-2B | combined | online | 3 | 56% | 64% | 0.170 | 0.412 | 100 sampel |

### MS-MGCA-Qwen3-VL

| Run | Eksperimen | Backbone | Dataset | Aug | Akurasi | Akurasi norm. | CER | WER | Eval |
|---|---|---|---|---|---|---|---|---|---|
| `run_20261112-104833_pc_5070_e6_full_model_8B_rxhand_offline` | E6 full model | 8B | rxhand | offline | 71% | 75% | 0.138 | 0.303 | 100 sampel |
| `run_20261111-231207_pc_5070_e1_baseline_2B_rxhand_offline` | E1 baseline | 2B | rxhand | offline | 59% | 70% | 0.152 | 0.404 | 100 sampel |

Catatan:
- E6 selesai 10 epoch (commit `9e35fd7`, 22 Nov). Evaluasi final di `eval_metrics.json` memakai checkpoint dengan val loss terendah: akurasi 58%, akurasi norm. 68%, CER 0.177, WER 0.422. Angka 71% di tabel adalah puncak selama training, di checkpoint lain.
- E6: val loss naik dari 0.63 ke 1.15 di akhir training (overfitting).
- `experiments/MS-MGCA-Qwen3-VL/results/E1_baseline/val/eval_summary.json` (50 sampel) memberi akurasi 16% dan CER 0.471, jauh di bawah angka training. Perlu dicek checkpoint mana yang dipakai.

### YOLO (klasifikasi kata, 1.600 kelas)

| Run | Model | Dataset | Aug | Epoch | Akurasi | Akurasi norm. | CER | WER | Eval |
|---|---|---|---|---|---|---|---|---|---|
| `run_20261021-054427_pc_5070_offline` | YOLO26x | combined | offline | 29 | 61.5% | 65.8% | 0.236 | 0.409 | full val |
| `run_20261020-210121_pc_5070_offline` | YOLO26m | combined | offline | 14 | 59.9% | 64.3% | 0.252 | 0.429 | full val |
| `run_20261020-213734_pc_5070_offline` | YOLO26l | combined | offline | 21 | 59.7% | 63.2% | 0.256 | 0.426 | full val |
| `run_20261020-190128_pc_5070_offline` | YOLO26s | combined | offline | 14 | 58.9% | 63.1% | 0.268 | 0.440 | full val |
| `run_20261014-095445_pc_5070` | YOLOv8m | combined | online | 33 | 58.7% | 63.4% | 0.265 | 0.442 | full val |
| `run_20261020-192818_pc_5070_offline` | YOLO26n | combined | offline | 18 | 57.8% | 62.4% | 0.271 | 0.444 | full val |

YOLO hanya bisa menebak kata yang ada di label training, jadi akurasinya punya batas atas (~79% pada split ini).

### CRNN

| Run | Dataset | Aug | Akurasi | CER | WER | Eval |
|---|---|---|---|---|---|---|
| `run_20261110-220305_pc_5070_rxhand_offline` | rxhand | offline | 37.9% | 0.356 | 0.640 | full val |
| `run_20261110-200934_pc_5070_dhp_offline` | dhp | offline | 0% | 0.636 | 0.719 | full val |

## Dimulai, tidak selesai

| Run | Keterangan |
|---|---|
| CRNN `run_20261110-104639_pc_5070_dhp_offline` | Berhenti di awal (CER 0.93) |
| MS-MGCA E6 2B `run_20261118-155311_laptop_3050ti_rxhand_offline` | Hanya `run_config.json` |
| MS-MGCA E6 8B `run_20261118-154716_laptop_3050ti_rxhand_offline` | Hanya `run_config.json`, 8B tidak muat di 4GB |

## Belum dijalankan

### Sudah didukung `main.py`

| Model | Perintah |
|---|---|
| Qwen3-VL-4B | `python main.py --model qwen-vl --checkpoint Qwen/Qwen3-VL-4B-Instruct` |
| MS-MGCA E2 multiscale | `python main.py --model ms-mgca --experiment e2_multiscale` |
| MS-MGCA E3 cross-attention | `python main.py --model ms-mgca --experiment e3_cross_attention` |
| MS-MGCA E4 confidence | `python main.py --model ms-mgca --experiment e4_confidence` |
| MS-MGCA E5 multiscale + cross-attn | `python main.py --model ms-mgca --experiment e5_multiscale_crossattn` |

### Perlu loader baru di `models/models.py`

| Model | Pembuat | Ukuran yang realistis (8–16GB) |
|---|---|---|
| Llama 3.2 Vision | Meta | 11B (QLoRA, butuh 12GB+) |
| PaliGemma / Gemma 3 / Gemma 4 | Google DeepMind | 3B–12B |
| LLaVA / LLaVA-NeXT | Komunitas | 7B |
| Pixtral | Mistral AI | 12B (QLoRA, butuh 12GB+) |
| Phi-4 Multimodal | Microsoft | ~5.6B |
| InternVL | OpenGVLab | 1B–8B |
| DeepSeek Janus | DeepSeek | 1B–7B |
| CogVLM | THUDM | 17B (butuh 16GB) |
| Moondream2 | vikhyatk | ~2B |

## Catatan perbandingan

Angka di atas belum setara antar model:
- Dataset berbeda: Qwen/YOLO `combined`, MS-MGCA `rxhand`, CRNN `dhp`/`rxhand`.
- Jumlah sampel eval berbeda: Qwen/MS-MGCA 100 sampel, YOLO/CRNN seluruh val set.

Untuk paper, latih ulang semua model dengan dataset dan split yang sama (`--dataset rxhand --heldout-test`, lihat aturan split di [how_to_run.md](how_to_run.md)). Cukup mengulang `infer` saja tidak bisa: model lama pernah melihat gambar `Test_Set` saat training.

## Catatan per model

### Qwen3-VL-2B-Instruct
- Terbaik di 25 epoch (66%). Run 50 epoch tidak lebih baik.
- `runs/checkpoints/` berisi satu run 2B tanpa nama folder.
- Rencana: latih ulang dengan `--heldout-test`.

### Qwen3-VL-4B-Instruct
- Run pertama dengan `--heldout-test` (pc_3070ti, offline aug). Early stopping di step 10000, checkpoint terbaik step 8000.
- Val 66.5% vs test 40.8%: 59% gambar test berisi kata yang ada di train (akurasi norm. 68%), 41% kata baru (20%). Val acak dari Train_Set berbagi kosakata dengan train, jadi angka val terlalu optimis.
- Val loss naik sejak step 2500 (0.61 ke 0.90) walau CER terus turun.
- Zero-shot (tanpa LoRA) di test: akurasi 19.7%, CER 0.50.

### Qwen3-VL-8B-Instruct
- CER terendah di antara run Qwen (0.121). Butuh GPU 12GB+ (4-bit QLoRA), tidak disarankan di GPU 8GB.
- Rencana: latih ulang dengan `--heldout-test`.

### MS-MGCA-Qwen3-VL
- Kode model ada di `experiments/MS-MGCA-Qwen3-VL/`, dijalankan lewat `main.py --model ms-mgca`.
- E6 8B overfitting; evaluasi final (58%) di bawah puncak training (71%).
- Hasil `evaluate.py` E1 (16%) memakai split lain (seed 3407), jadi tidak sebanding.
- Rencana: jalankan E1–E6 dengan `--heldout-test` (`run_ms_mgca_ablations.py --heldout-test`).

### YOLO26 (n/s/m/l/x) dan YOLOv8m
- Terbaik YOLO26x (61.5%). Ukuran lain berbeda tipis (57.8–59.9%).
- Hanya bisa menebak kata yang ada di label training, jadi akurasinya punya batas atas.
- Ukuran model diatur lewat `YOLO26_CHECKPOINT` di `main.py`, belum ada argumen CLI.
- Rencana: latih ulang YOLO26x dengan `--heldout-test`.

### CRNN
- Gagal di DHP (0%); dugaan: data terlalu sedikit untuk melatih dari nol. RxHand 37.9%.
- Run DHP `run_20261110-104639` berhenti di awal, tidak dipakai.
- Rencana: latih ulang di RxHand dengan `--heldout-test` sebagai baseline paper.

## Cara memperbarui file ini

Panduan menjalankan training ada di [how_to_run.md](how_to_run.md) dan [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md).

Setelah run selesai:
1. Commit file kecil di `runs/<model>/<run>/` (CSV, JSON, plot). Bobot model otomatis diabaikan git.
2. Tambahkan satu baris di tabel yang sesuai, dan pindahkan model dari "Belum dijalankan" bila perlu.
3. Tulis alasan, masalah, atau temuan di "Catatan per model".
4. Plot yang dipakai di paper disalin ke `paper/manuscript/fig/`.
