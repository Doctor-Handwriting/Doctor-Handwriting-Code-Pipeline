# Experiments

Folder ini berisi kode model eksperimen, bukan hasil training.

```
experiments/
└── MS-MGCA-Qwen3-VL/   # model usulan: arsitektur, config E1–E6, dataset, training, test
```

`MS-MGCA-Qwen3-VL/` dipakai langsung oleh `models/ms_mgca.py` dan `main.py --model ms-mgca`. Jangan dipindah atau diganti namanya.

| Yang dicari | Lokasi |
|---|---|
| Hasil lengkap satu run (metrik, plot, config) | `runs/<model>/<run>/` |
| Ringkasan semua model, hasil, dataset, dan catatan | [docs/MODEL_STATUS.md](../docs/MODEL_STATUS.md) |
| Cara menjalankan | [docs/how_to_run.md](../docs/how_to_run.md) |
