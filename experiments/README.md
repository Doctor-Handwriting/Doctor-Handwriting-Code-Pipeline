# Experiments — progress per model

One folder per model, named exactly like the pipeline's `cfg.model_name`, so it matches
`runs/<model_name>/` (which is git-ignored because it holds checkpoints).

```
experiments/
├── Qwen3-VL-2B-Instruct/
├── Qwen3-VL-4B-Instruct/
├── Qwen3-VL-8B-Instruct/
├── CRNN/
└── yolo26m-cls/          # add yolo26n-cls/, yolo26s-cls/, ... if other sizes are trained
    ├── README.md         # progress table: one row per run
    └── run_<YYYYmmdd-HHMMSS>_<profile>/   # copied from runs/<model>/<run>/
        ├── run_config.json
        ├── eval_metrics.json / inference_metrics.json
        └── plots/*.png
```

## After every finished run
1. Create `experiments/<model_name>/<run_name>/` (same run name as in `runs/`).
2. Copy **only** the small files: `run_config.json`, `*_metrics.json`, `plots/`.
   Never copy `checkpoints/` (too large for git).
3. Add one row to that model's `README.md` table (who, date, dataset, metrics, status).
4. Plots used in the paper go to `paper/manuscript/figures/`.
