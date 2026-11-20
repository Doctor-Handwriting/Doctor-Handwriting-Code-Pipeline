"""Run the MS-MGCA-Qwen3-VL ablation (E1-E6, optionally the K sweep) end to end, one `python main.py` process per
stage so the GPU is released between runs. Every run uses the same backbone, dataset, augmentation and seed.

    python run_ms_mgca_ablations.py                          # E1..E6: train -> calibrate (E4/E6) -> infer
    python run_ms_mgca_ablations.py --experiments e1_baseline e6_full_model
    python run_ms_mgca_ablations.py --sweep-k 1 4 8          # E6 with K = 1, 4, 8 (override --sweep-experiment)
    python run_ms_mgca_ablations.py --checkpoint Qwen/Qwen3-VL-4B-Instruct --dataset rxhand --augmentation online
    python run_ms_mgca_ablations.py --dry-run                # print the commands only
    python run_ms_mgca_ablations.py --resume --epochs 10     # continue after an interruption (same settings as before)

Afterwards: python compare_ms_mgca.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ALL_EXPERIMENTS = ["e1_baseline", "e2_multiscale", "e3_cross_attention", "e4_confidence",
                   "e5_multiscale_crossattn", "e6_full_model"]
HAS_CONFIDENCE = {"e4_confidence", "e6_full_model"}
HAS_CROSS_ATTENTION = {"e3_cross_attention", "e5_multiscale_crossattn", "e6_full_model"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", choices=ALL_EXPERIMENTS, default=ALL_EXPERIMENTS)
    p.add_argument("--sweep-k", nargs="+", type=int, metavar="K",
                   help="instead of --experiments, run --sweep-experiment once per K (e.g. 1 4 8)")
    p.add_argument("--sweep-experiment", choices=sorted(HAS_CROSS_ATTENTION), default="e6_full_model")
    p.add_argument("--stages", nargs="+", choices=["train", "calibrate", "infer"], default=["train", "calibrate", "infer"])
    p.add_argument("--checkpoint", default="Qwen/Qwen3-VL-2B-Instruct",
                   help="Qwen3-VL backbone, the same for every run (default: %(default)s)")
    p.add_argument("--dataset", choices=["combined", "dhp", "rxhand"])
    p.add_argument("--augmentation", choices=["online", "offline"])
    p.add_argument("--profile", help="hardware profile (default: HARDWARE_PROFILE in main.py)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--resume", action="store_true",
                   help="pick up where an earlier invocation stopped: an interrupted training continues from its "
                        "latest checkpoint, finished stages are skipped (use the same --epochs as the original run)")
    p.add_argument("--keep-going", action="store_true", help="continue with the next run when one fails")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = [(args.sweep_experiment, k) for k in args.sweep_k] if args.sweep_k else [(e, None) for e in args.experiments]
    common = ["--model", "ms-mgca"]
    for flag, value in (("--checkpoint", args.checkpoint), ("--dataset", args.dataset),
                        ("--augmentation", args.augmentation), ("--profile", args.profile), ("--epochs", args.epochs)):
        if value is not None:
            common += [flag, str(value)]
    if args.resume:
        common.append("--resume")

    failures = []
    for experiment, k in jobs:
        for stage in args.stages:
            if stage == "calibrate" and experiment not in HAS_CONFIDENCE:
                continue  # no confidence head to fit
            cmd = [sys.executable, "main.py", *common, "--experiment", experiment, "--stage", stage]
            if k is not None:
                cmd += ["--queries", str(k)]
            print("\n>>> " + " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            if subprocess.run(cmd, cwd=ROOT).returncode != 0:
                failures.append(f"{experiment}{'' if k is None else f' K={k}'} [{stage}]")
                print(f"!!! {failures[-1]} failed", flush=True)
                if not args.keep_going:
                    print("Stopping (use --keep-going to continue with the remaining runs).")
                    return 1
                break  # skip this experiment's later stages: they need the failed one

    if failures:
        print("\nFailed: " + ", ".join(failures))
        return 1
    print("\nAll runs finished. Build the comparison with: python compare_ms_mgca.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
