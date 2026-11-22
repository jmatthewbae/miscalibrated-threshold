"""Run every stage in order. Output is also logged to outputs/run_log.txt.

    python run_all.py
    python run_all.py --epochs 2    # smoke test
"""

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime

import config as C

PY = sys.executable
LOG = C.OUT / "run_log.txt"


def log(msg: str = "") -> None:
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def step(title: str, args: list[str]) -> None:
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)

    t0 = time.time()
    proc = subprocess.Popen(
        [PY, "-u", *args], cwd=C.ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        log(line.rstrip())
    proc.wait()

    if proc.returncode != 0:
        log(f"\nFAILED: {' '.join(args)} (exit {proc.returncode})")
        sys.exit(proc.returncode)
    log(f"({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cv", action="store_true",
                    help="Reuse existing Run A fold predictions.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs.")
    args = ap.parse_args()

    extra = ["--epochs", str(args.epochs)] if args.epochs else []

    log("\n" + "#" * 78)
    log(f"# Miscalibrated Threshold experiment")
    log(f"# started {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"# {platform.platform()}  python {platform.python_version()}")
    log("#" * 78)

    t_start = time.time()

    step("1. Build splits", ["build_splits.py"])

    if not args.skip_cv:
        for fold in range(C.N_FOLDS):
            step(f"2.{fold + 1} Run A cross-validation fold {fold} "
                 f"(threshold fitting only)",
                 ["train.py", "--run", "run_a", "--fold", str(fold), *extra])
    else:
        log("\nSkipping cross-validation as requested.")

    step("3.1 Train final Run A (skewed, FST I-II only)",
         ["train.py", "--run", "run_a", *extra])
    step("3.2 Train final Run B (balanced across bands)",
         ["train.py", "--run", "run_b", *extra])

    step("4. Evaluate both runs on held-out DDI", ["evaluate.py"])
    step("5. Figures", ["figures.py"])

    log("\n" + "#" * 78)
    log(f"# Done in {(time.time() - t_start) / 60:.1f} minutes")
    log(f"# Results  {C.RESULT_DIR}")
    log(f"# Figures  {C.FIG_DIR}")
    log(f"# Log      {LOG}")
    log("#" * 78)


if __name__ == "__main__":
    main()
