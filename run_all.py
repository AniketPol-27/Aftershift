#!/usr/bin/env python3
"""Cross-platform driver for the full AFTERSHIFT battery (Windows/macOS/Linux).

Replaces scripts/run_scaleup.sh + scripts/package_results.sh when bash is not
available, and works from a FRESH clone (trains everything from scratch).

Usage:
    python run_all.py --stage validate    # 2-min correctness gate (run this FIRST)
    python run_all.py --stage all         # full battery, ~3-4 h, unattended
    python run_all.py --stage train       # individual stages (resume-safe)
    python run_all.py --stage package     # -> results_bundle_<timestamp>.zip

Stages: validate | train | sweep | detection | recovery | selfid | plots | package | all
Training is resume-safe: a model whose runs/<name>/meta.json shows it converged
(final eval >= 450) is skipped on re-runs.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

BUDGETS = "0,500,1000,2000,5000,10000,20000,40000,80000"
SHIFTS = "1.0,1.5,2.0,2.5,3.0"
SEEDS = "0,1,2,3,4,5,6,7,8,9"
SEEDS_SHORT = "0,1,2"


def sh(args, desc):
    print(f"\n{'=' * 60}\n=== {desc}\n$ {' '.join(args)}\n{'=' * 60}", flush=True)
    t0 = time.time()
    r = subprocess.run(args, cwd=ROOT)
    print(f"--- {desc}: exit {r.returncode} in {(time.time() - t0) / 60:.1f} min ---", flush=True)
    if r.returncode != 0:
        print(f"!! {desc} FAILED - capture this output. Continuing with other stages "
              f"(partial results are fine; the driver is resume-safe).", flush=True)
    return r.returncode


def meta_ok(name: str, threshold: float = 450.0) -> bool:
    p = os.path.join(ROOT, "runs", name, "meta.json")
    if not os.path.exists(p):
        return False
    try:
        m = json.load(open(p))
        fe = m.get("final_eval", {}) or {}
        return m.get("converged") is True or fe.get("mean_return", 0.0) >= threshold
    except Exception:
        return False


def stage_validate():
    return sh([PY, "-m", "aftershift.selfid", "--validate"], "VALIDATE (correctness gate)")


def stage_train():
    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    for s in range(5):
        if meta_ok(f"sac_s{s}"):
            print(f"[skip] sac_s{s} already converged"); continue
        sh([PY, "-m", "aftershift.train", "--algo", "sac", "--seed", str(s),
            "--steps", "50000", "--name", f"sac_s{s}"], f"train sac_s{s}")
        sh([PY, "-m", "aftershift.converge", "--run", f"runs/sac_s{s}",
            "--threshold", "450", "--max-extra", "75000"], f"converge sac_s{s}")
    for s in range(5):
        if meta_ok(f"ppo_s{s}"):
            print(f"[skip] ppo_s{s} already converged"); continue
        sh([PY, "-m", "aftershift.train", "--algo", "ppo", "--seed", str(s),
            "--steps", "200000", "--name", f"ppo_s{s}"], f"train ppo_s{s}")
        sh([PY, "-m", "aftershift.converge", "--run", f"runs/ppo_s{s}",
            "--threshold", "450", "--chunk", "100000", "--max-extra", "200000"],
           f"converge ppo_s{s}")
    for s in range(5):
        if meta_ok(f"sacdr_s{s}"):
            print(f"[skip] sacdr_s{s} already converged"); continue
        sh([PY, "-m", "aftershift.train", "--algo", "sac", "--seed", str(s),
            "--steps", "60000", "--dr", "--name", f"sacdr_s{s}"], f"train sacdr_s{s}")
        sh([PY, "-m", "aftershift.converge", "--run", f"runs/sacdr_s{s}",
            "--threshold", "450", "--max-extra", "50000"], f"converge sacdr_s{s}")
    if not meta_ok("sacoracle_pl2_s0"):
        sh([PY, "-m", "aftershift.train", "--algo", "sac", "--seed", "0", "--steps", "100000",
            "--context", '{"pole_len": 2.0}', "--name", "sacoracle_pl2_s0"], "train oracle")
        sh([PY, "-m", "aftershift.converge", "--run", "runs/sacoracle_pl2_s0",
            "--threshold", "450", "--max-extra", "150000"], "converge oracle")
    else:
        print("[skip] oracle already converged")
    return 0


ALL_MODELS = ([f"runs/sac_s{s}" for s in range(5)]
              + [f"runs/ppo_s{s}" for s in range(5)]
              + [f"runs/sacdr_s{s}" for s in range(5)])


def stage_sweep():
    return sh([PY, "-m", "aftershift.evaluate", "degradation",
               "--models", *ALL_MODELS,
               "--episodes", "20", "--out", "results/degradation_20ep.csv"],
              "DEGRADATION sweep (20 eps/point)")


def stage_detection():
    rc = 0
    for m in ("sac_s0", "sac_s1"):
        rc |= sh([PY, "-m", "aftershift.evaluate", "detection",
                  "--model", f"runs/{m}", "--episodes", "100", "--cal-episodes", "100",
                  "--t-shift", "100", "--out", f"results/detection_{m}.csv"],
                 f"DETECTION {m} (100 eps/shift)")
    return rc


def stage_recovery():
    rc = 0
    for s in SEEDS_SHORT.split(","):
        rc |= sh([PY, "-m", "aftershift.recovery", "--model", f"runs/sac_s{s}",
                  "--budgets", BUDGETS, "--shift", '{"pole_len": 2.0}',
                  "--episodes", "20", "--oracle", "runs/sacoracle_pl2_s0",
                  "--out", "results/recovery_20ep.csv",
                  "--summary", f"results/recovery_summary20_s{s}.json"],
                 f"RECOVERY sac_s{s} (budgets up to 80k)")
    return rc


def stage_selfid():
    rc = 0
    rc |= sh([PY, "-m", "aftershift.selfid", "--shifts", SHIFTS, "--seeds", SEEDS,
              "--out", "results/selfid_grid.csv"], "SELF-ID grid identifier (10 seeds)")
    rc |= sh([PY, "-m", "aftershift.selfid", "--shifts", SHIFTS, "--seeds", SEEDS,
              "--identifier", "ekf", "--out", "results/selfid_ekf.csv"],
             "SELF-ID EKF ablation")
    rc |= sh([PY, "-m", "aftershift.selfid", "--shifts", SHIFTS, "--seeds", SEEDS,
              "--no-dither", "--out", "results/selfid_nodither.csv"],
             "SELF-ID no-dither ablation")
    rc |= sh([PY, "-m", "aftershift.selfid", "--shifts", SHIFTS, "--seeds", SEEDS,
              "--oracle-detect", "--out", "results/selfid_oracledetect.csv"],
             "SELF-ID oracle-detection ablation")
    return rc


def stage_plots():
    return sh([PY, "-m", "aftershift.plots"], "PLOTS")


def stage_package():
    import zipfile
    out = os.path.join(ROOT, f"results_bundle_{datetime.datetime.now():%Y%m%d_%H%M}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        res = os.path.join(ROOT, "results")
        for dirpath, _, files in os.walk(res):
            for fn in files:
                full = os.path.join(dirpath, fn)
                z.write(full, os.path.relpath(full, ROOT))
        runsdir = os.path.join(ROOT, "runs")
        if os.path.isdir(runsdir):
            for d in sorted(os.listdir(runsdir)):
                mp = os.path.join(runsdir, d, "meta.json")
                if os.path.exists(mp):
                    z.write(mp, f"runs_meta/{d}.json")
        z.writestr("envinfo.txt",
                   f"{sys.version}\n{platform.platform()}\n"
                   f"driver: run_all.py\n")
    print(f"\nBundle written: {out}")
    print("Results bundle written. Checkpoints (runs/) are excluded on purpose.")
    return 0


STAGES = dict(validate=stage_validate, train=stage_train, sweep=stage_sweep,
              detection=stage_detection, recovery=stage_recovery, selfid=stage_selfid,
              plots=stage_plots, package=stage_package)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    t0 = time.time()
    if args.stage == "all":
        if stage_validate() != 0:
            sys.exit("VALIDATION FAILED - fix before running the battery "
                     "(send me the traceback).")
        for name in ("train", "sweep", "detection", "recovery", "selfid", "plots", "package"):
            STAGES[name]()
    else:
        STAGES[args.stage]()
    print(f"\nDone. Total: {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
