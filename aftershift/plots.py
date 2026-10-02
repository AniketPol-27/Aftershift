"""Figures and summary tables for the AFTERSHIFT pilot."""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DR_ABS = {"masscart": (0.5, 2.0), "force_mag": (5.0, 15.0), "pole_len": (0.25, 1.0),
          "cart_friction": (0.0, 1.0)}
NOMINAL = {"masscart": 1.0, "force_mag": 10.0, "pole_len": 0.5, "cart_friction": 0.0}
PARAMS = ["masscart", "force_mag", "pole_len", "cart_friction"]


def family(model: str) -> str:
    return model.rsplit("_", 1)[0]


def plot_degradation(csv_path: str, out_png: str):
    df = pd.read_csv(csv_path)
    df["family"] = df["model"].map(family)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, p in zip(axes.flat, PARAMS):
        sub = df[df["param"] == p]
        for fam, g in sub.groupby("family"):
            m = g.groupby("value")["norm_return"].agg(["mean", "std", "count"])
            ax.plot(m.index, m["mean"], marker="o", label=f"{fam} (n={int(m['count'].max())})")
            ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.15)
        lo, hi = DR_ABS[p]
        if hi > lo:
            ax.axvspan(lo, hi, alpha=0.08, color="gray")
        ax.axvline(NOMINAL[p], color="k", ls=":", lw=1)
        ax.set_title(p)
        ax.set_xlabel(f"{p} (nominal={NOMINAL[p]}, gray = DR range)")
        ax.set_ylabel("normalized return")
        ax.set_ylim(-0.05, 1.05)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("AFTERSHIFT pilot v0.1 — degradation under post-training context shift "
                 "(mean ± std across seeds; 5 eps/point)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"[plots] wrote {out_png}")


def plot_recovery(csv_path: str, out_png: str):
    df = pd.read_csv(csv_path)
    oracle = df[df["model"].str.startswith("oracle")]
    ft = df[~df["model"].str.startswith("oracle")]
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, g in ft.groupby("model"):
        g = g.sort_values("budget")
        ax.plot(g["budget"], g["norm_return"], marker="o", label=m)
    if len(oracle):
        ax.axhline(oracle["norm_return"].iloc[0], color="k", ls="--",
                   label=f"retrained oracle ({oracle['model'].iloc[0]})")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("post-shift fine-tuning interactions")
    ax.set_ylabel("normalized return")
    ax.set_title("Budget-matched recovery after pole_len 0.5 -> 2.0 (SAC fine-tuning)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"[plots] wrote {out_png}")


def detection_table(df, out_md: str):
    lines = ["| model | shift | detect rate | fail rate | pre-failure detect | median lead (steps) |",
             "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        model = r.get("model", "model") if "model" in getattr(df, "columns", []) else "model"
        lines.append(
            f"| {model} | {r['param']} -> {r['shift_to']} | {r['detect_rate']:.2f} "
            f"| {r['fail_rate']:.2f} "
            f"| {r['prefail_detect_rate'] if pd.notna(r['prefail_detect_rate']) else 'n/a'} "
            f"| {r['median_lead'] if pd.notna(r['median_lead']) else 'n/a'} |")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[plots] wrote {out_md}")


def main():
    os.makedirs("results/figures", exist_ok=True)
    deg = "results/degradation_20ep.csv" if os.path.exists("results/degradation_20ep.csv") \
        else "results/degradation.csv"
    if os.path.exists(deg):
        plot_degradation(deg, "results/figures/degradation.png")
    rec = "results/recovery_20ep.csv" if os.path.exists("results/recovery_20ep.csv") \
        else "results/recovery.csv"
    if os.path.exists(rec):
        plot_recovery(rec, "results/figures/recovery.png")
    import glob
    det_files = [f for f in glob.glob("results/detection*.csv") if "_episodes" not in f]
    if det_files:
        df = pd.concat([pd.read_csv(f) for f in sorted(det_files)], ignore_index=True)
        if "model" not in df.columns:
            df["model"] = "sac"
        detection_table(df, "results/figures/detection_table.md")


if __name__ == "__main__":
    main()
