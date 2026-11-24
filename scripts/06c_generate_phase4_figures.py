#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Phase 4 figures: training curves, paired replay Pareto, and online GPU latency vs accuracy."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "phase4"
FIG_DIR = OUT_DIR / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Learning curves
    curves_path = OUT_DIR / "training_curves.csv"
    if curves_path.exists():
        df_c = pd.read_csv(curves_path)
        agg = df_c.groupby("epoch").agg(
            tr_m=("train_loss", "mean"), tr_s=("train_loss", "std"),
            cal_m=("cal_loss", "mean"), cal_s=("cal_loss", "std"),
            br_m=("cal_brier", "mean"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        ax.plot(agg["epoch"], agg["tr_m"], label="Train Loss (5 seeds)", color="#004488", lw=2.0)
        ax.fill_between(agg["epoch"], agg["tr_m"] - agg["tr_s"].fillna(0), agg["tr_m"] + agg["tr_s"].fillna(0), color="#004488", alpha=0.18)
        ax.plot(agg["epoch"], agg["cal_m"], label="Calibration Loss", color="#cc3311", lw=2.0, ls="--")
        ax.fill_between(agg["epoch"], agg["cal_m"] - agg["cal_s"].fillna(0), agg["cal_m"] + agg["cal_s"].fillna(0), color="#cc3311", alpha=0.16)
        ax.plot(agg["epoch"], agg["br_m"], label="Calibration Brier", color="#ee7733", lw=1.6, ls=":")
        ax.set_title("Phase 4 Real-Trace GRU Stopping Policy (5 Seeds)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss / Brier")
        ax.grid(True, alpha=0.3, ls="--")
        ax.legend()
        fig.savefig(FIG_DIR / "fig1_phase4_learning_curves.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / "fig1_phase4_learning_curves.pdf", bbox_inches="tight")
        plt.close(fig)

    # 2. Paired Replay Pareto
    replay_path = OUT_DIR / "replay_results.json"
    if replay_path.exists():
        rep = json.loads(replay_path.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for ax, ds_k, title in [(axes[0], "gsm8k", "GSM8K Holdout Replay (n=96)"), (axes[1], "math500", "MATH-500 Holdout Replay (n=96)")]:
            for st, info in rep["datasets"][ds_k].items():
                ax.scatter(info["avg_tokens"], info["accuracy_pct"], s=90 if st == "gru" else 65, label=f"{st} ({info['avg_tokens']}t, {info['accuracy_pct']}%)")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Avg Tokens per Question")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=8)
        fig.savefig(FIG_DIR / "fig2_phase4_replay_pareto.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / "fig2_phase4_replay_pareto.pdf", bbox_inches="tight")
        plt.close(fig)

    # 3. Online GPU Latency vs Accuracy
    online_path = OUT_DIR / "online_gpu_results.json"
    if online_path.exists():
        onl = json.loads(online_path.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for ax, ds_k, title in [(axes[0], "gsm8k", "GSM8K Online GPU (n=48)"), (axes[1], "math500", "MATH-500 Online GPU (n=48)")]:
            for m, info in onl["datasets"][ds_k].items():
                ax.scatter(info["avg_latency_sec"], info["accuracy_pct"], s=95 if m == "gru" else 65, label=f"{m} ({info['avg_latency_sec']}s, {info['accuracy_pct']}%)")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("End-to-End Wall-Clock Latency (sec)")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=8)
        fig.savefig(FIG_DIR / "fig3_phase4_online_gpu_pareto.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / "fig3_phase4_online_gpu_pareto.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"[Figures] Saved Phase 4 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
