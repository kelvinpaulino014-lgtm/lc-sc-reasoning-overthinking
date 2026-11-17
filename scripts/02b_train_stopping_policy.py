#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 02b (Phase 3 - Part A): Multi-Seed, 5-Fold CV Deep Neural Training
of the Compact `NeuralStoppingPolicy` (<3K params) strictly on `val_calibration`.

Logs continuous progress to `results/phase3_deep_training/training_progress.log`,
saves checkpoints, epoch-by-epoch learning curves, and generates learning-curve plots.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stopping_policy import NeuralStoppingPolicy, build_step_features_from_trace

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "phase3_deep_training"
CKPT_DIR = OUT_DIR / "checkpoints"
SEEDS = [42, 101, 2026, 777, 999]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_msg(log_path: Path, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def prepare_padded_tensors(
    df_sub: pd.DataFrame,
    l_star: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds padded [N, T_max, 6] features, targets, and valid step masks."""
    seqs, s_tgts, c_tgts = [], [], []
    max_t = 1
    for _, row in df_sub.iterrows():
        steps = json.loads(row["steps_json"])
        if not steps:
            continue
        f, st, ct = build_step_features_from_trace(
            steps, l_star=l_star, gold_answer=str(row["gold_answer"])
        )
        seqs.append(f)
        s_tgts.append(st)
        c_tgts.append(ct)
        max_t = max(max_t, f.shape[0])

    n = len(seqs)
    X = np.zeros((n, max_t, 6), dtype=np.float32)
    y_stop = np.zeros((n, max_t), dtype=np.float32)
    y_corr = np.zeros((n, max_t), dtype=np.float32)
    mask = np.zeros((n, max_t), dtype=np.float32)

    for i, (f, st, ct) in enumerate(zip(seqs, s_tgts, c_tgts)):
        t_len = f.shape[0]
        X[i, :t_len] = f
        y_stop[i, :t_len] = st
        y_corr[i, :t_len] = ct
        mask[i, :t_len] = 1.0

    return (
        torch.from_numpy(X),
        torch.from_numpy(y_stop),
        torch.from_numpy(y_corr),
        torch.from_numpy(mask),
    )


def compute_masked_loss(
    logits: torch.Tensor,
    y_stop: torch.Tensor,
    y_corr: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = nn.BCEWithLogitsLoss(reduction="none")
    stop_bce = bce(logits[..., 0], y_stop)
    corr_bce = bce(logits[..., 1], y_corr)
    corr_prob = torch.sigmoid(logits[..., 1])
    brier = (corr_prob - y_corr) ** 2

    denom = torch.clamp(mask.sum(), min=1.0)
    loss_stop = (stop_bce * mask).sum() / denom
    loss_corr = ((0.5 * corr_bce + 0.5 * brier) * mask).sum() / denom
    loss_brier = (brier * mask).sum() / denom
    return loss_stop + loss_corr, loss_stop, loss_brier


def train_single_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    l_star: float,
    seed: int,
    device: str,
    max_epochs: int = 55,
    patience: int = 8,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    set_seed(seed)
    X_tr, ys_tr, yc_tr, m_tr = prepare_padded_tensors(df_tr, l_star)
    X_va, ys_va, yc_va, m_va = prepare_padded_tensors(df_va, l_star)

    X_tr, ys_tr, yc_tr, m_tr = X_tr.to(device), ys_tr.to(device), yc_tr.to(device), m_tr.to(device)
    X_va, ys_va, yc_va, m_va = X_va.to(device), ys_va.to(device), yc_va.to(device), m_va.to(device)

    model = NeuralStoppingPolicy(input_dim=6, hidden_dim=16, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    wait = 0
    curve_rows: List[Dict[str, Any]] = []

    batch_size = 128
    n_tr = X_tr.shape[0]

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        tr_loss_acc = 0.0
        tr_batches = 0
        for start in range(0, n_tr, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits, _ = model(X_tr[idx])
            loss, _, _ = compute_masked_loss(logits, ys_tr[idx], yc_tr[idx], m_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss_acc += float(loss.item())
            tr_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits, _ = model(X_va)
            va_loss, _, va_brier = compute_masked_loss(va_logits, ys_va, yc_va, m_va)
            va_loss_f = float(va_loss.item())
            va_brier_f = float(va_brier.item())

        tr_loss_f = tr_loss_acc / max(1, tr_batches)
        curve_rows.append({
            "epoch": epoch,
            "train_loss": tr_loss_f,
            "val_loss": va_loss_f,
            "val_brier": va_brier_f,
        })

        if va_loss_f < best_val_loss - 1e-4:
            best_val_loss = va_loss_f
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Evaluate best checkpoint on validation fold
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        va_logits, _ = model(X_va)
        va_probs = torch.sigmoid(va_logits).cpu().numpy()
    m_np = m_va.cpu().numpy().astype(bool)
    ys_np = ys_va.cpu().numpy()[m_np]
    yc_np = yc_va.cpu().numpy()[m_np]
    p_stop = va_probs[..., 0][m_np]
    p_corr = va_probs[..., 1][m_np]

    stop_f1 = float(f1_score(ys_np.astype(int), (p_stop >= 0.5).astype(int), zero_division=0))
    corr_brier = float(brier_score_loss(yc_np.astype(int), np.clip(p_corr, 0.0, 1.0)))
    try:
        corr_auc = float(roc_auc_score(yc_np.astype(int), p_corr))
    except Exception:
        corr_auc = 0.5

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "val_stop_f1": round(stop_f1, 4),
        "val_corr_brier": round(corr_brier, 4),
        "val_corr_auc": round(corr_auc, 4),
        "param_count": model.count_parameters(),
    }
    return summary, curve_rows, best_state


def plot_learning_curves(df_curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, ds_name, color in [
        (axes[0], "GSM8K", "#004488"),
        (axes[1], "MATH-500", "#117733"),
    ]:
        sub = df_curves[df_curves["dataset"] == ds_name]
        agg = sub.groupby("epoch").agg(
            tr_mean=("train_loss", "mean"),
            tr_std=("train_loss", "std"),
            va_mean=("val_loss", "mean"),
            va_std=("val_loss", "std"),
            br_mean=("val_brier", "mean"),
        ).reset_index()
        ep = agg["epoch"].values
        ax.plot(ep, agg["tr_mean"], label="Train Loss (BCE+Brier)", color=color, lw=2.0)
        ax.fill_between(
            ep,
            agg["tr_mean"] - agg["tr_std"].fillna(0),
            agg["tr_mean"] + agg["tr_std"].fillna(0),
            color=color,
            alpha=0.18,
        )
        ax.plot(ep, agg["va_mean"], label="Val CV Loss (5-Fold x 5-Seed)", color="#cc3311", lw=2.0, linestyle="--")
        ax.fill_between(
            ep,
            agg["va_mean"] - agg["va_std"].fillna(0),
            agg["va_mean"] + agg["va_std"].fillna(0),
            color="#cc3311",
            alpha=0.16,
        )
        ax.plot(ep, agg["br_mean"], label="Val Brier Score", color="#ee7733", lw=1.6, linestyle=":")
        ax.set_title(f"{ds_name} — NeuralStoppingPolicy (1,490 params)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss / Brier")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Phase 3 Deep Neural Stopping Policy: Multi-Seed 5-Fold CV Learning Curves (val_calibration)",
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(OUT_DIR / "fig_phase3_learning_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_phase3_learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "training_progress.log"
    if log_path.exists():
        log_path.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    param_cnt = NeuralStoppingPolicy().count_parameters()
    log_msg(log_path, f"Starting Phase 3 NeuralStoppingPolicy training on device={device}, params={param_cnt} (<3000 OK)")

    all_curves: List[Dict[str, Any]] = []
    metrics_out: Dict[str, Any] = {"param_count": param_cnt, "seeds": SEEDS, "datasets": {}}

    for ds_key, ds_name, parquet_path in [
        ("gsm8k", "GSM8K", RESULTS_DIR / "traces_gsm8k.parquet"),
        ("math500", "MATH-500", RESULTS_DIR / "traces_math500.parquet"),
    ]:
        df = pd.read_parquet(parquet_path)
        df_val = df[df["split"] == "val_calibration"].copy()
        qids = list(df_val["question_id"].unique())
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_summaries: List[Dict[str, Any]] = []
        best_overall_loss = float("inf")
        best_overall_state: Dict[str, torch.Tensor] = {}
        best_l_star = 140.0

        for seed in SEEDS:
            for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(qids)):
                tr_q = {qids[i] for i in tr_idx}
                va_q = {qids[i] for i in va_idx}
                d_tr = df_val[df_val["question_id"].isin(tr_q)]
                d_va = df_val[df_val["question_id"].isin(va_q)]

                corr_tr = d_tr[d_tr["is_correct_full"] == 1]
                q20, q75 = np.percentile(corr_tr["total_tokens"], [20, 75])
                opt = corr_tr[(corr_tr["total_tokens"] >= q20) & (corr_tr["total_tokens"] <= q75)]
                l_star_fold = float(opt["total_tokens"].median()) if len(opt) > 0 else 140.0

                summary, curves, state = train_single_fold(d_tr, d_va, l_star=l_star_fold, seed=seed, device=device)
                summary.update({"seed": seed, "fold": fold_idx, "l_star": round(l_star_fold, 2)})
                fold_summaries.append(summary)

                for c in curves:
                    c.update({"dataset": ds_name, "seed": seed, "fold": fold_idx})
                    all_curves.append(c)

                if summary["best_val_loss"] < best_overall_loss:
                    best_overall_loss = summary["best_val_loss"]
                    best_overall_state = state
                    best_l_star = l_star_fold

                log_msg(
                    log_path,
                    f"[{ds_name}] seed={seed} fold={fold_idx} | best_ep={summary['best_epoch']} "
                    f"val_loss={summary['best_val_loss']:.4f} stop_F1={summary['val_stop_f1']:.4f} "
                    f"brier={summary['val_corr_brier']:.4f} auc={summary['val_corr_auc']:.4f}",
                )

        # Save best checkpoint for real online GPU inference
        ckpt_path = CKPT_DIR / f"{ds_key}_best_policy.pt"
        torch.save(
            {
                "state_dict": best_overall_state,
                "l_star": best_l_star,
                "l_min": max(45, int(0.50 * best_l_star)),
                "stop_threshold": 0.58,
                "param_count": param_cnt,
            },
            ckpt_path,
        )

        df_sum = pd.DataFrame(fold_summaries)
        metrics_out["datasets"][ds_key] = {
            "n_val_questions": len(qids),
            "n_val_traces": len(df_val),
            "l_star_mean": round(float(df_sum["l_star"].mean()), 2),
            "val_loss_mean": round(float(df_sum["best_val_loss"].mean()), 5),
            "val_loss_std": round(float(df_sum["best_val_loss"].std()), 5),
            "val_stop_f1_mean": round(float(df_sum["val_stop_f1"].mean()), 4),
            "val_stop_f1_std": round(float(df_sum["val_stop_f1"].std()), 4),
            "val_brier_mean": round(float(df_sum["val_corr_brier"].mean()), 4),
            "val_brier_std": round(float(df_sum["val_corr_brier"].std()), 4),
            "val_auc_mean": round(float(df_sum["val_corr_auc"].mean()), 4),
            "val_auc_std": round(float(df_sum["val_corr_auc"].std()), 4),
            "checkpoint_path": str(ckpt_path),
            "runs": fold_summaries,
        }
        log_msg(
            log_path,
            f"=== [{ds_name}] 5-Fold x 5-Seed Summary: val_loss={metrics_out['datasets'][ds_key]['val_loss_mean']:.4f}+/-{metrics_out['datasets'][ds_key]['val_loss_std']:.4f}, "
            f"stop_F1={metrics_out['datasets'][ds_key]['val_stop_f1_mean']:.4f}, Brier={metrics_out['datasets'][ds_key]['val_brier_mean']:.4f}, "
            f"AUC={metrics_out['datasets'][ds_key]['val_auc_mean']:.4f} ===",
        )

    df_curves = pd.DataFrame(all_curves)
    df_curves.to_csv(OUT_DIR / "training_curves.csv", index=False)
    with open(OUT_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    plot_learning_curves(df_curves)
    log_msg(log_path, "Phase 3 Part A training complete! Saved checkpoints, metrics, and learning curves.")


if __name__ == "__main__":
    main()
