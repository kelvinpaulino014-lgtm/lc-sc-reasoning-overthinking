#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 CPU Training, Threshold Calibration & Paired Holdout Replay (`06_train_real_policy.py`).
Follows `docs/08_Protocolo_Fase4.md` and `src/models/real_trace_policy.py` strictly:
- Trains LogisticRegression + NeuralStoppingPolicy (GRU, 5 seeds) on `train` split only.
- Selects & freezes thresholds on `calibration` split only.
- Evaluates paired replay on all 96 `holdout` questions/dataset with paired Bootstrap 95% CI
  and conservative Wilson/Newcombe bound for rare paired disagreements.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.entropy_tracker import bootstrap_ci, compute_ece
from src.models.real_trace_policy import (
    FEATURES,
    MAX_TOKENS,
    STRIDE,
    FittedPolicy,
    Window,
    complete_answer,
    consensus_after_two,
    load_policy,
    make_window,
    sha256_file,
    vote,
)
from src.models.stopping_policy import NeuralStoppingPolicy
from src.utils.answer_extractor import is_equivalent

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SEEDS = [42, 101, 2026, 777, 999]
OUT_DIR = ROOT / "results" / "phase4"
POLICY_DIR = OUT_DIR / "policy"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_phase4_records(out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    labels: Dict[str, Dict[str, Any]] = {}
    for line in (out_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["question_id"]] = row

    records: List[Dict[str, Any]] = []
    for line in (out_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records, labels


def build_trace_windows(trace: Dict[str, Any], tokenizer: Any) -> List[Window]:
    token_ids: List[int] = trace["token_ids"]
    entropies: List[float] = trace["entropies"]
    logprobs: List[float] = trace["logprobs"]
    total = min(len(token_ids), MAX_TOKENS)
    windows: List[Window] = []
    prev_w: Optional[Window] = None
    start = 0
    while start < total:
        end = min(start + STRIDE, total)
        prefix_text = tokenizer.decode(token_ids[:end], skip_special_tokens=True)
        w = make_window(
            prefix_text=prefix_text,
            entropy_window=entropies[start:end],
            logprob_window=logprobs[start:end],
            n_tokens=end,
            previous_window=prev_w,
        )
        windows.append(w)
        prev_w = w
        start = end
    return windows


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired_diff_ci(
    vals_a: Sequence[float],
    vals_b: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired difference (vals_a - vals_b) with bootstrap 95% CI and conservative loss bound."""
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    diff = a - b
    mean_d, lo_d, hi_d = bootstrap_ci(diff.tolist(), n_boot=n_boot, seed=seed)
    # Conservative Newcombe-like upper bound on paired losses (a < b)
    n = len(diff)
    losses = int(np.sum(diff < -1e-9))
    _, loss_hi = wilson_ci(losses, n)
    return {
        "mean": round(mean_d, 4),
        "ci_low": round(lo_d, 4),
        "ci_high": round(hi_d, 4),
        "n_losses": losses,
        "n_gains": int(np.sum(diff > 1e-9)),
        "conservative_loss_upper_pct": round(loss_hi * 100.0, 2),
    }


def simulate_question_replay(
    q_windows: List[List[Window]],
    q_full_answers: List[Optional[str]],
    q_full_tokens: List[int],
    strategy: str,
    policy: Optional[FittedPolicy] = None,
    gold: Optional[str] = None,
) -> Dict[str, Any]:
    """Simulates a strategy on 4 paired traces of one question."""
    used_answers: List[Optional[str]] = []
    used_tokens: List[int] = []
    stop_reasons: List[str] = []
    stopped_early_l = 0
    lost_correct = 0
    improved_incorrect = 0

    def run_one_trace(idx: int) -> Tuple[Optional[str], int, str]:
        wins = q_windows[idx]
        full_ans = q_full_answers[idx]
        full_tok = q_full_tokens[idx]

        if strategy in ("sc4", "consensus2"):
            return full_ans, full_tok, "full"
        if strategy == "first_box":
            for w in wins:
                if w.answer is not None:
                    return w.answer, w.n_tokens, "first_box"
            return full_ans, full_tok, "full"
        if strategy == "fixed_cap":
            cap_tok = MAX_TOKENS // 2
            for w in wins:
                if w.n_tokens >= cap_tok or w is wins[-1]:
                    return w.answer, w.n_tokens, "fixed_cap"
            return full_ans, full_tok, "full"
        if strategy in ("logistic", "gru"):
            assert policy is not None
            probs = policy.predict_windows(wins, strategy)
            for w, p_row in zip(wins, probs):
                if policy.should_stop(w, p_row, strategy):
                    return w.answer, w.n_tokens, f"{strategy}_stop"
            return full_ans, full_tok, "full"
        raise ValueError(f"Unknown strategy: {strategy}")

    # Batch 1: first 2 traces
    for i in range(2):
        ans_i, tok_i, reason_i = run_one_trace(i)
        used_answers.append(ans_i)
        used_tokens.append(tok_i)
        stop_reasons.append(reason_i)
        if reason_i != "full":
            stopped_early_l += 1
        if gold is not None:
            if is_equivalent(q_full_answers[i], gold) and not is_equivalent(ans_i, gold):
                lost_correct += 1
            if not is_equivalent(q_full_answers[i], gold) and is_equivalent(ans_i, gold):
                improved_incorrect += 1

    # Consensus after 2 for all early-N strategies
    if strategy != "sc4" and consensus_after_two(used_answers):
        pred, cov = vote(used_answers)
        return {
            "pred": pred,
            "confidence": cov,
            "tokens": sum(used_tokens),
            "n_traces": 2,
            "stopped_early_l": stopped_early_l,
            "lost_correct": lost_correct,
            "improved_incorrect": improved_incorrect,
        }

    # Batch 2: remaining 2 traces
    for i in range(2, 4):
        ans_i, tok_i, reason_i = run_one_trace(i)
        used_answers.append(ans_i)
        used_tokens.append(tok_i)
        stop_reasons.append(reason_i)
        if reason_i != "full":
            stopped_early_l += 1
        if gold is not None:
            if is_equivalent(q_full_answers[i], gold) and not is_equivalent(ans_i, gold):
                lost_correct += 1
            if not is_equivalent(q_full_answers[i], gold) and is_equivalent(ans_i, gold):
                improved_incorrect += 1

    pred, cov = vote(used_answers)
    return {
        "pred": pred,
        "confidence": cov,
        "tokens": sum(used_tokens),
        "n_traces": 4,
        "stopped_early_l": stopped_early_l,
        "lost_correct": lost_correct,
        "improved_incorrect": improved_incorrect,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_DIR.mkdir(parents=True, exist_ok=True)

    records, labels = load_phase4_records(OUT_DIR)
    print(f"[Phase 4 Train] Loaded {len(records)} records from traces.jsonl")

    model_path = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Precompute causal windows for all records
    prepared: List[Dict[str, Any]] = []
    for rec in records:
        qid = rec["question_id"]
        gold = labels[qid]["answer_gold"]
        q_wins: List[List[Window]] = []
        q_full_ans: List[Optional[str]] = []
        q_full_toks: List[int] = []
        for tr in rec["traces"]:
            wins = build_trace_windows(tr, tokenizer)
            q_wins.append(wins)
            q_full_ans.append(complete_answer(tr["text"]))
            q_full_toks.append(len(tr["token_ids"]))
        prepared.append({
            "question_id": qid,
            "dataset": rec["dataset"],
            "split": rec["split"],
            "gold": gold,
            "windows": q_wins,
            "full_answers": q_full_ans,
            "full_tokens": q_full_toks,
        })

    train_items = [x for x in prepared if x["split"] == "train"]
    cal_items = [x for x in prepared if x["split"] == "calibration"]
    holdout_items = [x for x in prepared if x["split"] == "holdout"]
    print(f"[Splits] train={len(train_items)}, calibration={len(cal_items)}, holdout={len(holdout_items)}")

    # Build window feature matrices on train & calibration
    def build_seq_tensors(items: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        Xs, Ys_safe, Ys_corr = [], [], []
        for item in items:
            gold = item["gold"]
            for wins, full_ans in zip(item["windows"], item["full_answers"]):
                full_ok = is_equivalent(full_ans, gold)
                x_arr = np.asarray([w.features for w in wins], dtype=np.float32)
                y_c = np.asarray([1.0 if (w.answer is not None and is_equivalent(w.answer, gold)) else 0.0 for w in wins], dtype=np.float32)
                y_s = np.asarray([1.0 if (w.answer is not None and (is_equivalent(w.answer, gold) or not full_ok)) else 0.0 for w in wins], dtype=np.float32)
                Xs.append(x_arr)
                Ys_safe.append(y_s)
                Ys_corr.append(y_c)
        return Xs, Ys_safe, Ys_corr

    X_tr_list, ys_tr_list, yc_tr_list = build_seq_tensors(train_items)
    X_cal_list, ys_cal_list, yc_cal_list = build_seq_tensors(cal_items)

    scaler = StandardScaler()
    scaler.fit(np.vstack(X_tr_list))

    # 1. Fit LogisticRegression on train
    X_tr_flat = scaler.transform(np.vstack(X_tr_list))
    ys_tr_flat = np.concatenate(ys_tr_list).astype(int)
    yc_tr_flat = np.concatenate(yc_tr_list).astype(int)

    clf_safe = LogisticRegression(C=1.0, max_iter=500).fit(X_tr_flat, ys_tr_flat)
    clf_corr = LogisticRegression(C=1.0, max_iter=500).fit(X_tr_flat, yc_tr_flat)
    log_coef = np.vstack([clf_safe.coef_[0], clf_corr.coef_[0]]).tolist()
    log_intercept = [float(clf_safe.intercept_[0]), float(clf_corr.intercept_[0])]

    # 2. Train NeuralStoppingPolicy (GRU) across 5 seeds on train, early-stopping on calibration
    def pad_sequences(x_list: List[np.ndarray], ys_list: List[np.ndarray], yc_list: List[np.ndarray]):
        n = len(x_list)
        t_max = max(len(x) for x in x_list)
        Xp = np.zeros((n, t_max, len(FEATURES)), dtype=np.float32)
        Ysp = np.zeros((n, t_max), dtype=np.float32)
        Ycp = np.zeros((n, t_max), dtype=np.float32)
        Mp = np.zeros((n, t_max), dtype=np.float32)
        for i, (x, ys, yc) in enumerate(zip(x_list, ys_list, yc_list)):
            t = len(x)
            Xp[i, :t] = scaler.transform(x)
            Ysp[i, :t] = ys
            Ycp[i, :t] = yc
            Mp[i, :t] = 1.0
        return torch.from_numpy(Xp), torch.from_numpy(Ysp), torch.from_numpy(Ycp), torch.from_numpy(Mp)

    Xp_tr, Ysp_tr, Ycp_tr, Mp_tr = pad_sequences(X_tr_list, ys_tr_list, yc_tr_list)
    Xp_cal, Ysp_cal, Ycp_cal, Mp_cal = pad_sequences(X_cal_list, ys_cal_list, yc_cal_list)

    bce = nn.BCEWithLogitsLoss(reduction="none")
    curve_rows: List[Dict[str, Any]] = []
    seed_models: Dict[int, NeuralStoppingPolicy] = {}
    seed_cal_losses: Dict[int, float] = {}

    for seed in SEEDS:
        set_seed(seed)
        model = NeuralStoppingPolicy(input_dim=len(FEATURES), hidden_dim=16, dropout=0.1)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        best_loss = float("inf")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0

        for ep in range(1, 51):
            model.train()
            perm = torch.randperm(Xp_tr.shape[0])
            tr_acc, nb = 0.0, 0
            for s in range(0, Xp_tr.shape[0], 64):
                idx = perm[s : s + 64]
                opt.zero_grad()
                logits, _ = model(Xp_tr[idx])
                denom = torch.clamp(Mp_tr[idx].sum(), min=1.0)
                l_s = (bce(logits[..., 0], Ysp_tr[idx]) * Mp_tr[idx]).sum() / denom
                l_c = (bce(logits[..., 1], Ycp_tr[idx]) * Mp_tr[idx]).sum() / denom
                brier = (((torch.sigmoid(logits[..., 1]) - Ycp_tr[idx]) ** 2) * Mp_tr[idx]).sum() / denom
                loss = l_s + 0.5 * l_c + 0.5 * brier
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_acc += float(loss.item())
                nb += 1
            sched.step()

            model.eval()
            with torch.no_grad():
                logits_c, _ = model(Xp_cal)
                denom_c = torch.clamp(Mp_cal.sum(), min=1.0)
                ls_c = (bce(logits_c[..., 0], Ysp_cal) * Mp_cal).sum() / denom_c
                lc_c = (bce(logits_c[..., 1], Ycp_cal) * Mp_cal).sum() / denom_c
                br_c = (((torch.sigmoid(logits_c[..., 1]) - Ycp_cal) ** 2) * Mp_cal).sum() / denom_c
                cal_loss = float((ls_c + 0.5 * lc_c + 0.5 * br_c).item())

            curve_rows.append({
                "seed": seed,
                "epoch": ep,
                "train_loss": round(tr_acc / max(1, nb), 5),
                "cal_loss": round(cal_loss, 5),
                "cal_brier": round(float(br_c.item()), 5),
            })

            if cal_loss < best_loss - 1e-4:
                best_loss = cal_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= 8:
                    break

        model.load_state_dict(best_state)
        model.eval()
        seed_models[seed] = model
        seed_cal_losses[seed] = best_loss
        np.savez(POLICY_DIR / f"gru_seed{seed}.npz", **{k: v.cpu().numpy() for k, v in best_state.items()})
        print(f"  [GRU seed={seed}] best_cal_loss={best_loss:.4f}")

    deployment_seed = min(SEEDS, key=lambda s: seed_cal_losses[s])

    # 3. Select thresholds on calibration split (never touch holdout!)
    sc4_cal_acc = float(np.mean([
        int(is_equivalent(simulate_question_replay(x["windows"], x["full_answers"], x["full_tokens"], "sc4")["pred"], x["gold"]))
        for x in cal_items
    ]))

    def select_thresholds(family: str, model: NeuralStoppingPolicy) -> List[float]:
        best_pair = [0.99, 0.99]  # fallback: conserve full budget
        best_tok = float("inf")
        best_acc = -1.0
        for t_s in (0.50, 0.60, 0.70, 0.80, 0.90, 0.99):
            for t_c in (0.50, 0.60, 0.70, 0.80, 0.90, 0.99):
                cfg_tmp = {
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "logistic_coef": log_coef,
                    "logistic_intercept": log_intercept,
                    "thresholds": {family: [t_s, t_c]},
                }
                fp = FittedPolicy(cfg_tmp, model)
                accs, toks = [], []
                for x in cal_items:
                    r = simulate_question_replay(x["windows"], x["full_answers"], x["full_tokens"], family, fp, x["gold"])
                    accs.append(int(is_equivalent(r["pred"], x["gold"])))
                    toks.append(r["tokens"])
                m_acc, m_tok = float(np.mean(accs)), float(np.mean(toks))
                if m_acc >= sc4_cal_acc - 0.02 and (m_tok < best_tok or (m_tok == best_tok and m_acc > best_acc)):
                    best_tok = m_tok
                    best_acc = m_acc
                    best_pair = [t_s, t_c]
        return best_pair

    log_thresh = select_thresholds("logistic", seed_models[deployment_seed])
    gru_thresh_by_seed = {str(s): select_thresholds("gru", seed_models[s]) for s in SEEDS}
    print(f"[Calibration] SC-4 cal_acc={sc4_cal_acc*100:.2f}%, log_thresh={log_thresh}, gru_thresh[{deployment_seed}]={gru_thresh_by_seed[str(deployment_seed)]}")

    config = {
        "features": list(FEATURES),
        "stride": STRIDE,
        "max_tokens": MAX_TOKENS,
        "seeds": SEEDS,
        "deployment_seed": deployment_seed,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "logistic_coef": log_coef,
        "logistic_intercept": log_intercept,
        "thresholds": {"logistic": log_thresh},
        "gru_thresholds_by_seed": gru_thresh_by_seed,
    }
    cfg_path = POLICY_DIR / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    sha_map = {"config.json": sha256_file(cfg_path)}
    for s in SEEDS:
        fn = f"gru_seed{s}.npz"
        sha_map[fn] = sha256_file(POLICY_DIR / fn)
    (POLICY_DIR / "frozen.json").write_text(json.dumps({"sha256": sha_map}, indent=2) + "\n", encoding="utf-8")

    # 4. Paired Holdout Replay (96 questions/dataset = 192 total)
    fitted = load_policy(POLICY_DIR, seed=deployment_seed)
    strategies = ["sc4", "consensus2", "first_box", "fixed_cap", "logistic", "gru"]
    replay_out: Dict[str, Any] = {"deployment_seed": deployment_seed, "datasets": {}}
    csv_rows: List[Dict[str, Any]] = []

    for ds_key in ("gsm8k", "math500", "overall"):
        sub = holdout_items if ds_key == "overall" else [x for x in holdout_items if x["dataset"] == ds_key]
        ds_res: Dict[str, Any] = {}
        raw_per_q: Dict[str, Dict[str, List[Any]]] = {}

        for st in strategies:
            corrs, toks, n_trs, confs, l_stops, lost_c, imp_i, no_ans = [], [], [], [], [], [], [], []
            for x in sub:
                r = simulate_question_replay(x["windows"], x["full_answers"], x["full_tokens"], st, fitted, x["gold"])
                corrs.append(int(is_equivalent(r["pred"], x["gold"])))
                toks.append(r["tokens"])
                n_trs.append(r["n_traces"])
                confs.append(r["confidence"])
                l_stops.append(r["stopped_early_l"])
                lost_c.append(r["lost_correct"])
                imp_i.append(r["improved_incorrect"])
                no_ans.append(int(r["pred"] is None))
            raw_per_q[st] = {"corrs": corrs, "toks": toks}

            acc_m, acc_lo, acc_hi = bootstrap_ci(corrs, n_boot=2000, seed=42)
            tok_m, tok_lo, tok_hi = bootstrap_ci(toks, n_boot=2000, seed=42)
            ece_v = compute_ece(np.array(corrs, dtype=float), np.clip(np.array(confs), 0.01, 0.99), n_bins=8)
            brier_v = float(brier_score_loss(corrs, np.clip(confs, 0.01, 0.99)))
            try:
                auc_v = float(roc_auc_score(corrs, confs)) * 100.0 if len(set(corrs)) > 1 else 50.0
            except Exception:
                auc_v = 50.0

            ds_res[st] = {
                "n_questions": len(sub),
                "accuracy_pct": round(acc_m * 100.0, 2),
                "ci95_low_pct": round(acc_lo * 100.0, 2),
                "ci95_high_pct": round(acc_hi * 100.0, 2),
                "avg_tokens": round(tok_m, 1),
                "tokens_ci95": [round(tok_lo, 1), round(tok_hi, 1)],
                "avg_traces": round(float(np.mean(n_trs)), 2),
                "ece_pct": round(ece_v * 100.0, 2),
                "brier_score": round(brier_v, 4),
                "roc_auc_pct": round(auc_v, 2),
                "l_stopped_traces": int(sum(l_stops)),
                "correct_candidate_lost": int(sum(lost_c)),
                "incorrect_candidate_improved": int(sum(imp_i)),
                "no_explicit_answer": int(sum(no_ans)),
            }

        sc4_tok = ds_res["sc4"]["avg_tokens"]
        for st in strategies:
            ds_res[st]["token_saving_vs_sc4_pct"] = round((1.0 - ds_res[st]["avg_tokens"] / max(1.0, sc4_tok)) * 100.0, 1)
            ds_res[st]["paired_acc_diff_vs_sc4_pp"] = paired_diff_ci(
                [c * 100.0 for c in raw_per_q[st]["corrs"]],
                [c * 100.0 for c in raw_per_q["sc4"]["corrs"]],
            )
            ds_res[st]["paired_acc_diff_vs_consensus2_pp"] = paired_diff_ci(
                [c * 100.0 for c in raw_per_q[st]["corrs"]],
                [c * 100.0 for c in raw_per_q["consensus2"]["corrs"]],
            )
            csv_rows.append({
                "Dataset": ds_key,
                "Strategy": st,
                "Accuracy_Pct": ds_res[st]["accuracy_pct"],
                "CI95_Low": ds_res[st]["ci95_low_pct"],
                "CI95_High": ds_res[st]["ci95_high_pct"],
                "Avg_Tokens": ds_res[st]["avg_tokens"],
                "Token_Saving_vs_SC4_Pct": ds_res[st]["token_saving_vs_sc4_pct"],
                "Avg_Traces": ds_res[st]["avg_traces"],
                "ECE_Pct": ds_res[st]["ece_pct"],
                "ROC_AUC_Pct": ds_res[st]["roc_auc_pct"],
                "Delta_Acc_vs_SC4_pp": ds_res[st]["paired_acc_diff_vs_sc4_pp"]["mean"],
            })

        replay_out["datasets"][ds_key] = ds_res

    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "training_curves.csv", index=False)
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "replay_summary.csv", index=False)
    (OUT_DIR / "replay_results.json").write_text(json.dumps(replay_out, indent=2) + "\n", encoding="utf-8")
    print("\n=== PHASE 4 PAIRED HOLDOUT REPLAY ===")
    print(pd.DataFrame(csv_rows).to_string(index=False))


if __name__ == "__main__":
    main()
