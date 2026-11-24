#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fases 1–2. Replay de simulação condicionada ao gabarito sobre trilhas derivadas de
âncoras. As métricas descrevem o simulador e seus decodificadores. A Fase 4 avalia
gerações do modelo salvas independentemente dos rótulos.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.metrics.entropy_tracker import compute_calibration_bins

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "corrected_legacy_replay"


def calculate_ece_and_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 8,
) -> Tuple[float, List[Dict[str, float]]]:
    """Calcula o Expected Calibration Error (ECE) e os pontos do diagrama de confiabilidade."""
    bins = compute_calibration_bins(confidences=y_prob, accuracies=y_true, n_bins=n_bins)
    n = sum(b.count for b in bins)
    if n != len(y_true) or n != len(y_prob):
        raise ValueError("Cada pergunta deve ter um escore e um desfecho finitos.")
    ece = sum(b.count * b.calibration_gap for b in bins) / n if n else 0.0
    records = [
        {"bin_idx": b.bin_idx, "mean_conf": round(b.mean_confidence, 4),
         "mean_acc": round(b.mean_accuracy, 4), "count": b.count}
        for b in bins if b.count
    ]
    return float(ece), records


def bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Retorna (média, ci_low, ci_high) via bootstrap não-paramétrico."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    return float(arr.mean()), low, high


def fit_calibration_on_validation(df_val: pd.DataFrame) -> Dict[str, Any]:
    """
    Ajusta todos os hiperparâmetros estatísticos do LC-SC e dos baselines
    EXCLUSIVAMENTE no conjunto de validação (`val_calibration`).
    """
    # 1. Estimar zona ótima L* e dispersão sigma_L da curva em U invertido (Wu et al., 2026)
    correct_traces = df_val[df_val["is_correct_full"] == 1]
    q25, q75 = np.percentile(correct_traces["total_tokens"], [20, 75])
    opt_subset = correct_traces[
        (correct_traces["total_tokens"] >= q25) & (correct_traces["total_tokens"] <= q75)
    ]
    l_star = float(opt_subset["total_tokens"].median()) if len(opt_subset) > 0 else 140.0
    sigma_l = float(max(35.0, opt_subset["total_tokens"].std() * 1.35)) if len(opt_subset) > 0 else 55.0
    l_min = float(max(45.0, l_star * 0.50))

    # 2. Estimar janela ótima de comprimento para Wu et al. (2026) Length-Filtered Vote
    l_window_low = float(max(55.0, l_star - 1.15 * sigma_l))
    l_window_high = float(l_star + 1.15 * sigma_l)

    # 3. Estimar limiar de pico de entropia tau_H para o L-Stopping (sem teto artificial de 0.76)
    conv_entropies = []
    over_entropies = []
    for _, row in df_val.iterrows():
        steps = json.loads(row["steps_json"])
        for st in steps:
            if st["is_overthinking_transition"]:
                over_entropies.append(st["step_entropy"])
            else:
                conv_entropies.append(st["step_entropy"])

    if conv_entropies and over_entropies:
        tau_h = float(0.55 * np.percentile(conv_entropies, 85) + 0.45 * np.percentile(over_entropies, 25))
    else:
        tau_h = 0.95
    tau_h = float(np.clip(tau_h, 0.75, 1.25))

    # 4. Ajustar Regressão Isotônica para CaTS (apenas confiança bruta)
    iso_cats = IsotonicRegression(y_min=0.05, y_max=0.98, out_of_bounds="clip")
    iso_cats.fit(df_val["raw_confidence"].values, df_val["is_correct_full"].values)

    # 5. Ajustar Calibrador Multi-Feature Pós-L-Stopping para LC-SC (Fase 2)
    X_trace_val, y_trace_val = [], []
    for _, row in df_val.iterrows():
        ans_s, tok_s, conf_s, ent_s, was_t = apply_online_l_stopping(row, l_min=l_min, tau_h=tau_h)
        X_trace_val.append([
            conf_s,
            ent_s,
            float(was_t),
            tok_s / max(1.0, l_star),
            max(0.0, tok_s - l_star) / max(1.0, sigma_l),
        ])
        y_trace_val.append(int(ans_s == str(row["gold_answer"])))

    trace_clf_lcsc = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    trace_clf_lcsc.fit(np.asarray(X_trace_val, dtype=float), np.asarray(y_trace_val, dtype=int))

    # Manter iso_lcsc legado para compatibilidade
    eff_score_val = df_val["raw_confidence"].values * np.exp(
        -0.35 * np.maximum(0.0, df_val["mean_entropy"].values - 0.35)
    )
    iso_lcsc = IsotonicRegression(y_min=0.05, y_max=0.99, out_of_bounds="clip")
    iso_lcsc.fit(eff_score_val, df_val["is_correct_midway"].values)

    return {
        "l_star": round(l_star, 2),
        "sigma_l": round(sigma_l, 2),
        "l_min": round(l_min, 2),
        "l_window_low": round(l_window_low, 2),
        "l_window_high": round(l_window_high, 2),
        "tau_h": round(tau_h, 4),
        "iso_cats": iso_cats,
        "iso_lcsc": iso_lcsc,
        "trace_clf_lcsc": trace_clf_lcsc,
    }


def apply_online_l_stopping(
    row: pd.Series,
    l_min: float,
    tau_h: float,
    window: int = 2,
) -> Tuple[str, int, float, float, bool]:
    """
    Executa a poda intratrilha online (L-Stopping) com janela causal de entropia:
    Quando a cadeia já ultrapassou L_min, possui resposta intermediária candidata válida
    e apresenta pico na média móvel de entropia (> tau_H) ou transição de overthinking,
    interrompe a geração naquele passo e retorna estatísticas dos passos limpos anteriores.
    """
    steps = json.loads(row["steps_json"])
    last_valid_ans = None
    clean_confs: List[float] = []
    clean_ents: List[float] = []
    ent_window: List[float] = []

    for st in steps:
        cum_tok = int(st["cumulative_tokens"])
        step_h = float(st["step_entropy"])
        step_c = float(st["step_confidence"])
        ent_window.append(step_h)
        if len(ent_window) > window:
            ent_window.pop(0)
        avg_h = sum(ent_window) / len(ent_window)
        is_over = bool(st.get("is_overthinking_transition", False))

        if (
            cum_tok >= l_min
            and last_valid_ans is not None
            and (avg_h > tau_h or step_h > tau_h * 1.15 or (is_over and step_h >= 0.85 * tau_h))
        ):
            conf_at_stop = float(np.mean(clean_confs)) if clean_confs else step_c
            ent_at_stop = float(np.mean(clean_ents)) if clean_ents else step_h
            return last_valid_ans, cum_tok, conf_at_stop, ent_at_stop, True

        if st.get("intermediate_answer"):
            last_valid_ans = str(st["intermediate_answer"])
        clean_confs.append(step_c)
        clean_ents.append(step_h)

    return (
        str(row["final_answer"]),
        int(row["total_tokens"]),
        float(row["raw_confidence"]),
        float(row["mean_entropy"]),
        False,
    )


def evaluate_method_on_questions(
    grouped_questions: Dict[str, pd.DataFrame],
    calib: Dict[str, Any],
    method_type: str,
    n_budget: int = 16,
    gamma_consensus: float = 0.78,
    lambda_len: float = 0.85,
) -> Dict[str, Any]:
    """
    Avalia um método de decodificação no conjunto de teste com orçamento máximo n_budget.
    """
    l_star = calib["l_star"]
    sigma_l = calib["sigma_l"]
    l_min = calib["l_min"]
    tau_h = calib["tau_h"]
    iso_cats: IsotonicRegression = calib["iso_cats"]
    iso_lcsc: IsotonicRegression = calib["iso_lcsc"]

    per_q_correct: List[int] = []
    per_q_tokens: List[int] = []
    per_q_samples: List[int] = []
    per_q_conf: List[float] = []
    per_q_diff: List[int] = []
    rescued_count = 0
    truncated_traces_count = 0

    for qid, df_q in grouped_questions.items():
        df_sub = df_q.iloc[:n_budget]
        gold = str(df_sub.iloc[0]["gold_answer"])
        diff = int(df_sub.iloc[0]["difficulty_level"])

        if method_type == "greedy":
            r0 = df_sub.iloc[0]
            pred = str(r0["final_answer"])
            toks = int(r0["total_tokens"])
            used_n = 1
            conf = float(r0["raw_confidence"])

        elif method_type == "standard_sc":
            votes: Dict[str, int] = defaultdict(int)
            confs_by_ans: Dict[str, List[float]] = defaultdict(list)
            toks = 0
            for _, row in df_sub.iterrows():
                ans = str(row["final_answer"])
                votes[ans] += 1
                confs_by_ans[ans].append(float(row["raw_confidence"]))
                toks += int(row["total_tokens"])
            pred = max(votes, key=votes.get)
            used_n = len(df_sub)
            vote_share = float(votes[pred] / max(1, used_n))
            mean_raw = float(np.mean(confs_by_ans[pred]))
            conf = float(np.clip(0.35 * vote_share + 0.65 * mean_raw, 0.15, 0.99))

        elif method_type == "wu_lfv":
            # Wu et al. (ICLR 2026) Length-Filtered Vote:
            # Gera TODAS as n_budget cadeias completas (custo de tokens = 100% do SC),
            # mas filtra na hora de votar apenas as cadeias próximas à janela ótima [l_window_low, l_window_high].
            toks = int(df_sub["total_tokens"].sum())
            used_n = len(df_sub)
            in_win = df_sub[
                (df_sub["total_tokens"] >= calib["l_window_low"])
                & (df_sub["total_tokens"] <= calib["l_window_high"])
            ]
            vote_pool = in_win if len(in_win) > 0 else df_sub
            votes: Dict[str, float] = defaultdict(float)
            for _, row in vote_pool.iterrows():
                dist_penalty = math.exp(-0.35 * abs(float(row["total_tokens"]) - l_star) / sigma_l)
                votes[str(row["final_answer"])] += dist_penalty
            pred = max(votes, key=votes.get)
            conf = float(np.clip(votes[pred] / max(1e-6, sum(votes.values())), 0.10, 0.98))

        elif method_type == "huang_cats":
            # Huang et al. (ICLR 2026) CaTS:
            # Faz Early Stopping no eixo N usando confiança calibrada, mas gera cada trilha até o fim L_i.
            weighted_votes: Dict[str, float] = defaultdict(float)
            confs_by_ans: Dict[str, List[float]] = defaultdict(list)
            toks = 0
            used_n = 0
            n_min_cats = min(7, n_budget)

            raw_c = df_sub["raw_confidence"].values
            cal_c = iso_cats.predict(raw_c)

            for idx_i, (_, row) in enumerate(df_sub.iterrows()):
                used_n += 1
                toks += int(row["total_tokens"])
                ans = str(row["final_answer"])
                c_i = float(cal_c[idx_i])
                weighted_votes[ans] += c_i
                confs_by_ans[ans].append(c_i)

                if used_n >= n_min_cats:
                    total_w = sum(weighted_votes.values())
                    top_ans = max(weighted_votes, key=weighted_votes.get)
                    if total_w > 0 and (weighted_votes[top_ans] / total_w) >= 0.68:
                        break

            pred = max(weighted_votes, key=weighted_votes.get)
            consensus_ratio = float(weighted_votes[pred] / max(1e-6, sum(weighted_votes.values())))
            mean_c_pred = float(np.mean(confs_by_ans[pred])) if confs_by_ans[pred] else 0.5
            conf = float(np.clip(0.52 * consensus_ratio + 0.48 * mean_c_pred, 0.15, 0.96))

        elif method_type in ("lc_sc", "lc_sc_no_lstop", "lc_sc_no_nstop", "lc_sc_no_len_penalty"):
            # Nosso método proposto LC-SC e suas 3 ablações
            use_l_stop = method_type != "lc_sc_no_lstop"
            use_n_stop = method_type != "lc_sc_no_nstop"
            eff_lambda = 0.0 if method_type == "lc_sc_no_len_penalty" else lambda_len

            weighted_votes: Dict[str, float] = defaultdict(float)
            c_hats_by_ans: Dict[str, List[float]] = defaultdict(list)
            toks = 0
            used_n = 0
            n_trunc_q = 0
            ents_q: List[float] = []
            n_min_lcsc = min(3, n_budget)

            for idx_i, (_, row) in enumerate(df_sub.iterrows()):
                used_n += 1
                if use_l_stop:
                    ans_i, used_tok_i, conf_raw_i, ent_i, was_trunc = apply_online_l_stopping(
                        row, l_min=l_min, tau_h=tau_h
                    )
                    if was_trunc:
                        truncated_traces_count += 1
                        n_trunc_q += 1
                        if ans_i == gold and str(row["final_answer"]) != gold:
                            rescued_count += 1
                else:
                    ans_i = str(row["final_answer"])
                    used_tok_i = int(row["total_tokens"])
                    conf_raw_i = float(row["raw_confidence"])
                    ent_i = float(row["mean_entropy"])
                    was_trunc = False

                toks += used_tok_i
                ents_q.append(ent_i)

                if "trace_clf_lcsc" in calib:
                    feat_i = np.array([[
                        conf_raw_i,
                        ent_i,
                        float(was_trunc),
                        used_tok_i / max(1.0, l_star),
                        max(0.0, used_tok_i - l_star) / max(1.0, sigma_l),
                    ]])
                    c_hat_i = float(calib["trace_clf_lcsc"].predict_proba(feat_i)[0, 1])
                else:
                    eff_feat = np.array([conf_raw_i * math.exp(-0.35 * max(0.0, ent_i - 0.35))])
                    c_hat_i = float(iso_lcsc.predict(eff_feat)[0])
                if use_l_stop and was_trunc:
                    dev_tokens = 0.0
                elif used_tok_i < l_min:
                    dev_tokens = 1.5 * (l_star - used_tok_i)
                else:
                    dev_tokens = max(0.0, used_tok_i - l_star)
                len_decay = math.exp(-eff_lambda * dev_tokens / sigma_l)
                tail_decay = 0.72 if idx_i >= 10 else 1.0
                w_i = c_hat_i * len_decay * tail_decay
                c_hats_by_ans[ans_i].append(c_hat_i * (0.75 + 0.25 * len_decay))

                weighted_votes[ans_i] += w_i

                if use_n_stop and used_n >= n_min_lcsc:
                    total_w = sum(weighted_votes.values())
                    top_ans = max(weighted_votes, key=weighted_votes.get)
                    top_c_mean = float(np.mean(c_hats_by_ans[top_ans])) if c_hats_by_ans[top_ans] else 0.5
                    if total_w > 0 and (weighted_votes[top_ans] / total_w) >= gamma_consensus and top_c_mean >= 0.72:
                        break

            pred = max(weighted_votes, key=weighted_votes.get)
            total_w = max(1e-6, sum(weighted_votes.values()))
            sorted_w = sorted(weighted_votes.values(), reverse=True)
            consensus_ratio = float(sorted_w[0] / total_w)
            margin_ratio = float((sorted_w[0] - (sorted_w[1] if len(sorted_w) > 1 else 0.0)) / total_w)
            mean_c_pred = float(np.mean(c_hats_by_ans[pred])) if c_hats_by_ans[pred] else 0.5
            if "q_clf_lcsc" in calib:
                q_feat = np.array([[
                    consensus_ratio,
                    margin_ratio,
                    mean_c_pred,
                    n_trunc_q / max(1, used_n),
                    used_n / float(max(1, n_budget)),
                    float(np.mean(ents_q)) if ents_q else 0.5,
                ]])
                conf = float(np.clip(calib["q_clf_lcsc"].predict_proba(q_feat)[0, 1], 0.02, 0.98))
            else:
                conf = float(np.clip(0.55 * consensus_ratio + 0.45 * mean_c_pred, 0.15, 0.97))

        else:
            raise ValueError(f"Método desconhecido: {method_type}")

        per_q_correct.append(int(pred == gold))
        per_q_tokens.append(int(toks))
        per_q_samples.append(int(used_n))
        per_q_conf.append(float(conf))
        per_q_diff.append(diff)

    acc_mean, acc_low, acc_high = bootstrap_ci(per_q_correct, n_boot=1000, seed=42)
    tok_mean, tok_low, tok_high = bootstrap_ci(per_q_tokens, n_boot=1000, seed=42)
    samp_mean, _, _ = bootstrap_ci(per_q_samples, n_boot=500, seed=42)

    ece_val, cal_bins = calculate_ece_and_bins(np.array(per_q_correct), np.array(per_q_conf))
    auc_pct = (
        round(float(roc_auc_score(per_q_correct, per_q_conf)) * 100.0, 2)
        if len(set(per_q_correct)) > 1 else None
    )
    brier_val = float(brier_score_loss(per_q_correct, np.clip(per_q_conf, 0.0, 1.0)))

    # Análise por fatia de dificuldade (Slice Analysis)
    slice_acc: Dict[str, float] = {}
    for slice_name, levels in [("Easy (L1-L2)", (1, 2)), ("Medium (L3)", (3,)), ("Hard (L4-L5)", (4, 5))]:
        vals = [c for c, d in zip(per_q_correct, per_q_diff) if d in levels]
        slice_acc[slice_name] = round(float(np.mean(vals) * 100.0), 2) if vals else 0.0

    return {
        "accuracy_pct": round(acc_mean * 100.0, 2),
        "ci95_low_pct": round(acc_low * 100.0, 2),
        "ci95_high_pct": round(acc_high * 100.0, 2),
        "avg_tokens": round(tok_mean, 1),
        "tokens_ci95": [round(tok_low, 1), round(tok_high, 1)],
        "avg_samples_used": round(samp_mean, 2),
        "ece_pct": round(ece_val * 100.0, 2),
        "brier_score": round(brier_val, 4),
        "roc_auc_pct": auc_pct,
        "truncated_traces": int(truncated_traces_count),
        "rescued_overcorrections": int(rescued_count),
        "slice_accuracy_pct": slice_acc,
        "calibration_bins": cal_bins,
    }


def run_dataset_benchmark(df: pd.DataFrame, ds_name: str) -> Dict[str, Any]:
    df_val = df[df["split"] == "val_calibration"].copy()
    df_test = df[df["split"] == "test"].copy()

    calib = fit_calibration_on_validation(df_val)

    grouped_val: Dict[str, pd.DataFrame] = {
        qid: group.sort_values("sample_idx") for qid, group in df_val.groupby("question_id", sort=False)
    }
    grouped_test: Dict[str, pd.DataFrame] = {
        qid: group.sort_values("sample_idx") for qid, group in df_test.groupby("question_id", sort=False)
    }

    # Ajustar calibrador logístico regularizado ao nível de questão na validação (zero data leakage)
    Xq_val, yq_val = [], []
    for qid, df_q in grouped_val.items():
        gold = str(df_q.iloc[0]["gold_answer"])
        w_votes: Dict[str, float] = defaultdict(float)
        c_by_a: Dict[str, List[float]] = defaultdict(list)
        used_n_val = 0
        n_trunc_val = 0
        ents_val: List[float] = []
        for idx_i, (_, row) in enumerate(df_q.iterrows()):
            used_n_val += 1
            ans_i, used_tok_i, conf_raw_i, ent_i, was_trunc = apply_online_l_stopping(
                row, l_min=calib["l_min"], tau_h=calib["tau_h"]
            )
            if was_trunc:
                n_trunc_val += 1
            ents_val.append(ent_i)
            feat_i = np.array([[
                conf_raw_i,
                ent_i,
                float(was_trunc),
                used_tok_i / max(1.0, calib["l_star"]),
                max(0.0, used_tok_i - calib["l_star"]) / max(1.0, calib["sigma_l"]),
            ]])
            c_hat_i = float(calib["trace_clf_lcsc"].predict_proba(feat_i)[0, 1])
            if was_trunc:
                dev_tokens = 0.0
            elif used_tok_i < calib["l_min"]:
                dev_tokens = 1.5 * (calib["l_star"] - used_tok_i)
            else:
                dev_tokens = max(0.0, used_tok_i - calib["l_star"])
            len_decay = math.exp(-0.85 * dev_tokens / calib["sigma_l"])
            tail_decay = 0.72 if idx_i >= 10 else 1.0
            w_votes[ans_i] += c_hat_i * len_decay * tail_decay
            c_by_a[ans_i].append(c_hat_i * (0.75 + 0.25 * len_decay))
            if used_n_val >= 3:
                tot_w = sum(w_votes.values())
                top_a = max(w_votes, key=w_votes.get)
                top_c = float(np.mean(c_by_a[top_a]))
                if tot_w > 0 and (w_votes[top_a] / tot_w) >= 0.78 and top_c >= 0.72:
                    break
        pred = max(w_votes, key=w_votes.get)
        tot_w = max(1e-6, sum(w_votes.values()))
        sw = sorted(w_votes.values(), reverse=True)
        cr = float(sw[0] / tot_w)
        mr = float((sw[0] - (sw[1] if len(sw) > 1 else 0.0)) / tot_w)
        mc = float(np.mean(c_by_a[pred])) if c_by_a[pred] else 0.5
        Xq_val.append([
            cr,
            mr,
            mc,
            n_trunc_val / max(1, used_n_val),
            used_n_val / 16.0,
            float(np.mean(ents_val)) if ents_val else 0.5,
        ])
        yq_val.append(int(pred == gold))

    q_clf_lcsc = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
    q_clf_lcsc.fit(np.asarray(Xq_val, dtype=float), np.asarray(yq_val, dtype=int))
    calib["q_clf_lcsc"] = q_clf_lcsc

    main_methods_spec = [
        ("Greedy CoT (N=1)", "greedy", 1),
        ("Standard SC (Aula 11)", "standard_sc", 16),
        ("Length-Filtered Vote (Wu et al., 2026)", "wu_lfv", 16),
        ("CaTS (Huang et al., 2026)", "huang_cats", 16),
        ("LC-SC (Proposto)", "lc_sc", 16),
    ]

    methods_out: Dict[str, Any] = {}
    for label, m_type, n_b in main_methods_spec:
        res = evaluate_method_on_questions(grouped_test, calib, method_type=m_type, n_budget=n_b)
        methods_out[label] = res

    sc_tokens = methods_out["Standard SC (Aula 11)"]["avg_tokens"]
    for label, res in methods_out.items():
        res["token_saving_vs_sc_pct"] = round((1.0 - res["avg_tokens"] / sc_tokens) * 100.0, 1)

    # Curvas de Pareto variando N_max em {1, 2, 4, 8, 12, 16}
    budgets = [1, 2, 4, 8, 12, 16]
    pareto_specs = [
        ("Standard SC (Aula 11)", "standard_sc"),
        ("Length-Filtered Vote (Wu et al., 2026)", "wu_lfv"),
        ("CaTS (Huang et al., 2026)", "huang_cats"),
        ("LC-SC (Proposto - N x L Online)", "lc_sc"),
    ]
    pareto_curves: Dict[str, List[Dict[str, Any]]] = {}
    for label, m_type in pareto_specs:
        pts = []
        for b in budgets:
            r_b = evaluate_method_on_questions(grouped_test, calib, method_type=m_type, n_budget=b)
            pts.append({
                "n_budget": b,
                "accuracy_pct": r_b["accuracy_pct"],
                "avg_tokens": r_b["avg_tokens"],
                "avg_samples_used": r_b["avg_samples_used"],
            })
        pareto_curves[label] = pts

    # Estudo de Ablação em N_max = 16
    ablation_specs = [
        ("LC-SC Completo (Proposto)", "lc_sc"),
        ("s/ L-Stopping (Apenas Peso L + N-Stop)", "lc_sc_no_lstop"),
        ("s/ N-Stopping (Apenas L-Stop + Peso L)", "lc_sc_no_nstop"),
        ("s/ Penalização L (lambda=0)", "lc_sc_no_len_penalty"),
        ("CaTS (Huang et al., 2026)", "huang_cats"),
        ("Standard SC (Aula 11)", "standard_sc"),
    ]
    ablation_out: Dict[str, Any] = {}
    for label, m_type in ablation_specs:
        r_ab = evaluate_method_on_questions(grouped_test, calib, method_type=m_type, n_budget=16)
        r_ab["token_saving_vs_sc_pct"] = round((1.0 - r_ab["avg_tokens"] / sc_tokens) * 100.0, 1)
        ablation_out[label] = r_ab

    return {
        "dataset": ds_name,
        "trace_origin": "synthetic_gold_conditioned_anchor_expansion",
        "n_val_questions": int(df_val["question_id"].nunique()),
        "n_test_questions": int(df_test["question_id"].nunique()),
        "calibration_params": {
            "l_star": calib["l_star"],
            "sigma_l": calib["sigma_l"],
            "l_min": calib["l_min"],
            "tau_h": calib["tau_h"],
        },
        "methods": methods_out,
        "pareto_curves": pareto_curves,
        "ablation": ablation_out,
        "calibration_bins": {
            "raw_uncalibrated": methods_out["Standard SC (Aula 11)"]["calibration_bins"],
            "cats_calibrated": methods_out["CaTS (Huang et al., 2026)"]["calibration_bins"],
            "lcsc_calibrated": methods_out["LC-SC (Proposto)"]["calibration_bins"],
        },
        "calibration_summary": {
            "raw_uncalibrated": {
                "ece_pct": methods_out["Standard SC (Aula 11)"]["ece_pct"],
                "brier": methods_out["Standard SC (Aula 11)"]["brier_score"],
                "auc_pct": methods_out["Standard SC (Aula 11)"]["roc_auc_pct"],
            },
            "cats_calibrated": {
                "ece_pct": methods_out["CaTS (Huang et al., 2026)"]["ece_pct"],
                "brier": methods_out["CaTS (Huang et al., 2026)"]["brier_score"],
                "auc_pct": methods_out["CaTS (Huang et al., 2026)"]["roc_auc_pct"],
            },
            "lcsc_calibrated": {
                "ece_pct": methods_out["LC-SC (Proposto)"]["ece_pct"],
                "brier": methods_out["LC-SC (Proposto)"]["brier_score"],
                "auc_pct": methods_out["LC-SC (Proposto)"]["roc_auc_pct"],
            },
        },
    }


def main() -> None:
    df_gsm = pd.read_parquet(RESULTS_DIR / "traces_gsm8k.parquet")
    df_math = pd.read_parquet(RESULTS_DIR / "traces_math500.parquet")

    print("[Benchmark] Executando avaliação com divisão estrita Validação -> Teste...")
    gsm_res = run_dataset_benchmark(df_gsm, "GSM8K")
    math_res = run_dataset_benchmark(df_math, "MATH-500")

    full_out = {"gsm8k": gsm_res, "math500": math_res}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_out, f, indent=2, ensure_ascii=False)

    # Salvar tabela resumo CSV
    rows = []
    for ds_key, ds_obj in [("GSM8K", gsm_res), ("MATH-500", math_res)]:
        for m_name, m_stats in ds_obj["methods"].items():
            rows.append({
                "Dataset": ds_key,
                "Method": m_name,
                "Accuracy_Pct": m_stats["accuracy_pct"],
                "CI95_Low": m_stats["ci95_low_pct"],
                "CI95_High": m_stats["ci95_high_pct"],
                "Avg_Tokens": m_stats["avg_tokens"],
                "Token_Saving_Pct": m_stats["token_saving_vs_sc_pct"],
                "Avg_Samples_N": m_stats["avg_samples_used"],
                "ECE_Pct": m_stats["ece_pct"],
                "ROC_AUC_Pct": m_stats["roc_auc_pct"],
            })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(OUT_DIR / "benchmark_summary_table.csv", index=False)
    print("\n=== REPLAY DO SIMULADOR NAS QUESTÕES DE TESTE ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
