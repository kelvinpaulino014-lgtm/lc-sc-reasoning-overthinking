#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 03b (Phase 3 - Part B): Real Online GPU Inference Benchmark
on RTX 3050 6GB using `Qwen/Qwen2.5-1.5B-Instruct` (FP16) and
`EntropyNeuralStoppingCriteria` (transformers.StoppingCriteria).

Evaluates Greedy CoT, Standard SC (N=8), CaTS (N_max=8), and LC-SC Phase 3 (N_max=8)
on a stratified `test` batch of GSM8K and MATH-500, measuring real wall-clock latency,
exact generated tokens, peak VRAM (MB), throughput (tok/s), accuracy, ECE, Brier, and ROC-AUC.
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.entropy_tracker import bootstrap_ci, compute_ece
from src.models.stopping_policy import EntropyNeuralStoppingCriteria, NeuralStoppingPolicy
from src.utils.answer_extractor import extract_answer, is_equivalent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "phase3_deep_training"
CKPT_DIR = OUT_DIR / "checkpoints"


def log_msg(log_path: Path, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sample_stratified_test(df_splits: pd.DataFrame, n_total: int = 30, seed: int = 42) -> pd.DataFrame:
    df_test = df_splits[df_splits["split"] == "test"].copy()
    if n_total >= len(df_test):
        return df_test.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    counts = df_test["difficulty_level"].value_counts(normalize=True)
    remaining = n_total
    levels = sorted(counts.index.tolist())
    for i, lvl in enumerate(levels):
        sub = df_test[df_test["difficulty_level"] == lvl]
        k = remaining if i == len(levels) - 1 else max(1, int(round(counts[lvl] * n_total)))
        k = min(k, len(sub), remaining)
        if k > 0:
            idx = rng.choice(len(sub), size=k, replace=False)
            parts.append(sub.iloc[idx])
            remaining -= k
    return pd.concat(parts, ignore_index=True)


def build_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a precise math assistant. Solve the problem step by step and put your final numeric answer inside \\boxed{}.",
        },
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_batch_traces(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
) -> Tuple[List[str], List[int], List[float], float]:
    """Generates `num_samples` traces on GPU, returning texts, exact token counts, raw confs, and elapsed sec."""
    prompt_len = prompt_ids.shape[1]
    t0 = time.perf_counter()
    gen_kwargs: Dict[str, Any] = {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_samples,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
    if stopping_criteria is not None:
        gen_kwargs["stopping_criteria"] = stopping_criteria

    out = model.generate(**gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    seqs = out.sequences
    scores = out.scores
    eos_id = tokenizer.eos_token_id

    texts: List[str] = []
    tok_counts: List[int] = []
    raw_confs: List[float] = []

    # Compute mean chosen token probability per sequence
    n_steps = len(scores)
    for b in range(seqs.shape[0]):
        gen_ids = seqs[b, prompt_len:]
        # Count exact non-pad/non-post-EOS tokens
        valid_len = 0
        for tid in gen_ids.tolist():
            valid_len += 1
            if eos_id is not None and tid == eos_id:
                break
        valid_len = max(1, min(valid_len, n_steps))
        tok_counts.append(valid_len)
        texts.append(tokenizer.decode(gen_ids[:valid_len], skip_special_tokens=True))

        logprobs = []
        for t in range(valid_len):
            lp_t = F.log_softmax(scores[t][b].float(), dim=-1)
            logprobs.append(float(lp_t[gen_ids[t]].item()))
        mean_lp = float(np.mean(logprobs)) if logprobs else -0.5
        raw_confs.append(float(np.clip(math.exp(mean_lp), 0.05, 0.99)))

    return texts, tok_counts, raw_confs, elapsed


def evaluate_dataset_gpu(
    ds_key: str,
    ds_name: str,
    model: Any,
    tokenizer: Any,
    n_questions: int,
    max_new_tokens: int,
    log_path: Path,
) -> Dict[str, Any]:
    df_splits = pd.read_parquet(DATA_DIR / f"{ds_key}_splits.parquet")
    df_traces = pd.read_parquet(RESULTS_DIR / f"traces_{ds_key}.parquet")
    df_val = df_traces[df_traces["split"] == "val_calibration"]

    # Fit CaTS isotonic calibrator strictly on val_calibration
    iso_cats = IsotonicRegression(y_min=0.05, y_max=0.98, out_of_bounds="clip")
    iso_cats.fit(df_val["raw_confidence"].values, df_val["is_correct_full"].values)

    # Load Phase 3 NeuralStoppingPolicy checkpoint
    ckpt = torch.load(CKPT_DIR / f"{ds_key}_best_policy.pt", map_location="cuda", weights_only=False)
    policy = NeuralStoppingPolicy(input_dim=6, hidden_dim=16, dropout=0.1).to("cuda").eval()
    policy.load_state_dict(ckpt["state_dict"])
    l_star = float(ckpt["l_star"])
    l_min = int(ckpt["l_min"])
    stop_threshold = float(ckpt["stop_threshold"])

    df_test_batch = sample_stratified_test(df_splits, n_total=n_questions, seed=42)
    log_msg(log_path, f"[{ds_name}] Loaded {len(df_test_batch)} stratified test questions (l_star={l_star:.1f}, l_min={l_min})")

    methods_data: Dict[str, Dict[str, List[Any]]] = {
        m: {"correct": [], "tokens": [], "samples": [], "latency": [], "conf": []}
        for m in ("Greedy CoT (N=1)", "Standard SC (N=8)", "CaTS (Huang et al., 2026)", "LC-SC Phase 3 (Neural GPU)")
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for q_idx, row in df_test_batch.iterrows():
        gold = str(row["answer_gold"])
        prompt_text = build_prompt(tokenizer, str(row["question"]))
        enc = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        p_ids = enc["input_ids"]
        p_mask = enc["attention_mask"]
        p_len = p_ids.shape[1]

        # 1. Greedy CoT (N=1)
        texts_g, toks_g, confs_g, lat_g = generate_batch_traces(
            model, tokenizer, p_ids, p_mask, num_samples=1, max_new_tokens=max_new_tokens, do_sample=False
        )
        ans_g = extract_answer(texts_g[0], benchmark=ds_key) or ""
        methods_data["Greedy CoT (N=1)"]["correct"].append(int(is_equivalent(ans_g, gold)))
        methods_data["Greedy CoT (N=1)"]["tokens"].append(toks_g[0])
        methods_data["Greedy CoT (N=1)"]["samples"].append(1)
        methods_data["Greedy CoT (N=1)"]["latency"].append(lat_g)
        methods_data["Greedy CoT (N=1)"]["conf"].append(confs_g[0])

        # 2. Standard SC (N=8) -> two batches of 4 for RTX 3050 6GB VRAM safety
        sc_texts, sc_toks, sc_confs = [], [], []
        sc_lat = 0.0
        for _ in range(2):
            t_b, k_b, c_b, l_b = generate_batch_traces(
                model, tokenizer, p_ids, p_mask, num_samples=4, max_new_tokens=max_new_tokens, do_sample=True
            )
            sc_texts.extend(t_b)
            sc_toks.extend(k_b)
            sc_confs.extend(c_b)
            sc_lat += l_b
        votes_sc: Dict[str, int] = defaultdict(int)
        for txt in sc_texts:
            a = extract_answer(txt, benchmark=ds_key) or "__NONE__"
            votes_sc[a] += 1
        pred_sc = max(votes_sc, key=votes_sc.get)
        conf_sc = float(np.clip(0.35 * (votes_sc[pred_sc] / 8.0) + 0.65 * float(np.mean(sc_confs)), 0.15, 0.99))
        methods_data["Standard SC (N=8)"]["correct"].append(int(is_equivalent(pred_sc, gold)))
        methods_data["Standard SC (N=8)"]["tokens"].append(int(sum(sc_toks)))
        methods_data["Standard SC (N=8)"]["samples"].append(8)
        methods_data["Standard SC (N=8)"]["latency"].append(sc_lat)
        methods_data["Standard SC (N=8)"]["conf"].append(conf_sc)

        # 3. CaTS (Huang et al., 2026) -> adaptive N in {4, 8} without L-Stopping
        t_c1, k_c1, c_c1, l_c1 = generate_batch_traces(
            model, tokenizer, p_ids, p_mask, num_samples=4, max_new_tokens=max_new_tokens, do_sample=True
        )
        cal_c1 = iso_cats.predict(np.array(c_c1))
        w_cats: Dict[str, float] = defaultdict(float)
        for txt, cc in zip(t_c1, cal_c1):
            w_cats[extract_answer(txt, benchmark=ds_key) or "__NONE__"] += float(cc)
        tot_wc = max(1e-6, sum(w_cats.values()))
        top_cats = max(w_cats, key=w_cats.get)
        cats_toks, cats_n, cats_lat = int(sum(k_c1)), 4, l_c1
        if (w_cats[top_cats] / tot_wc) < 0.72:
            t_c2, k_c2, c_c2, l_c2 = generate_batch_traces(
                model, tokenizer, p_ids, p_mask, num_samples=4, max_new_tokens=max_new_tokens, do_sample=True
            )
            cal_c2 = iso_cats.predict(np.array(c_c2))
            for txt, cc in zip(t_c2, cal_c2):
                w_cats[extract_answer(txt, benchmark=ds_key) or "__NONE__"] += float(cc)
            cats_toks += int(sum(k_c2))
            cats_n = 8
            cats_lat += l_c2
            tot_wc = max(1e-6, sum(w_cats.values()))
            top_cats = max(w_cats, key=w_cats.get)
        conf_cats = float(np.clip(w_cats[top_cats] / tot_wc, 0.15, 0.96))
        methods_data["CaTS (Huang et al., 2026)"]["correct"].append(int(is_equivalent(top_cats, gold)))
        methods_data["CaTS (Huang et al., 2026)"]["tokens"].append(cats_toks)
        methods_data["CaTS (Huang et al., 2026)"]["samples"].append(cats_n)
        methods_data["CaTS (Huang et al., 2026)"]["latency"].append(cats_lat)
        methods_data["CaTS (Huang et al., 2026)"]["conf"].append(conf_cats)

        # 4. LC-SC Phase 3 (Real Online GPU Neural StoppingCriteria + Adaptive N in {4, 8})
        w_lcsc: Dict[str, float] = defaultdict(float)
        c_lcsc_by_ans: Dict[str, List[float]] = defaultdict(list)
        lcsc_toks, lcsc_n, lcsc_lat = 0, 0, 0.0
        for batch_n in (4, 4):
            criteria = EntropyNeuralStoppingCriteria(
                policy=policy,
                tokenizer=tokenizer,
                prompt_len=p_len,
                l_star=l_star,
                l_min=l_min,
                stop_threshold=stop_threshold,
                step_stride=24,
                device="cuda",
            )
            t_l, k_l, c_l, l_l = generate_batch_traces(
                model,
                tokenizer,
                p_ids,
                p_mask,
                num_samples=batch_n,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                stopping_criteria=StoppingCriteriaList([criteria]),
            )
            lcsc_toks += int(sum(k_l))
            lcsc_n += batch_n
            lcsc_lat += l_l
            for b_i, (txt, tok_cnt, raw_c) in enumerate(zip(t_l, k_l, c_l)):
                ans_i = extract_answer(txt, benchmark=ds_key)
                if ans_i is None:
                    continue
                corr_p = criteria.last_corr_probs[b_i] if criteria.initialized else raw_c
                c_eff = 0.65 * corr_p + 0.35 * raw_c
                dev = max(0.0, float(tok_cnt) - l_star)
                w_i = c_eff * math.exp(-0.85 * dev / max(35.0, 0.35 * l_star))
                w_lcsc[ans_i] += w_i
                c_lcsc_by_ans[ans_i].append(c_eff)

            if w_lcsc:
                tot_wl = max(1e-6, sum(w_lcsc.values()))
                top_lcsc = max(w_lcsc, key=w_lcsc.get)
                lead_ratio = w_lcsc[top_lcsc] / tot_wl
                top_c_mean = float(np.mean(c_lcsc_by_ans[top_lcsc]))
                if lcsc_n >= 4 and lead_ratio >= 0.72 and top_c_mean >= 0.55:
                    break

        if w_lcsc:
            tot_wl = max(1e-6, sum(w_lcsc.values()))
            top_lcsc = max(w_lcsc, key=w_lcsc.get)
            lead_ratio = w_lcsc[top_lcsc] / tot_wl
            top_c_mean = float(np.mean(c_lcsc_by_ans[top_lcsc]))
            conf_lcsc = float(np.clip(0.55 * lead_ratio + 0.45 * top_c_mean, 0.10, 0.98))
        else:
            top_lcsc = "__NONE__"
            conf_lcsc = 0.15
        methods_data["LC-SC Phase 3 (Neural GPU)"]["correct"].append(int(is_equivalent(top_lcsc, gold)))
        methods_data["LC-SC Phase 3 (Neural GPU)"]["tokens"].append(lcsc_toks)
        methods_data["LC-SC Phase 3 (Neural GPU)"]["samples"].append(lcsc_n)
        methods_data["LC-SC Phase 3 (Neural GPU)"]["latency"].append(lcsc_lat)
        methods_data["LC-SC Phase 3 (Neural GPU)"]["conf"].append(conf_lcsc)

        if (q_idx + 1) % 5 == 0 or (q_idx + 1) == len(df_test_batch):
            log_msg(
                log_path,
                f"  [{ds_name}] {q_idx+1}/{len(df_test_batch)} | "
                f"SC Acc={np.mean(methods_data['Standard SC (N=8)']['correct'])*100:.1f}% ({np.mean(methods_data['Standard SC (N=8)']['latency']):.1f}s) | "
                f"LC-SC P3 Acc={np.mean(methods_data['LC-SC Phase 3 (Neural GPU)']['correct'])*100:.1f}% ({np.mean(methods_data['LC-SC Phase 3 (Neural GPU)']['latency']):.1f}s)",
            )

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0**2) if torch.cuda.is_available() else 0.0

    summary_out: Dict[str, Any] = {"peak_vram_mb": round(peak_vram_mb, 1), "n_questions": len(df_test_batch), "methods": {}}
    sc_avg_tok = float(np.mean(methods_data["Standard SC (N=8)"]["tokens"]))
    sc_avg_lat = float(np.mean(methods_data["Standard SC (N=8)"]["latency"]))

    for m_name, d in methods_data.items():
        acc_m, acc_lo, acc_hi = bootstrap_ci(d["correct"], n_boot=1000, seed=42)
        tok_m, _, _ = bootstrap_ci(d["tokens"], n_boot=1000, seed=42)
        lat_m = float(np.mean(d["latency"]))
        y_true = np.asarray(d["correct"], dtype=float)
        y_prob = np.clip(np.asarray(d["conf"], dtype=float), 0.01, 0.99)
        ece_val = compute_ece(y_true, y_prob, n_bins=8)
        brier_val = float(brier_score_loss(y_true, y_prob))
        try:
            auc_val = float(roc_auc_score(y_true, y_prob)) * 100.0 if len(np.unique(y_true)) > 1 else 50.0
        except Exception:
            auc_val = 50.0

        summary_out["methods"][m_name] = {
            "accuracy_pct": round(acc_m * 100.0, 2),
            "ci95_low_pct": round(acc_lo * 100.0, 2),
            "ci95_high_pct": round(acc_hi * 100.0, 2),
            "avg_tokens": round(tok_m, 1),
            "token_saving_vs_sc_pct": round((1.0 - tok_m / max(1.0, sc_avg_tok)) * 100.0, 1),
            "avg_samples_used": round(float(np.mean(d["samples"])), 2),
            "avg_latency_sec": round(lat_m, 2),
            "speedup_vs_sc": round(sc_avg_lat / max(0.01, lat_m), 2),
            "throughput_tok_per_sec": round(tok_m / max(0.01, lat_m), 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "ece_pct": round(ece_val * 100.0, 2),
            "brier_score": round(brier_val, 4),
            "roc_auc_pct": round(auc_val, 2),
        }
    return summary_out


def plot_gpu_latency_pareto(res_all: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {
        "Greedy CoT (N=1)": "#777777",
        "Standard SC (N=8)": "#cc3311",
        "CaTS (Huang et al., 2026)": "#ee7733",
        "LC-SC Phase 3 (Neural GPU)": "#004488",
    }
    markers = {
        "Greedy CoT (N=1)": "s",
        "Standard SC (N=8)": "o",
        "CaTS (Huang et al., 2026)": "^",
        "LC-SC Phase 3 (Neural GPU)": "*",
    }
    for ax, ds_key, title in [
        (axes[0], "gsm8k", "GSM8K — Real Online GPU (RTX 3050 6GB)"),
        (axes[1], "math500", "MATH-500 — Real Online GPU (RTX 3050 6GB)"),
    ]:
        methods = res_all[ds_key]["methods"]
        for m_name, st in methods.items():
            ms = 14 if "LC-SC" in m_name else 8
            ax.scatter(
                st["avg_latency_sec"],
                st["accuracy_pct"],
                color=colors[m_name],
                marker=markers[m_name],
                s=ms**2,
                label=f"{m_name} ({st['avg_latency_sec']}s, {st['accuracy_pct']}%)",
                zorder=4 if "LC-SC" in m_name else 3,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Wall-Clock Latency per Question (sec)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=8.0)

    fig.suptitle(
        "Phase 3 Real Online GPU Benchmark (Qwen2.5-1.5B-Instruct FP16): Accuracy vs. Wall-Clock Latency",
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(OUT_DIR / "fig_phase3_gpu_latency_pareto.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_phase3_gpu_latency_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-questions", type=int, default=20, help="Stratified test questions per dataset")
    parser.add_argument("--max-new-tokens", type=int, default=384, help="Max new tokens per trace")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "online_gpu_progress.log"
    if log_path.exists():
        log_path.unlink()

    log_msg(log_path, f"Loading Qwen/Qwen2.5-1.5B-Instruct (FP16) on CUDA (n_questions={args.n_questions}/dataset)...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()

    gsm_res = evaluate_dataset_gpu("gsm8k", "GSM8K", model, tokenizer, args.n_questions, args.max_new_tokens, log_path)
    math_res = evaluate_dataset_gpu("math500", "MATH-500", model, tokenizer, args.n_questions, args.max_new_tokens, log_path)

    full_out = {"gsm8k": gsm_res, "math500": math_res}
    with open(OUT_DIR / "online_gpu_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(full_out, f, indent=2, ensure_ascii=False)

    rows = []
    for ds_label, ds_obj in [("GSM8K", gsm_res), ("MATH-500", math_res)]:
        for m_name, m_st in ds_obj["methods"].items():
            rows.append({"Dataset": ds_label, "Method": m_name, **m_st})
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(OUT_DIR / "online_gpu_benchmark_summary.csv", index=False)
    plot_gpu_latency_pareto(full_out)
    log_msg(log_path, "Phase 3 Part B Real Online GPU Benchmark COMPLETE!")
    print("\n" + df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
