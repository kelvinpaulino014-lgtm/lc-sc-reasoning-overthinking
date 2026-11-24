#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 Real Online GPU Benchmark (`06b_eval_phase4_online.py`) on RTX 3050 6GB.
Follows `docs/08_Protocolo_Fase4.md`:
- Evaluates the first 48 `holdout` questions per dataset in frozen manifest order.
- Compares 5 online methods: `sc4_4x1`, `sc4_2x2`, `consensus2`, `logistic`, `gru`.
- Uses `RealPolicyStoppingCriteria` (causal 24-token windows via `make_window` + `FittedPolicy`),
  paired question/batch seeds, one warmup question per method, counterbalanced order,
  and GPU synchronization for exact end-to-end wall-clock latency and Peak VRAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.entropy_tracker import bootstrap_ci, compute_ece
from src.models.real_trace_policy import (
    MAX_TOKENS,
    STRIDE,
    FittedPolicy,
    Window,
    complete_answer,
    consensus_after_two,
    load_policy,
    make_window,
    vote,
)
from src.utils.answer_extractor import is_equivalent

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PROMPT = "You are a precise math assistant. Solve the problem step by step and put your final answer inside \\boxed{}."
GEN_CFG = dict(max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.7, top_p=0.95, top_k=20, repetition_penalty=1.1)
OUT_DIR = ROOT / "results" / "phase4"
POLICY_DIR = OUT_DIR / "policy"


def batch_seed(dataset: str, qid: str, batch_idx: int) -> int:
    key = f"phase4-online-v1|{dataset}|{qid}|b{batch_idx}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


class RealPolicyStoppingCriteria(StoppingCriteria):
    """Causal 24-token-window StoppingCriteria backed by `FittedPolicy`."""

    def __init__(
        self,
        policy: Optional[FittedPolicy],
        family: Optional[str],
        tokenizer: Any,
        prompt_len: int,
    ) -> None:
        self.policy = policy
        self.family = family
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.initialized = False
        self.stopped: List[bool] = []
        self.stop_reason: List[str] = []
        self.stop_tokens: List[int] = []
        self.ent_buf: List[List[float]] = []
        self.lp_buf: List[List[float]] = []
        self.windows: List[List[Window]] = []

    def _init(self, bsz: int) -> None:
        self.stopped = [False] * bsz
        self.stop_reason = ["eos_or_cap"] * bsz
        self.stop_tokens = [0] * bsz
        self.ent_buf = [[] for _ in range(bsz)]
        self.lp_buf = [[] for _ in range(bsz)]
        self.windows = [[] for _ in range(bsz)]
        self.initialized = True

    @torch.inference_mode()
    def __call__(self, input_ids: torch.LongTensor, scores: Tuple[torch.FloatTensor, ...], **kwargs: Any) -> torch.BoolTensor:
        bsz = input_ids.shape[0]
        if not self.initialized:
            self._init(bsz)
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len <= 0 or not scores or self.policy is None or self.family is None:
            return torch.tensor(self.stopped, dtype=torch.bool, device=input_ids.device)

        logits_t = scores[-1].float()
        log_p = F.log_softmax(logits_t, dim=-1)
        probs = log_p.exp()
        ent_t = torch.nan_to_num(torch.special.entr(probs).sum(dim=-1), nan=0.0, posinf=2.0, neginf=0.0)
        picked = input_ids[:, -1].unsqueeze(-1)
        lp_t = log_p.gather(-1, picked).squeeze(-1)

        for b in range(bsz):
            if self.stopped[b]:
                continue
            tok_id = int(input_ids[b, -1].item())
            lp_val = float(lp_t[b].item())
            if tok_id in (self.eos_id, self.pad_id) or not np.isfinite(lp_val):
                self.stopped[b] = True
                self.stop_tokens[b] = gen_len
                continue
            self.ent_buf[b].append(float(ent_t[b].item()))
            self.lp_buf[b].append(lp_val)
            if len(self.ent_buf[b]) == STRIDE or gen_len == MAX_TOKENS:
                prev_w = self.windows[b][-1] if self.windows[b] else None
                prefix_txt = self.tokenizer.decode(input_ids[b, self.prompt_len :], skip_special_tokens=True)
                w = make_window(prefix_txt, self.ent_buf[b], self.lp_buf[b], gen_len, prev_w)
                self.windows[b].append(w)
                self.ent_buf[b].clear()
                self.lp_buf[b].clear()
                p_arr = self.policy.predict_windows(self.windows[b], self.family)
                if self.policy.should_stop(w, p_arr[-1], self.family):
                    self.stopped[b] = True
                    self.stop_reason[b] = f"{self.family}_stop"
                    self.stop_tokens[b] = gen_len

        return torch.tensor(self.stopped, dtype=torch.bool, device=input_ids.device)


@torch.inference_mode()
def run_one_method_on_question(
    row: Dict[str, Any],
    gold: str,
    method: str,
    model: Any,
    tok: Any,
    policy: FittedPolicy,
) -> Dict[str, Any]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": row["question"].strip()}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    p_len = ids.shape[1]
    mask = torch.ones_like(ids)

    batch_sizes = (4,) if method == "sc4_4x1" else (2, 2)
    family = method if method in ("logistic", "gru") else None
    pol_arg = policy if family is not None else None

    answers: List[Optional[str]] = []
    tokens_used: List[int] = []
    reasons: List[str] = []

    for b_idx, n_ret in enumerate(batch_sizes):
        torch.manual_seed(batch_seed(row["dataset"], row["question_id"], b_idx))
        crit = RealPolicyStoppingCriteria(pol_arg, family, tok, p_len)
        out = model.generate(
            input_ids=ids,
            attention_mask=mask,
            num_return_sequences=n_ret,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            output_scores=family is not None,
            return_dict_in_generate=True,
            stopping_criteria=StoppingCriteriaList([crit]) if family is not None else None,
            **GEN_CFG,
        )
        seqs = out.sequences[:, p_len:].tolist()
        for i, seq in enumerate(seqs):
            eos = tok.eos_token_id in seq
            n_tok = seq.index(tok.eos_token_id) + 1 if eos else len(seq)
            if crit.initialized and crit.stopped[i]:
                n_tok = min(n_tok, crit.stop_tokens[i])
                reasons.append(crit.stop_reason[i])
            elif not eos and n_tok >= MAX_TOKENS:
                reasons.append("budget_cap")
            else:
                reasons.append("eos")
            txt = tok.decode(seq[:n_tok], skip_special_tokens=True)
            answers.append(complete_answer(txt))
            tokens_used.append(n_tok)

        if method != "sc4_2x2" and method != "sc4_4x1" and b_idx == 0 and consensus_after_two(answers):
            break

    pred, cov = vote(answers)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "question_id": row["question_id"],
        "dataset": row["dataset"],
        "method": method,
        "correct": int(is_equivalent(pred, gold)),
        "pred": pred,
        "confidence": cov,
        "tokens": int(sum(tokens_used)),
        "n_traces": len(answers),
        "elapsed_sec": elapsed,
        "reasons": reasons,
        "no_explicit_answer": int(pred is None),
        "capped_by_budget": int(any(r == "budget_cap" for r in reasons)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-holdout", type=int, default=48, help="First N holdout questions per dataset")
    args = ap.parse_args()

    inputs = [json.loads(l) for l in (OUT_DIR / "inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = {r["question_id"]: r["answer_gold"] for r in map(json.loads, (OUT_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines()) if r}

    holdout_by_ds: Dict[str, List[Dict[str, Any]]] = {"gsm8k": [], "math500": []}
    for r in inputs:
        if r["split"] == "holdout" and len(holdout_by_ds[r["dataset"]]) < args.n_holdout:
            holdout_by_ds[r["dataset"]].append(r)

    warmup_q = next(r for r in inputs if r["split"] == "calibration")

    model_path = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda", local_files_only=True)
    model.eval()
    policy = load_policy(POLICY_DIR)

    methods = ["sc4_2x2", "sc4_4x1", "consensus2", "logistic", "gru"]
    for m in methods:
        _ = run_one_method_on_question(warmup_q, labels[warmup_q["question_id"]], m, model, tok, policy)

    torch.cuda.reset_peak_memory_stats()
    all_records: List[Dict[str, Any]] = []
    for ds_key in ("gsm8k", "math500"):
        qs = holdout_by_ds[ds_key]
        for idx, row in enumerate(qs):
            order = methods[idx % len(methods) :] + methods[: idx % len(methods)]
            for m in order:
                rec = run_one_method_on_question(row, labels[row["question_id"]], m, model, tok, policy)
                all_records.append(rec)
            if (idx + 1) % 8 == 0 or (idx + 1) == len(qs):
                print(f"[{ds_key}] {idx+1}/{len(qs)} online GPU questions complete", flush=True)

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
    (OUT_DIR / "online_gpu_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_records) + "\n", encoding="utf-8"
    )

    df = pd.DataFrame(all_records)
    summary_rows: List[Dict[str, Any]] = []
    out_json: Dict[str, Any] = {"peak_vram_mb": round(peak_vram_mb, 1), "n_per_dataset": args.n_holdout, "datasets": {}}

    for ds_key in ("gsm8k", "math500", "overall"):
        sub = df if ds_key == "overall" else df[df["dataset"] == ds_key]
        sc_lat = float(sub[sub["method"] == "sc4_2x2"]["elapsed_sec"].mean())
        sc_tok = float(sub[sub["method"] == "sc4_2x2"]["tokens"].mean())
        ds_out: Dict[str, Any] = {}
        for m in methods:
            sm = sub[sub["method"] == m]
            acc_m, acc_lo, acc_hi = bootstrap_ci(sm["correct"].tolist(), n_boot=2000, seed=42)
            lat_m, lat_lo, lat_hi = bootstrap_ci(sm["elapsed_sec"].tolist(), n_boot=2000, seed=42)
            tok_m = float(sm["tokens"].mean())
            ece_v = compute_ece(sm["correct"].values.astype(float), np.clip(sm["confidence"].values, 0.01, 0.99), n_bins=8)
            ds_out[m] = {
                "n_questions": len(sm),
                "accuracy_pct": round(acc_m * 100.0, 2),
                "ci95_low_pct": round(acc_lo * 100.0, 2),
                "ci95_high_pct": round(acc_hi * 100.0, 2),
                "avg_tokens": round(tok_m, 1),
                "token_saving_vs_sc4_2x2_pct": round((1.0 - tok_m / max(1.0, sc_tok)) * 100.0, 1),
                "avg_traces": round(float(sm["n_traces"].mean()), 2),
                "avg_latency_sec": round(lat_m, 2),
                "latency_ratio_vs_sc4_2x2": round(lat_m / max(1e-6, sc_lat), 3),
                "ece_pct": round(ece_v * 100.0, 2),
                "no_explicit_answer": int(sm["no_explicit_answer"].sum()),
                "capped_by_budget": int(sm["capped_by_budget"].sum()),
            }
            summary_rows.append({"Dataset": ds_key, "Method": m, **ds_out[m], "peak_vram_mb": round(peak_vram_mb, 1)})
        out_json["datasets"][ds_key] = ds_out

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "online_gpu_summary.csv", index=False)
    (OUT_DIR / "online_gpu_results.json").write_text(json.dumps(out_json, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
