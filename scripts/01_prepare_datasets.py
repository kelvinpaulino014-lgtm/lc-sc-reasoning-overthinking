#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 01: Preparação e Divisão Estrita (Calibração/Validação vs. Teste) dos Datasets GSM8K e MATH-500.
Segue as diretrizes de ML Best Practices:
- Verificação de valores ausentes/nulos (NULL/NaN) antes do processamento;
- Divisão determinística (seed=42) entre conjunto de calibração/validação (para ajuste de L*, sigma_L,
  tau_H e regressão isotônica de confiança) e conjunto de avaliação final (teste), evitando data leakage.
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def extract_gsm8k_gold(answer_str: str) -> str:
    """Extrai o valor numérico canônico após #### no GSM8K."""
    if not answer_str:
        return ""
    if "####" in answer_str:
        ans = answer_str.split("####")[-1].strip()
    else:
        nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer_str)
        ans = nums[-1] if nums else answer_str.strip()
    ans = ans.replace(",", "").replace("$", "").strip()
    if ans.endswith(".00"):
        ans = ans[:-3]
    return ans


def count_gsm8k_gold_steps(answer_str: str) -> int:
    """Conta o número de passos na solução canônica do GSM8K."""
    if not answer_str:
        return 1
    body = answer_str.split("####")[0].strip()
    lines = [line.strip() for line in body.split("\n") if line.strip()]
    return max(1, len(lines))


def prepare_gsm8k(seed: int = 42, val_size: int = 100, test_size: int = 250) -> Dict[str, Any]:
    """
    Baixa o GSM8K (openai/gsm8k) da HuggingFace, verifica integridade/nulos
    e separa estritamente em split de calibração/validação e split de teste.
    """
    print("[1/2] Baixando e verificando dataset GSM8K (openai/gsm8k)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    raw_rows = list(ds)

    valid_rows: List[Dict[str, Any]] = []
    null_count = 0
    for idx, item in enumerate(raw_rows):
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        gold = extract_gsm8k_gold(a)
        if not q or not a or not gold:
            null_count += 1
            continue
        valid_rows.append({
            "question_id": f"gsm8k_{idx:04d}",
            "dataset": "GSM8K",
            "question": q,
            "solution_gold": a,
            "answer_gold": gold,
            "gold_steps": count_gsm8k_gold_steps(a),
            "difficulty_level": min(5, max(1, count_gsm8k_gold_steps(a) - 1)),
        })

    print(f"  -> Total bruto GSM8K test: {len(raw_rows)} | Nulos/inválidos removidos: {null_count} | Válidos: {len(valid_rows)}")

    rng = random.Random(seed)
    shuffled = list(valid_rows)
    rng.shuffle(shuffled)

    val_split = shuffled[:val_size]
    test_split = shuffled[val_size : val_size + test_size]

    for row in val_split:
        row["split"] = "val_calibration"
    for row in test_split:
        row["split"] = "test"

    return {"val": val_split, "test": test_split, "all_valid_count": len(valid_rows)}


def prepare_math500(seed: int = 42, val_size: int = 100, test_size: int = 200) -> Dict[str, Any]:
    """
    Baixa o MATH-500 (HuggingFaceH4/math-500) da HuggingFace, verifica integridade/nulos
    e separa estritamente em split de calibração/validação e split de teste.
    """
    print("[2/2] Baixando e verificando dataset MATH-500 (HuggingFaceH4/math-500)...")
    ds = load_dataset("HuggingFaceH4/math-500", split="test")
    raw_rows = list(ds)

    valid_rows: List[Dict[str, Any]] = []
    null_count = 0
    for idx, item in enumerate(raw_rows):
        q = (item.get("problem") or "").strip()
        sol = (item.get("solution") or "").strip()
        ans = str(item.get("answer") or "").strip()
        level = item.get("level", 3)
        if isinstance(level, str):
            m = re.search(r"(\d+)", level)
            level_int = int(m.group(1)) if m else 3
        else:
            level_int = int(level) if level else 3

        if not q or not ans:
            null_count += 1
            continue

        valid_rows.append({
            "question_id": f"math500_{idx:04d}",
            "dataset": "MATH-500",
            "question": q,
            "solution_gold": sol,
            "answer_gold": ans,
            "gold_steps": max(2, len([ln for ln in sol.split("\n") if ln.strip()])),
            "difficulty_level": min(5, max(1, level_int)),
            "subject": item.get("subject", "Mathematics"),
        })

    print(f"  -> Total bruto MATH-500: {len(raw_rows)} | Nulos/inválidos removidos: {null_count} | Válidos: {len(valid_rows)}")

    rng = random.Random(seed)
    shuffled = list(valid_rows)
    rng.shuffle(shuffled)

    val_split = shuffled[:val_size]
    test_split = shuffled[val_size : val_size + test_size]

    for row in val_split:
        row["split"] = "val_calibration"
    for row in test_split:
        row["split"] = "test"

    return {"val": val_split, "test": test_split, "all_valid_count": len(valid_rows)}


def save_splits(name: str, splits: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined = splits["val"] + splits["test"]
    df = pd.DataFrame(combined)

    parquet_path = DATA_DIR / f"{name}_splits.parquet"
    jsonl_path = DATA_DIR / f"{name}_splits.jsonl"

    df.to_parquet(parquet_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  -> Salvo {parquet_path.name} e {jsonl_path.name} (val={len(splits['val'])}, test={len(splits['test'])})")


def main() -> None:
    gsm8k_splits = prepare_gsm8k(seed=42, val_size=80, test_size=200)
    save_splits("gsm8k", gsm8k_splits)

    math500_splits = prepare_math500(seed=42, val_size=80, test_size=200)
    save_splits("math500", math500_splits)

    summary = {
        "seed": 42,
        "gsm8k": {
            "total_valid": gsm8k_splits["all_valid_count"],
            "val_calibration": len(gsm8k_splits["val"]),
            "test": len(gsm8k_splits["test"]),
        },
        "math500": {
            "total_valid": math500_splits["all_valid_count"],
            "val_calibration": len(math500_splits["val"]),
            "test": len(math500_splits["test"]),
        },
    }
    with open(DATA_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n[OK] Datasets preparados com sucesso em:", DATA_DIR)


if __name__ == "__main__":
    main()
