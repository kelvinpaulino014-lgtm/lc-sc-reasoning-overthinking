#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suíte de Testes 02: Integridade dos Dados, Zero Data Leakage e Estrutura das Trilhas CoT na GPU.
Verifica se os conjuntos de validação (`val_calibration`) e teste (`test`) são estritamente disjuntos
e se as 8.960 cadeias CoT geradas em Parquet possuem esquema completo e propriedades físicas válidas.
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


class TestDataAndTracesIntegrity(unittest.TestCase):
    """Testes de integridade de dados, ausência de vazamento (data leakage) e estrutura de trilhas."""

    def test_splits_files_exist_and_counts(self) -> None:
        """Verifica se os arquivos de splits de GSM8K e MATH-500 existem e têm 80 val + 200 test = 280 questões."""
        for ds_key in ("gsm8k", "math500"):
            split_path = DATA_DIR / f"{ds_key}_splits.parquet"
            self.assertTrue(split_path.exists(), f"Arquivo ausente: {split_path}")
            df = pd.read_parquet(split_path)
            self.assertEqual(len(df), 280, f"Total incorreto de questões em {ds_key}")
            val_cnt = int((df["split"] == "val_calibration").sum())
            test_cnt = int((df["split"] == "test").sum())
            self.assertEqual(val_cnt, 80, f"Contagem de val_calibration incorreta em {ds_key}")
            self.assertEqual(test_cnt, 200, f"Contagem de test incorreta em {ds_key}")

    def test_zero_data_leakage_between_val_and_test(self) -> None:
        """Garante interseção vazia (zero data leakage) de IDs e textos entre val_calibration e test."""
        for ds_key in ("gsm8k", "math500"):
            df = pd.read_parquet(DATA_DIR / f"{ds_key}_splits.parquet")
            val_df = df[df["split"] == "val_calibration"]
            test_df = df[df["split"] == "test"]

            overlap_ids = set(val_df["question_id"]).intersection(set(test_df["question_id"]))
            self.assertEqual(len(overlap_ids), 0, f"Vazamento de IDs detectado em {ds_key}: {overlap_ids}")

            overlap_texts = set(val_df["question"].str.strip()).intersection(set(test_df["question"].str.strip()))
            self.assertEqual(len(overlap_texts), 0, f"Vazamento de enunciados detectado em {ds_key}")

    def test_traces_parquet_schema_and_completeness(self) -> None:
        """Verifica esquema, ausência de NaNs e integridade das 4.480 trilhas por benchmark."""
        required_cols = {
            "question_id",
            "dataset",
            "split",
            "difficulty_level",
            "sample_idx",
            "regime_type",
            "gold_answer",
            "final_answer",
            "is_correct_full",
            "midway_answer",
            "is_correct_midway",
            "overcorrected",
            "total_tokens",
            "raw_confidence",
            "mean_entropy",
            "max_entropy",
            "anchor_gpu_tokens",
            "anchor_gpu_entropy",
            "steps_json",
        }
        for ds_key in ("gsm8k", "math500"):
            trace_path = RESULTS_DIR / f"traces_{ds_key}.parquet"
            self.assertTrue(trace_path.exists(), f"Cache Parquet ausente: {trace_path}")
            df = pd.read_parquet(trace_path)
            self.assertEqual(len(df), 4480, f"Esperado 4480 trilhas (280x16) em {ds_key}, obtido {len(df)}")
            self.assertTrue(required_cols.issubset(set(df.columns)), f"Colunas faltando em {ds_key}")

            for col in required_cols:
                na_count = int(df[col].isna().sum())
                self.assertEqual(na_count, 0, f"Coluna {col} possui {na_count} valores NaN em {ds_key}")

            self.assertTrue((df["total_tokens"] >= 16).all())
            self.assertTrue(((df["raw_confidence"] > 0.0) & (df["raw_confidence"] < 1.0)).all())
            self.assertTrue((df["mean_entropy"] > 0.0).all())
            self.assertTrue((df["anchor_gpu_entropy"] > 0.0).all())

            # Inspeciona amostra de JSONs de passos intermediários
            sample_rows = df.sample(n=40, random_state=42)
            for _, row in sample_rows.iterrows():
                steps = json.loads(row["steps_json"])
                self.assertGreaterEqual(len(steps), 2)
                toks = [int(s["cumulative_tokens"]) for s in steps]
                self.assertEqual(toks, sorted(toks), "cumulative_tokens deve ser monotonicamente crescente")
                self.assertEqual(toks[-1], int(row["total_tokens"]))

    def test_inverted_u_curve_and_overcorrection_phenomena_in_traces(self) -> None:
        """Verifica se os dados exibem a curva em U invertido (Wu et al., 2026) e sobrecorreção (Hao et al., 2026)."""
        for ds_key in ("gsm8k", "math500"):
            df = pd.read_parquet(RESULTS_DIR / f"traces_{ds_key}.parquet")
            opt_acc = float(df[df["regime_type"] == "optimal_zone"]["is_correct_full"].mean())
            over_acc = float(df[df["regime_type"] == "overthinking"]["is_correct_full"].mean())
            under_acc = float(df[df["regime_type"] == "underthinking"]["is_correct_full"].mean())

            self.assertGreater(opt_acc, over_acc + 0.25, f"Curva U invertido violada em {ds_key}")
            self.assertGreater(opt_acc, under_acc + 0.10, f"Sub-pensamento deveria ter menor acurácia em {ds_key}")

            overcorr_count = int(df["overcorrected"].sum())
            self.assertGreater(overcorr_count, 200, f"Fenômeno de sobrecorreção insuficiente em {ds_key}")


if __name__ == "__main__":
    unittest.main()
