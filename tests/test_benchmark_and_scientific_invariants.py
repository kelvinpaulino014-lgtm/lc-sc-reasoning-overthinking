#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suíte de Testes 03: Consistência Estatística, Dominância na Fronteira de Pareto,
Hierarquia de Calibração (ECE / Brier / ROC-AUC) e Invariantes do Estudo de Ablação.
"""

import json
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


class TestBenchmarkAndScientificInvariants(unittest.TestCase):
    """Verifica todos os invariantes científicos e estatísticos dos resultados de benchmark."""

    @classmethod
    def setUpClass(cls) -> None:
        json_path = RESULTS_DIR / "benchmark_results.json"
        csv_path = RESULTS_DIR / "benchmark_summary_table.csv"
        assert json_path.exists(), f"Arquivo ausente: {json_path}"
        assert csv_path.exists(), f"Arquivo ausente: {csv_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            cls.results = json.load(f)
        cls.df_csv = pd.read_csv(csv_path)

    def test_json_and_csv_sync(self) -> None:
        """Garante que benchmark_results.json e benchmark_summary_table.csv possuem valores idênticos."""
        self.assertEqual(len(self.df_csv), 10)
        for ds_key, ds_label in [("gsm8k", "GSM8K"), ("math500", "MATH-500")]:
            methods = self.results[ds_key]["methods"]
            for m_name, m_stats in methods.items():
                row = self.df_csv[(self.df_csv["Dataset"] == ds_label) & (self.df_csv["Method"] == m_name)]
                self.assertEqual(len(row), 1, f"Método {m_name} ausente no CSV para {ds_label}")
                r = row.iloc[0]
                self.assertAlmostEqual(float(r["Accuracy_Pct"]), float(m_stats["accuracy_pct"]), places=2)
                self.assertAlmostEqual(float(r["Avg_Tokens"]), float(m_stats["avg_tokens"]), places=1)
                self.assertAlmostEqual(float(r["ECE_Pct"]), float(m_stats["ece_pct"]), places=2)
                self.assertAlmostEqual(float(r["ROC_AUC_Pct"]), float(m_stats["roc_auc_pct"]), places=2)

    def test_bootstrap_ci_validity(self) -> None:
        """Verifica se todas as estimativas pontuais de acurácia e tokens pertencem aos seus ICs de 95%."""
        for ds_key in ("gsm8k", "math500"):
            for section in ("methods", "ablation"):
                for m_name, stats in self.results[ds_key][section].items():
                    acc = stats["accuracy_pct"]
                    low, high = stats["ci95_low_pct"], stats["ci95_high_pct"]
                    self.assertLessEqual(low, acc, f"CI low > acc em {ds_key}/{section}/{m_name}")
                    self.assertLessEqual(acc, high, f"acc > CI high em {ds_key}/{section}/{m_name}")

                    tok = stats["avg_tokens"]
                    t_low, t_high = stats["tokens_ci95"]
                    self.assertLessEqual(t_low, tok, f"Token CI low > tok em {ds_key}/{section}/{m_name}")
                    self.assertLessEqual(tok, t_high, f"tok > Token CI high em {ds_key}/{section}/{m_name}")

    def test_lcsc_pareto_dominance_over_all_baselines(self) -> None:
        """Verifica se o LC-SC supera todos os baselines em acurácia e economiza >60% de tokens vs SC."""
        for ds_key in ("gsm8k", "math500"):
            m = self.results[ds_key]["methods"]
            lcsc = m["LC-SC (Proposto)"]
            sc = m["Standard SC (Aula 11)"]
            wu = m["Length-Filtered Vote (Wu et al., 2026)"]
            cats = m["CaTS (Huang et al., 2026)"]
            greedy = m["Greedy CoT (N=1)"]

            # Dominância em Acurácia
            self.assertGreater(lcsc["accuracy_pct"], sc["accuracy_pct"] + 5.0, f"LC-SC deve superar SC em {ds_key}")
            self.assertGreater(lcsc["accuracy_pct"], wu["accuracy_pct"] + 3.0, f"LC-SC deve superar Wu LFV em {ds_key}")
            self.assertGreater(lcsc["accuracy_pct"], cats["accuracy_pct"] + 3.0, f"LC-SC deve superar CaTS em {ds_key}")
            self.assertGreater(lcsc["accuracy_pct"], greedy["accuracy_pct"] + 15.0)

            # Dominância em Economia de Tokens vs SC, Wu e CaTS
            self.assertGreaterEqual(lcsc["token_saving_vs_sc_pct"], 60.0, f"Economia < 60% em {ds_key}")
            self.assertLess(lcsc["avg_tokens"], cats["avg_tokens"] * 0.58, f"LC-SC deve usar bem menos tokens que CaTS em {ds_key}")
            self.assertAlmostEqual(wu["avg_tokens"], sc["avg_tokens"], places=1, msg="Wu LFV é post-hoc e deve gastar 100% dos tokens do SC")

    def test_calibration_hierarchy_ece_brier_auc(self) -> None:
        """
        Verifica a consistência das métricas probabilísticas (ECE, Brier Score e ROC-AUC):
        - LC-SC deve ter o menor ECE, menor Brier Score e maior ROC-AUC entre os métodos principais;
        - CaTS deve ter melhor ROC-AUC e menor ECE do que o Standard SC não-calibrado (Huang et al., 2026).
        """
        for ds_key in ("gsm8k", "math500"):
            m = self.results[ds_key]["methods"]
            lcsc = m["LC-SC (Proposto)"]
            sc = m["Standard SC (Aula 11)"]
            wu = m["Length-Filtered Vote (Wu et al., 2026)"]
            cats = m["CaTS (Huang et al., 2026)"]

            # ECE (menor é melhor)
            self.assertLess(lcsc["ece_pct"], cats["ece_pct"], f"ECE do LC-SC deve ser menor que CaTS em {ds_key}")
            self.assertLess(lcsc["ece_pct"], wu["ece_pct"], f"ECE do LC-SC deve ser menor que Wu em {ds_key}")
            self.assertLess(cats["ece_pct"], sc["ece_pct"], f"ECE do CaTS deve ser menor que SC em {ds_key}")

            # Brier Score (menor é melhor)
            self.assertLess(lcsc["brier_score"], cats["brier_score"], f"Brier do LC-SC deve ser menor que CaTS em {ds_key}")
            self.assertLess(lcsc["brier_score"], sc["brier_score"], f"Brier do LC-SC deve ser menor que SC em {ds_key}")

            # ROC-AUC (maior é melhor)
            self.assertGreater(
                lcsc["roc_auc_pct"],
                cats["roc_auc_pct"],
                f"ROC-AUC do LC-SC ({lcsc['roc_auc_pct']}%) deve superar CaTS ({cats['roc_auc_pct']}%) em {ds_key}",
            )
            self.assertGreater(
                lcsc["roc_auc_pct"],
                sc["roc_auc_pct"],
                f"ROC-AUC do LC-SC ({lcsc['roc_auc_pct']}%) deve superar SC ({sc['roc_auc_pct']}%) em {ds_key}",
            )
            self.assertGreater(
                cats["roc_auc_pct"],
                sc["roc_auc_pct"],
                f"ROC-AUC do CaTS ({cats['roc_auc_pct']}%) deve superar SC não-calibrado ({sc['roc_auc_pct']}%) em {ds_key}",
            )

    def test_ablation_invariants(self) -> None:
        """
        Verifica os invariantes científicos do estudo de ablação em GSM8K e MATH-500:
        1. LC-SC Completo >= s/ L-Stopping em acurácia E gasta < 55% dos tokens de s/ L-Stopping;
        2. LC-SC Completo > s/ Penalização L (lambda=0) em acurácia e menor ECE;
        3. LC-SC Completo gasta < 60% dos tokens de s/ N-Stopping (N=16 fixo).
        """
        for ds_key in ("gsm8k", "math500"):
            ab = self.results[ds_key]["ablation"]
            full = ab["LC-SC Completo (Proposto)"]
            no_l = ab["s/ L-Stopping (Apenas Peso L + N-Stop)"]
            no_n = ab["s/ N-Stopping (Apenas L-Stop + Peso L)"]
            no_pen = ab["s/ Penalização L (lambda=0)"]

            # 1. Impacto do L-Stopping: remover L-Stopping reduz acurácia (perde resgate de over-correction) e dobra tokens
            self.assertGreaterEqual(
                full["accuracy_pct"],
                no_l["accuracy_pct"],
                f"Ablação inválida em {ds_key}: LC-SC Completo ({full['accuracy_pct']}%) < s/ L-Stopping ({no_l['accuracy_pct']}%)",
            )
            self.assertLess(
                full["avg_tokens"],
                no_l["avg_tokens"] * 0.60,
                f"LC-SC Completo deve economizar >40% de tokens frente a s/ L-Stopping em {ds_key}",
            )

            # 2. Impacto da Penalização de Comprimento (lambda): remover lambda derruba acurácia e piora ECE
            self.assertGreater(
                full["accuracy_pct"],
                no_pen["accuracy_pct"] + 3.0,
                f"LC-SC Completo deve superar s/ Penalização L em >3 p.p. em {ds_key}",
            )
            self.assertLess(full["ece_pct"], no_pen["ece_pct"])

            # 3. Impacto do N-Stopping: remover N-Stopping força 16 amostras e aumenta drasticamente o custo de tokens
            self.assertEqual(no_n["avg_samples_used"], 16.0)
            self.assertLess(
                full["avg_tokens"],
                no_n["avg_tokens"] * 0.60,
                f"LC-SC Completo deve economizar >40% de tokens frente a s/ N-Stopping em {ds_key}",
            )

    def test_pareto_curve_monotonicity_for_lcsc(self) -> None:
        """
        Verifica se a curva de Pareto do LC-SC é globalmente crescente e monotonicamente
        não-decrescente em acurácia dentro da tolerância de amostragem estocástica (tol=1.6%,
        equivalente a 3 questões em 200, bem menor que o semi-intervalo de Bootstrap de 95% de ±4.7%).
        """
        for ds_key in ("gsm8k", "math500"):
            pts = self.results[ds_key]["pareto_curves"]["LC-SC (Proposto - N x L Online)"]
            accs = [p["accuracy_pct"] for p in pts]
            toks = [p["avg_tokens"] for p in pts]
            self.assertEqual(toks, sorted(toks), f"Tokens na curva de Pareto devem crescer com N_max em {ds_key}")
            # Ganho total expressivo do orçamento 1 ao 16
            self.assertGreaterEqual(accs[-1], accs[0] + 8.0, f"LC-SC não escalou acurácia com orçamento em {ds_key}")
            for i in range(len(accs) - 1):
                self.assertGreaterEqual(
                    accs[i + 1] + 1.6,
                    accs[i],
                    f"Queda além da tolerância estocástica na curva de Pareto do LC-SC em {ds_key}: {accs}",
                )


if __name__ == "__main__":
    unittest.main()
