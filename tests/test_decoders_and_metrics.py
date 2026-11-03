"""
Suíte completa de testes unitários para os módulos de extração de respostas,
métricas de entropia/calibração e decodificadores (Baselines ICLR 2026 + LC-SC e Ablações).

Compatível tanto com `pytest` quanto com `python -m unittest`.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

# Garante que o diretório raiz do Projeto_2 esteja no sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decoders import (
    ConfidenceCalibrator,
    CoTStep,
    CoTTrace,
    GreedyCoTDecoder,
    HuangCaTSDecoder,
    LCSC_Ablation_NoLenPenalty,
    LCSC_Ablation_NoLStop,
    LCSC_Ablation_NoNStop,
    LCSCDecoder,
    StandardSCDecoder,
    WuLengthFilteredVoteDecoder,
    build_all_decoders,
    load_traces_jsonl,
    save_traces_jsonl,
)
from src.metrics import (
    EntropyTracker,
    bootstrap_ci,
    compute_answer_entropy,
    compute_auc_roc,
    compute_brier_score,
    compute_calibration_bins,
    compute_ece,
    compute_token_entropy,
    compute_windowed_entropy,
    fit_inverted_u_curve,
)
from src.utils import (
    detect_overthinking_transition,
    extract_answer,
    extract_intermediate_answer,
    extract_last_boxed,
    is_equivalent,
    normalize_math_answer,
    segment_cot_steps,
)


def _make_overthinking_trace(
    question_id: str,
    sample_idx: int,
    correct_ans: str = "42",
    corrupted_ans: str = "84",
) -> CoTTrace:
    """
    Cria uma trilha sintética representativa do fenômeno de sobrecorreção (Hao et al., 2026):
    - Passos 0 e 1 derivam a resposta correta (`42`) com baixa entropia (0.18 - 0.22) em 95 tokens.
    - Passos 2 e 3 entram em hesitação/overthinking (`step_entropy = 0.92 > tau_H`),
      corrompem a resposta para `84` e gastam 280 tokens totais.
    """
    steps = [
        CoTStep(
            step_idx=0,
            cumulative_tokens=45,
            step_tokens=45,
            step_entropy=0.18,
            step_confidence=0.90,
            intermediate_answer=None,
            is_overthinking_transition=False,
            text="First, let us compute 6 * 7 step by step.",
        ),
        CoTStep(
            step_idx=1,
            cumulative_tokens=95,
            step_tokens=50,
            step_entropy=0.21,
            step_confidence=0.92,
            intermediate_answer=correct_ans,
            is_overthinking_transition=False,
            text=f"Multiplying 6 by 7 gives = {correct_ans}. So the answer is {correct_ans}.",
        ),
        CoTStep(
            step_idx=2,
            cumulative_tokens=165,
            step_tokens=70,
            step_entropy=0.95,
            step_confidence=0.45,
            intermediate_answer=None,
            is_overthinking_transition=True,
            text="Wait, let me double check! Hmm, should we multiply by 2 again?",
        ),
        CoTStep(
            step_idx=3,
            cumulative_tokens=280,
            step_tokens=115,
            step_entropy=0.88,
            step_confidence=0.52,
            intermediate_answer=corrupted_ans,
            is_overthinking_transition=True,
            text=f"Alternatively, maybe 42 * 2 = {corrupted_ans}. #### {corrupted_ans}",
        ),
    ]
    return CoTTrace(
        question_id=question_id,
        sample_idx=sample_idx,
        full_text="\n\n".join(s.text for s in steps),
        total_tokens=280,
        final_answer=corrupted_ans,
        raw_confidence=0.68,
        mean_entropy=0.55,
        steps=steps,
        gold_answer=correct_ans,
    )


def _make_clean_trace(
    question_id: str,
    sample_idx: int,
    answer: str = "42",
    gold_answer: str = "42",
    tokens: int = 110,
    confidence: float = 0.91,
    entropy: float = 0.20,
) -> CoTTrace:
    """Cria uma trilha limpa e concisa (dentro da zona ótima L*)."""
    steps = [
        CoTStep(
            step_idx=0,
            cumulative_tokens=tokens // 2,
            step_tokens=tokens // 2,
            step_entropy=entropy,
            step_confidence=confidence,
            intermediate_answer=None,
            is_overthinking_transition=False,
            text="Setting up the equation directly.",
        ),
        CoTStep(
            step_idx=1,
            cumulative_tokens=tokens,
            step_tokens=tokens - (tokens // 2),
            step_entropy=entropy,
            step_confidence=confidence,
            intermediate_answer=answer,
            is_overthinking_transition=False,
            text=f"Therefore, the final answer is \\boxed{{{answer}}}.",
        ),
    ]
    return CoTTrace(
        question_id=question_id,
        sample_idx=sample_idx,
        full_text="\n\n".join(s.text for s in steps),
        total_tokens=tokens,
        final_answer=answer,
        raw_confidence=confidence,
        mean_entropy=entropy,
        steps=steps,
        gold_answer=gold_answer,
    )


class TestAnswerExtractor(unittest.TestCase):
    """Testes para extração de respostas, equivalência algébrica e segmentação de CoT."""

    def test_extract_gsm8k_hash_and_boxed(self) -> None:
        self.assertEqual(extract_answer("Step 1: 10+5=15.\n#### 15"), "15")
        self.assertEqual(extract_answer("Total cost is $1,250.00\n#### $1,250.00"), "1250")
        self.assertEqual(
            extract_answer("We find x = \\boxed{\\frac{3}{4}} after simplifying."),
            "3/4",
        )
        self.assertEqual(
            extract_last_boxed("Intermediate \\boxed{10} and final \\boxed{\\frac{1+\\sqrt{2}}{2}}"),
            "\\frac{1+\\sqrt{2}}{2}",
        )

    def test_extract_after_think_and_cues(self) -> None:
        text = "<think>\nWait, 20 * 3 = 60, no 20 + 3 = 23.\n</think>\nPortanto, a resposta final é 23."
        self.assertEqual(extract_answer(text), "23")

    def test_equivalence_and_normalization(self) -> None:
        self.assertTrue(is_equivalent("0.5", "\\frac{1}{2}"))
        self.assertTrue(is_equivalent("\\boxed{1,000.00}", "1000"))
        self.assertTrue(is_equivalent("-42.0", "-42"))
        self.assertTrue(is_equivalent("\\dfrac{6}{4}", "1.5"))
        self.assertFalse(is_equivalent("42", "43"))
        self.assertFalse(is_equivalent(None, "42"))
        self.assertIsNone(normalize_math_answer("   "))

    def test_segment_cot_steps_and_overthinking_detection(self) -> None:
        cot_text = (
            "First, 12 * 3 = 36.\n\n"
            "So the answer is 36.\n\n"
            "Wait, let me double check whether we need to add 4.\n\n"
            "Hmm, alternatively 36 + 4 = 40. #### 40"
        )
        steps = segment_cot_steps(cot_text, total_tokens=120)
        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[-1].cumulative_tokens, 120)
        self.assertEqual(steps[1].intermediate_answer, "36")
        self.assertFalse(steps[0].is_overthinking_transition)
        self.assertTrue(steps[2].is_overthinking_transition)
        self.assertTrue(detect_overthinking_transition("Wait, let's double check this."))
        self.assertEqual(extract_intermediate_answer("Thus 7 * 8 = 56"), "56")


class TestEntropyAndCalibrationMetrics(unittest.TestCase):
    """Testes para entropia de Shannon, ECE, Brier Score, AUC-ROC, Bootstrap e Curva U."""

    def test_token_shannon_entropy(self) -> None:
        # Distribuição uniforme sobre 4 tokens => H = ln(4)
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        h_unif = compute_token_entropy(uniform_probs, from_logits=False)
        self.assertAlmostEqual(float(h_unif), math.log(4.0), places=6)

        # Distribuição determinística (one-hot) => H = 0
        one_hot = np.array([1.0, 0.0, 0.0, 0.0])
        h_det = compute_token_entropy(one_hot, from_logits=False)
        self.assertAlmostEqual(float(h_det), 0.0, places=6)

        # Matriz 2D de logits
        logits_2d = np.array([[0.0, 0.0, 0.0, 0.0], [100.0, -100.0, -100.0, -100.0]])
        h_2d = compute_token_entropy(logits_2d, from_logits=True)
        self.assertEqual(h_2d.shape, (2,))
        self.assertAlmostEqual(float(h_2d[0]), math.log(4.0), places=5)
        self.assertAlmostEqual(float(h_2d[1]), 0.0, places=5)

    def test_windowed_entropy_and_tracker(self) -> None:
        series = [0.2, 0.4, 0.6, 0.8, 1.0]
        win = compute_windowed_entropy(series, window_size=3)
        self.assertAlmostEqual(float(win[0]), 0.2, places=6)
        self.assertAlmostEqual(float(win[1]), 0.3, places=6)
        self.assertAlmostEqual(float(win[2]), 0.4, places=6)
        self.assertAlmostEqual(float(win[4]), 0.8, places=6)

        tracker = EntropyTracker(window_size=2, tau_h=0.60, l_min=50)
        self.assertFalse(tracker.update(0.20, num_tokens=30, has_candidate_answer=False))
        self.assertFalse(tracker.update(0.25, num_tokens=30, has_candidate_answer=True))
        self.assertTrue(tracker.update(0.85, num_tokens=25, has_candidate_answer=True))
        self.assertTrue(tracker.spike_detected)

    def test_calibration_metrics_and_missing_values(self) -> None:
        # Caso perfeitamente calibrado
        confs = [0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, np.nan]
        accs = [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
        ece = compute_ece(confs, accs, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=6)

        brier = compute_brier_score([1.0, 0.0], [1, 0])
        self.assertAlmostEqual(brier, 0.0, places=6)

        auc_perfect = compute_auc_roc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self.assertAlmostEqual(auc_perfect, 1.0, places=6)

        bins = compute_calibration_bins(confs, accs, n_bins=10)
        self.assertEqual(len(bins), 10)

    def test_bootstrap_ci(self) -> None:
        rng = np.random.default_rng(123)
        sample = rng.normal(loc=0.75, scale=0.1, size=100).tolist()
        mean_val, ci_low, ci_high = bootstrap_ci(sample, n_boot=1000, ci=0.95, seed=42)
        self.assertLess(ci_low, mean_val)
        self.assertGreater(ci_high, mean_val)
        res = bootstrap_ci(sample, n_boot=500, ci=0.95, seed=42)
        self.assertAlmostEqual(res.mean, res["mean"], places=8)
        self.assertGreater(res.half_width, 0.0)

    def test_inverted_u_curve_fit(self) -> None:
        # Gera dados sintéticos seguindo uma curva em U invertido com pico em L* = 160
        lengths = np.linspace(50, 350, 120)
        probs = np.clip(0.90 - 2.6e-5 * (lengths - 160.0) ** 2, 0.05, 0.95)
        accs = (probs >= 0.65).astype(float)
        fit_res = fit_inverted_u_curve(lengths, accs, n_bins=6)
        self.assertTrue(fit_res.is_concave)
        self.assertTrue(125.0 <= fit_res.l_star <= 195.0)
        self.assertGreater(fit_res.sigma_l, 15.0)
        self.assertGreater(fit_res.overthinking_drop, 0.0)


class TestDecodersAndAblations(unittest.TestCase):
    """Testes para os 5 decodificadores principais e as 3 variantes de ablação."""

    def setUp(self) -> None:
        # Constrói conjunto de validação e conjunto de teste disjuntos
        self.val_traces: List[CoTTrace] = []
        for q_idx in range(6):
            qid = f"val_q{q_idx}"
            for s_idx in range(6):
                if s_idx < 4:
                    self.val_traces.append(
                        _make_clean_trace(qid, s_idx, answer="42", gold_answer="42", tokens=110 + 5 * s_idx)
                    )
                else:
                    self.val_traces.append(
                        _make_overthinking_trace(qid, s_idx, correct_ans="42", corrupted_ans="84")
                    )

        # Questão de teste onde 3 de 6 trilhas sofrem over-correction (trocando 42 por 84 no final),
        # 2 trilhas são limpas com resposta 42, e 1 trilha longa erra direto para 84.
        # - No Standard SC / Wu / CaTS sem L-Stop: 4 trilhas terminam com "84" vs 2 com "42" -> votam "84" (ERRADO) ou gastam muitos tokens!
        # - No LC-SC (Proposto): o L-Stopping trunca as 3 trilhas de over-correction no passo 2,
        #   recuperando a resposta intermediária pré-sobrecorreção ("42") e economizando tokens!
        self.test_q_traces: List[CoTTrace] = [
            _make_clean_trace("test_q1", 0, answer="42", gold_answer="42", tokens=105, confidence=0.92),
            _make_overthinking_trace("test_q1", 1, correct_ans="42", corrupted_ans="84"),
            _make_overthinking_trace("test_q1", 2, correct_ans="42", corrupted_ans="84"),
            _make_overthinking_trace("test_q1", 3, correct_ans="42", corrupted_ans="84"),
            _make_clean_trace("test_q1", 4, answer="42", gold_answer="42", tokens=115, confidence=0.90),
            _make_clean_trace("test_q1", 5, answer="84", gold_answer="42", tokens=290, confidence=0.85, entropy=0.60),
        ]

    def test_trace_serialization_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traces.jsonl"
            save_traces_jsonl(self.test_q_traces, path)
            loaded = load_traces_jsonl(path)
            self.assertEqual(len(loaded), len(self.test_q_traces))
            self.assertEqual(loaded[1].steps[1].intermediate_answer, "42")

    def test_greedy_and_standard_sc_baselines(self) -> None:
        greedy = GreedyCoTDecoder()
        pred_g = greedy.predict_question(self.test_q_traces, gold_answer="42")
        self.assertEqual(pred_g.num_samples_used, 1)
        self.assertEqual(pred_g.total_tokens_used, 105)
        self.assertTrue(pred_g.is_correct)

        sc = StandardSCDecoder(n_max=6)
        pred_sc = sc.predict_question(self.test_q_traces, gold_answer="42")
        self.assertEqual(pred_sc.num_samples_used, 6)
        self.assertEqual(pred_sc.total_tokens_used, sum(t.total_tokens for t in self.test_q_traces))
        # Sem L-Stopping, 4 trilhas têm final_answer="84" e apenas 2 têm "42"
        self.assertEqual(pred_sc.predicted_answer, "84")
        self.assertFalse(pred_sc.is_correct)

    def test_wu_length_filtered_vote_posthoc_cost(self) -> None:
        wu = WuLengthFilteredVoteDecoder(n_max=6, window_fraction=0.5)
        wu.fit(self.val_traces)
        pred_wu = wu.predict_question(self.test_q_traces, gold_answer="42")
        # Wu et al. (2026) é post-hoc: usa 100% dos tokens de todas as trilhas
        self.assertEqual(pred_wu.total_tokens_used, sum(t.total_tokens for t in self.test_q_traces))
        self.assertEqual(pred_wu.token_savings_ratio, 0.0)

    def test_huang_cats_early_n_stop_without_l_stop(self) -> None:
        cats = HuangCaTSDecoder(n_max=6, n_min=2, gamma=0.70)
        cats.fit(self.val_traces)
        clean_consensus = [
            _make_clean_trace("test_q2", i, answer="42", gold_answer="42", tokens=120, confidence=0.93)
            for i in range(6)
        ]
        pred_cats = cats.predict_question(clean_consensus, gold_answer="42")
        self.assertEqual(pred_cats.num_samples_used, 2)
        self.assertTrue(pred_cats.stopped_early_n)
        self.assertEqual(pred_cats.num_l_stopped_traces, 0)

    def test_lcsc_recovers_pre_overcorrection_and_saves_tokens(self) -> None:
        lcsc = LCSCDecoder(
            n_max=6,
            n_min=3,
            l_min=48,
            l_star=120.0,
            sigma_l=50.0,
            tau_h=0.60,
            lambda_penalty=0.50,
            gamma=0.75,
            auto_calibrate_tau_h=False,
        )
        lcsc.fit(self.val_traces)

        # Verifica fórmula exata do peso exponencial por comprimento
        w_short = lcsc.compute_trace_weight(0.80, used_tokens=100)
        w_long = lcsc.compute_trace_weight(0.80, used_tokens=220)
        expected_w_long = 0.80 * math.exp(-0.50 * ((220.0 - lcsc.l_star) / lcsc.sigma_l))
        self.assertAlmostEqual(w_short, 0.80, places=6)
        self.assertAlmostEqual(w_long, expected_w_long, places=6)
        self.assertLess(w_long, w_short)

        pred_lcsc = lcsc.predict_question(self.test_q_traces, gold_answer="42")
        # O LC-SC interrompe as trilhas com over-correction, recupera "42" e para precocemente em N=3!
        self.assertEqual(pred_lcsc.predicted_answer, "42")
        self.assertTrue(pred_lcsc.is_correct)
        self.assertGreater(pred_lcsc.num_l_stopped_traces, 0)
        self.assertTrue(pred_lcsc.stopped_early_n)
        self.assertLess(pred_lcsc.total_tokens_used, pred_lcsc.full_tokens_budget)
        self.assertGreater(pred_lcsc.token_savings_ratio, 0.40)

    def test_ablation_variants_behavior(self) -> None:
        no_l = LCSC_Ablation_NoLStop(n_max=6, n_min=3, tau_h=0.60, gamma=0.75, auto_calibrate_tau_h=False)
        no_n = LCSC_Ablation_NoNStop(n_max=6, n_min=3, tau_h=0.60, gamma=0.75, auto_calibrate_tau_h=False)
        no_pen = LCSC_Ablation_NoLenPenalty(n_max=6, n_min=3, tau_h=0.60, gamma=0.75, auto_calibrate_tau_h=False)

        for dec in (no_l, no_n, no_pen):
            dec.fit(self.val_traces)

        res_no_l = no_l.predict_question(self.test_q_traces, gold_answer="42")
        res_no_n = no_n.predict_question(self.test_q_traces, gold_answer="42")
        res_no_pen = no_pen.predict_question(self.test_q_traces, gold_answer="42")

        # 1. NoLStop nunca faz L-Stop
        self.assertEqual(res_no_l.num_l_stopped_traces, 0)
        # 2. NoNStop sempre usa todas as N_max=6 trilhas, mas faz L-Stop nas 3 trilhas com overthinking
        self.assertEqual(res_no_n.num_samples_used, 6)
        self.assertFalse(res_no_n.stopped_early_n)
        self.assertEqual(res_no_n.num_l_stopped_traces, 3)
        # 3. NoLenPenalty tem lambda=0 (peso igual à confiança calibrada independente de L)
        self.assertEqual(no_pen.lambda_penalty, 0.0)
        self.assertAlmostEqual(no_pen.compute_trace_weight(0.85, used_tokens=999), 0.85, places=6)
        self.assertTrue(res_no_pen.is_correct)

    def test_build_all_decoders_and_evaluate_dataset(self) -> None:
        all_decs = build_all_decoders(n_max=6, include_ablations=True)
        self.assertEqual(len(all_decs), 8)
        for dec in all_decs.values():
            dec.fit(self.val_traces)
            summary = dec.evaluate_dataset(self.test_q_traces, n_boot=100, seed=42)
            self.assertEqual(summary["num_questions"], 1)
            self.assertIn("accuracy_ci", summary)
            self.assertIn("ece", summary)


if __name__ == "__main__":
    unittest.main()
