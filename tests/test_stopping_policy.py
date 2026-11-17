#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite 5: Unit tests for Phase 3 `NeuralStoppingPolicy` (<3K params),
step feature extraction, and real-time `EntropyNeuralStoppingCriteria`.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.stopping_policy import (
    EntropyNeuralStoppingCriteria,
    NeuralStoppingPolicy,
    build_step_features_from_trace,
)


class _DummyTokenizer:
    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        # Simulate text containing an explicit final-answer cue "\\boxed{42}"
        return "Step 1: 6 * 7 = 42. Therefore the answer is \\boxed{42}."


class _PartialDummyTokenizer:
    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        # Simulate incomplete CoT with intermediate numbers but NO final answer cue
        return "Step 1: 6 * 7 = 42. Next we need to subtract 5"


class TestNeuralStoppingPolicy(unittest.TestCase):
    """Verifies parameter budget, forward pass, feature builder, and online StoppingCriteria."""

    def test_parameter_budget_under_3000(self) -> None:
        policy = NeuralStoppingPolicy(input_dim=6, hidden_dim=16, dropout=0.1)
        param_count = policy.count_parameters()
        self.assertLess(param_count, 3000, f"Policy exceeds 3K params: {param_count}")
        self.assertGreater(param_count, 500)

        x = torch.randn(4, 7, 6)
        logits, hn = policy(x)
        self.assertEqual(logits.shape, (4, 7, 2))
        self.assertEqual(hn.shape, (1, 4, 16))

        stop_p, corr_p, h_next = policy.step_predict(x[:, :1, :], None)
        self.assertEqual(stop_p.shape, (4,))
        self.assertEqual(corr_p.shape, (4,))
        self.assertTrue(torch.all((stop_p >= 0.0) & (stop_p <= 1.0)))
        self.assertTrue(torch.all((corr_p >= 0.0) & (corr_p <= 1.0)))
        self.assertEqual(h_next.shape, (1, 4, 16))

    def test_build_step_features_from_trace(self) -> None:
        steps = [
            {"step_entropy": 0.25, "step_confidence": 0.90, "cumulative_tokens": 40, "intermediate_answer": "42", "is_overthinking_transition": False},
            {"step_entropy": 1.40, "step_confidence": 0.45, "cumulative_tokens": 90, "intermediate_answer": "99", "is_overthinking_transition": True},
        ]
        feats, stop_t, corr_t = build_step_features_from_trace(steps, l_star=100.0, gold_answer="42")
        self.assertEqual(feats.shape, (2, 6))
        self.assertEqual(stop_t.tolist(), [0.0, 1.0])
        self.assertEqual(corr_t.tolist(), [1.0, 0.0])

    def test_online_stopping_criteria_triggers(self) -> None:
        policy = NeuralStoppingPolicy(input_dim=6, hidden_dim=16, dropout=0.0)
        # Force stop_logit bias high so threshold triggers deterministically
        with torch.no_grad():
            policy.head[-1].bias.data = torch.tensor([5.0, 2.0])

        criteria = EntropyNeuralStoppingCriteria(
            policy=policy,
            tokenizer=_DummyTokenizer(),
            prompt_len=10,
            l_star=50.0,
            l_min=4,
            stop_threshold=0.5,
            step_stride=2,
            device="cpu",
        )

        vocab = 32
        # Simulate 4 generated tokens beyond prompt_len=10
        for step in range(1, 5):
            input_ids = torch.randint(0, vocab, (2, 10 + step))
            scores = tuple(torch.randn(2, vocab) for _ in range(step))
            done = criteria(input_ids, scores)
            if step < 4:
                self.assertFalse(bool(done.all()))
            else:
                self.assertTrue(bool(done.all()))
                self.assertEqual(criteria.stop_tokens, [4, 4])

    def test_incomplete_cot_does_not_stop_prematurely(self) -> None:
        policy = NeuralStoppingPolicy(input_dim=6, hidden_dim=16, dropout=0.0)
        with torch.no_grad():
            policy.head[-1].bias.data = torch.tensor([5.0, 2.0])
        criteria = EntropyNeuralStoppingCriteria(
            policy=policy,
            tokenizer=_PartialDummyTokenizer(),
            prompt_len=10,
            l_star=50.0,
            l_min=4,
            stop_threshold=0.5,
            step_stride=2,
            device="cpu",
        )
        vocab = 32
        for step in range(1, 7):
            input_ids = torch.randint(0, vocab, (2, 10 + step))
            scores = tuple(torch.randn(2, vocab) for _ in range(step))
            done = criteria(input_ids, scores)
            self.assertFalse(bool(done.any()))


if __name__ == "__main__":
    unittest.main()
