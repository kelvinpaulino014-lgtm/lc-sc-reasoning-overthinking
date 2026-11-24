#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Phase 4 causal real_trace_policy contract."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.real_trace_policy import (
    FEATURES,
    MAX_TOKENS,
    STRIDE,
    FittedPolicy,
    complete_answer,
    consensus_after_two,
    load_policy,
    make_window,
    sha256_file,
    vote,
)
from src.models.stopping_policy import NeuralStoppingPolicy


class TestRealTracePolicy(unittest.TestCase):
    def test_complete_answer_requires_closed_boxed(self) -> None:
        self.assertIsNone(complete_answer("Step 1: 6 * 7 = 42"))
        self.assertIsNone(complete_answer("Therefore \\boxed{42"))
        self.assertEqual(complete_answer("Therefore \\boxed{42}"), "42")
        self.assertEqual(complete_answer("First \\boxed{10}, actually \\boxed{42}"), "42")

    def test_make_window_causality_and_validation(self) -> None:
        w1 = make_window(
            prefix_text="Step 1: compute 6 * 7 = 42.",
            entropy_window=[0.3] * 24,
            logprob_window=[-0.2] * 24,
            n_tokens=24,
            previous_window=None,
        )
        self.assertEqual(w1.n_tokens, 24)
        self.assertIsNone(w1.answer)
        self.assertEqual(len(w1.features), 6)
        self.assertAlmostEqual(w1.features[0], 0.3, places=5)
        self.assertAlmostEqual(w1.features[1], 0.0, places=5)
        self.assertEqual(w1.features[4], 0.0)
        self.assertEqual(w1.features[5], 0.0)

        w2 = make_window(
            prefix_text="Step 1: 42. Answer: \\boxed{42}",
            entropy_window=[0.5] * 24,
            logprob_window=[-0.1] * 24,
            n_tokens=48,
            previous_window=w1,
        )
        self.assertEqual(w2.answer, "42")
        self.assertAlmostEqual(w2.features[1], 0.2, places=5)
        self.assertEqual(w2.features[4], 1.0)
        self.assertEqual(w2.features[5], 0.0)

        w3 = make_window(
            prefix_text="Answer: \\boxed{42} wait",
            entropy_window=[0.4] * 12,
            logprob_window=[-0.15] * 12,
            n_tokens=60,
            previous_window=w2,
        )
        self.assertEqual(w3.features[5], 1.0)

        # Invalid window length or non-finite values must raise ValueError
        with self.assertRaises(ValueError):
            make_window("abc", [0.1] * 10, [-0.1] * 10, n_tokens=24, previous_window=None)
        with self.assertRaises(ValueError):
            make_window("abc", [float("nan")] * 24, [-0.1] * 24, n_tokens=24, previous_window=None)

    def test_consensus_after_two_and_vote(self) -> None:
        self.assertTrue(consensus_after_two(["42", "42"]))
        self.assertFalse(consensus_after_two([None, None]))
        self.assertFalse(consensus_after_two(["42", "43"]))
        self.assertFalse(consensus_after_two(["42"]))

        ans, cov = vote([None, "7", "8", "7"])
        self.assertEqual(ans, "7")
        self.assertAlmostEqual(cov, 0.5)
        self.assertEqual(vote([None, None]), (None, 0.0))

    def test_fitted_policy_causality_and_load_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            model = NeuralStoppingPolicy()
            arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            npz_path = d / "gru_seed42.npz"
            np.savez(npz_path, **arrays)

            cfg = {
                "features": list(FEATURES),
                "stride": STRIDE,
                "max_tokens": MAX_TOKENS,
                "seeds": [42],
                "deployment_seed": 42,
                "scaler_mean": [0.0] * 6,
                "scaler_scale": [1.0] * 6,
                "logistic_coef": [[0.1] * 6, [0.1] * 6],
                "logistic_intercept": [0.0, 0.0],
                "thresholds": {"logistic": [0.5, 0.5]},
                "gru_thresholds_by_seed": {"42": [0.5, 0.5]},
            }
            cfg_path = d / "config.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf8")
            frozen = {"sha256": {"config.json": sha256_file(cfg_path), "gru_seed42.npz": sha256_file(npz_path)}}
            (d / "frozen.json").write_text(json.dumps(frozen), encoding="utf8")

            fitted = load_policy(d)
            w1 = make_window("x", [0.2] * 24, [-0.1] * 24, 24)
            w2 = make_window("x \\boxed{5}", [0.4] * 24, [-0.2] * 24, 48, w1)
            w3 = make_window("x \\boxed{5} extra", [1.2] * 24, [-0.8] * 24, 72, w2)

            p_prefix = fitted.predict_windows([w1, w2], "gru")
            p_full = fitted.predict_windows([w1, w2, w3], "gru")
            # Causal invariance: future window w3 must not alter predictions at w1, w2
            np.testing.assert_allclose(p_prefix, p_full[:2], atol=1e-6)
            self.assertEqual(fitted.predict_windows([w1, w2], "logistic").shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
