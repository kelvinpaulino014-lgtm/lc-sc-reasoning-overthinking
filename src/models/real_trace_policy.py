"""Causal, gold-free features and deployment contract for the Phase 4 policy."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from src.models.stopping_policy import NeuralStoppingPolicy
from src.utils.answer_extractor import extract_last_boxed, normalize_math_answer


STRIDE = 24
MAX_TOKENS = 384
FEATURES = (
    "entropy", "delta_entropy", "mean_logprob", "length_fraction",
    "has_complete_answer", "same_as_previous_window",
)


@dataclass(frozen=True)
class Window:
    n_tokens: int
    answer: Optional[str]
    features: tuple[float, ...]


def complete_answer(text: str) -> Optional[str]:
    """Only accept a completed latest box, in accordance with the generation prompt."""
    raw = extract_last_boxed(text, require_closed=True)
    return normalize_math_answer(raw) if raw else None


def make_window(
    prefix_text: str,
    entropy_window: Sequence[float],
    logprob_window: Sequence[float],
    n_tokens: int,
    previous_window: Optional[Window] = None,
) -> Window:
    """Create one decision point from this prefix and the *new* tokens only."""
    prev_tokens = previous_window.n_tokens if previous_window else 0
    window_length = n_tokens - prev_tokens
    if (not 1 <= window_length <= STRIDE or n_tokens > MAX_TOKENS
            or len(entropy_window) != window_length or len(logprob_window) != window_length):
        raise ValueError("Window inputs must cover precisely the new, non-padded token positions.")
    entropy = float(np.mean(entropy_window))
    logp = float(np.mean(logprob_window))
    if not np.isfinite(entropy) or not np.isfinite(logp):
        raise ValueError("Non-finite token score in a real trace.")
    answer = complete_answer(prefix_text)
    same_answer = answer is not None and previous_window is not None and answer == previous_window.answer
    previous_entropy = previous_window.features[0] if previous_window else entropy
    features = (entropy, entropy - previous_entropy, logp, n_tokens / MAX_TOKENS,
                float(answer is not None), float(same_answer))
    return Window(n_tokens, answer, features)


def vote(answers: Sequence[Optional[str]]) -> tuple[Optional[str], float]:
    """Plurality over valid proposals, first valid vote breaks ties; coverage enters confidence."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None, 0.0
    answer, support = counts.most_common(1)[0]
    return answer, support / len(answers)


def consensus_after_two(answers: Sequence[Optional[str]]) -> bool:
    return len(answers) == 2 and answers[0] is not None and answers[0] == answers[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FittedPolicy:
    def __init__(self, config: dict, model: NeuralStoppingPolicy):
        self.config = config
        self.model = model.eval().cpu()
        self.thresholds = config["thresholds"]
        self.mean = np.asarray(config["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(config["scaler_scale"], dtype=np.float32)

    @torch.inference_mode()
    def predict_windows(self, windows: Sequence[Window], family: str) -> np.ndarray:
        """Sequential probabilities [P(no accuracy harm), P(current candidate correct)]."""
        features = np.asarray([w.features for w in windows], dtype=np.float32)
        x = (features - self.mean) / self.scale
        if family == "gru":
            logits, _ = self.model(torch.from_numpy(x)[None])
            return logits[0].sigmoid().numpy()
        if family == "logistic":
            coeff = np.asarray(self.config["logistic_coef"])
            bias = np.asarray(self.config["logistic_intercept"])
            return 1.0 / (1.0 + np.exp(-np.clip(x @ coeff.T + bias, -40, 40)))
        raise ValueError(f"Unknown model family: {family}")

    def should_stop(self, window: Window, probabilities: Sequence[float], family: str) -> bool:
        safety_threshold, correct_threshold = self.thresholds[family]
        return (window.answer is not None and probabilities[0] >= safety_threshold
                and probabilities[1] >= correct_threshold)


def load_policy(directory: Path, seed: Optional[int] = None) -> FittedPolicy:
    """Load fingerprinted numeric arrays without executing serialized Python objects."""
    directory = Path(directory)
    frozen = json.loads((directory / "frozen.json").read_text(encoding="utf8"))
    for filename, expected in frozen["sha256"].items():
        if sha256_file(directory / filename) != expected:
            raise ValueError(f"Frozen training artifact changed: {filename}")
    config = json.loads((directory / "config.json").read_text(encoding="utf8"))
    if config["features"] != list(FEATURES) or config["stride"] != STRIDE or config["max_tokens"] != MAX_TOKENS:
        raise ValueError("Frozen feature definition differs from the runtime.")
    seed = config["deployment_seed"] if seed is None else seed
    if seed not in config["seeds"]:
        raise ValueError("Seed absent from frozen training.")
    config["thresholds"] = dict(config["thresholds"])
    config["thresholds"]["gru"] = config["gru_thresholds_by_seed"][str(seed)]
    model = NeuralStoppingPolicy()
    with np.load(directory / f"gru_seed{seed}.npz", allow_pickle=False) as arrays:
        model.load_state_dict({name: torch.from_numpy(arrays[name]) for name in arrays.files})
    return FittedPolicy(config, model)
