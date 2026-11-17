"""
Compact Neural Stopping Policy (<3,000 parameters) and Real-Time
HuggingFace `StoppingCriteria` for Online GPU Length-Calibrated Self-Consistency (LC-SC Phase 3).

Inspired by Hao et al. (ICLR 2026, RL of Thoughts) and Huang et al. (ICLR 2026, CaTS):
- Uses a 1-layer GRU + LayerNorm + 2-head MLP (<1,500 params) over causal step-level
  entropy/confidence/length/cue trajectories.
- Outputs:
  1) `stop_prob`: probability that halting generation at step t avoids overthinking/over-correction.
  2) `corr_prob`: calibrated probability that the current intermediate candidate answer is correct.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import StoppingCriteria

from ..utils.answer_extractor import extract_answer

_HESITATION_RE = re.compile(
    r"\b(wait|alternatively|hmm|let's double check|re-evaluate|recalculate|maybe|on second thought|but wait)\b",
    flags=re.IGNORECASE,
)

_EXPLICIT_ANSWER_CUE_RE = re.compile(
    r"(\\boxed\{[^{}]+\}|####\s*[-+]?\d|(?:therefore|thus|so),?\s+(?:the\s+)?(?:final\s+)?answer\s+is\s+[-+]?\d)",
    flags=re.IGNORECASE,
)


class NeuralStoppingPolicy(nn.Module):
    """
    Compact sequential GRU + MLP stopping policy (<3,000 parameters).

    Input feature vector per step (dim=6):
        [step_entropy, delta_entropy, step_logprob, norm_cum_tokens, has_answer_cue, is_overthinking_transition]
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [B, T, input_dim]
            h0: Optional hidden state [1, B, hidden_dim]
        Returns:
            logits: [B, T, 2] where [..., 0] is stop_logit and [..., 1] is corr_logit
            hn: [1, B, hidden_dim]
        """
        out, hn = self.gru(x, h0)
        out = self.norm(out)
        logits = self.head(out)
        return logits, hn

    @torch.no_grad()
    def step_predict(
        self,
        feat_t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step online inference for a batch [B, 1, input_dim].
        Returns: (stop_prob [B], corr_prob [B], h_next [1, B, H])
        """
        logits, h_next = self.forward(feat_t, h_prev)
        probs = torch.sigmoid(logits[:, -1, :])
        return probs[:, 0], probs[:, 1], h_next


def build_step_features_from_trace(
    steps: Sequence[Dict[str, Any]],
    l_star: float,
    gold_answer: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a trace's step list into:
      - features: [T, 6] float32 array
      - stop_targets: [T] float32 array (1.0 if step is at/after overthinking onset or post-answer spike)
      - corr_targets: [T] float32 array (1.0 if current candidate answer == gold_answer)
    """
    n_steps = len(steps)
    feats = np.zeros((n_steps, 6), dtype=np.float32)
    stop_targets = np.zeros((n_steps,), dtype=np.float32)
    corr_targets = np.zeros((n_steps,), dtype=np.float32)

    prev_h = 0.0
    last_ans: Optional[str] = None
    seen_correct_then_lost = False

    for t, st in enumerate(steps):
        h_t = float(st.get("step_entropy", 0.0))
        delta_h = h_t - prev_h if t > 0 else 0.0
        prev_h = h_t

        conf_t = float(np.clip(st.get("step_confidence", 0.5), 1e-4, 0.9999))
        logprob_t = float(math.log(conf_t))
        cum_tok = float(st.get("cumulative_tokens", 0))
        norm_len = cum_tok / max(1.0, float(l_star))

        inter_ans = st.get("intermediate_answer")
        if inter_ans is not None and str(inter_ans).strip() != "":
            last_ans = str(inter_ans).strip()
        has_cue = 1.0 if last_ans is not None else 0.0
        is_over = 1.0 if bool(st.get("is_overthinking_transition", False)) else 0.0

        feats[t] = [h_t, delta_h, logprob_t, norm_len, has_cue, is_over]

        if gold_answer is not None:
            is_c = 1.0 if (last_ans is not None and str(last_ans) == str(gold_answer)) else 0.0
            corr_targets[t] = is_c
            if t > 0 and corr_targets[t - 1] == 1.0 and is_c == 0.0:
                seen_correct_then_lost = True
            # Optimal to stop once we have a candidate answer AND either overthinking starts or answer is stable & past 0.85*L*
            if has_cue > 0.5 and (is_over > 0.5 or seen_correct_then_lost or (norm_len >= 0.85 and is_c == 1.0)):
                stop_targets[t] = 1.0
        else:
            if has_cue > 0.5 and is_over > 0.5:
                stop_targets[t] = 1.0

    return feats, stop_targets, corr_targets


class EntropyNeuralStoppingCriteria(StoppingCriteria):
    """
    Real-time HuggingFace `StoppingCriteria` driven by `NeuralStoppingPolicy` on GPU.

    Tracks token-level Shannon entropy H_t and chosen token log-probabilities directly
    from `scores` during `model.generate(...)`. Every `step_stride` tokens (or on
    sentence/newline boundaries after `l_min`), updates the GRU policy and halts
    finished sequences early before overthinking occurs.
    """

    def __init__(
        self,
        policy: NeuralStoppingPolicy,
        tokenizer: Any,
        prompt_len: int,
        l_star: float,
        l_min: int = 48,
        stop_threshold: float = 0.62,
        step_stride: int = 20,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.policy = policy.to(device).eval()
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.l_star = float(max(1.0, l_star))
        self.l_min = int(max(1, l_min))
        self.stop_threshold = float(stop_threshold)
        self.step_stride = int(max(1, step_stride))
        self.device = device

        # Per-batch-item online state
        self.initialized = False
        self.batch_size = 0
        self.stopped: List[bool] = []
        self.stop_tokens: List[Optional[int]] = []
        self.token_entropies: List[List[float]] = []
        self.token_logprobs: List[List[float]] = []
        self.prev_step_entropy: List[float] = []
        self.last_eval_idx: List[int] = []
        self.h_states: List[Optional[torch.Tensor]] = []
        self.last_stop_probs: List[float] = []
        self.last_corr_probs: List[float] = []

    def _init_batch(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.stopped = [False] * batch_size
        self.stop_tokens = [None] * batch_size
        self.token_entropies = [[] for _ in range(batch_size)]
        self.token_logprobs = [[] for _ in range(batch_size)]
        self.prev_step_entropy = [0.0] * batch_size
        self.last_eval_idx = [0] * batch_size
        self.h_states = [None] * batch_size
        self.last_stop_probs = [0.0] * batch_size
        self.last_corr_probs = [0.5] * batch_size
        self.initialized = True

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: Tuple[torch.FloatTensor, ...],
        **kwargs: Any,
    ) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        if not self.initialized:
            self._init_batch(batch_size)

        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len <= 0 or not scores:
            return torch.tensor(self.stopped, dtype=torch.bool, device=input_ids.device)

        # Compute exact Shannon entropy & logprob for the latest generated token
        logits_t = scores[-1].float()  # [B, V]
        probs_t = F.softmax(logits_t, dim=-1)
        log_probs_t = F.log_softmax(logits_t, dim=-1)
        entropy_t = torch.special.entr(probs_t).sum(dim=-1)
        entropy_t = torch.nan_to_num(entropy_t, nan=0.42, posinf=1.5, neginf=0.0)

        last_tok_ids = input_ids[:, -1].unsqueeze(-1)
        chosen_lp = log_probs_t.gather(dim=-1, index=last_tok_ids).squeeze(-1)

        for b in range(batch_size):
            if self.stopped[b]:
                continue
            self.token_entropies[b].append(float(entropy_t[b].item()))
            self.token_logprobs[b].append(float(chosen_lp[b].item()))

            # Evaluate policy every `step_stride` tokens once past `l_min`
            if gen_len - self.last_eval_idx[b] >= self.step_stride:
                start_i = self.last_eval_idx[b]
                step_ents = self.token_entropies[b][start_i:]
                step_lps = self.token_logprobs[b][start_i:]
                h_t = float(np.mean(step_ents)) if step_ents else 0.0
                delta_h = h_t - self.prev_step_entropy[b] if start_i > 0 else 0.0
                self.prev_step_entropy[b] = h_t
                lp_t = float(np.mean(step_lps)) if step_lps else -0.5
                norm_len = float(gen_len) / self.l_star

                if gen_len >= self.l_min:
                    decoded_so_far = self.tokenizer.decode(
                        input_ids[b, self.prompt_len :], skip_special_tokens=True
                    )
                    has_cue = 1.0 if _EXPLICIT_ANSWER_CUE_RE.search(decoded_so_far) else 0.0
                    recent_text = self.tokenizer.decode(
                        input_ids[b, max(self.prompt_len, input_ids.shape[1] - self.step_stride) :],
                        skip_special_tokens=True,
                    )
                    is_over = 1.0 if _HESITATION_RE.search(recent_text) else 0.0
                else:
                    has_cue = 0.0
                    is_over = 0.0

                feat = torch.tensor(
                    [[[h_t, delta_h, lp_t, norm_len, has_cue, is_over]]],
                    dtype=torch.float32,
                    device=self.device,
                )
                stop_p, corr_p, h_next = self.policy.step_predict(feat, self.h_states[b])
                self.h_states[b] = h_next
                self.last_stop_probs[b] = float(stop_p[0].item())
                self.last_corr_probs[b] = float(corr_p[0].item())
                self.last_eval_idx[b] = gen_len

                if gen_len >= self.l_min and has_cue > 0.5 and self.last_stop_probs[b] >= self.stop_threshold:
                    self.stopped[b] = True
                    self.stop_tokens[b] = gen_len

        return torch.tensor(self.stopped, dtype=torch.bool, device=input_ids.device)
