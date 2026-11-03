"""
Estruturas de dados fundamentais (`CoTStep`, `CoTTrace`, `QuestionPrediction`) e
implementação dos 4 decodificadores de linha de base (Baselines) para avaliação
comparativa no GSM8K e MATH-500:

1. `GreedyCoTDecoder`: Decodificação gulosa ($N=1$, sem parada precoce).
2. `StandardSCDecoder`: Self-Consistency clássico (Wang et al., 2023; Aula 11, Slide 17).
3. `WuLengthFilteredVoteDecoder`: Length-Filtered Vote post-hoc (Wu et al., ICLR 2026, Algoritmo 1, p. 27).
4. `HuangCaTSDecoder`: Calibrated Test-Time Scaling com parada precoce no eixo $N$ (Huang et al., ICLR 2026).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..metrics.entropy_tracker import (
    bootstrap_ci,
    compute_answer_entropy,
    compute_auc_roc,
    compute_brier_score,
    compute_calibration_bins,
    compute_ece,
    fit_inverted_u_curve,
)
from ..utils.answer_extractor import (
    extract_answer,
    is_equivalent,
    normalize_math_answer,
    segment_cot_steps,
)


@dataclass
class CoTStep:
    """
    Representa um passo intermediário de raciocínio dentro de uma trilha CoT.
    Funciona tanto em geração online (PyTorch/HuggingFace) quanto em simulação
    exata sobre cache de trilhas.
    """

    step_idx: int
    cumulative_tokens: int
    step_entropy: float
    step_confidence: float = 0.80
    intermediate_answer: Optional[str] = None
    is_overthinking_transition: bool = False
    text: str = ""
    step_tokens: int = 0

    def __post_init__(self) -> None:
        self.step_idx = int(self.step_idx)
        self.cumulative_tokens = max(0, int(self.cumulative_tokens))
        self.step_tokens = max(0, int(self.step_tokens))
        if self.step_entropy is None or math.isnan(float(self.step_entropy)):
            self.step_entropy = 0.0
        else:
            self.step_entropy = max(0.0, float(self.step_entropy))

        if self.step_confidence is None or math.isnan(float(self.step_confidence)):
            self.step_confidence = float(max(0.05, min(0.99, math.exp(-0.45 * self.step_entropy))))
        else:
            self.step_confidence = float(np.clip(float(self.step_confidence), 1e-4, 1.0 - 1e-4))

        if self.intermediate_answer is not None:
            norm = normalize_math_answer(str(self.intermediate_answer))
            self.intermediate_answer = norm if norm is not None else str(self.intermediate_answer).strip()

        self.is_overthinking_transition = bool(self.is_overthinking_transition)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "cumulative_tokens": self.cumulative_tokens,
            "step_tokens": self.step_tokens,
            "step_entropy": self.step_entropy,
            "step_confidence": self.step_confidence,
            "intermediate_answer": self.intermediate_answer,
            "is_overthinking_transition": self.is_overthinking_transition,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], "CoTStep"]) -> "CoTStep":
        if isinstance(data, CoTStep):
            return data
        return cls(
            step_idx=int(data.get("step_idx", 0)),
            cumulative_tokens=int(data.get("cumulative_tokens", data.get("tokens", 0))),
            step_entropy=float(data.get("step_entropy", data.get("entropy", 0.0))),
            step_confidence=float(
                data.get(
                    "step_confidence",
                    data.get("confidence", math.exp(-0.45 * max(0.0, float(data.get("step_entropy", 0.0))))),
                )
            ),
            intermediate_answer=data.get("intermediate_answer"),
            is_overthinking_transition=bool(data.get("is_overthinking_transition", False)),
            text=str(data.get("text", "")),
            step_tokens=int(data.get("step_tokens", 0)),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class CoTTrace:
    """
    Estrutura de dados padrão para uma trilha de Cadeia de Pensamento (CoT).
    Armazena metadados da questão, comprimento total em tokens, resposta final,
    confiança bruta, entropia média e a lista detalhada de passos (`steps`).
    """

    question_id: str
    sample_idx: int
    full_text: str
    total_tokens: int
    final_answer: Optional[str]
    raw_confidence: float
    mean_entropy: float
    steps: List[CoTStep] = field(default_factory=list)
    gold_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    benchmark: str = "gsm8k"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question_id = str(self.question_id)
        self.sample_idx = int(self.sample_idx)
        self.full_text = str(self.full_text or "")

        # Converte dicionários ou objetos SegmentedStep em CoTStep
        converted_steps: List[CoTStep] = []
        if self.steps:
            prev_cum = 0
            for idx, s in enumerate(self.steps):
                if isinstance(s, CoTStep):
                    step_obj = s
                elif isinstance(s, dict):
                    step_obj = CoTStep.from_dict(s)
                else:
                    step_obj = CoTStep.from_dict(s.to_dict() if hasattr(s, "to_dict") else vars(s))
                if step_obj.step_tokens <= 0 and step_obj.cumulative_tokens > prev_cum:
                    step_obj.step_tokens = step_obj.cumulative_tokens - prev_cum
                prev_cum = step_obj.cumulative_tokens
                converted_steps.append(step_obj)
        elif self.full_text.strip():
            # Se nenhum passo foi passado mas há texto, segmenta automaticamente
            seg_steps = segment_cot_steps(
                self.full_text,
                total_tokens=self.total_tokens if self.total_tokens and self.total_tokens > 0 else None,
                default_entropy=self.mean_entropy if self.mean_entropy and not math.isnan(self.mean_entropy) else 0.25,
                default_confidence=self.raw_confidence if self.raw_confidence and not math.isnan(self.raw_confidence) else 0.80,
            )
            converted_steps = [CoTStep.from_dict(s.to_dict()) for s in seg_steps]
        self.steps = converted_steps

        if self.total_tokens is None or int(self.total_tokens) <= 0:
            if self.steps:
                self.total_tokens = max(1, int(self.steps[-1].cumulative_tokens))
            else:
                self.total_tokens = max(1, len(self.full_text.split()))
        else:
            self.total_tokens = int(self.total_tokens)

        if self.final_answer is None and self.full_text.strip():
            self.final_answer = extract_answer(self.full_text, benchmark=self.benchmark)
        elif self.final_answer is not None:
            norm_ans = normalize_math_answer(str(self.final_answer))
            self.final_answer = norm_ans if norm_ans is not None else str(self.final_answer).strip()

        if self.mean_entropy is None or math.isnan(float(self.mean_entropy)):
            if self.steps:
                self.mean_entropy = float(np.mean([s.step_entropy for s in self.steps]))
            else:
                self.mean_entropy = 0.25
        else:
            self.mean_entropy = max(0.0, float(self.mean_entropy))

        if self.raw_confidence is None or math.isnan(float(self.raw_confidence)):
            if self.steps:
                self.raw_confidence = float(np.mean([s.step_confidence for s in self.steps]))
            else:
                self.raw_confidence = float(max(0.05, min(0.99, math.exp(-0.45 * self.mean_entropy))))
        else:
            self.raw_confidence = float(np.clip(float(self.raw_confidence), 1e-4, 1.0 - 1e-4))

        if self.gold_answer is not None:
            norm_gold = normalize_math_answer(str(self.gold_answer))
            self.gold_answer = norm_gold if norm_gold is not None else str(self.gold_answer).strip()
            if self.is_correct is None:
                self.is_correct = is_equivalent(self.final_answer, self.gold_answer)

    def evaluate_correctness(self, gold_answer: Optional[str] = None) -> bool:
        target_gold = gold_answer if gold_answer is not None else self.gold_answer
        if target_gold is not None:
            return is_equivalent(self.final_answer, target_gold)
        return bool(self.is_correct) if self.is_correct is not None else False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "sample_idx": self.sample_idx,
            "full_text": self.full_text,
            "total_tokens": self.total_tokens,
            "final_answer": self.final_answer,
            "raw_confidence": self.raw_confidence,
            "mean_entropy": self.mean_entropy,
            "steps": [s.to_dict() for s in self.steps],
            "gold_answer": self.gold_answer,
            "is_correct": self.is_correct,
            "benchmark": self.benchmark,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], "CoTTrace"]) -> "CoTTrace":
        if isinstance(data, CoTTrace):
            return data
        raw_steps = data.get("steps", [])
        if isinstance(raw_steps, str):
            try:
                raw_steps = json.loads(raw_steps)
            except Exception:
                raw_steps = []
        steps_objs = [CoTStep.from_dict(s) for s in raw_steps] if raw_steps else []
        return cls(
            question_id=str(data.get("question_id", "q_0")),
            sample_idx=int(data.get("sample_idx", 0)),
            full_text=str(data.get("full_text", data.get("response", ""))),
            total_tokens=int(data.get("total_tokens", 0)),
            final_answer=data.get("final_answer", data.get("answer")),
            raw_confidence=float(data.get("raw_confidence", data.get("confidence", 0.8))),
            mean_entropy=float(data.get("mean_entropy", data.get("entropy", 0.25))),
            steps=steps_objs,
            gold_answer=data.get("gold_answer", data.get("correct_answer")),
            is_correct=data.get("is_correct"),
            benchmark=str(data.get("benchmark", "gsm8k")),
            metadata=dict(data.get("metadata", {})),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class QuestionPrediction:
    """
    Resultado padronizado da decodificação de uma única questão por qualquer método.
    """

    question_id: str
    method_name: str
    predicted_answer: Optional[str]
    gold_answer: Optional[str]
    is_correct: bool
    num_samples_used: int
    total_tokens_used: int
    full_tokens_budget: int
    winning_confidence: float
    num_l_stopped_traces: int = 0
    stopped_early_n: bool = False
    vote_distribution: Dict[str, float] = field(default_factory=dict)
    per_trace_tokens_used: List[int] = field(default_factory=list)
    per_trace_answers: List[Optional[str]] = field(default_factory=list)

    @property
    def token_savings_ratio(self) -> float:
        if self.full_tokens_budget <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.total_tokens_used / float(self.full_tokens_budget)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "method_name": self.method_name,
            "predicted_answer": self.predicted_answer,
            "gold_answer": self.gold_answer,
            "is_correct": bool(self.is_correct),
            "num_samples_used": int(self.num_samples_used),
            "total_tokens_used": int(self.total_tokens_used),
            "full_tokens_budget": int(self.full_tokens_budget),
            "token_savings_ratio": float(self.token_savings_ratio),
            "winning_confidence": float(self.winning_confidence),
            "num_l_stopped_traces": int(self.num_l_stopped_traces),
            "stopped_early_n": bool(self.stopped_early_n),
            "vote_distribution": dict(self.vote_distribution),
            "per_trace_tokens_used": list(self.per_trace_tokens_used),
            "per_trace_answers": list(self.per_trace_answers),
        }

    def __getitem__(self, key: str) -> Any:
        if key == "token_savings_ratio":
            return self.token_savings_ratio
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "token_savings_ratio":
            return self.token_savings_ratio
        return getattr(self, key, default)


class ConfidenceCalibrator:
    """
    Calibrador supervisionado de confiança (Platt Scaling Logístico + Regressão Isotônica via PAVA)
    ajustado estritamente sobre o conjunto de validação (sem vazamento para o conjunto de teste).

    Suporta calibração univariada (apenas `raw_confidence`) ou bivariada
    (`raw_confidence` e `entropy`).
    """

    def __init__(self, method: str = "hybrid") -> None:
        if method not in ("platt", "isotonic", "hybrid"):
            raise ValueError("method deve ser 'platt', 'isotonic' ou 'hybrid'.")
        self.method = method
        self.is_fitted: bool = False
        # Pesos do modelo logístico: logit(p) = w0 + w1 * logit(c_raw) + w2 * H
        self.w0: float = 0.0
        self.w1: float = 1.0
        self.w2: float = 0.0
        # Pontos da função degrau isotônica (PAVA)
        self.iso_x: Optional[np.ndarray] = None
        self.iso_y: Optional[np.ndarray] = None

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        p_clip = np.clip(p, eps, 1.0 - eps)
        return np.log(p_clip / (1.0 - p_clip))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z_clip = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    @staticmethod
    def _fit_pava(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pool-Adjacent-Violators Algorithm (PAVA) exato para Regressão Isotônica
        monotonamente não-decrescente em $O(n)$.
        """
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ys = y[order]

        # Cada bloco é [soma_y, peso, x_min, x_max]
        blocks: List[List[float]] = []
        for xi, yi in zip(xs, ys):
            blocks.append([float(yi), 1.0, float(xi), float(xi)])
            while len(blocks) >= 2:
                mean_prev = blocks[-2][0] / blocks[-2][1]
                mean_curr = blocks[-1][0] / blocks[-1][1]
                if mean_prev > mean_curr:
                    b_last = blocks.pop()
                    blocks[-1][0] += b_last[0]
                    blocks[-1][1] += b_last[1]
                    blocks[-1][3] = b_last[3]
                else:
                    break

        knot_x: List[float] = []
        knot_y: List[float] = []
        for sum_y, cnt, xmin, xmax in blocks:
            val = float(np.clip(sum_y / cnt, 0.02, 0.98))
            knot_x.extend([xmin, xmax])
            knot_y.extend([val, val])

        # Garante estrita monotonicidade em x para np.interp
        ux, idx = np.unique(np.asarray(knot_x, dtype=float), return_index=True)
        uy = np.asarray(knot_y, dtype=float)[idx]
        return ux, uy

    def fit(
        self,
        raw_confidences: Sequence[float],
        correctness: Sequence[Union[bool, int, float]],
        entropies: Optional[Sequence[float]] = None,
    ) -> "ConfidenceCalibrator":
        confs = np.asarray(raw_confidences, dtype=float)
        ys = np.asarray(correctness, dtype=float)
        mask = ~(np.isnan(confs) | np.isinf(confs) | np.isnan(ys) | np.isinf(ys))
        if entropies is not None:
            ents = np.asarray(entropies, dtype=float)
            mask = mask & ~(np.isnan(ents) | np.isinf(ents))
            ents = ents[mask]
        else:
            ents = None

        confs = np.clip(confs[mask], 1e-4, 1.0 - 1e-4)
        ys = np.clip(ys[mask], 0.0, 1.0)

        if len(confs) < 4 or np.allclose(ys, ys[0]):
            self.w0 = 0.0
            self.w1 = 1.0
            self.w2 = 0.0
            self.is_fitted = True
            return self

        # 1. Ajuste Logístico (Platt Scaling com regularização L2 leve via Newton-Raphson)
        x1 = self._logit(confs)
        x2 = ents if ents is not None else np.zeros_like(x1)
        use_ent = ents is not None and float(np.std(x2)) > 1e-5

        if use_ent:
            X = np.column_stack([np.ones_like(x1), x1, x2])
            w = np.array([0.0, 0.8, -0.2], dtype=float)
        else:
            X = np.column_stack([np.ones_like(x1), x1])
            w = np.array([0.0, 1.0], dtype=float)

        reg = 1e-2
        for _ in range(25):
            logits = X @ w
            preds = self._sigmoid(logits)
            grad = (X.T @ (preds - ys)) / len(ys) + reg * np.r_[0.0, w[1:]]
            weights_diag = np.clip(preds * (1.0 - preds), 1e-4, None)
            hess = (X.T * weights_diag) @ X / len(ys) + reg * np.diag(np.r_[0.0, np.ones(len(w) - 1)])
            try:
                step = np.linalg.solve(hess, grad)
                w = w - step
                if float(np.max(np.abs(step))) < 1e-6:
                    break
            except np.linalg.LinAlgError:
                break

        self.w0 = float(w[0])
        self.w1 = float(max(0.05, w[1]))  # Garante associação positiva com a confiança
        self.w2 = float(min(0.0, w[2])) if use_ent else 0.0  # Entropia maior não deve inflar confiança

        # 2. Ajuste Isotônico (PAVA) sobre os scores logísticos (ou confianças brutas)
        platt_scores = self._predict_platt(confs, ents)
        if self.method in ("isotonic", "hybrid") and len(confs) >= 6:
            base_x = confs if self.method == "isotonic" else platt_scores
            self.iso_x, self.iso_y = self._fit_pava(base_x, ys)

        self.is_fitted = True
        return self

    def _predict_platt(
        self,
        confs: np.ndarray,
        entropies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x1 = self._logit(confs)
        if entropies is not None:
            logits = self.w0 + self.w1 * x1 + self.w2 * entropies
        else:
            logits = self.w0 + self.w1 * x1
        return self._sigmoid(logits)

    def predict(
        self,
        raw_confidence: Union[float, Sequence[float], np.ndarray],
        entropy: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        scalar_input = np.ndim(raw_confidence) == 0
        c_arr = np.asarray(raw_confidence, dtype=float)
        c_arr = np.nan_to_num(c_arr, nan=0.5, posinf=0.99, neginf=0.01)
        c_arr = np.clip(c_arr, 1e-4, 1.0 - 1e-4)

        e_arr: Optional[np.ndarray] = None
        if entropy is not None:
            e_arr = np.asarray(entropy, dtype=float)
            e_arr = np.nan_to_num(e_arr, nan=0.25, posinf=2.0, neginf=0.0)

        if not self.is_fitted:
            out = c_arr
        else:
            platt = self._predict_platt(c_arr, e_arr)
            if self.method == "platt" or self.iso_x is None or self.iso_y is None:
                out = platt
            elif self.method == "isotonic":
                out = np.interp(c_arr, self.iso_x, self.iso_y)
            else:
                iso_pred = np.interp(platt, self.iso_x, self.iso_y)
                out = 0.5 * platt + 0.5 * iso_pred

        out = np.clip(out, 0.01, 0.99)
        if scalar_input:
            return float(np.reshape(out, (-1,))[0])
        return out


def group_traces_by_question(
    traces_or_dict: Union[Dict[str, Sequence[CoTTrace]], Sequence[Union[CoTTrace, Dict[str, Any]]]]
) -> Dict[str, List[CoTTrace]]:
    """
    Converte uma lista plana de `CoTTrace` (ou dicionário `{question_id: [traces]}`)
    em um dicionário ordenado `{question_id: [CoTTrace, ...]}` com trilhas ordenadas por `sample_idx`.
    """
    grouped: Dict[str, List[CoTTrace]] = defaultdict(list)
    if isinstance(traces_or_dict, dict):
        for qid, t_list in traces_or_dict.items():
            for t in t_list:
                grouped[str(qid)].append(CoTTrace.from_dict(t))
    else:
        for t in traces_or_dict:
            t_obj = CoTTrace.from_dict(t)
            grouped[t_obj.question_id].append(t_obj)

    for qid in grouped:
        grouped[qid].sort(key=lambda tr: tr.sample_idx)
    return dict(grouped)


class BaseCoTDecoder:
    """
    Interface base para todos os decodificadores de raciocínio em tempo de teste.
    """

    def __init__(self, method_name: str, n_max: int = 16) -> None:
        self.method_name = method_name
        self.n_max = max(1, int(n_max))
        self.is_fitted: bool = False

    def fit(
        self,
        val_traces: Union[Dict[str, Sequence[CoTTrace]], Sequence[CoTTrace]],
        gold_answers: Optional[Dict[str, str]] = None,
    ) -> "BaseCoTDecoder":
        """Ajusta hiperparâmetros/calibradores estritamente no split de validação."""
        self.is_fitted = True
        return self

    def predict_question(
        self,
        traces: Sequence[Union[CoTTrace, Dict[str, Any]]],
        gold_answer: Optional[str] = None,
    ) -> QuestionPrediction:
        """Executa a estratégia de decodificação sobre as trilhas candidatas de uma questão."""
        raise NotImplementedError

    def decode(
        self,
        traces: Sequence[Union[CoTTrace, Dict[str, Any]]],
        gold_answer: Optional[str] = None,
    ) -> QuestionPrediction:
        """Alias para `predict_question`."""
        return self.predict_question(traces, gold_answer=gold_answer)

    def evaluate_dataset(
        self,
        test_traces: Union[Dict[str, Sequence[CoTTrace]], Sequence[CoTTrace]],
        gold_answers: Optional[Dict[str, str]] = None,
        n_boot: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Avalia o decodificador em todas as questões de um conjunto de teste (ex.: GSM8K ou MATH-500),
        retornando acurácia, consumo médio de tokens, amostras utilizadas, ECE, Brier Score,
        AUC-ROC e intervalos de confiança de 95% via Bootstrap.
        """
        grouped = group_traces_by_question(test_traces)
        predictions: List[QuestionPrediction] = []

        for qid, q_traces in grouped.items():
            if not q_traces:
                continue
            gold = (
                gold_answers.get(qid)
                if gold_answers is not None and qid in gold_answers
                else q_traces[0].gold_answer
            )
            pred = self.predict_question(q_traces, gold_answer=gold)
            predictions.append(pred)

        if not predictions:
            return {
                "method_name": self.method_name,
                "num_questions": 0,
                "accuracy": 0.0,
                "accuracy_ci": (0.0, 0.0, 0.0),
                "mean_tokens": 0.0,
                "tokens_ci": (0.0, 0.0, 0.0),
                "mean_samples_used": 0.0,
                "token_savings_ratio": 0.0,
                "ece": 0.0,
                "brier_score": 0.0,
                "auc_roc": 0.5,
                "predictions": [],
            }

        acc_list = [1.0 if p.is_correct else 0.0 for p in predictions]
        tok_list = [float(p.total_tokens_used) for p in predictions]
        full_tok_list = [float(p.full_tokens_budget) for p in predictions]
        samples_list = [float(p.num_samples_used) for p in predictions]
        conf_list = [float(p.winning_confidence) for p in predictions]
        l_stop_counts = [int(p.num_l_stopped_traces) for p in predictions]

        acc_boot = bootstrap_ci(acc_list, n_boot=n_boot, ci=ci, seed=seed)
        tok_boot = bootstrap_ci(tok_list, n_boot=n_boot, ci=ci, seed=seed)
        samp_boot = bootstrap_ci(samples_list, n_boot=n_boot, ci=ci, seed=seed)

        ece_val = compute_ece(conf_list, acc_list, n_bins=10)
        brier_val = compute_brier_score(conf_list, acc_list)
        auc_val = compute_auc_roc(conf_list, acc_list)
        cal_bins = compute_calibration_bins(conf_list, acc_list, n_bins=10)

        total_used_sum = sum(tok_list)
        total_full_sum = sum(full_tok_list)
        overall_savings = (
            max(0.0, 1.0 - (total_used_sum / total_full_sum)) if total_full_sum > 0 else 0.0
        )

        return {
            "method_name": self.method_name,
            "num_questions": len(predictions),
            "accuracy": acc_boot.mean,
            "accuracy_ci": acc_boot,
            "accuracy_ci_low": acc_boot.ci_low,
            "accuracy_ci_high": acc_boot.ci_high,
            "mean_tokens": tok_boot.mean,
            "tokens_ci": tok_boot,
            "tokens_ci_low": tok_boot.ci_low,
            "tokens_ci_high": tok_boot.ci_high,
            "mean_samples_used": samp_boot.mean,
            "samples_ci": samp_boot,
            "token_savings_ratio": float(overall_savings),
            "total_l_stopped_traces": int(sum(l_stop_counts)),
            "early_n_stop_rate": float(np.mean([1.0 if p.stopped_early_n else 0.0 for p in predictions])),
            "ece": float(ece_val),
            "brier_score": float(brier_val),
            "auc_roc": float(auc_val),
            "calibration_bins": [b.to_dict() for b in cal_bins],
            "predictions": predictions,
        }


class GreedyCoTDecoder(BaseCoTDecoder):
    """
    Método 1: Greedy Chain-of-Thought ($N=1$, primeira trilha gerada sem parada precoce).
    Serve como linha de base de custo mínimo por amostra única, porém vulnerável a
    overthinking na trilha única e sem ganho de ensemble.
    """

    def __init__(self) -> None:
        super().__init__(method_name="Greedy CoT (N=1)", n_max=1)

    def predict_question(
        self,
        traces: Sequence[Union[CoTTrace, Dict[str, Any]]],
        gold_answer: Optional[str] = None,
    ) -> QuestionPrediction:
        trace_objs = [CoTTrace.from_dict(t) for t in traces]
        if not trace_objs:
            raise ValueError("A lista de trilhas não pode estar vazia.")

        first = trace_objs[0]
        target_gold = gold_answer if gold_answer is not None else first.gold_answer
        pred_ans = first.final_answer
        correct = is_equivalent(pred_ans, target_gold) if target_gold is not None else bool(first.is_correct)
        full_budget = sum(t.total_tokens for t in trace_objs)

        return QuestionPrediction(
            question_id=first.question_id,
            method_name=self.method_name,
            predicted_answer=pred_ans,
            gold_answer=target_gold,
            is_correct=bool(correct),
            num_samples_used=1,
            total_tokens_used=first.total_tokens,
            full_tokens_budget=full_budget,
            winning_confidence=float(np.clip(first.raw_confidence, 0.01, 0.99)),
            num_l_stopped_traces=0,
            stopped_early_n=False,
            vote_distribution={str(pred_ans): 1.0} if pred_ans is not None else {},
            per_trace_tokens_used=[first.total_tokens],
            per_trace_answers=[pred_ans],
        )


class StandardSCDecoder(BaseCoTDecoder):
    """
    Método 2: Standard Self-Consistency (Wang et al., 2023; Aula 11, Slide 17).
    Amostra todas as $N$ trilhas completas até o final e realiza votação majoritária
    simples (não-ponderada). Custo computacional: 100% dos tokens das $N$ trilhas.
    """

    def __init__(self, n_max: int = 16) -> None:
        super().__init__(method_name="Standard Self-Consistency (SC)", n_max=n_max)

    def predict_question(
        self,
        traces: Sequence[Union[CoTTrace, Dict[str, Any]]],
        gold_answer: Optional[str] = None,
    ) -> QuestionPrediction:
        trace_objs = [CoTTrace.from_dict(t) for t in traces[: self.n_max]]
        if not trace_objs:
            raise ValueError("A lista de trilhas não pode estar vazia.")

        target_gold = gold_answer if gold_answer is not None else trace_objs[0].gold_answer
        vote_counts: Dict[str, int] = defaultdict(int)
        conf_sums: Dict[str, float] = defaultdict(float)

        for t in trace_objs:
            if t.final_answer is not None:
                vote_counts[t.final_answer] += 1
                conf_sums[t.final_answer] += t.raw_confidence

        if vote_counts:
            # Desempate determinístico pela maior confiança acumulada
            best_ans = max(
                vote_counts.keys(),
                key=lambda a: (vote_counts[a], conf_sums[a]),
            )
            total_valid = sum(vote_counts.values())
            consensus_ratio = vote_counts[best_ans] / float(total_valid)
            mean_raw_conf = conf_sums[best_ans] / float(vote_counts[best_ans])
            winning_conf = float(np.clip(0.5 * consensus_ratio + 0.5 * mean_raw_conf, 0.01, 0.99))
            vote_dist = {k: v / float(total_valid) for k, v in vote_counts.items()}
        else:
            best_ans = None
            winning_conf = 0.01
            vote_dist = {}

        total_tokens = sum(t.total_tokens for t in trace_objs)
        correct = is_equivalent(best_ans, target_gold) if target_gold is not None else False

        return QuestionPrediction(
            question_id=trace_objs[0].question_id,
            method_name=self.method_name,
            predicted_answer=best_ans,
            gold_answer=target_gold,
            is_correct=bool(correct),
            num_samples_used=len(trace_objs),
            total_tokens_used=total_tokens,
            full_tokens_budget=total_tokens,
            winning_confidence=winning_conf,
            num_l_stopped_traces=0,
            stopped_early_n=False,
            vote_distribution=vote_dist,
            per_trace_tokens_used=[t.total_tokens for t in trace_objs],
            per_trace_answers=[t.final_answer for t in trace_objs],
        )


class WuLengthFilteredVoteDecoder(BaseCoTDecoder):
    """
    Método 3: Length-Filtered Vote (Wu et al., ICLR 2026 - *When More is Less*, Algoritmo 1, p. 27).

    Funcionamento:
    - No split de validação (`fit`), estima a curva em U invertido ($L^*$ e $\\sigma_L$).
    - Na inferência (`predict_question`), gera **todas as $N$ trilhas completas** até o fim
      (portanto, o custo computacional real em tokens é **100% dos tokens das $N$ trilhas**,
      pois a filtragem é estritamente *post-hoc*).
    - Ordena as $N$ trilhas por comprimento $L_{(1)} \\le \\dots \\le L_{(N)}$, avalia janelas
      candidatas de comprimento e seleciona a janela $W^*$ que minimiza a entropia de Shannon
      das respostas $H_{\\text{ans}}(W)$ próxima à zona ótima $L^*$ (descartando trilhas com
      *overthinking* extremo na hora de votar).
    """

    def __init__(
        self,
        n_max: int = 16,
        window_fraction: float = 0.5,
        min_window_traces: int = 2,
        length_prior_weight: float = 0.35,
    ) -> None:
        super().__init__(method_name="Length-Filtered Vote (Wu et al., 2026)", n_max=n_max)
        self.window_fraction = float(np.clip(window_fraction, 0.2, 1.0))
        self.min_window_traces = max(1, int(min_window_traces))
        self.length_prior_weight = float(max(0.0, length_prior_weight))
        self.l_star: float = 180.0
        self.sigma_l: float = 65.0
        self.u_curve_fit: Optional[InvertedUFitResult] = None

    def fit(
        self,
        val_traces: Union[Dict[str, Sequence[CoTTrace]], Sequence[CoTTrace]],
        gold_answers: Optional[Dict[str, str]] = None,
    ) -> "WuLengthFilteredVoteDecoder":
        grouped = group_traces_by_question(val_traces)
        lengths: List[float] = []
        accs: List[float] = []

        for qid, q_traces in grouped.items():
            gold = (
                gold_answers.get(qid)
                if gold_answers is not None and qid in gold_answers
                else (q_traces[0].gold_answer if q_traces else None)
            )
            for t in q_traces[: self.n_max]:
                lengths.append(float(t.total_tokens))
                accs.append(1.0 if t.evaluate_correctness(gold) else 0.0)

        self.u_curve_fit = fit_inverted_u_curve(lengths, accs)
        self.l_star = self.u_curve_fit.l_star
        self.sigma_l = self.u_curve_fit.sigma_l
        self.is_fitted = True
        return self

    def predict_question(
        self,
        traces: Sequence[Union[CoTTrace, Dict[str, Any]]],
        gold_answer: Optional[str] = None,
    ) -> QuestionPrediction:
        trace_objs = [CoTTrace.from_dict(t) for t in traces[: self.n_max]]
        if not trace_objs:
            raise ValueError("A lista de trilhas não pode estar vazia.")

        target_gold = gold_answer if gold_answer is not None else trace_objs[0].gold_answer
        valid_traces = [t for t in trace_objs if t.final_answer is not None]
        if not valid_traces:
            valid_traces = trace_objs

        # Ordena por comprimento L crescente (Algoritmo 1 de Wu et al., 2026)
        sorted_traces = sorted(valid_traces, key=lambda t: t.total_tokens)
        n_valid = len(sorted_traces)
        win_size = max(self.min_window_traces, int(round(self.window_fraction * n_valid)))
        win_size = min(win_size, n_valid)

        candidate_windows: List[List[CoTTrace]] = []
        if n_valid <= win_size:
            candidate_windows.append(sorted_traces)
        else:
            for start_i in range(0, n_valid - win_size + 1):
                candidate_windows.append(sorted_traces[start_i : start_i + win_size])

        # Adiciona também a janela centrada em L* aprendido na validação (se possuir >= min_window_traces)
        in_band = [
            t
            for t in sorted_traces
            if (self.l_star - 1.25 * self.sigma_l) <= t.total_tokens <= (self.l_star + 1.25 * self.sigma_l)
        ]
        if len(in_band) >= self.min_window_traces:
            candidate_windows.append(in_band)

        # Seleciona a janela de menor entropia de resposta combinada à proximidade de L*
        best_window = candidate_windows[0]
        best_score = float("inf")
        for w_traces in candidate_windows:
            ans_ent = compute_answer_entropy([t.final_answer for t in w_traces])
            mean_len = float(np.mean([t.total_tokens for t in w_traces]))
            # Penaliza janelas cujo comprimento médio excede fortemente L* (zona de overthinking)
            len_deviation = max(0.0, mean_len - self.l_star) / max(1.0, self.sigma_l)
            dist_center = abs(mean_len - self.l_star) / max(1.0, self.sigma_l)
            score = ans_ent + self.length_prior_weight * (0.7 * len_deviation + 0.3 * dist_center)
            if score < best_score:
                best_score = score
                best_window = w_traces

        # Votação majoritária dentro da janela selecionada W*
        vote_counts: Dict[str, int] = defaultdict(int)
        conf_sums: Dict[str, float] = defaultdict(float)
        for t in best_window:
            if t.final_answer is not None:
                vote_counts[t.final_answer] += 1
                conf_sums[t.final_answer] += t.raw_confidence

        if vote_counts:
            best_ans = max(vote_counts.keys(), key=lambda a: (vote_counts[a], conf_sums[a]))
            total_w = sum(vote_counts.values())
            consensus = vote_counts[best_ans] / float(total_w)
            mean_c = conf_sums[best_ans] / float(vote_counts[best_ans])
            winning_conf = float(np.clip(0.5 * consensus + 0.5 * mean_c, 0.01, 0.99))
            vote_dist = {k: v / float(total_w) for k, v in vote_counts.items()}
        else:
            best_ans = trace_objs[0].final_answer
            winning_conf = float(np.clip(trace_objs[0].raw_confidence, 0.01, 0.99))
            vote_dist = {}

        # Custo computacional post-hoc: TODAS as N trilhas foram geradas integralmente!
        total_tokens_all = sum(t.total_tokens for t in trace_objs)
        correct = is_equivalent(best_ans, target_gold) if target_gold is not None else False

        return QuestionPrediction(
            question_id=trace_objs[0].question_id,
            method_name=self.method_name,
            predicted_answer=best_ans,
            gold_answer=target_gold,
            is_correct=bool(correct),
            num_samples_used=len(trace_objs),
            total_tokens_used=total_tokens_all,
            full_tokens_budget=total_tokens_all,
            winning_confidence=winning_conf,
            num_l_stopped_traces=0,
            stopped_early_n=False,
            vote_distribution=vote_dist,
            per_trace_tokens_used=[t.total_tokens for t in trace_objs],
            per_trace_answers=[t.final_answer for t in trace_objs],
        )


class HuangCaTSDecoder(BaseCoTDecoder):
    """
    Método 4: CaTS - Calibrated Test-Time Scaling (Huang et al., ICLR 2026).

    Funcionamento:
    - Calibra a confiança $\\hat{c}_i$ no conjunto de validação (`fit`) usando regressão
      isotônica / Platt scaling sobre a confiança bruta da trilha completa.
    - Durante a inferência (`predict_question`), amostra trilhas sequencialmente $i = 1, \\dots, N_{\\max}$
      e interrompe a geração de novas trilhas ($N$-Stopping) assim que o consenso ponderado
      atinge o limiar $\\gamma$ (para $i \\ge N_{\\min}$).
    - Limitação documentada em Huang et al. (2026): gera cada trilha $i$ até o seu final
      completo $L_i^{\\text{full}}$ sem truncar overthinking no eixo $L$, ficando sujeito a
      trilhas longas superconfiantes e ao custo integral de tokens das trilhas amostradas.
    """

    def __init__(
        self,
        n_max: int = 16,
        n_min: int = 3,
        gamma: float = 0.72,
        calibration_method: str = "hybrid",
    ) -> None:
        super().__init__(method_name="CaTS (Huang et al., 2026)", n_max=n_max)
        self.n_min = max(1, int(n_min))
        self.gamma = float(np.clip(gamma, 0.1, 1.0))
        self.calibrator = ConfidenceCalibrator(method=calibration_method)

    def fit(
        self,
        val_traces: Union[Dict[str, Sequence[CoTTrace]], Sequence[CoTTrace]],
        gold_answers: Optional[Dict[str, str]] = None,
    ) -> "HuangCaTSDecoder":
        grouped = group_traces_by_question(val_traces)
        raw_confs: List[float] = []
        correctness: List[float] = []

        for qid, q_traces in grouped.items():
            gold = (
                gold_answers.get(qid)
                if gold_answers is not None and qid in gold_answers
                else (q_traces[0].gold_answer if q_traces else None)
            )
            for t in q_traces[: self.n_max]:
                raw_confs.append(float(t.raw_confidence))
                correctness.append(1.0 if t.evaluate_correctness(gold) else 0.0)

        # CaTS calibra apenas a confiança final sem controlar o comprimento L_i
        self.calibrator.fit(raw_confs, correctness, entropies=None)
        self.is_fitted = True
        return self

    def predict_question(
        self,
        traces: Sequence[Union[CoTTrace, Dict[str, Any]]],
        gold_answer: Optional[str] = None,
    ) -> QuestionPrediction:
        trace_objs = [CoTTrace.from_dict(t) for t in traces[: self.n_max]]
        if not trace_objs:
            raise ValueError("A lista de trilhas não pode estar vazia.")

        target_gold = gold_answer if gold_answer is not None else trace_objs[0].gold_answer
        full_budget = sum(t.total_tokens for t in trace_objs)

        weighted_votes: Dict[str, float] = defaultdict(float)
        cal_confs_by_ans: Dict[str, List[float]] = defaultdict(list)
        per_trace_tokens: List[int] = []
        per_trace_answers: List[Optional[str]] = []
        stopped_early_n = False

        for idx, t in enumerate(trace_objs):
            sample_num = idx + 1
            # CaTS não trunca o comprimento individual L_i
            per_trace_tokens.append(t.total_tokens)
            per_trace_answers.append(t.final_answer)

            c_hat = float(self.calibrator.predict(t.raw_confidence))
            if t.final_answer is not None:
                weighted_votes[t.final_answer] += c_hat
                cal_confs_by_ans[t.final_answer].append(c_hat)

            if sample_num >= self.n_min and weighted_votes:
                total_w = sum(weighted_votes.values())
                top_ans = max(weighted_votes.keys(), key=lambda a: weighted_votes[a])
                ratio = weighted_votes[top_ans] / total_w if total_w > 0 else 0.0
                if ratio >= self.gamma:
                    if sample_num < len(trace_objs):
                        stopped_early_n = True
                    break

        if weighted_votes:
            best_ans = max(weighted_votes.keys(), key=lambda a: weighted_votes[a])
            total_w = sum(weighted_votes.values())
            consensus_ratio = weighted_votes[best_ans] / total_w if total_w > 0 else 0.5
            mean_c_hat = float(np.mean(cal_confs_by_ans[best_ans])) if cal_confs_by_ans[best_ans] else 0.5
            winning_conf = float(np.clip(0.5 * consensus_ratio + 0.5 * mean_c_hat, 0.01, 0.99))
            vote_dist = {k: v / total_w for k, v in weighted_votes.items()} if total_w > 0 else {}
        else:
            best_ans = trace_objs[0].final_answer
            winning_conf = float(self.calibrator.predict(trace_objs[0].raw_confidence))
            vote_dist = {}

        tokens_used = sum(per_trace_tokens)
        correct = is_equivalent(best_ans, target_gold) if target_gold is not None else False

        return QuestionPrediction(
            question_id=trace_objs[0].question_id,
            method_name=self.method_name,
            predicted_answer=best_ans,
            gold_answer=target_gold,
            is_correct=bool(correct),
            num_samples_used=len(per_trace_tokens),
            total_tokens_used=tokens_used,
            full_tokens_budget=full_budget,
            winning_confidence=winning_conf,
            num_l_stopped_traces=0,
            stopped_early_n=stopped_early_n,
            vote_distribution=vote_dist,
            per_trace_tokens_used=per_trace_tokens,
            per_trace_answers=per_trace_answers,
        )


def save_traces_jsonl(traces: Iterable[CoTTrace], path: Union[str, Path]) -> None:
    """Salva uma coleção de `CoTTrace` em formato JSON Lines (`.jsonl`)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for tr in traces:
            obj = tr.to_dict() if isinstance(tr, CoTTrace) else CoTTrace.from_dict(tr).to_dict()
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_traces_jsonl(path: Union[str, Path]) -> List[CoTTrace]:
    """Carrega uma lista de `CoTTrace` a partir de um arquivo `.jsonl` ou `.json`."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo de trilhas não encontrado: {in_path}")

    text = in_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("[") or text.startswith("{") and "\n" not in text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [CoTTrace.from_dict(item) for item in parsed]
            if isinstance(parsed, dict) and "traces" in parsed:
                return [CoTTrace.from_dict(item) for item in parsed["traces"]]
        except Exception:
            pass

    traces: List[CoTTrace] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        traces.append(CoTTrace.from_dict(json.loads(line)))
    return traces
