"""
Módulo de rastreamento de Entropia de Shannon no nível de token/passo, métricas
de calibração probabilística (ECE, Brier Score, AUC-ROC), intervalos de confiança
via Bootstrap não-paramétrico e ajuste da curva em U invertido de Wu et al. (ICLR 2026).

Boas práticas de ML aplicadas:
- Tratamento explícito e seguro de valores ausentes (`None`, `NaN`, `Inf`).
- Separação estrita entre ajuste nos dados de validação/calibração e avaliação no teste.
- Reprodutibilidade determinística via gerador `np.random.default_rng(seed)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np


class _BootstrapTupleBase(NamedTuple):
    """Tupla nomeada base `(mean, ci_low, ci_high)` compatível com desempacotamento."""

    mean: float
    ci_low: float
    ci_high: float


class BootstrapCIResult(_BootstrapTupleBase):
    """
    Resultado do cálculo de Intervalo de Confiança via Bootstrap.
    Suporta simultaneamente:
    - Desempacotamento de tupla: `mean, low, high = bootstrap_ci(values)`
    - Acesso por atributos: `res.mean`, `res.ci_low`, `res.ci_high`, `res.lower`, `res.upper`, `res.half_width`
    - Acesso estilo dicionário: `res["mean"]`, `res["ci_low"]`, `res["ci_high"]`, `res["half_width"]`
    """

    @property
    def lower(self) -> float:
        return self.ci_low

    @property
    def upper(self) -> float:
        return self.ci_high

    @property
    def ci_lower(self) -> float:
        return self.ci_low

    @property
    def ci_upper(self) -> float:
        return self.ci_high

    @property
    def half_width(self) -> float:
        return 0.5 * (self.ci_high - self.ci_low)

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": float(self.mean),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "lower": float(self.ci_low),
            "upper": float(self.ci_high),
            "half_width": float(self.half_width),
        }

    def __getitem__(self, key: Union[int, slice, str]) -> Any:
        if isinstance(key, str):
            mapping = self.to_dict()
            if key in mapping:
                return mapping[key]
            raise KeyError(f"Chave '{key}' não encontrada em BootstrapCIResult.")
        return super().__getitem__(key)


@dataclass
class CalibrationBinStat:
    """Estatísticas de um bin individual do diagrama de confiabilidade (ECE)."""

    bin_idx: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    mean_accuracy: float
    calibration_gap: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_idx": self.bin_idx,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "mean_accuracy": self.mean_accuracy,
            "calibration_gap": self.calibration_gap,
        }


@dataclass
class InvertedUFitResult:
    """
    Parâmetros estimados da curva em U invertido (Acurácia vs. Comprimento L do CoT)
    conforme Wu et al. (ICLR 2026) no conjunto de validação.
    """

    l_star: float
    sigma_l: float
    l_star_parametric: float
    l_star_nonparametric: float
    beta_coeffs: Tuple[float, float, float]  # (beta_0, beta_1, beta_2) para beta_0 + beta_1*L + beta_2*L^2
    is_concave: bool
    peak_accuracy: float
    overthinking_drop: float
    bin_centers: List[float] = field(default_factory=list)
    bin_accuracies: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)
    recommended_tau_h: float = 0.65

    def predict_accuracy(self, length: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Prediz a acurácia esperada no modelo quadrático ajustado (limitada a [0, 1])."""
        b0, b1, b2 = self.beta_coeffs
        l_arr = np.asarray(length, dtype=float)
        pred = b0 + b1 * l_arr + b2 * (l_arr**2)
        pred_clipped = np.clip(pred, 0.0, 1.0)
        if np.ndim(length) == 0:
            return float(pred_clipped)
        return pred_clipped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l_star": float(self.l_star),
            "sigma_l": float(self.sigma_l),
            "l_star_parametric": float(self.l_star_parametric),
            "l_star_nonparametric": float(self.l_star_nonparametric),
            "beta_coeffs": [float(c) for c in self.beta_coeffs],
            "is_concave": bool(self.is_concave),
            "peak_accuracy": float(self.peak_accuracy),
            "overthinking_drop": float(self.overthinking_drop),
            "bin_centers": [float(x) for x in self.bin_centers],
            "bin_accuracies": [float(x) for x in self.bin_accuracies],
            "bin_counts": [int(c) for c in self.bin_counts],
            "recommended_tau_h": float(self.recommended_tau_h),
        }


def _clean_paired_arrays(
    arr1: Sequence[Any],
    arr2: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtra pares com valores ausentes (`None`, `NaN`, `Inf`) preservando alinhamento.
    """
    if len(arr1) != len(arr2):
        raise ValueError(
            f"As sequências devem ter o mesmo comprimento (recebido {len(arr1)} vs {len(arr2)})."
        )
    valid_x: List[float] = []
    valid_y: List[float] = []
    for x, y in zip(arr1, arr2):
        if x is None or y is None:
            continue
        try:
            fx = float(x)
            fy = float(y)
        except (TypeError, ValueError):
            continue
        if math.isnan(fx) or math.isinf(fx) or math.isnan(fy) or math.isinf(fy):
            continue
        valid_x.append(fx)
        valid_y.append(fy)
    return np.asarray(valid_x, dtype=float), np.asarray(valid_y, dtype=float)


def _clean_single_array(values: Iterable[Any]) -> np.ndarray:
    """Remove valores `None`, `NaN` ou `Inf` de uma sequência unidimensional."""
    cleaned: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not (math.isnan(fv) or math.isinf(fv)):
            cleaned.append(fv)
    return np.asarray(cleaned, dtype=float)


def compute_token_entropy(
    logits_or_probs: Any,
    from_logits: bool = False,
    eps: float = 1e-12,
    base: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calcula a Entropia de Shannon no nível de token:
        H_t = - \\sum_{v \\in V} p_t(v) \\log p_t(v)

    Parâmetros:
    - `logits_or_probs`: vetor 1D `(vocab_size,)` ou matriz 2D `(seq_len, vocab_size)`
      (aceita listas Python, `np.ndarray` ou tensores PyTorch).
    - `from_logits`: se `True`, aplica log-softmax numericamente estável antes do cálculo.
      Se `False`, interpreta a entrada como distribuição de probabilidades (normalizando
      caso a soma difira ligeiramente de 1).
    - `eps`: constante de estabilidade numérica para evitar `log(0)`.
    - `base`: base do logaritmo (`None` para log natural em nats; `2.0` para bits).
    """
    if hasattr(logits_or_probs, "detach"):
        arr = logits_or_probs.detach().cpu().numpy().astype(np.float64)
    else:
        arr = np.asarray(logits_or_probs, dtype=np.float64)

    if arr.size == 0:
        return 0.0 if arr.ndim <= 1 else np.zeros(0, dtype=np.float64)

    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[np.newaxis, :]

    # Substitui eventuais NaNs por valores neutros seguros
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)

    if from_logits:
        shifted = arr - np.max(arr, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.clip(np.sum(exp_logits, axis=-1, keepdims=True), eps, None)
    else:
        probs = np.clip(arr, 0.0, None)
        row_sums = np.sum(probs, axis=-1, keepdims=True)
        # Se a entrada parecer conter logits negativos ou não somar ~1, aplica softmax
        if np.any(arr < -1e-6) or np.any(np.abs(row_sums - 1.0) > 1e-2):
            if np.any(arr < -1e-6):
                shifted = arr - np.max(arr, axis=-1, keepdims=True)
                exp_logits = np.exp(shifted)
                probs = exp_logits / np.clip(np.sum(exp_logits, axis=-1, keepdims=True), eps, None)
            else:
                probs = np.where(row_sums > eps, probs / row_sums, 1.0 / arr.shape[-1])

    probs_safe = np.clip(probs, eps, 1.0)
    # Zera termos onde p == 0 exatamente
    ent = -np.sum(np.where(probs > 0.0, probs * np.log(probs_safe), 0.0), axis=-1)

    if base is not None and base > 0 and base != math.e:
        ent = ent / math.log(base)

    ent = np.clip(ent, 0.0, None)
    if is_1d:
        return float(ent[0])
    return ent


def compute_windowed_entropy(
    entropies: Sequence[float],
    window_size: int = 5,
    strict_window: bool = False,
) -> np.ndarray:
    """
    Calcula a entropia suavizada em janela móvel causal:
        \\bar{H}_k(t) = \\frac{1}{k} \\sum_{j=t-k+1}^{t} H_j

    Parâmetros:
    - `entropies`: sequência de entropias $H_1, \\dots, H_T$.
    - `window_size`: tamanho da janela $k \\ge 1$.
    - `strict_window`: se `False` (padrão), para $t < k$ calcula a média parcial dos
      primeiros $t$ elementos. Se `True`, retorna `NaN` para $t < k$.
    """
    if window_size <= 0:
        raise ValueError("window_size (k) deve ser >= 1.")

    arr = np.asarray(entropies, dtype=float)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=float)

    # Imputação causal simples caso existam NaNs isolados na série de entropia
    clean_arr = arr.copy()
    mask_nan = np.isnan(clean_arr) | np.isinf(clean_arr)
    if np.all(mask_nan):
        return np.zeros(n, dtype=float)
    if np.any(mask_nan):
        valid_median = float(np.nanmedian(clean_arr[~mask_nan]))
        for i in range(n):
            if mask_nan[i]:
                clean_arr[i] = clean_arr[i - 1] if i > 0 else valid_median

    out = np.zeros(n, dtype=float)
    cumsum = np.cumsum(np.insert(clean_arr, 0, 0.0))
    for t in range(n):
        if t + 1 < window_size:
            if strict_window:
                out[t] = np.nan
            else:
                out[t] = (cumsum[t + 1] - cumsum[0]) / float(t + 1)
        else:
            start_idx = t + 1 - window_size
            out[t] = (cumsum[t + 1] - cumsum[start_idx]) / float(window_size)
    return out


def compute_answer_entropy(answers: Sequence[Optional[str]]) -> float:
    """
    Calcula a entropia de Shannon $H_{\\text{ans}} = -\\sum_a p(a)\\log p(a)$ sobre as
    respostas finais extraídas de um conjunto de trilhas (usado no *Length-Filtered Vote*
    de Wu et al., ICLR 2026, Algoritmo 1).
    """
    valid = [str(a).strip() for a in answers if a is not None and str(a).strip() != ""]
    if len(valid) <= 1:
        return 0.0
    counts: Dict[str, int] = {}
    for a in valid:
        counts[a] = counts.get(a, 0) + 1
    total = float(len(valid))
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return float(compute_token_entropy(probs, from_logits=False))


class EntropyTracker:
    """
    Rastreador online de entropia de Shannon para monitoramento intratrilha ($L$-Stopping).
    Mantém o histórico de entropias por token/passo e sinaliza picos sustentados
    $\\bar{H}_k(t) > \\tau_H$ após um comprimento mínimo $L_{\\min}$.
    """

    def __init__(
        self,
        window_size: int = 3,
        tau_h: float = 0.65,
        l_min: int = 48,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size deve ser >= 1.")
        self.window_size = int(window_size)
        self.tau_h = float(tau_h)
        self.l_min = int(l_min)
        self.entropies: List[float] = []
        self.cumulative_tokens: int = 0
        self.spike_detected: bool = False
        self.spike_step_idx: Optional[int] = None

    def reset(self) -> None:
        self.entropies.clear()
        self.cumulative_tokens = 0
        self.spike_detected = False
        self.spike_step_idx = None

    def update(
        self,
        entropy_value: float,
        num_tokens: int = 1,
        has_candidate_answer: bool = True,
    ) -> bool:
        """
        Registra uma nova leitura de entropia (de um token ou passo lógico) e retorna
        `True` se o gatilho de parada precoce intratrilha ($L$-Stopping) foi ativado.
        """
        val = 0.0 if (entropy_value is None or math.isnan(float(entropy_value))) else max(0.0, float(entropy_value))
        self.entropies.append(val)
        self.cumulative_tokens += max(0, int(num_tokens))

        current_window_ent = self.current_window_entropy
        if (
            not self.spike_detected
            and self.cumulative_tokens >= self.l_min
            and has_candidate_answer
            and (val > self.tau_h or current_window_ent > self.tau_h)
        ):
            self.spike_detected = True
            self.spike_step_idx = len(self.entropies) - 1
            return True
        return False

    @property
    def current_window_entropy(self) -> float:
        if not self.entropies:
            return 0.0
        w = self.entropies[-self.window_size :]
        return float(sum(w) / len(w))

    @property
    def mean_entropy(self) -> float:
        if not self.entropies:
            return 0.0
        return float(sum(self.entropies) / len(self.entropies))

    @property
    def max_entropy(self) -> float:
        if not self.entropies:
            return 0.0
        return float(max(self.entropies))


def compute_calibration_bins(
    confidences: Sequence[float],
    accuracies: Sequence[Union[bool, int, float]],
    n_bins: int = 10,
) -> List[CalibrationBinStat]:
    """
    Calcula as estatísticas detalhadas de cada um dos `M = n_bins` intervalos em `[0, 1]`
    para construção do Diagrama de Confiabilidade (Reliability Diagram) e ECE.
    """
    if n_bins <= 0:
        raise ValueError("n_bins deve ser >= 1.")

    confs, accs = _clean_paired_arrays(confidences, accuracies)
    confs = np.clip(confs, 0.0, 1.0)
    accs = np.clip(accs, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_out: List[CalibrationBinStat] = []

    for m in range(n_bins):
        low = float(bin_edges[m])
        high = float(bin_edges[m + 1])
        if m == 0:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs > low) & (confs <= high)

        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_c = float(np.mean(confs[mask]))
            mean_a = float(np.mean(accs[mask]))
            gap = abs(mean_a - mean_c)
        else:
            mean_c = 0.5 * (low + high)
            mean_a = 0.0
            gap = 0.0

        bins_out.append(
            CalibrationBinStat(
                bin_idx=m,
                lower_bound=low,
                upper_bound=high,
                count=cnt,
                mean_confidence=mean_c,
                mean_accuracy=mean_a,
                calibration_gap=gap,
            )
        )
    return bins_out


def compute_ece(
    confidences: Sequence[float],
    accuracies: Sequence[Union[bool, int, float]],
    n_bins: int = 10,
) -> float:
    """
    Calcula o Expected Calibration Error (ECE) com `M = n_bins` bins de largura igual:
        ECE = \\sum_{m=1}^{M} \\frac{|B_m|}{n} |\\text{acc}(B_m) - \\text{conf}(B_m)|
    """
    confs, accs = _clean_paired_arrays(confidences, accuracies)
    n = len(confs)
    if n == 0:
        return 0.0

    bins = compute_calibration_bins(confs, accs, n_bins=n_bins)
    ece = sum((b.count / float(n)) * b.calibration_gap for b in bins if b.count > 0)
    return float(ece)


def compute_brier_score(
    confidences: Sequence[float],
    accuracies: Sequence[Union[bool, int, float]],
) -> float:
    """
    Calcula o Brier Score médio:
        BS = \\frac{1}{n} \\sum_{i=1}^{n} (\\hat{c}_i - y_i)^2
    """
    confs, accs = _clean_paired_arrays(confidences, accuracies)
    if len(confs) == 0:
        return 0.0
    confs = np.clip(confs, 0.0, 1.0)
    accs = np.clip(accs, 0.0, 1.0)
    return float(np.mean((confs - accs) ** 2))


def compute_auc_roc(
    confidences: Sequence[float],
    accuracies: Sequence[Union[bool, int, float]],
) -> float:
    """
    Calcula a Área sob a Curva ROC (AUC-ROC) da confiança para prever o acerto $y_i \\in \\{0, 1\\}$.
    Usa a estatística exata de Mann-Whitney U com tratamento de empates (equivalente a
    `sklearn.metrics.roc_auc_score` até precisão de máquina $10^{-15}$).
    Retorna `0.5` caso haja apenas uma classe presente ou entrada vazia.
    """
    confs, accs = _clean_paired_arrays(confidences, accuracies)
    if len(confs) < 2:
        return 0.5

    y_bin = (accs >= 0.5).astype(int)
    n_pos = int(np.sum(y_bin == 1))
    n_neg = int(np.sum(y_bin == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Ordena por score crescente e calcula postos médios (average ranks) para empates
    order = np.argsort(confs, kind="mergesort")
    sorted_scores = confs[order]
    sorted_y = y_bin[order]

    n = len(confs)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # Posto médio (1-indexed) para os elementos empatados entre i e j
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    sum_pos_ranks = float(np.sum(ranks[sorted_y == 1]))
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def bootstrap_ci(
    values: Sequence[Union[float, int, bool]],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapCIResult:
    """
    Calcula a média amostral e o Intervalo de Confiança não-paramétrico via Bootstrap
    (padrão: 95% de confiança com `n_boot=1000` reamostragens com reposição).

    Retorna `BootstrapCIResult(mean, ci_low, ci_high)`, que pode ser desempacotado
    como tupla `(mean, low, high)` ou acessado por atributos/chaves.
    """
    clean_vals = _clean_single_array(values)
    n = len(clean_vals)
    if n == 0:
        return BootstrapCIResult(mean=0.0, ci_low=0.0, ci_high=0.0)

    point_mean = float(np.mean(clean_vals))
    if n == 1 or n_boot <= 1 or np.allclose(clean_vals, clean_vals[0]):
        return BootstrapCIResult(mean=point_mean, ci_low=point_mean, ci_high=point_mean)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = np.mean(clean_vals[indices], axis=1)

    alpha = max(0.0, min(1.0, 1.0 - float(ci)))
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low = float(np.percentile(boot_means, low_pct))
    ci_high = float(np.percentile(boot_means, high_pct))

    return BootstrapCIResult(mean=point_mean, ci_low=ci_low, ci_high=ci_high)


def fit_inverted_u_curve(
    lengths: Sequence[Union[int, float]],
    accuracies: Sequence[Union[bool, int, float]],
    entropies: Optional[Sequence[float]] = None,
    n_bins: int = 8,
    min_bin_samples: int = 2,
) -> InvertedUFitResult:
    """
    Ajusta de forma paramétrica (regressão quadrática côncava ponderada) e
    não-paramétrica (bins empíricos suavizados) a curva em U invertido de
    Wu et al. (ICLR 2026) estritamente sobre o conjunto de validação.

    Estima sem vazamento de dados:
    - $L^*$: comprimento ótimo (em tokens) onde a acurácia atinge o máximo.
    - $\\sigma_L$: dispersão característica da janela ótima de comprimento.
    - $\\tau_H$: limiar recomendado de entropia para detecção de sobrecorreção.
    """
    lens, accs = _clean_paired_arrays(lengths, accuracies)
    if len(lens) == 0:
        return InvertedUFitResult(
            l_star=180.0,
            sigma_l=60.0,
            l_star_parametric=180.0,
            l_star_nonparametric=180.0,
            beta_coeffs=(0.5, 0.0, 0.0),
            is_concave=False,
            peak_accuracy=0.0,
            overthinking_drop=0.0,
        )

    lens = np.clip(lens, 1.0, None)
    accs = np.clip(accs, 0.0, 1.0)
    n = len(lens)

    if n < 4 or np.allclose(lens, lens[0]):
        l_mean = float(np.mean(lens))
        sig = max(20.0, float(np.std(lens)) if n > 1 else 0.25 * l_mean)
        return InvertedUFitResult(
            l_star=l_mean,
            sigma_l=sig,
            l_star_parametric=l_mean,
            l_star_nonparametric=l_mean,
            beta_coeffs=(float(np.mean(accs)), 0.0, 0.0),
            is_concave=False,
            peak_accuracy=float(np.mean(accs)),
            overthinking_drop=0.0,
            bin_centers=[l_mean],
            bin_accuracies=[float(np.mean(accs))],
            bin_counts=[n],
        )

    # 1. Estimador Não-Paramétrico por Bins de Comprimento
    eff_bins = max(3, min(int(n_bins), n // max(1, min_bin_samples)))
    l_min_val, l_max_val = float(np.min(lens)), float(np.max(lens))
    edges = np.linspace(l_min_val, l_max_val, eff_bins + 1)

    bin_centers: List[float] = []
    bin_accuracies: List[float] = []
    bin_counts: List[int] = []

    for b in range(eff_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (lens >= lo) & (lens <= hi) if b == 0 else (lens > lo) & (lens <= hi)
        cnt = int(np.sum(mask))
        if cnt >= max(1, min_bin_samples):
            bin_centers.append(float(np.mean(lens[mask])))
            bin_accuracies.append(float(np.mean(accs[mask])))
            bin_counts.append(cnt)

    if not bin_centers:
        bin_centers = [float(np.mean(lens))]
        bin_accuracies = [float(np.mean(accs))]
        bin_counts = [n]

    # Suavização leve de 3 pontos nos bins para robustez contra ruído amostral
    smoothed_acc = np.asarray(bin_accuracies, dtype=float).copy()
    if len(smoothed_acc) >= 3:
        for i in range(1, len(smoothed_acc) - 1):
            smoothed_acc[i] = 0.25 * bin_accuracies[i - 1] + 0.50 * bin_accuracies[i] + 0.25 * bin_accuracies[i + 1]

    max_smooth = float(np.max(smoothed_acc))
    top_mask = smoothed_acc >= (max_smooth - 0.025)
    if np.any(top_mask):
        top_centers = np.asarray(bin_centers, dtype=float)[top_mask]
        top_weights = np.clip(smoothed_acc[top_mask], 1e-3, None) * np.asarray(bin_counts, dtype=float)[top_mask]
        l_star_nonparam = float(np.average(top_centers, weights=top_weights))
    else:
        best_bin_idx = int(np.argmax(smoothed_acc))
        l_star_nonparam = float(bin_centers[best_bin_idx])
    peak_acc = float(max(bin_accuracies))
    tail_acc = float(bin_accuracies[-1]) if len(bin_accuracies) > 1 else peak_acc
    overthinking_drop = max(0.0, peak_acc - tail_acc)

    # 2. Estimador Paramétrico Quadrático: acc(L) = beta_0 + beta_1 * L + beta_2 * L^2
    # Normaliza L para evitar mal-condicionamento numérico na matriz de Vandermonde
    l_scale = max(1.0, float(np.max(lens)))
    u = lens / l_scale
    design = np.column_stack([np.ones_like(u), u, u**2])
    try:
        coeffs_u, _, _, _ = np.linalg.lstsq(design, accs, rcond=None)
        b0 = float(coeffs_u[0])
        b1 = float(coeffs_u[1]) / l_scale
        b2 = float(coeffs_u[2]) / (l_scale**2)
    except np.linalg.LinAlgError:
        b0, b1, b2 = float(np.mean(accs)), 0.0, 0.0

    is_concave = b2 < -1e-10
    p10, p85 = float(np.percentile(lens, 10)), float(np.percentile(lens, 85))
    if is_concave:
        vertex = -b1 / (2.0 * b2)
        l_star_param = float(np.clip(vertex, p10, p85))
    else:
        l_star_param = l_star_nonparam

    # Combinação robusta de L* (média ponderada entre paramétrico e não-paramétrico)
    l_star = 0.5 * l_star_param + 0.5 * l_star_nonparam if is_concave else l_star_nonparam
    l_star = float(np.clip(l_star, p10, p85))

    # Estimativa de sigma_L: dispersão das trilhas corretas (ou próximas a L*)
    correct_lens = lens[accs >= 0.5]
    if len(correct_lens) >= 3:
        mad = float(np.median(np.abs(correct_lens - np.median(correct_lens))))
        std_robust = 1.4826 * mad if mad > 1e-6 else float(np.std(correct_lens))
        sigma_l = max(float(np.std(correct_lens)), std_robust)
    else:
        sigma_l = float(np.std(lens))
    sigma_l = float(max(sigma_l, 0.15 * l_star, 16.0))

    # Recomendação de tau_H a partir das entropias de validação (se fornecidas)
    recommended_tau_h = 0.65
    if entropies is not None:
        _, valid_ents = _clean_paired_arrays(lens, entropies)
        if len(valid_ents) >= 4:
            # Limiar entre o percentil 70 das trilhas corretas e as trilhas longas
            recommended_tau_h = float(np.clip(np.percentile(valid_ents, 70), 0.30, 1.50))

    return InvertedUFitResult(
        l_star=l_star,
        sigma_l=sigma_l,
        l_star_parametric=l_star_param,
        l_star_nonparametric=l_star_nonparam,
        beta_coeffs=(b0, b1, b2),
        is_concave=is_concave,
        peak_accuracy=peak_acc,
        overthinking_drop=overthinking_drop,
        bin_centers=bin_centers,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        recommended_tau_h=recommended_tau_h,
    )
