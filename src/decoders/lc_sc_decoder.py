"""
Implementação do método proposto **LC-SC (*Length-Calibrated Self-Consistency*)**
e de suas 3 variantes de estudo de ablação:
- `LCSCDecoder`: Método completo proposto (L-Stopping + Ponderação por Comprimento + N-Stopping).
- `LCSC_Ablation_NoLStop`: Sem truncamento online de comprimento ($L$-Stopping desativado).
- `LCSC_Ablation_NoNStop`: Sem parada precoce de amostras ($N$-Stopping desativado, $N=N_{\\max}$ fixo).
- `LCSC_Ablation_NoLenPenalty`: Sem penalização exponencial de comprimento no voto ($\\lambda = 0$).

Fundamentação Teórica (ICLR 2026):
1. WU et al. (2026) - Curva em U invertido da acurácia vs. comprimento $L$ do CoT.
2. HUANG et al. (2026) - Escalonamento calibrado em tempo de teste (CaTS) no eixo $N$.
3. HAO et al. (2026) - Fenômeno de sobrecorreção (*over-correction*) em SLMs após picos de incerteza.
4. FENG et al. (2026, Oral) - Diluição de atenção pelo acúmulo excessivo de tokens em contexto.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..metrics.entropy_tracker import (
    EntropyTracker,
    InvertedUFitResult,
    compute_token_entropy,
    compute_windowed_entropy,
    fit_inverted_u_curve,
)
from ..utils.answer_extractor import (
    detect_overthinking_transition,
    extract_answer,
    extract_intermediate_answer,
    is_equivalent,
    segment_cot_steps,
)
from .baselines import (
    BaseCoTDecoder,
    ConfidenceCalibrator,
    CoTStep,
    CoTTrace,
    QuestionPrediction,
    group_traces_by_question,
)


@dataclass
class TruncatedTraceResult:
    """
    Estado resultante de uma trilha CoT após o monitoramento intratrilha ($L$-Stopping).
    """

    extracted_answer: Optional[str]
    used_tokens: int
    full_tokens: int
    effective_raw_confidence: float
    effective_mean_entropy: float
    was_l_stopped: bool
    stopped_step_idx: Optional[int]


class LCSCDecoder(BaseCoTDecoder):
    """
    **Método 5 (Proposto): Length-Calibrated Self-Consistency (LC-SC)**

    Controla simultaneamente e de forma online as duas dimensões de custo de
    Test-Time Compute ($N \\times L$):
    1. **Calibração no Split de Validação (`fit`):** Aprende o calibrador de confiança
       $\\hat{c}_i$, o comprimento ótimo $L^*$, a escala de dispersão $\\sigma_L$ e o
       limiar de pico de entropia $\\tau_H$ estritamente nos dados de validação.
    2. **$L$-Stopping Online (Poda Intratrilha Guiada por Entropia):** Monitora os passos
       da cadeia $i$ após $L_{\\min}$ tokens. Ao detectar um pico de entropia
       ($\\text{step\\_entropy} > \\tau_H$ ou média móvel $\\bar{H}_k > \\tau_H$) após a
       derivação já ter alcançado uma resposta intermediária candidata válida, interrompe
       a trilha no passo $t_{\\text{stop}}$, recupera a resposta pré-sobrecorreção e
       economiza todos os tokens subsequentes ($L_i^{\\text{used}} = L_{t_{\\text{stop}}} \\ll L_i^{\\text{full}}$).
    3. **Votação Calibrada com Penalização Suave de Comprimento:**
       $$w_i = \\hat{c}_i \\cdot \\exp\\left(-\\lambda \\frac{\\max(0, L_i^{\\text{used}} - L^*)}{\\sigma_L}\\right)$$
    4. **$N$-Stopping Calibrado (Orçamento Adaptativo de Amostras):** Após cada trilha
       $i \\ge N_{\\min}$, verifica se a resposta líder acumula fração de peso
       $\\frac{\\max_a W(a)}{\\sum_b W(b)} \\ge \\gamma$. Se sim, interrompe a amostragem
       em $N_{\\text{adap}} = i$.
    """

    def __init__(
        self,
        n_max: int = 16,
        n_min: int = 3,
        l_min: int = 48,
        l_star: float = 180.0,
        sigma_l: float = 65.0,
        tau_h: float = 0.95,
        lambda_penalty: float = 0.85,
        gamma: float = 0.78,
        entropy_window: int = 2,
        enable_l_stop: bool = True,
        enable_n_stop: bool = True,
        auto_calibrate_tau_h: bool = True,
        calibration_method: str = "hybrid",
        method_name: str = "LC-SC (Proposed)",
    ) -> None:
        super().__init__(method_name=method_name, n_max=n_max)
        self.n_min = max(1, int(n_min))
        self.l_min = max(1, int(l_min))
        self.l_star = float(l_star)
        self.sigma_l = max(1.0, float(sigma_l))
        self.tau_h = float(tau_h)
        self.lambda_penalty = max(0.0, float(lambda_penalty))
        self.gamma = float(np.clip(gamma, 0.1, 1.0))
        self.entropy_window = max(1, int(entropy_window))
        self.enable_l_stop = bool(enable_l_stop)
        self.enable_n_stop = bool(enable_n_stop)
        self.auto_calibrate_tau_h = bool(auto_calibrate_tau_h)
        self.calibrator = ConfidenceCalibrator(method=calibration_method)
        self.u_curve_fit: Optional[InvertedUFitResult] = None

    def compute_trace_weight(self, calibrated_confidence: float, used_tokens: int) -> float:
        """
        Calcula o peso da trilha $i$ na votação ponderada do LC-SC:
            w_i = \\hat{c}_i \\cdot \\exp\\left(-\\lambda \\frac{\\max(0, L_i^{\\text{used}} - L^*)}{\\sigma_L}\\right)
        """
        c_hat = float(np.clip(calibrated_confidence, 1e-4, 1.0))
        if self.lambda_penalty <= 0.0:
            return c_hat
        excess_tokens = max(0.0, float(used_tokens) - float(self.l_star))
        exponent = -self.lambda_penalty * (excess_tokens / max(1e-6, float(self.sigma_l)))
        exponent = max(-50.0, min(0.0, exponent))
        return float(c_hat * math.exp(exponent))

    def truncate_trace_online(self, trace: CoTTrace) -> TruncatedTraceResult:
        """
        Executa (ou simula de maneira exata sobre os passos da trilha) o mecanismo de
        $L$-Stopping online guiado por entropia de Shannon.

        Retorna `TruncatedTraceResult` contendo a resposta extraída (pré-sobrecorreção
        caso truncada), o número de tokens efetivamente consumidos $L_i^{\\text{used}}$,
        a confiança e entropia até o ponto de parada e o indicador `was_l_stopped`.
        """
        if not self.enable_l_stop or not trace.steps:
            return TruncatedTraceResult(
                extracted_answer=trace.final_answer,
                used_tokens=trace.total_tokens,
                full_tokens=trace.total_tokens,
                effective_raw_confidence=trace.raw_confidence,
                effective_mean_entropy=trace.mean_entropy,
                was_l_stopped=False,
                stopped_step_idx=None,
            )

        tracker = EntropyTracker(
            window_size=self.entropy_window,
            tau_h=self.tau_h,
            l_min=self.l_min,
        )

        last_valid_answer: Optional[str] = None
        clean_confidences: List[float] = []
        clean_entropies: List[float] = []
        prev_cum_tokens = 0
        num_steps = len(trace.steps)

        for idx, step in enumerate(trace.steps):
            step_tokens = max(1, step.cumulative_tokens - prev_cum_tokens)
            has_candidate_before = last_valid_answer is not None

            # Verifica se este passo dispara o gatilho de pico de entropia APÓS já termos
            # atingido uma resposta intermediária candidata válida em passo anterior
            spike_triggered = tracker.update(
                entropy_value=step.step_entropy,
                num_tokens=step_tokens,
                has_candidate_answer=has_candidate_before,
            )

            # Se o passo é explicitamente uma transição de overthinking ("Wait", "Alternatively", etc.)
            # com entropia elevada (> 0.85 * tau_h) e já temos resposta candidata válida, também trunca
            overthinking_spike = (
                has_candidate_before
                and step.cumulative_tokens >= self.l_min
                and step.is_overthinking_transition
                and step.step_entropy >= 0.85 * self.tau_h
            )

            if spike_triggered or overthinking_spike:
                # Se o pico ocorreu no último passo da lista, contabiliza apenas os tokens
                # até o início do pico (garantindo L_used < L_full quando o passo final é longo)
                if idx < num_steps - 1 and step.cumulative_tokens < trace.total_tokens:
                    used_tok = step.cumulative_tokens
                else:
                    buffer_tok = max(1, min(16, step_tokens // 3))
                    used_tok = min(trace.total_tokens - 1, prev_cum_tokens + buffer_tok)
                    used_tok = max(1, used_tok)

                eff_conf = (
                    float(np.mean(clean_confidences))
                    if clean_confidences
                    else float(max(trace.raw_confidence, step.step_confidence))
                )
                eff_ent = (
                    float(np.mean(clean_entropies))
                    if clean_entropies
                    else float(min(trace.mean_entropy, step.step_entropy))
                )

                return TruncatedTraceResult(
                    extracted_answer=last_valid_answer,
                    used_tokens=int(used_tok),
                    full_tokens=trace.total_tokens,
                    effective_raw_confidence=eff_conf,
                    effective_mean_entropy=eff_ent,
                    was_l_stopped=True,
                    stopped_step_idx=idx,
                )

            # Passo aceito como parte da derivação limpa: atualiza resposta intermediária candidata
            if step.intermediate_answer is not None:
                last_valid_answer = step.intermediate_answer
            clean_confidences.append(step.step_confidence)
            clean_entropies.append(step.step_entropy)
            prev_cum_tokens = step.cumulative_tokens

        # Se nenhum pico de sobrecorreção ocorreu, usa a trilha completa
        final_ans = trace.final_answer if trace.final_answer is not None else last_valid_answer
        return TruncatedTraceResult(
            extracted_answer=final_ans,
            used_tokens=trace.total_tokens,
            full_tokens=trace.total_tokens,
            effective_raw_confidence=trace.raw_confidence,
            effective_mean_entropy=trace.mean_entropy,
            was_l_stopped=False,
            stopped_step_idx=None,
        )

    def fit(
        self,
        val_traces: Union[Dict[str, Sequence[CoTTrace]], Sequence[CoTTrace]],
        gold_answers: Optional[Dict[str, str]] = None,
    ) -> "LCSCDecoder":
        """
        Ajusta estritamente nos dados de validação (sem vazamento para o conjunto de teste):
        1. $L^*$ e $\\sigma_L$ via ajuste da curva em U invertido de Wu et al. (2026).
        2. Limiar de pico de entropia $\\tau_H$ separando passos de derivação normal vs.
           passos de hesitação/sobrecorreção na validação (se `auto_calibrate_tau_h=True`).
        3. Calibrador de confiança $\\hat{c}_i = g(c_i^{\\text{raw}}, \\bar{H}_i)$.
        """
        grouped = group_traces_by_question(val_traces)
        lengths: List[float] = []
        accs: List[float] = []
        raw_confs: List[float] = []
        mean_ents: List[float] = []
        normal_step_ents: List[float] = []
        overthinking_step_ents: List[float] = []

        for qid, q_traces in grouped.items():
            gold = (
                gold_answers.get(qid)
                if gold_answers is not None and qid in gold_answers
                else (q_traces[0].gold_answer if q_traces else None)
            )
            for t in q_traces[: self.n_max]:
                is_corr = 1.0 if t.evaluate_correctness(gold) else 0.0
                lengths.append(float(t.total_tokens))
                accs.append(is_corr)
                raw_confs.append(float(t.raw_confidence))
                mean_ents.append(float(t.mean_entropy))

                seen_valid_ans = False
                for st in t.steps:
                    if seen_valid_ans and (
                        st.is_overthinking_transition
                        or (gold is not None and st.intermediate_answer is not None and not is_equivalent(st.intermediate_answer, gold))
                    ):
                        overthinking_step_ents.append(float(st.step_entropy))
                    else:
                        normal_step_ents.append(float(st.step_entropy))
                    if st.intermediate_answer is not None:
                        seen_valid_ans = True

        # 1. Estima L* e sigma_L na validação
        if lengths:
            self.u_curve_fit = fit_inverted_u_curve(lengths, accs, entropies=mean_ents)
            self.l_star = float(self.u_curve_fit.l_star)
            self.sigma_l = float(self.u_curve_fit.sigma_l)
            # Atualiza L_min de forma conservadora caso L* seja muito pequeno nos dados
            self.l_min = int(min(self.l_min, max(12, int(0.35 * self.l_star))))

        # 2. Calibra tau_H nos passos de validação se auto_calibrate_tau_h estiver ativo
        if self.auto_calibrate_tau_h and normal_step_ents:
            norm_arr = np.asarray(normal_step_ents, dtype=float)
            p80_normal = float(np.percentile(norm_arr, 80))
            if overthinking_step_ents:
                over_arr = np.asarray(overthinking_step_ents, dtype=float)
                p30_over = float(np.percentile(over_arr, 30))
                if p30_over > p80_normal:
                    self.tau_h = float(0.5 * (p80_normal + p30_over))
                else:
                    self.tau_h = float(np.clip(np.percentile(norm_arr, 85), 0.35, 1.25))
            else:
                self.tau_h = float(np.clip(np.percentile(norm_arr, 82), 0.35, 1.25))

        # 3. Ajusta o calibrador de confiança supervisionado na validação
        # Se L-Stopping estiver ativo, calibra nas estatísticas pós-L-Stopping da validação
        cal_confs: List[float] = []
        cal_ents: List[float] = []
        cal_accs: List[float] = []
        for qid, q_traces in grouped.items():
            gold = (
                gold_answers.get(qid)
                if gold_answers is not None and qid in gold_answers
                else (q_traces[0].gold_answer if q_traces else None)
            )
            for t in q_traces[: self.n_max]:
                trunc = self.truncate_trace_online(t)
                cal_confs.append(trunc.effective_raw_confidence)
                cal_ents.append(trunc.effective_mean_entropy)
                if gold is not None:
                    cal_accs.append(1.0 if is_equivalent(trunc.extracted_answer, gold) else 0.0)
                else:
                    cal_accs.append(1.0 if t.is_correct else 0.0)

        if cal_confs:
            self.calibrator.fit(cal_confs, cal_accs, entropies=cal_ents)

        self.is_fitted = True
        return self

    def predict_question(
        self,
        traces: Sequence[Union[CoTTrace, Dict[str, Any]]],
        gold_answer: Optional[str] = None,
    ) -> QuestionPrediction:
        """
        Executa a decodificação LC-SC completa em uma questão:
        - Para cada trilha $i = 1, \\dots, N_{\\max}$:
          (a) Aplica $L$-Stopping online para podar picos de entropia pós-resposta candidata.
          (b) Calcula a confiança calibrada $\\hat{c}_i$ e o peso penalizado por comprimento $w_i$.
          (c) Acumula $W(a) \\leftarrow W(a) + w_i$ e verifica o critério de $N$-Stopping ($\\ge \\gamma$).
        """
        trace_objs = [CoTTrace.from_dict(t) for t in traces[: self.n_max]]
        if not trace_objs:
            raise ValueError("A lista de trilhas não pode estar vazia.")

        target_gold = gold_answer if gold_answer is not None else trace_objs[0].gold_answer
        full_budget = sum(t.total_tokens for t in trace_objs)

        weighted_votes: Dict[str, float] = defaultdict(float)
        calibrated_confs_by_ans: Dict[str, List[float]] = defaultdict(list)
        per_trace_tokens: List[int] = []
        per_trace_answers: List[Optional[str]] = []
        num_l_stopped = 0
        stopped_early_n = False

        for idx, t in enumerate(trace_objs):
            sample_num = idx + 1
            trunc = self.truncate_trace_online(t)
            if trunc.was_l_stopped:
                num_l_stopped += 1

            per_trace_tokens.append(trunc.used_tokens)
            per_trace_answers.append(trunc.extracted_answer)

            c_hat = float(
                self.calibrator.predict(
                    trunc.effective_raw_confidence,
                    entropy=trunc.effective_mean_entropy,
                )
            )
            w_i = self.compute_trace_weight(c_hat, trunc.used_tokens)

            if trunc.extracted_answer is not None:
                weighted_votes[trunc.extracted_answer] += w_i
                calibrated_confs_by_ans[trunc.extracted_answer].append(c_hat)

            # (c) N-Stopping Calibrado (Orçamento Adaptativo de Amostras)
            if self.enable_n_stop and sample_num >= self.n_min and weighted_votes:
                total_w = sum(weighted_votes.values())
                sorted_weights = sorted(weighted_votes.values(), reverse=True)
                top_w = sorted_weights[0]
                second_w = sorted_weights[1] if len(sorted_weights) > 1 else 0.0
                lead_ratio = top_w / total_w if total_w > 0 else 0.0

                # Verifica consenso ponderado >= gamma ou maioria inalcançável na segunda metade do orçamento
                remaining_samples = len(trace_objs) - sample_num
                avg_w = total_w / float(sample_num)
                insurmountable = (
                    sample_num >= max(self.n_min + 1, len(trace_objs) // 2)
                    and (top_w - second_w) > remaining_samples * avg_w * 0.95
                )

                if lead_ratio >= self.gamma or insurmountable:
                    if sample_num < len(trace_objs):
                        stopped_early_n = True
                    break

        if weighted_votes:
            best_ans = max(weighted_votes.keys(), key=lambda a: weighted_votes[a])
            total_w = sum(weighted_votes.values())
            consensus_ratio = weighted_votes[best_ans] / total_w if total_w > 0 else 0.5
            mean_c_hat = (
                float(np.mean(calibrated_confs_by_ans[best_ans]))
                if calibrated_confs_by_ans[best_ans]
                else 0.5
            )
            winning_conf = float(np.clip(0.55 * consensus_ratio + 0.45 * mean_c_hat, 0.01, 0.99))
            vote_dist = {k: v / total_w for k, v in weighted_votes.items()} if total_w > 0 else {}
        else:
            best_ans = trace_objs[0].final_answer
            winning_conf = float(
                self.calibrator.predict(
                    trace_objs[0].raw_confidence,
                    entropy=trace_objs[0].mean_entropy,
                )
            )
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
            num_l_stopped_traces=num_l_stopped,
            stopped_early_n=stopped_early_n,
            vote_distribution=vote_dist,
            per_trace_tokens_used=per_trace_tokens,
            per_trace_answers=per_trace_answers,
        )


class LCSC_Ablation_NoLStop(LCSCDecoder):
    """
    Variante de Ablação 1 (`LC-SC w/o L-Stopping`):
    Desativa a poda intratrilha ($L$-Stopping = `False`), mantendo apenas a ponderação
    exponencial por comprimento ($\\lambda > 0$) e a parada precoce de amostras ($N$-Stopping = `True`).
    Demonstra o impacto de não cortar a geração de tokens durante o overthinking.
    """

    def __init__(
        self,
        n_max: int = 16,
        n_min: int = 3,
        l_min: int = 48,
        l_star: float = 180.0,
        sigma_l: float = 65.0,
        tau_h: float = 0.65,
        lambda_penalty: float = 0.50,
        gamma: float = 0.72,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_max=n_max,
            n_min=n_min,
            l_min=l_min,
            l_star=l_star,
            sigma_l=sigma_l,
            tau_h=tau_h,
            lambda_penalty=lambda_penalty,
            gamma=gamma,
            enable_l_stop=False,
            enable_n_stop=True,
            method_name="LC-SC (w/o L-Stop)",
            **kwargs,
        )


class LCSC_Ablation_NoNStop(LCSCDecoder):
    """
    Variante de Ablação 2 (`LC-SC w/o N-Stopping`):
    Desativa a parada precoce no eixo de amostras ($N$-Stopping = `False`, usando sempre
    $N = N_{\\max}$ fixo), mas mantém o truncamento intratrilha ($L$-Stopping = `True`) e a
    ponderação por comprimento ($\\lambda > 0$).
    """

    def __init__(
        self,
        n_max: int = 16,
        n_min: int = 3,
        l_min: int = 48,
        l_star: float = 180.0,
        sigma_l: float = 65.0,
        tau_h: float = 0.65,
        lambda_penalty: float = 0.50,
        gamma: float = 0.72,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_max=n_max,
            n_min=n_min,
            l_min=l_min,
            l_star=l_star,
            sigma_l=sigma_l,
            tau_h=tau_h,
            lambda_penalty=lambda_penalty,
            gamma=gamma,
            enable_l_stop=True,
            enable_n_stop=False,
            method_name="LC-SC (w/o N-Stop)",
            **kwargs,
        )


class LCSC_Ablation_NoLenPenalty(LCSCDecoder):
    """
    Variante de Ablação 3 (`LC-SC w/o Length Penalty`, $\\lambda = 0$):
    Define $\\lambda = 0$, mantendo o truncamento intratrilha ($L$-Stopping = `True`) e a
    parada precoce de amostras ($N$-Stopping = `True`), porém sem decaimento exponencial
    de comprimento no peso de votação.
    """

    def __init__(
        self,
        n_max: int = 16,
        n_min: int = 3,
        l_min: int = 48,
        l_star: float = 180.0,
        sigma_l: float = 65.0,
        tau_h: float = 0.65,
        gamma: float = 0.72,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("lambda_penalty", None)
        super().__init__(
            n_max=n_max,
            n_min=n_min,
            l_min=l_min,
            l_star=l_star,
            sigma_l=sigma_l,
            tau_h=tau_h,
            lambda_penalty=0.0,
            gamma=gamma,
            enable_l_stop=True,
            enable_n_stop=True,
            method_name="LC-SC (w/o Length Penalty, lambda=0)",
            **kwargs,
        )


def build_all_decoders(
    n_max: int = 16,
    n_min: int = 3,
    l_min: int = 48,
    l_star: float = 180.0,
    sigma_l: float = 65.0,
    tau_h: float = 0.65,
    lambda_penalty: float = 0.50,
    gamma: float = 0.72,
    include_ablations: bool = True,
) -> Dict[str, BaseCoTDecoder]:
    """
    Instancia o conjunto completo dos 5 decodificadores principais + 3 variantes de ablação.
    """
    from .baselines import (
        GreedyCoTDecoder,
        HuangCaTSDecoder,
        StandardSCDecoder,
        WuLengthFilteredVoteDecoder,
    )

    decoders: Dict[str, BaseCoTDecoder] = {
        "greedy_cot": GreedyCoTDecoder(),
        "standard_sc": StandardSCDecoder(n_max=n_max),
        "wu_lfv": WuLengthFilteredVoteDecoder(n_max=n_max),
        "huang_cats": HuangCaTSDecoder(n_max=n_max, n_min=n_min, gamma=gamma),
        "lc_sc": LCSCDecoder(
            n_max=n_max,
            n_min=n_min,
            l_min=l_min,
            l_star=l_star,
            sigma_l=sigma_l,
            tau_h=tau_h,
            lambda_penalty=lambda_penalty,
            gamma=gamma,
        ),
    }

    if include_ablations:
        decoders["lc_sc_no_l_stop"] = LCSC_Ablation_NoLStop(
            n_max=n_max,
            n_min=n_min,
            l_min=l_min,
            l_star=l_star,
            sigma_l=sigma_l,
            tau_h=tau_h,
            lambda_penalty=lambda_penalty,
            gamma=gamma,
        )
        decoders["lc_sc_no_n_stop"] = LCSC_Ablation_NoNStop(
            n_max=n_max,
            n_min=n_min,
            l_min=l_min,
            l_star=l_star,
            sigma_l=sigma_l,
            tau_h=tau_h,
            lambda_penalty=lambda_penalty,
            gamma=gamma,
        )
        decoders["lc_sc_no_len_penalty"] = LCSC_Ablation_NoLenPenalty(
            n_max=n_max,
            n_min=n_min,
            l_min=l_min,
            l_star=l_star,
            sigma_l=sigma_l,
            tau_h=tau_h,
            gamma=gamma,
        )

    return decoders
