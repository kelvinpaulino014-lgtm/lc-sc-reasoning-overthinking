"""
Utilitários de extração de respostas matemáticas, normalização algébrica/numérica
e segmentação de passos de Cadeias de Pensamento (Chain-of-Thought - CoT) para
os benchmarks GSM8K e MATH-500.

Referências Científicas (ICLR 2026):
- WU et al. (2026): When More is Less: Understanding Chain-of-Thought Length in LLMs.
- HUANG et al. (2026): CaTS: Calibrated Test-Time Scaling for Efficient LLM Reasoning.
- HAO et al. (2026): RL of Thoughts: Navigating LLM Reasoning with Inference-time RL.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Expressões regulares pré-compiladas para marcadores de hesitação / overthinking
# documentados em Wu et al. (ICLR 2026) e Hao et al. (ICLR 2026)
OVERTHINKING_TRANSITION_PATTERN = re.compile(
    r"(?:^|\b)(?:"
    r"wait(?: a (?:second|minute|moment))?|"
    r"alternatively|"
    r"hmm+|"
    r"let(?:'s| us| me) (?:double[- ]?check|re-?check|recalculate|verify|reconsider|think again)|"
    r"hold on|"
    r"actually,\s*(?:wait|let)|"
    r"but wait|"
    r"on second thought|"
    r"espera|"
    r"pensando bem|"
    r"vamos (?:verificar|conferir|recalcular)|"
    r"será que"
    r")\b",
     flags=re.IGNORECASE,
)

# Marcadores de conclusão final (inglês e português)
FINAL_ANSWER_CUE_PATTERN = re.compile(
    r"(?:"
    r"(?:the\s+)?(?:final\s+)?answer\s+is|"
    r"therefore,?\s+the\s+(?:final\s+)?(?:answer|result|value)\s+is|"
    r"so\s+the\s+(?:final\s+)?answer\s+is|"
    r"(?:a\s+)?resposta\s+(?:final\s+)?(?:é|e)|"
    r"portanto,?\s+(?:a\s+)?resposta\s+(?:final\s+)?(?:é|e)|"
    r"logo,?\s+o\s+resultado\s+(?:é|e)"
    r")\s*[:=]?\s*([^\n.]+)",
    flags=re.IGNORECASE,
)

# Marcadores GSM8K: #### <resposta>
GSM8K_HASH_PATTERN = re.compile(r"####\s*([^\n]+)")

# Números com suporte a sinal, separador de milhar, decimais e frações simples
NUMBER_TOKEN_PATTERN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"
)


@dataclass
class SegmentedStep:
    """Representa um passo lógico extraído da segmentação de uma trilha CoT."""

    step_idx: int
    text: str
    step_tokens: int
    cumulative_tokens: int
    step_entropy: float
    step_confidence: float
    intermediate_answer: Optional[str]
    is_overthinking_transition: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": int(self.step_idx),
            "text": self.text,
            "step_tokens": int(self.step_tokens),
            "cumulative_tokens": int(self.cumulative_tokens),
            "step_entropy": float(self.step_entropy),
            "step_confidence": float(self.step_confidence),
            "intermediate_answer": self.intermediate_answer,
            "is_overthinking_transition": bool(self.is_overthinking_transition),
        }


def extract_last_boxed(text: str) -> Optional[str]:
    """
    Extrai o conteúdo do último comando `\\boxed{...}` ou `\\fbox{...}` no texto,
    respeitando chaves aninhadas (ex.: `\\boxed{\\frac{1}{2}}`).
    """
    if not text or not isinstance(text, str):
        return None

    idx = text.rfind("\\boxed")
    cmd_len = 6
    if idx < 0:
        idx = text.rfind("\\fbox")
        cmd_len = 5
        if idx < 0:
            return None

    i = idx + cmd_len
    while i < len(text) and text[i].isspace():
        i += 1

    if i >= len(text) or text[i] != "{":
        # Caso \boxed sem chaves explícitas (ex: \boxed 42)
        match = re.match(r"\\(?:boxed|fbox)\s+([^\s$}]+)", text[idx:])
        return match.group(1).strip() if match else None

    open_start = i + 1
    depth = 0
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_start:i].strip()
        i += 1

    # Caso a geração tenha sido truncada antes de fechar a última chave,
    # tenta recuperar o conteúdo após `\boxed{`
    tail = text[open_start:].strip()
    return tail if tail else None


def _fix_latex_fracs(string: str) -> str:
    """Converte `\\frac12` ou `\\frac1{2}` para `\\frac{1}{2}`."""
    substrs = string.split("\\frac")
    if len(substrs) <= 1:
        return string
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if not substr:
            continue
        if substr[0] == "{":
            new_str += substr
        else:
            if len(substr) < 2:
                return string
            a = substr[0]
            b = substr[1]
            if b != "{":
                post = substr[2:] if len(substr) > 2 else ""
                new_str += f"{{{a}}}{{{b}}}{post}"
            else:
                post = substr[2:] if len(substr) > 2 else ""
                new_str += f"{{{a}}}{b}{post}"
    return new_str


def _fix_latex_sqrt(string: str) -> str:
    """Converte `\\sqrt3` para `\\sqrt{3}`."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += f"\\sqrt{{{split[0]}}}{split[1:]}"
        else:
            new_string += f"\\sqrt{split}"
    return new_string


def normalize_math_answer(raw_answer: Optional[str]) -> Optional[str]:
    """
    Normaliza uma string matemática/numérica para comparação canônica entre
    GSM8K e MATH-500.

    Operações:
    1. Desempacota `\\boxed{...}` ou `####` se ainda presentes.
    2. Remove unidades à direita (`\\text{...}`, `\\mbox{...}`, moedas `$`, `%`, graus).
    3. Padroniza comandos LaTeX (`\\dfrac`, `\\tfrac` -> `\\frac`, `\\left`, `\\right`).
    4. Remove vírgulas separadoras de milhar (`1,234` -> `1234`).
    5. Converte números inteiros/decimais/frações exatas para forma numérica canônica
       quando possível (ex.: `42.00` -> `42`, `\\frac{6}{4}` -> `3/2`).
    """
    if raw_answer is None:
        return None
    if not isinstance(raw_answer, str):
        raw_answer = str(raw_answer)

    s = raw_answer.strip()
    if not s:
        return None

    # Se contiver ####, extrai o que vem depois
    hash_match = GSM8K_HASH_PATTERN.search(s)
    if hash_match:
        s = hash_match.group(1).strip()

    # Se contiver \boxed{...}, extrai o conteúdo interno
    boxed = extract_last_boxed(s)
    if boxed is not None:
        s = boxed

    # Remove delimitadores de modo matemático e pontuação final
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = s.replace("\\$", "").replace("$", "").replace("R$", "")
    s = s.replace("\n", " ").replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "")

    # Remove blocos \text{...} ou \mbox{...} que representam unidades
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox)\s*\{([^{}]*)\}", lambda m: m.group(1) if len(m.group(1).strip()) <= 1 else "", s)

    # Remove atribuições iniciais simples como "x = 5" ou "ans = 5"
    if "=" in s:
        parts = s.split("=")
        if len(parts) == 2 and len(parts[0].strip()) <= 6:
            s = parts[1].strip()

    # Remove pontuação final solta (ponto final, vírgula ou ponto e vírgula no fim)
    s = s.rstrip(".,;:")

    # Remove vírgulas de milhar entre dígitos (ex: 1,234,567 -> 1234567)
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\b|[^\d]))", "", s)

    # Remove unidades textuais comuns no GSM8K (ex: "42 dollars", "15 apples")
    num_only_match = re.match(
        r"^([-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s+[a-zA-Z\s]+$",
        s.strip(),
    )
    if num_only_match:
        s = num_only_match.group(1)

    s = _fix_latex_sqrt(s)
    s = _fix_latex_fracs(s)

    # Converte \frac{a}{b} simples (numéricos) para a/b
    frac_num_match = re.fullmatch(
        r"([-+]?)\s*\\frac\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\{\s*([-+]?\d+(?:\.\d+)?)\s*\}",
        s.strip(),
    )
    if frac_num_match:
        sign, num_s, den_s = frac_num_match.groups()
        s = f"{sign}{num_s}/{den_s}"

    # Remove espaços em branco internos para forma canônica
    s_no_space = re.sub(r"\s+", "", s)
    if not s_no_space:
        return None

    if s_no_space.startswith("."):
        s_no_space = "0" + s_no_space
    elif s_no_space.startswith("-."):
        s_no_space = "-0." + s_no_space[2:]

    # Tenta normalizar como número racional/inteiro/decimal exato
    numeric_val = _try_parse_rational_or_float(s_no_space)
    if numeric_val is not None:
        frac_val, float_val = numeric_val
        if frac_val is not None:
            if frac_val.denominator == 1:
                return str(frac_val.numerator)
            # Para decimais exatos ou frações irredutíveis: se veio como decimal no GSM8K,
            # ainda podemos representar fração irredutível ou decimal limpo.
            # Mas se o denominador divide uma potência de 10 pequena (ex: 0.5 == 1/2),
            # padronizar em string numérica ou comparar via _try_parse_rational_or_float.
            # Aqui retornamos inteiro se denominador=1; caso contrário, se era decimal,
            # removemos zeros à direita; se era fração, retornamos num/den irredutível.
            if "/" in s_no_space:
                return f"{frac_val.numerator}/{frac_val.denominator}"
            # Decimal: remove zeros finais desnecessários
            if "." in s_no_space:
                cleaned_dec = f"{float_val:.10f}".rstrip("0").rstrip(".")
                return "0" if cleaned_dec in ("-0", "") else cleaned_dec
            return str(frac_val.numerator)

    return s_no_space


def _try_parse_rational_or_float(s: str) -> Optional[Tuple[Optional[Fraction], float]]:
    """
    Tenta converter uma string já limpa em `(Fraction, float)`.
    Retorna `None` se a string for uma expressão simbólica não puramente numérica.
    """
    if not s:
        return None
    try:
        if "/" in s and not any(c in s for c in ("\\", "{", "}", "^", "sqrt", "pi")):
            parts = s.split("/")
            if len(parts) == 2:
                num = Fraction(parts[0])
                den = Fraction(parts[1])
                if den == 0:
                    return None
                frac = num / den
                return frac, float(frac)
        # Número decimal ou inteiro direto
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s):
            fval = float(s)
            if math.isnan(fval) or math.isinf(fval):
                return None
            try:
                frac = Fraction(s).limit_denominator(10**9)
            except (ValueError, OverflowError):
                frac = None
            return frac, fval
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return None


def extract_answer(text: Optional[str], benchmark: str = "auto") -> Optional[str]:
    """
    Extrai a resposta final de uma cadeia de pensamento (GSM8K ou MATH-500).

    Ordem de prioridade:
    1. Marcador explícito do GSM8K (`#### <resposta>`).
    2. Último bloco `\\boxed{...}` ou `\\fbox{...}` (padrão MATH-500 e DeepSeek/Qwen).
    3. Expressão após marcadores explícitos de conclusão ("The answer is ...",
       "Portanto, a resposta final é ...").
    4. Se houver tag de fechamento `</think>`, busca o último número/expressão no
       segmento pós-`</think>`.
    5. Último número válido presente no texto.
    """
    if text is None or not isinstance(text, str):
        return None
    cleaned_text = text.strip()
    if not cleaned_text:
        return None

    # 1. GSM8K #### <num>
    hash_matches = GSM8K_HASH_PATTERN.findall(cleaned_text)
    if hash_matches:
        candidate = normalize_math_answer(hash_matches[-1])
        if candidate is not None:
            return candidate

    # 2. \boxed{...} ou \fbox{...}
    boxed = extract_last_boxed(cleaned_text)
    if boxed is not None:
        candidate = normalize_math_answer(boxed)
        if candidate is not None:
            return candidate

    # Se houver </think>, prioriza o texto após </think> para cues e números finais
    post_think_text = cleaned_text
    if "</think>" in cleaned_text:
        after_think = cleaned_text.split("</think>")[-1].strip()
        if after_think:
            post_think_text = after_think

    # 3. Frases de conclusão ("the answer is X", "a resposta é X")
    cue_matches = FINAL_ANSWER_CUE_PATTERN.findall(post_think_text)
    if not cue_matches and post_think_text is not cleaned_text:
        cue_matches = FINAL_ANSWER_CUE_PATTERN.findall(cleaned_text)
    if cue_matches:
        raw_cue = cue_matches[-1].strip()
        # Extrai \boxed se houver ou primeiro/último número dentro do trecho
        boxed_cue = extract_last_boxed(raw_cue)
        if boxed_cue:
            return normalize_math_answer(boxed_cue)
        nums_in_cue = NUMBER_TOKEN_PATTERN.findall(raw_cue)
        if nums_in_cue:
            return normalize_math_answer(nums_in_cue[-1])
        norm_cue = normalize_math_answer(raw_cue)
        if norm_cue:
            return norm_cue

    # 4. Fallback: último número no trecho pós-</think> (ou no texto completo)
    nums = NUMBER_TOKEN_PATTERN.findall(post_think_text)
    if not nums and post_think_text is not cleaned_text:
        nums = NUMBER_TOKEN_PATTERN.findall(cleaned_text)
    if nums:
        return normalize_math_answer(nums[-1])

    return None


def is_equivalent(
    pred: Optional[str],
    gold: Optional[str],
    tol: float = 1e-6,
) -> bool:
    """
    Verifica equivalência matemática/numérica entre a resposta predita `pred`
    e o gabarito `gold`.

    Suporta:
    - Igualdade canônica de strings normalizadas.
    - Equivalência entre frações e decimais (`1/2` == `0.5`, `\\frac{3}{4}` == `0.75`).
    - Equivalência em ponto flutuante dentro da tolerância `tol`.
    - Equivalência simbólica via `sympy` (quando disponível) para expressões LaTeX
      do MATH-500 (ex.: `\\frac{\\sqrt{2}}{2}` vs `1/\\sqrt{2}`).
    """
    if pred is None or gold is None:
        return False

    norm_pred = normalize_math_answer(pred)
    norm_gold = normalize_math_answer(gold)

    if norm_pred is None or norm_gold is None:
        return False

    if norm_pred == norm_gold:
        return True

    # Comparação racional / ponto flutuante
    parsed_pred = _try_parse_rational_or_float(norm_pred)
    parsed_gold = _try_parse_rational_or_float(norm_gold)

    if parsed_pred is not None and parsed_gold is not None:
        frac_p, float_p = parsed_pred
        frac_g, float_g = parsed_gold
        if frac_p is not None and frac_g is not None and frac_p == frac_g:
            return True
        if math.isclose(float_p, float_g, rel_tol=tol, abs_tol=tol):
            return True
        return False

    # Comparação de expressões simbólicas simples (ex: pi, sqrt) ou via SymPy
    val_p = _eval_simple_symbolic(norm_pred)
    val_g = _eval_simple_symbolic(norm_gold)
    if val_p is not None and val_g is not None:
        if math.isclose(val_p, val_g, rel_tol=tol, abs_tol=tol):
            return True

    # Fallback opcional com SymPy se instalado no ambiente
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex

        expr_p = parse_latex(norm_pred)
        expr_g = parse_latex(norm_gold)
        if expr_p.equals(expr_g):
            return True
        diff = simplify(expr_p - expr_g)
        if diff == 0:
            return True
        num_diff = abs(complex(expr_p.evalf()) - complex(expr_g.evalf()))
        if num_diff <= tol:
            return True
    except Exception:
        pass

    return False


def _eval_simple_symbolic(expr: str) -> Optional[float]:
    """
    Avalia expressões matemáticas elementares contendo `\\sqrt{...}`, `\\frac{...}{...}`
    e `\\pi` sem depender obrigatoriamente de parser externo.
    """
    try:
        s = expr
        if not any(tok in s for tok in ("\\sqrt", "\\frac", "\\pi", "/")):
            return None
        s = s.replace("\\pi", f"({math.pi})")
        # Substitui \sqrt{a} por math.sqrt(a)
        for _ in range(4):
            s = re.sub(
                r"\\sqrt\{([^{}]+)\}",
                r"math.sqrt(\1)",
                s,
            )
            s = re.sub(
                r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
                r"((\1)/(\2))",
                s,
            )
        if re.fullmatch(r"[0-9+\-*/().eE\s]|math\.sqrt", re.sub(r"math\.sqrt", "", s)):
            pass
        # Verifica segurança dos caracteres antes de avaliar
        allowed = s.replace("math.sqrt", "")
        if not re.fullmatch(r"[0-9+\-*/().\s]+", allowed):
            return None
        val = float(eval(s, {"__builtins__": {}}, {"math": math}))  # noqa: S307
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def detect_overthinking_transition(step_text: str) -> bool:
    """
    Retorna `True` se o passo de raciocínio contiver marcadores linguísticos
    típicos de hesitação circular / sobrecorreção (*overthinking* / *over-correction*),
    conforme caracterizado por Wu et al. (ICLR 2026) e Hao et al. (ICLR 2026).
    """
    if not step_text or not isinstance(step_text, str):
        return False
    return bool(OVERTHINKING_TRANSITION_PATTERN.search(step_text))


def extract_intermediate_answer(step_text: str) -> Optional[str]:
    """
    Extrai uma resposta candidata parcial/intermediária derivada dentro de um passo
    de raciocínio CoT.

    Reconhece:
    - `#### <num>` ou `\\boxed{...}` dentro do passo.
    - Frases conclusivas intermediárias ("answer is X", "so X", "portanto X", "= X").
    - Resultados de igualdades aritméticas no final de frases (ex.: `3 * 14 = 42`).
    """
    if not step_text or not isinstance(step_text, str):
        return None

    # 1. Se o passo já contém #### ou \boxed
    hash_m = GSM8K_HASH_PATTERN.findall(step_text)
    if hash_m:
        return normalize_math_answer(hash_m[-1])

    boxed = extract_last_boxed(step_text)
    if boxed is not None:
        return normalize_math_answer(boxed)

    # 2. Frases explícitas de conclusão parcial
    cue_m = FINAL_ANSWER_CUE_PATTERN.findall(step_text)
    if cue_m:
        nums = NUMBER_TOKEN_PATTERN.findall(cue_m[-1])
        if nums:
            return normalize_math_answer(nums[-1])
        return normalize_math_answer(cue_m[-1])

    # 3. Expressões de igualdade matemática (ex.: "= 42" ou "is 42" ao final do passo)
    eq_matches = re.findall(
        r"(?:=|equals|\bis\b|\bgives\b|\bresults in\b|\bget\b|\btotal(?:s| is| of)?\b)\s*\$?\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?)",
        step_text,
        flags=re.IGNORECASE,
    )
    if eq_matches:
        return normalize_math_answer(eq_matches[-1])

    return None


def estimate_token_count(text: str) -> int:
    """
    Estima o número de tokens de um trecho de texto quando os IDs do tokenizador
    não são fornecidos diretamente (aproximação compatível com tokenizadores BPE
    da família Qwen/DeepSeek em expressões matemáticas e texto misto).
    """
    if not text or not isinstance(text, str):
        return 0
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, len(tokens)) if text.strip() else 0


def segment_cot_steps(
    full_text: str,
    token_entropies: Optional[Sequence[float]] = None,
    step_confidences: Optional[Sequence[float]] = None,
    total_tokens: Optional[int] = None,
    default_entropy: float = 0.25,
    default_confidence: float = 0.80,
) -> List[SegmentedStep]:
    """
    Segmenta uma cadeia de pensamento (CoT) completa em passos lógicos sequenciais,
    identificando comprimentos acumulados, entropia por passo, respostas parciais
    intermediárias e marcadores de transição de *overthinking*.

    Regras de divisão:
    1. Quebras de parágrafo (`\\n\\n`).
    2. Caso um bloco seja muito longo ou contenha marcadores de transição de
       hesitação no início de linhas (`\\nWait`, `\\nAlternatively`, `\\nHmm`,
       `\\nLet's double check`, `\\nPortanto`), divide também nessas fronteiras.
    """
    if not full_text or not isinstance(full_text, str) or not full_text.strip():
        return []

    # Divide por \n\n ou por quebras de linha imediatamente seguidas por marcadores de transição
    split_pattern = re.compile(
        r"\n\s*\n+|\n(?=\s*(?:Wait|Alternatively|Hmm|Let(?:'s| us| me) double|Hold on|Actually|But wait|On second thought|Portanto|Therefore|Thus|####|\\boxed|</think>))",
        flags=re.IGNORECASE,
    )
    raw_chunks = [chunk.strip() for chunk in split_pattern.split(full_text) if chunk and chunk.strip()]
    if not raw_chunks:
        raw_chunks = [full_text.strip()]

    raw_weights = [max(1, estimate_token_count(c)) for c in raw_chunks]
    sum_weights = sum(raw_weights)
    target_total_tokens = int(total_tokens) if (total_tokens is not None and total_tokens > 0) else sum_weights

    # Distribui total_tokens proporcionalmente aos comprimentos de cada passo
    step_token_counts: List[int] = []
    acc_assigned = 0
    for idx, w in enumerate(raw_weights):
        if idx == len(raw_weights) - 1:
            count = max(1, target_total_tokens - acc_assigned)
        else:
            count = max(1, int(round(target_total_tokens * (w / sum_weights))))
            if acc_assigned + count >= target_total_tokens:
                count = max(1, target_total_tokens - acc_assigned - (len(raw_weights) - 1 - idx))
        step_token_counts.append(count)
        acc_assigned += count

    # Se token_entropies foi fornecido, fatia as entropias por passo
    ent_list = list(token_entropies) if token_entropies is not None else None

    steps: List[SegmentedStep] = []
    cumulative = 0
    ent_cursor = 0

    for idx, (chunk, st_tokens) in enumerate(zip(raw_chunks, step_token_counts)):
        prev_cum = cumulative
        cumulative += st_tokens
        is_overthinking = detect_overthinking_transition(chunk)
        inter_ans = extract_intermediate_answer(chunk)

        if ent_list is not None and len(ent_list) > 0:
            if len(ent_list) == len(raw_chunks):
                step_ent = float(ent_list[idx])
            else:
                # Mapeia proporcionalmente ao intervalo de tokens
                start_i = int(round((prev_cum / max(1, cumulative if idx == len(raw_chunks) - 1 else target_total_tokens)) * len(ent_list)))
                end_i = int(round((cumulative / max(1, target_total_tokens)) * len(ent_list)))
                start_i = min(max(0, ent_cursor), len(ent_list) - 1)
                end_i = max(start_i + 1, min(len(ent_list), end_i))
                ent_cursor = end_i
                slice_vals = [float(x) for x in ent_list[start_i:end_i] if x is not None and not math.isnan(float(x))]
                step_ent = sum(slice_vals) / len(slice_vals) if slice_vals else float(default_entropy)
        else:
            step_ent = float(default_entropy)

        if step_confidences is not None and idx < len(step_confidences):
            step_conf = float(step_confidences[idx])
        else:
            # Relação inversa suave entre entropia do passo e confiança local
            step_conf = float(max(0.05, min(0.99, math.exp(-0.45 * max(0.0, step_ent))))) if token_entropies is not None else float(default_confidence)

        steps.append(
            SegmentedStep(
                step_idx=idx,
                text=chunk,
                step_tokens=st_tokens,
                cumulative_tokens=cumulative,
                step_entropy=step_ent,
                step_confidence=step_conf,
                intermediate_answer=inter_ans,
                is_overthinking_transition=is_overthinking,
            )
        )

    return steps
