"""
Subpacote `src.decoders`: estruturas de trilha CoT, 4 baselines da literatura (ICLR 2026)
e o decodificador proposto `LCSCDecoder` com suas 3 variantes de ablação.
"""

from .baselines import (
    BaseCoTDecoder,
    ConfidenceCalibrator,
    CoTStep,
    CoTTrace,
    GreedyCoTDecoder,
    HuangCaTSDecoder,
    QuestionPrediction,
    StandardSCDecoder,
    WuLengthFilteredVoteDecoder,
    group_traces_by_question,
    load_traces_jsonl,
    save_traces_jsonl,
)
from .lc_sc_decoder import (
    LCSC_Ablation_NoLenPenalty,
    LCSC_Ablation_NoLStop,
    LCSC_Ablation_NoNStop,
    LCSCDecoder,
    TruncatedTraceResult,
    build_all_decoders,
)

__all__ = [
    "BaseCoTDecoder",
    "ConfidenceCalibrator",
    "CoTStep",
    "CoTTrace",
    "GreedyCoTDecoder",
    "HuangCaTSDecoder",
    "LCSC_Ablation_NoLStop",
    "LCSC_Ablation_NoLenPenalty",
    "LCSC_Ablation_NoNStop",
    "LCSCDecoder",
    "QuestionPrediction",
    "StandardSCDecoder",
    "TruncatedTraceResult",
    "WuLengthFilteredVoteDecoder",
    "build_all_decoders",
    "group_traces_by_question",
    "load_traces_jsonl",
    "save_traces_jsonl",
]
