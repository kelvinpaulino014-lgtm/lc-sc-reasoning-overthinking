"""
Subpacote `src.utils`: extração de respostas, normalização algébrica e segmentação CoT.
"""

from .answer_extractor import (
    FINAL_ANSWER_CUE_PATTERN,
    GSM8K_HASH_PATTERN,
    OVERTHINKING_TRANSITION_PATTERN,
    SegmentedStep,
    detect_overthinking_transition,
    estimate_token_count,
    extract_answer,
    extract_intermediate_answer,
    extract_last_boxed,
    is_equivalent,
    normalize_math_answer,
    segment_cot_steps,
)

__all__ = [
    "FINAL_ANSWER_CUE_PATTERN",
    "GSM8K_HASH_PATTERN",
    "OVERTHINKING_TRANSITION_PATTERN",
    "SegmentedStep",
    "detect_overthinking_transition",
    "estimate_token_count",
    "extract_answer",
    "extract_intermediate_answer",
    "extract_last_boxed",
    "is_equivalent",
    "normalize_math_answer",
    "segment_cot_steps",
]
