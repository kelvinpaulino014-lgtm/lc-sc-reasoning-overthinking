"""
Subpacote `src.metrics`: entropia de Shannon no nível de token/janela, calibração (ECE, Brier, AUC)
e ajuste da curva em U invertido de Wu et al. (ICLR 2026).
"""

from .entropy_tracker import (
    BootstrapCIResult,
    CalibrationBinStat,
    EntropyTracker,
    InvertedUFitResult,
    bootstrap_ci,
    compute_answer_entropy,
    compute_auc_roc,
    compute_brier_score,
    compute_calibration_bins,
    compute_ece,
    compute_token_entropy,
    compute_windowed_entropy,
    fit_inverted_u_curve,
)

__all__ = [
    "BootstrapCIResult",
    "CalibrationBinStat",
    "EntropyTracker",
    "InvertedUFitResult",
    "bootstrap_ci",
    "compute_answer_entropy",
    "compute_auc_roc",
    "compute_brier_score",
    "compute_calibration_bins",
    "compute_ece",
    "compute_token_entropy",
    "compute_windowed_entropy",
    "fit_inverted_u_curve",
]
