"""Route B additive world-model modules for the fixed 2026-10-18 RLES snapshot.

No NautilusTrader or upstream DreamerV3 source is shipped in this package.
Main-process modules are deliberately importable without JAX.
"""
from .market import Candle, MarketSequence, as_candle
from .clock import MultiScaleClock, ClosedBucket
from .codec import OHLCVCodec, TrainOnlyStandardizer
from .sequences import SequenceBatch, make_windows
from .risk_targets import (
    RiskTargets,
    build_risk_targets,
    estimate_train_barrier_fractions,
    score_path_risk,
)
from .prior_forecast import build_prior_alignment_table, write_prior_alignment_table

__all__ = ["Candle", "MarketSequence", "as_candle", "MultiScaleClock", "ClosedBucket",
           "OHLCVCodec", "TrainOnlyStandardizer", "SequenceBatch", "make_windows",
           "RiskTargets", "build_risk_targets", "estimate_train_barrier_fractions",
           "score_path_risk",
           "build_prior_alignment_table", "write_prior_alignment_table"]
