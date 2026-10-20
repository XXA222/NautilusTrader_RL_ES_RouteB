"""Causal prior-to-observation alignment diagnostics for RouteB research runs."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np

from .market import MINUTE_NS


def build_prior_alignment_table(
    *,
    anchor_timestamps_ns: np.ndarray,
    target_timestamps_ns: np.ndarray,
    target_features: np.ndarray,
    valid_mask: np.ndarray,
    prior_mean: np.ndarray,
    prior_scale: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_scale: np.ndarray,
    prior_logits: np.ndarray,
    posterior_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Flatten aligned [batch,time] diagnostics without hiding invalid rows.

    The caller supplies model distributions computed from visible history and
    labels from the authorized Train or Validation reader. Valid target rows
    must be exactly the next minute after their anchor.
    """
    raw_anchor = np.asarray(anchor_timestamps_ns)
    raw_target = np.asarray(target_timestamps_ns)
    if (not np.issubdtype(raw_anchor.dtype, np.integer)
            or not np.issubdtype(raw_target.dtype, np.integer)):
        raise ValueError("alignment timestamps must be integer nanoseconds")
    anchor = raw_anchor.astype(np.int64, copy=False)
    target_ts = raw_target.astype(np.int64, copy=False)
    y = np.asarray(target_features, dtype=np.float32)
    mask = np.asarray(valid_mask)
    prior_mu = np.asarray(prior_mean, dtype=np.float32)
    prior_sd = np.asarray(prior_scale, dtype=np.float32)
    post_mu = np.asarray(posterior_mean, dtype=np.float32)
    post_sd = np.asarray(posterior_scale, dtype=np.float32)
    p_logits = np.asarray(prior_logits, dtype=np.float32)
    q_logits = np.asarray(posterior_logits, dtype=np.float32)
    shape = anchor.shape
    if (anchor.ndim < 1 or target_ts.shape != shape or y.shape != (*shape, 5)
            or mask.shape != shape or mask.dtype != np.dtype(bool)
            or prior_mu.shape != y.shape or prior_sd.shape != y.shape
            or post_mu.shape != y.shape or post_sd.shape != y.shape
            or p_logits.shape != q_logits.shape or p_logits.shape[:-2] != shape
            or p_logits.ndim != anchor.ndim + 2 or p_logits.shape[-1] < 2):
        raise ValueError("prior alignment inputs have inconsistent dimensions")
    if (not np.isfinite(y).all() or not np.isfinite(prior_mu).all()
            or not np.isfinite(prior_sd).all() or not np.isfinite(post_mu).all()
            or not np.isfinite(post_sd).all() or not np.isfinite(p_logits).all()
            or not np.isfinite(q_logits).all() or (prior_sd <= 0).any()
            or (post_sd <= 0).any()):
        raise ValueError("prior alignment values must be finite with positive scales")
    if np.any(mask & (target_ts != anchor + MINUTE_NS)):
        raise ValueError("valid prior targets must be the next one-minute observation")
    return {
        "anchor_timestamps_ns": anchor.reshape(-1).copy(),
        "target_timestamps_ns": target_ts.reshape(-1).copy(),
        "valid_mask": mask.reshape(-1).copy(),
        "target_features": y.reshape(-1, 5).copy(),
        "prior_mean": prior_mu.reshape(-1, 5).copy(),
        "prior_scale": prior_sd.reshape(-1, 5).copy(),
        "posterior_mean": post_mu.reshape(-1, 5).copy(),
        "posterior_scale": post_sd.reshape(-1, 5).copy(),
        "prior_logits": p_logits.reshape((-1, *p_logits.shape[-2:])).copy(),
        "posterior_logits": q_logits.reshape((-1, *q_logits.shape[-2:])).copy(),
    }


def write_prior_alignment_table(path: str | Path,
                                table: dict[str, np.ndarray]) -> tuple[Path, str]:
    """Atomically write an NPZ diagnostic table and return its SHA-256."""
    import hashlib

    required = {
        "anchor_timestamps_ns", "target_timestamps_ns", "valid_mask", "target_features",
        "prior_mean", "prior_scale", "posterior_mean", "posterior_scale",
        "prior_logits", "posterior_logits",
    }
    if set(table) != required:
        raise ValueError("alignment table members differ from the RouteB schema")
    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target_path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **table)  # type: ignore[arg-type]
    try:
        os.replace(temporary, target_path)
        digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    finally:
        temporary.unlink(missing_ok=True)
    return target_path, digest
