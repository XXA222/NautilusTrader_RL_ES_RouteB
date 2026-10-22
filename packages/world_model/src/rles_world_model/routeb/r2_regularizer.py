"""Train-only Barlow-style cross-correlation regularizer for RSSM latents.

Rows are sampled at a fixed within-sequence interval so adjacent minute tokens
do not masquerade as independent observations. The caller supplies a predictor
view and a stop-gradient target view and a mask that excludes padding/burn-in.
"""
from __future__ import annotations

import math


def cross_correlation_regularizer(prediction, target, mask, *, interval: int = 60,
                                  alpha: float = 0.005, eps: float = 1e-4):
    """Return scalar loss and diagnostics for aligned ``[batch,time,feature]`` views.

    The loss is ``sum((diag(C)-1)^2) + alpha*sum(offdiag(C)^2)`` where C is
    the masked, feature-standardized cross-correlation matrix over every
    ``interval``-th time position. Fewer than two valid rows yield zero loss.
    """
    import jax.numpy as jnp

    if type(interval) is not int or interval < 1:
        raise ValueError("R2 sampling interval must be a positive integer")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha < 0):
        raise ValueError("R2 off-diagonal weight must be nonnegative")
    if (isinstance(eps, bool) or not isinstance(eps, (int, float))
            or not math.isfinite(eps) or eps <= 0):
        raise ValueError("R2 epsilon must be positive")
    prediction = jnp.asarray(prediction, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    if (prediction.ndim != 3 or prediction.shape != target.shape
            or prediction.shape[:2] != mask.shape or prediction.shape[-1] < 2):
        raise ValueError("R2 views must be matching [B,T,D>=2] arrays with mask [B,T]")

    p = prediction[:, ::interval].reshape((-1, prediction.shape[-1]))
    e = target[:, ::interval].reshape((-1, target.shape[-1]))
    weights = mask[:, ::interval].reshape((-1,))
    count = weights.sum()
    denom = jnp.maximum(count, 1.0)
    p_mean = (p * weights[:, None]).sum(0) / denom
    e_mean = (e * weights[:, None]).sum(0) / denom
    p_centered = (p - p_mean) * weights[:, None]
    e_centered = (e - e_mean) * weights[:, None]
    p_var = jnp.square(p_centered).sum(0) / denom
    e_var = jnp.square(e_centered).sum(0) / denom
    p_norm = p_centered / jnp.sqrt(p_var[None, :] + eps)
    e_norm = e_centered / jnp.sqrt(e_var[None, :] + eps)
    correlation = (p_norm.T @ e_norm) / denom
    diagonal = jnp.diag(correlation)
    identity_loss = jnp.square(diagonal - 1.0).sum()
    off_diagonal = correlation - jnp.diag(diagonal)
    redundancy_loss = jnp.square(off_diagonal).sum()
    active = count >= 2.0
    loss = jnp.where(active, identity_loss + alpha * redundancy_loss, 0.0)
    # Gershgorin's strict diagonal-dominance bound is O(D^2), unlike a dense
    # SVD on every optimizer update. Non-dominant matrices receive the cap,
    # which records an unavailable/tightly ill-conditioned upper bound.
    absolute = jnp.abs(correlation)
    row_sum = absolute.sum(axis=1)
    diagonal_abs = jnp.abs(diagonal)
    row_margin = diagonal_abs - (row_sum - diagonal_abs)
    positive_margin = row_margin.min() > eps
    condition_bound = row_sum.max() / jnp.maximum(row_margin.min(), eps)
    condition = jnp.where(positive_margin, jnp.minimum(condition_bound, 1e6), 1e6)
    condition = jnp.where(active, condition, 0.0)
    p_variance = jnp.where(active, p_var.mean(), 0.0)
    e_variance = jnp.where(active, e_var.mean(), 0.0)
    return {
        "loss": loss,
        "valid_samples": count,
        "cross_correlation": correlation,
        "condition_number": condition,
        "prediction_variance": p_variance,
        "target_variance": e_variance,
        "identity_loss": jnp.where(active, identity_loss, 0.0),
        "redundancy_loss": jnp.where(active, redundancy_loss, 0.0),
    }
