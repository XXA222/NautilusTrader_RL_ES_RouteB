from __future__ import annotations

import pytest
import numpy as np

from rles_world_model.routeb.r2_regularizer import cross_correlation_regularizer
from rles_world_model.routeb.r2_ablation import score_r2_ablation_ensembles
from rles_world_model.routeb.jax_model import RSSMConfig


def test_r2_regularizer_samples_fixed_interval_and_has_finite_gradients():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    time = jnp.arange(180, dtype=jnp.float32)
    base = jnp.stack((jnp.sin(time / 7), jnp.sin(time / 7) * 0.8 + jnp.cos(time / 11),
                      jnp.cos(time / 5)), axis=-1)
    prediction = jnp.stack((base, base * 1.1 + 0.03), axis=0)
    target = jnp.stack((base * 0.9 + 0.1, base * 1.2 - 0.02), axis=0)
    mask = jnp.ones((2, 180), dtype=jnp.float32).at[1, 60].set(0)

    def objective(values):
        return cross_correlation_regularizer(
            values, target, mask, interval=60, alpha=0.005)["loss"]

    result = cross_correlation_regularizer(
        prediction, target, mask, interval=60, alpha=0.005)
    grad = jax.grad(objective)(prediction)
    assert float(result["valid_samples"]) == 5.0
    assert float(result["loss"]) > 0.0
    assert float(result["condition_number"]) >= 1.0
    assert float(result["prediction_variance"]) > 0.0
    assert jnp.isfinite(grad).all() and float(jnp.linalg.norm(grad)) > 0.0


def test_r2_regularizer_is_zero_when_fewer_than_two_rows_are_valid():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    values = jnp.arange(2 * 120 * 3, dtype=jnp.float32).reshape((2, 120, 3))
    mask = jnp.zeros((2, 120), dtype=jnp.float32).at[0, 0].set(1.0)
    result = cross_correlation_regularizer(values, values, mask, interval=60)
    assert float(result["valid_samples"]) == 1.0
    assert float(result["loss"]) == 0.0
    assert float(result["condition_number"]) == 0.0


def test_r2_regularizer_rejects_misaligned_or_invalid_configuration():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    values = jnp.ones((2, 8, 3), dtype=jnp.float32)
    with pytest.raises(ValueError, match="interval"):
        cross_correlation_regularizer(values, values, jnp.ones((2, 8)), interval=0)
    with pytest.raises(ValueError, match="matching"):
        cross_correlation_regularizer(values, values[:, :, :2], jnp.ones((2, 8)))


def test_r2_model_config_validates_regularizer_parameters():
    with pytest.raises(ValueError, match="configuration"):
        RSSMConfig(r2_regularizer_scale=-0.01)
    with pytest.raises(ValueError, match="configuration"):
        RSSMConfig(r2_regularizer_alpha=float("nan"))


def test_r3_ablation_scorer_uses_common_seeds_and_validation_paths():
    n, horizon, paths = 30, 15, 4
    anchors = np.arange(n, dtype=np.int64) * 120 * 60_000_000_000
    previous = np.full(n, 100.0)
    rng = np.random.default_rng(31)
    realized = np.empty((n, horizon, 5), dtype=np.float64)
    for row in range(n):
        close = previous[row]
        for step in range(horizon):
            opening = close
            close *= np.exp(rng.normal(0.0, 0.0004))
            realized[row, step] = (opening, max(opening, close) * 1.001,
                                   min(opening, close) * 0.999, close, 10.0)
    control = np.empty((n, paths, horizon, 5), dtype=np.float64)
    for row in range(n):
        close = previous[row]
        for step in range(horizon):
            opening = close
            close *= np.exp(rng.normal(0.0, 0.0003))
            control[row, :, step] = (opening, max(opening, close) * 1.001,
                                     min(opening, close) * 0.999, close, 10.0)
    forecast = {
        0.0: {seed: control.copy() for seed in (31, 102, 20261018)},
        0.01: {seed: control.copy() for seed in (31, 102, 20261018)},
        0.05: {seed: control.copy() for seed in (31, 102, 20261018)},
    }
    report = score_r2_ablation_ensembles(
        forecast, realized, previous, anchors,
        anchor_interval_ns=int(anchors[1] - anchors[0]),
        block_length=5, bootstrap_replicates=100)
    paired = report["comparisons"]["lambda_0.01_vs_0"]["pooled"]
    assert paired["path_log_rv_crps"]["mean_difference"] == 0.0
    assert paired["return_crps"]["mean_difference"] == 0.0
    assert report["test_payload_decoded"] is False
