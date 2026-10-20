from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_hierarchical_kl_weights_apply_once_per_component() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from rles_world_model.routeb.jax_hierarchy import masked_weighted_kl

    dynamics = jnp.asarray([[1.0, 3.0, 100.0]])
    representation = jnp.asarray([[2.0, 5.0, 200.0]])
    mask = jnp.asarray([[1.0, 1.0, 0.0]])

    def objective(dynamics_scale, representation_scale):
        return masked_weighted_kl(
            dynamics,
            representation,
            mask,
            dynamics_scale=dynamics_scale,
            representation_scale=representation_scale,
        )

    value = objective(0.25, 0.1)
    np.testing.assert_allclose(value, 0.25 * 2.0 + 0.1 * 3.5)
    dynamics_gradient, representation_gradient = jax.grad(objective, argnums=(0, 1))(
        0.25, 0.1,
    )
    np.testing.assert_allclose(dynamics_gradient, 2.0)
    np.testing.assert_allclose(representation_gradient, 3.5)


def test_risk_horizon_weights_emphasize_long_horizon_losses() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from rles_world_model.routeb.jax_model import _weighted_risk_loss

    continuous = jnp.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    barrier = jnp.zeros_like(continuous)
    weights = (0.5, 0.5, 1.0, 1.5, 1.5)

    value = _weighted_risk_loss(continuous, barrier, weights)
    gradient = jax.grad(lambda loss: _weighted_risk_loss(loss, barrier, weights))(continuous)

    np.testing.assert_allclose(value, 8.3)
    np.testing.assert_allclose(gradient, np.asarray(weights) / sum(weights))
