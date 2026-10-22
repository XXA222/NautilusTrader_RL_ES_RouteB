from __future__ import annotations

import numpy as np
import pytest

from rles_world_model.routeb.jax_gru import GRUConfig


def test_e1_gru_config_rejects_invalid_scale_or_capacity():
    with pytest.raises(ValueError, match="configuration"):
        GRUConfig(hidden=2)
    with pytest.raises(ValueError, match="configuration"):
        GRUConfig(min_scale=float("nan"))


def test_e1_train_loss_is_finite_and_respects_target_mask():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from rles_world_model.routeb.jax_gru import init_parameters, loss_and_metrics

    config = GRUConfig(hidden=8)
    params = init_parameters(17, config)
    features = jnp.linspace(-1.0, 1.0, 2 * 6 * 5).reshape(2, 6, 5)
    reset = jnp.zeros((2, 6), dtype=jnp.bool_).at[:, 0].set(True)
    mask = jnp.asarray([[False, False, True, True, True, True],
                        [False, False, True, True, True, True]])
    loss, metrics = loss_and_metrics(params, features, reset, mask, config)
    gradients = jax.grad(lambda values: loss_and_metrics(
        values, features, reset, mask, config)[0])(params)
    assert np.isfinite(float(loss))
    assert float(metrics["targets"]) == 8
    assert all(np.isfinite(np.asarray(leaf)).all()
               for leaf in jax.tree_util.tree_leaves(gradients))
