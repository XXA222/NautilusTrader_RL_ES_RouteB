"""Probabilistic GRU predictor for the preregistered E1 forecast baseline.

This compact baseline shares RouteB's Train-only sequence/scaler contract with
E2 but has no stochastic latent state or trading/account inputs. JAX is imported
inside functions so Python 3.14 host processes remain JAX-free.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from .randomness import validate_jax_seed


@dataclass(frozen=True)
class GRUConfig:
    hidden: int = 64
    feature_dim: int = 5
    min_scale: float = 0.02

    def __post_init__(self) -> None:
        if (type(self.hidden) is not int or self.hidden < 4
                or type(self.feature_dim) is not int or self.feature_dim != 5
                or type(self.min_scale) not in (int, float)
                or not math.isfinite(self.min_scale) or self.min_scale <= 0):
            raise ValueError("invalid E1 GRU configuration")


def init_parameters(seed: int, config: GRUConfig):
    """Create a deterministic, finite parameter PyTree for the E1 model."""
    import jax
    import jax.numpy as jnp

    validate_jax_seed(seed, name="GRU initialization seed")
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)
    hidden = config.hidden
    input_scale = 1.0 / math.sqrt(config.feature_dim + hidden)
    candidate_scale = 1.0 / math.sqrt(config.feature_dim + hidden)
    return {
        "gates_kernel": jax.random.normal(
            keys[0], (config.feature_dim + hidden, 2 * hidden), dtype=jnp.float32
        ) * input_scale,
        "gates_bias": jnp.zeros((2 * hidden,), dtype=jnp.float32),
        "input_kernel": jax.random.normal(
            keys[1], (config.feature_dim, hidden), dtype=jnp.float32
        ) * candidate_scale,
        "hidden_kernel": jax.random.normal(
            keys[2], (hidden, hidden), dtype=jnp.float32
        ) * candidate_scale,
        "candidate_bias": jnp.zeros((hidden,), dtype=jnp.float32),
        "head_kernel": jax.random.normal(
            keys[3], (hidden, 2 * config.feature_dim), dtype=jnp.float32
        ) / math.sqrt(hidden),
        "head_bias": jnp.zeros((2 * config.feature_dim,), dtype=jnp.float32),
    }


def _step(params, carry, feature, reset):
    import jax
    import jax.numpy as jnp

    carry = jnp.where(reset[..., None], 0.0, carry)
    gates = jnp.concatenate((feature, carry), axis=-1) @ params["gates_kernel"]
    update, reset_gate = jnp.split(gates + params["gates_bias"], 2, axis=-1)
    update = jax.nn.sigmoid(update)
    reset_gate = jax.nn.sigmoid(reset_gate)
    candidate = jnp.tanh(
        feature @ params["input_kernel"]
        + (reset_gate * carry) @ params["hidden_kernel"]
        + params["candidate_bias"]
    )
    return (1.0 - update) * candidate + update * carry


def _encode(params, features, reset):
    import jax
    import jax.numpy as jnp

    batch, length, _ = features.shape
    initial = jnp.zeros((batch, params["candidate_bias"].shape[0]), dtype=jnp.float32)

    def transition(carry, inputs):
        feature, is_reset = inputs
        next_carry = _step(params, carry, feature, is_reset)
        return next_carry, next_carry

    final, states = jax.lax.scan(
        transition,
        initial,
        (jnp.swapaxes(features, 0, 1), jnp.swapaxes(reset, 0, 1)),
    )
    return final, jnp.swapaxes(states, 0, 1)


def _distribution(params, states, config: GRUConfig):
    import jax
    import jax.numpy as jnp

    raw = states @ params["head_kernel"] + params["head_bias"]
    mean, raw_scale = jnp.split(raw, 2, axis=-1)
    scale = jax.nn.softplus(jnp.clip(raw_scale, -12.0, 8.0)) + config.min_scale
    return mean, scale


def loss_and_metrics(params, features, reset, target_mask, config: GRUConfig):
    """Teacher-forced one-step Gaussian NLL; the loss mask belongs to t+1."""
    import jax.numpy as jnp

    features = jnp.asarray(features, dtype=jnp.float32)
    reset = jnp.asarray(reset, dtype=jnp.bool_)
    target_mask = jnp.asarray(target_mask, dtype=jnp.float32)
    if features.ndim != 3 or features.shape[-1] != config.feature_dim:
        raise ValueError("expected E1 input [batch, time, 5]")
    if reset.shape != features.shape[:2] or target_mask.shape != reset.shape:
        raise ValueError("reset and target mask must align with E1 inputs")
    _, states = _encode(params, features, reset)
    mean, scale = _distribution(params, states[:, :-1], config)
    target = features[:, 1:]
    nll = (0.5 * jnp.square((target - mean) / scale)
           + jnp.log(scale) + 0.5 * math.log(2.0 * math.pi)).sum(axis=-1)
    valid = target_mask[:, 1:] * (~reset[:, 1:]).astype(jnp.float32)
    count = jnp.maximum(valid.sum(), 1.0)
    loss = (nll * valid).sum() / count
    return loss, {"loss": loss, "nll": loss, "targets": valid.sum()}


def forecast_standardized(params, history, *, steps: int, seed: int,
                          config: GRUConfig):
    """Sample recursive standardized feature paths, returning [B,H,5]."""
    import jax
    import jax.numpy as jnp

    history = jnp.asarray(history, dtype=jnp.float32)
    if (history.ndim != 3 or history.shape[-1] != config.feature_dim
            or history.shape[1] < 2 or not bool(jnp.isfinite(history).all())
            or type(steps) is not int or not 1 <= steps <= 60
            or type(seed) is not int):
        raise ValueError("invalid E1 forecast history, horizon, or seed")
    validate_jax_seed(seed, name="E1 forecast seed")
    reset = jnp.zeros(history.shape[:2], dtype=jnp.bool_).at[:, 0].set(True)
    carry, _ = _encode(params, history, reset)
    root_key = jax.random.PRNGKey(seed)

    def sample_step(previous, key):
        state, rng = previous
        mean, scale = _distribution(params, state, config)
        rng, noise_key = jax.random.split(rng)
        sample = mean + scale * jax.random.normal(noise_key, mean.shape, dtype=jnp.float32)
        next_state = _step(params, state, sample, jnp.zeros((sample.shape[0],), dtype=jnp.bool_))
        return (next_state, rng), sample

    (_, _), paths = jax.lax.scan(sample_step, (carry, root_key), xs=None, length=steps)
    return jnp.swapaxes(paths, 0, 1)


def next_distribution_standardized(params, history, *, config: GRUConfig):
    """Return the exact one-step diagonal Gaussian head parameters [B,5]."""
    import jax.numpy as jnp

    history = jnp.asarray(history, dtype=jnp.float32)
    if (history.ndim != 3 or history.shape[-1] != config.feature_dim
            or history.shape[1] < 2 or not bool(jnp.isfinite(history).all())):
        raise ValueError("history must be finite [batch,time,5]")
    reset = jnp.zeros(history.shape[:2], dtype=jnp.bool_).at[:, 0].set(True)
    _, states = _encode(params, history, reset)
    return _distribution(params, states[:, -1], config)
