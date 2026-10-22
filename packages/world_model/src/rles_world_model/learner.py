"""Train and infer the Route B market-only DreamerV3 RSSM.

This module is intentionally importable without JAX. ``create_learner`` imports
the pinned upstream checkout only inside the dedicated Python 3.12 worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

FORECAST_SUPPORT = {
    "log_return_min": -0.1,
    "log_return_max": 0.1,
    "wick_min": 0.0,
    "wick_max": 2.0,
    "log1p_volume_min": 0.0,
    "log1p_volume_max": 30.0,
}


@dataclass(frozen=True, slots=True)
class RSSMConfig:
    deter: int = 512
    hidden: int = 256
    stoch: int = 32
    classes: int = 4
    blocks: int = 8
    learning_rate: float = 3e-4
    kl_scale: float = 1.0
    hourly_aux_scale: float = 0.1
    free_nats: float = 1.0
    seed: int = 0
    input_features: int = 79

    def __post_init__(self) -> None:
        integer_fields = (self.deter, self.hidden, self.stoch, self.classes,
                          self.blocks, self.seed, self.input_features)
        if (any(type(value) is not int for value in integer_fields)
                or min(self.deter, self.hidden, self.stoch, self.classes, self.blocks) < 1
                or self.deter % self.blocks or self.learning_rate <= 0
                or not all(math.isfinite(value) for value in
                           (self.learning_rate, self.kl_scale, self.hourly_aux_scale,
                            self.free_nats))
                or self.kl_scale < 0 or self.hourly_aux_scale < 0 or self.free_nats < 0
                or self.input_features != 79):
            raise ValueError("invalid RSSM configuration")


def create_learner(upstream_checkout: str, config: RSSMConfig = RSSMConfig()) -> Any:
    """Build a TrainState-like object backed by the exact pinned DreamerV3 RSSM."""
    import sys
    from pathlib import Path

    checkout = str(Path(upstream_checkout).resolve())
    if checkout not in sys.path:
        sys.path.insert(0, checkout)
    from .upstream import UPSTREAM_COMMIT, inspect_upstream

    status = inspect_upstream(checkout)
    if status.status != "READY" or status.observed_commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"pinned DreamerV3 worker unavailable: {status.detail or status.status}")

    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    import optax
    from dreamerv3.rssm import RSSM
    from embodied.jax import nets as nn

    class MarketWorldModel(nj.Module):
        def _decode_features(self, latent, hour_state, day_state):
            hour_embed = self.sub("forecast_hour_state", nn.MLP, 1, config.hidden,
                                  act="gelu", norm="rms")(hour_state)
            day_embed = self.sub("forecast_day_state", nn.MLP, 1, config.hidden,
                                 act="gelu", norm="rms")(day_state)
            gate_input = nn.cast(jnp.concatenate([latent, hour_state, day_state], axis=-1))
            hour_gate = jax.nn.sigmoid(self.sub("forecast_hour_gate", nn.Linear, 1)(gate_input))
            day_gate = jax.nn.sigmoid(self.sub("forecast_day_gate", nn.Linear, 1)(gate_input))
            fused = jnp.concatenate([latent, hour_gate * hour_embed,
                                     day_gate * day_embed], axis=-1)
            decoded = self.sub("market_decoder", nn.MLP, 3, config.hidden,
                               act="gelu", norm="rms")(fused)
            return self.sub("market_head", nn.Linear, 23)(decoded)

        def _encode_market(self, observations):
            if observations.shape[-1] != config.input_features:
                raise ValueError(
                    f"market_features_v1 needs {config.input_features} features; "
                    f"received {observations.shape[-1]}"
                )
            minute = observations[..., :10]
            hour = observations[..., 10:21]
            day = observations[..., 21:33]
            calendar = observations[..., 33:39]
            hour_state = observations[..., 39:71]
            daily_state = observations[..., 71:79]
            minute_embed = self.sub("minute_encoder", nn.MLP, 1, config.hidden,
                                    act="gelu", norm="rms")(
                jnp.concatenate([minute, calendar], axis=-1))
            hour_embed = self.sub("hour_encoder", nn.MLP, 1, config.hidden,
                                  act="gelu", norm="rms")(
                jnp.concatenate([hour, hour_state], axis=-1))
            day_embed = self.sub("day_encoder", nn.MLP, 1, config.hidden,
                                 act="gelu", norm="rms")(
                jnp.concatenate([day, daily_state], axis=-1))
            context = observations
            context = nn.cast(context)
            hour_gate = jax.nn.sigmoid(self.sub("hour_gate", nn.Linear, 1)(context))
            day_gate = jax.nn.sigmoid(self.sub("day_gate", nn.Linear, 1)(context))
            fused = jnp.concatenate(
                [minute_embed, hour_gate * hour_embed, day_gate * day_embed], axis=-1,
            )
            return self.sub("cross_scale_fusion", nn.MLP, 2, config.hidden,
                            act="gelu", norm="rms")(fused)

        def __call__(self, observations, targets, hourly_targets, hourly_mask,
                     reset, *, training):
            action_space = {"market_dummy": elements.Space(np.float32, (1,))}
            tokens = self._encode_market(observations)
            rssm = self.sub(
                "market_rssm", RSSM, act_space=action_space,
                deter=config.deter, hidden=config.hidden, stoch=config.stoch,
                classes=config.classes, blocks=config.blocks,
                imglayers=2, obslayers=1, dynlayers=1, free_nats=config.free_nats,
            )
            actions = {"market_dummy": jnp.zeros((*reset.shape, 1), jnp.float32)}
            carry, entries, losses, feat, metrics = rssm.loss(
                rssm.initial(observations.shape[0]), tokens, actions, reset, training,
            )
            feature = jnp.concatenate(
                [feat["deter"], feat["stoch"].reshape((*reset.shape, -1))], axis=-1,
            )
            # The prior decoder conditions on observed slow-state snapshots;
            # rollout may advance these states from generated complete periods.
            params = self._decode_features(
                feature, observations[..., 39:71], observations[..., 71:79],
            )
            means, log_scales = params[..., :5], jnp.clip(params[..., 5:10], -5.0, 3.0)
            factors = params[..., 10:20].reshape((*reset.shape, 5, 2)) * 0.25
            zero_logits = params[..., 20:23]
            residual = targets - means
            diagonal = jnp.exp(log_scales)
            covariance = jnp.einsum("...ir,...jr->...ij", factors, factors)
            covariance += jnp.eye(5, dtype=jnp.float32) * diagonal[..., :, None] ** 2
            sign, logdet = jnp.linalg.slogdet(covariance)
            solved = jnp.linalg.solve(covariance, residual[..., None])[..., 0]
            mahalanobis = jnp.sum(residual * solved, axis=-1)
            gaussian_nll = 0.5 * (mahalanobis + logdet + 5.0 * jnp.log(2 * jnp.pi))
            positive = targets[..., 2:5] > 0
            hurdle_nll = jnp.sum(
                jnp.where(positive, jax.nn.softplus(-zero_logits),
                          jax.nn.softplus(zero_logits)), axis=-1,
            )
            # The joint low-rank likelihood captures residual dependence. Hurdle
            # terms encode exact zero wick and volume mass without clipping data.
            reconstruction = jnp.where(sign > 0, gaussian_nll, 1e6) + hurdle_nll
            kl = jnp.mean(losses["dyn"]) + jnp.mean(losses["rep"])
            hour_head = self.sub("hour_aux_decoder", nn.MLP, 2, config.hidden,
                                 act="gelu", norm="rms")
            hour_prediction = self.sub("hour_aux_head", nn.Linear, 5)(hour_head(feature))
            hour_error = jnp.mean(jnp.square(hour_prediction - hourly_targets), axis=-1)
            hourly_aux = jnp.sum(hour_error * hourly_mask) / jnp.maximum(jnp.sum(hourly_mask), 1)
            total = (jnp.mean(reconstruction) + config.kl_scale * kl
                     + config.hourly_aux_scale * hourly_aux)
            return total, (carry, feat, params, {
                "loss": total, "reconstruction": jnp.mean(reconstruction),
                "kl": kl, "dyn_entropy": metrics["dyn_ent"],
                "rep_entropy": metrics["rep_ent"], "hourly_aux_loss": hourly_aux,
                "hourly_aux_count": jnp.sum(hourly_mask),
            })

        def prior_step(self, carry):
            import elements
            import jax.numpy as jnp
            from dreamerv3.rssm import RSSM

            action_space = {"market_dummy": elements.Space(np.float32, (1,))}
            rssm = self.sub(
                "market_rssm", RSSM, act_space=action_space,
                deter=config.deter, hidden=config.hidden, stoch=config.stoch,
                classes=config.classes, blocks=config.blocks,
                imglayers=2, obslayers=1, dynlayers=1, free_nats=config.free_nats,
            )
            action = {"market_dummy": jnp.zeros((carry["deter"].shape[0], 1), jnp.float32)}
            carry, (features, _) = rssm.imagine(
                carry, action, length=1, training=False, single=True,
            )
            return carry, features

        def decode_prior(self, features, hour_state, day_state):
            latent = jnp.concatenate(
                [features["deter"], features["stoch"].reshape(
                    (*features["deter"].shape[:-1], -1))], axis=-1,
            )
            return self._decode_features(latent, hour_state, day_state)

        def initial_carry(self, batch_size: int):
            import elements
            import numpy as np
            from dreamerv3.rssm import RSSM

            action_space = {"market_dummy": elements.Space(np.float32, (1,))}
            rssm = self.sub(
                "market_rssm", RSSM, act_space=action_space,
                deter=config.deter, hidden=config.hidden, stoch=config.stoch,
                classes=config.classes, blocks=config.blocks,
                imglayers=2, obslayers=1, dynlayers=1, free_nats=config.free_nats,
            )
            return rssm.initial(batch_size)

        def posterior_step(self, carry, observation, reset):
            import elements
            import jax.numpy as jnp
            import numpy as np
            from dreamerv3.rssm import RSSM

            action_space = {"market_dummy": elements.Space(np.float32, (1,))}
            tokens = self._encode_market(observation)
            rssm = self.sub(
                "market_rssm", RSSM, act_space=action_space,
                deter=config.deter, hidden=config.hidden, stoch=config.stoch,
                classes=config.classes, blocks=config.blocks,
                imglayers=2, obslayers=1, dynlayers=1, free_nats=config.free_nats,
            )
            action = {"market_dummy": jnp.zeros((observation.shape[0], 1), jnp.float32)}
            return rssm.observe(carry, tokens, action, reset, False, single=True)

    model = MarketWorldModel(name="route_b_market_world_model")
    optimizer = optax.adam(config.learning_rate)
    return _Learner(model, config, optimizer, jax, jnp, nj, optax)


class _Learner:
    def __init__(self, model, config, optimizer, jax, jnp, nj, optax):
        self.model, self.config = model, config
        self.jax, self.jnp, self.nj, self.optimizer = jax, jnp, nj, optimizer
        self.optax = optax
        self.opt_state = None
        self.params = None
        self.step = 0
        self.input_feature_count = None

    def initialize(self, observations, targets, reset, hourly_targets=None, hourly_mask=None):
        jax = self.jax
        observations, targets, reset, hourly_targets, hourly_mask = self._validate_batch(
            observations, targets, reset, hourly_targets, hourly_mask,
        )
        self.params = self.nj.init(self.model)(
            {}, observations, targets, hourly_targets, hourly_mask, reset, training=True,
            seed=jax.random.PRNGKey(self.config.seed),
        )
        self.input_feature_count = int(observations.shape[-1])
        self.opt_state = self.optimizer.init(self.params)
        return self

    def train_step(self, observations, targets, reset, *, hourly_targets=None,
                   hourly_mask=None, seed: int | None = None):
        observations, targets, reset, hourly_targets, hourly_mask = self._validate_batch(
            observations, targets, reset, hourly_targets, hourly_mask,
        )
        if self.params is None:
            self.initialize(observations, targets, reset, hourly_targets, hourly_mask)
        jax, jnp, nj = self.jax, self.jnp, self.nj
        obs, target, resets = (jnp.asarray(observations, jnp.float32),
                               jnp.asarray(targets, jnp.float32),
                               jnp.asarray(reset, jnp.bool_))
        key = jax.random.PRNGKey(self.config.seed + self.step if seed is None else seed)

        def objective(params):
            (_, output) = nj.pure(self.model)(params, obs, target,
                                               hourly_targets, hourly_mask, resets,
                                               training=True, seed=key)
            return output[0], output[1][3]

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(self.params)
        updates, self.opt_state = self.optimizer.update(gradients, self.opt_state, self.params)
        self.params = self.optax.apply_updates(self.params, updates)
        self.step += 1
        return {name: float(jax.device_get(value)) for name, value in metrics.items()}

    def infer_posterior(self, observations, *, seed: int = 0):
        """Encode a causal warmup/observed sequence; no action or account input exists."""
        obs = self.jnp.asarray(observations, self.jnp.float32)
        if (obs.ndim != 3 or obs.shape[0] != 1 or obs.shape[1] < 1
                or obs.shape[-1] != self.config.input_features
                or not bool(self.jax.device_get(self.jnp.isfinite(obs).all()))):
            raise ValueError("posterior inference expects nonempty [batch, time, feature] inputs")
        carry = self.initial_carry(obs.shape[0])
        root = self.jax.random.PRNGKey(seed)
        features = []
        for index in range(obs.shape[1]):
            carry, feature = self.infer_step(
                carry, obs[:, index], reset=index == 0,
                seed=self.jax.random.fold_in(root, index),
            )
            features.append(feature)
        stacked = self.jax.tree.map(lambda *values: self.jnp.stack(values, axis=1), *features)
        return carry, stacked

    def initial_carry(self, batch_size: int = 1):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("RSSM batch size must be a positive integer")
        _, carry = self.nj.pure(self.model.initial_carry)(
            self.params, batch_size, seed=self.jax.random.PRNGKey(self.config.seed),
        )
        return carry

    def infer_step(self, carry, observation, *, reset: bool = False, seed: Any):
        """Apply one causal posterior transition for incremental-runtime parity."""
        obs = self.jnp.asarray(observation, self.jnp.float32)
        if obs.ndim != 2:
            raise ValueError("one posterior step expects [batch, feature] observations")
        resets = self.jnp.full((obs.shape[0],), reset, self.jnp.bool_)
        _, output = self.nj.pure(self.model.posterior_step)(
            self.params, carry, obs, resets, seed=seed,
        )
        return output[0], output[2]

    def prior_forecast(
        self, carry, *, steps: int, paths: int = 128, seed: int = 0,
        initial_scale_state: tuple[Any, Any] | None = None,
        scale_state_updater: Callable[[np.ndarray, int, np.ndarray, np.ndarray],
                                      tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        """Roll the market prior; optional synthetic path callbacks advance only completed slow periods."""
        if type(steps) is not int or steps < 1 or type(paths) is not int or paths < 1:
            raise ValueError("prior forecast steps and paths must be positive integers")
        jnp, jax, nj = self.jnp, self.jax, self.nj
        if carry["deter"].shape[0] != 1:
            raise ValueError("one prior forecast call requires exactly one observed market context")
        expanded = jax.tree.map(lambda value: jnp.repeat(value, paths, axis=0), carry)
        if initial_scale_state is None:
            hour_state = np.zeros((paths, 32), dtype=np.float32)
            day_state = np.zeros((paths, 8), dtype=np.float32)
        else:
            hour, day = (np.asarray(value, dtype=np.float32) for value in initial_scale_state)
            if hour.shape != (1, 32) or day.shape != (1, 8):
                raise ValueError("initial slow-state snapshots must have shapes [1,32] and [1,8]")
            hour_state = np.repeat(hour, paths, axis=0)
            day_state = np.repeat(day, paths, axis=0)
        root = jax.random.PRNGKey(seed)
        generated = []
        final_params = None
        for index in range(steps):
            step_key = jax.random.fold_in(root, index)
            _, (expanded, features) = nj.pure(self.model.prior_step)(
                self.params, expanded, seed=step_key,
            )
            _, params = nj.pure(self.model.decode_prior)(
                self.params, features, jnp.asarray(hour_state), jnp.asarray(day_state),
                seed=jax.random.fold_in(step_key, 1),
            )
            sample = self._sample_prior_parameters(
                params, seed=jax.random.fold_in(step_key, 2),
            )
            generated.append(sample)
            final_params = params
            if scale_state_updater is not None:
                hour_state, day_state = scale_state_updater(
                    np.asarray(jax.device_get(sample)), index,
                    hour_state, day_state,
                )
                hour_state = np.asarray(hour_state, dtype=np.float32)
                day_state = np.asarray(day_state, dtype=np.float32)
                if hour_state.shape != (paths, 32) or day_state.shape != (paths, 8):
                    raise ValueError("generated scale-state updater returned invalid per-path state shapes")
        samples = jnp.stack(generated, axis=1)
        return samples, {
            "carry": expanded, "parameters": final_params,
            "hour_state": hour_state, "day_state": day_state,
        }

    def _sample_prior_parameters(self, params, *, seed: Any):
        jnp, jax = self.jnp, self.jax
        means, log_scales = params[..., :5], jnp.clip(params[..., 5:10], -5, 3)
        factors = params[..., 10:20].reshape((*means.shape[:-1], 5, 2)) * 0.25
        logits = params[..., 20:23]
        eps_key, factor_key, hurdle_key = jax.random.split(seed, 3)
        eps = jax.random.normal(eps_key, means.shape, dtype=jnp.float32)
        factor_eps = jax.random.normal(factor_key, (*means.shape[:-1], 2), dtype=jnp.float32)
        sample = means + jnp.exp(log_scales) * eps + jnp.einsum("...ir,...r->...i", factors, factor_eps)
        sample = sample.at[..., :2].set(jnp.clip(
            sample[..., :2], FORECAST_SUPPORT["log_return_min"],
            FORECAST_SUPPORT["log_return_max"],
        ))
        positive = jnp.clip(
            sample[..., 2:5], FORECAST_SUPPORT["log1p_volume_min"],
            FORECAST_SUPPORT["log1p_volume_max"],
        )
        positive = positive.at[..., :2].set(jnp.clip(
            positive[..., :2], FORECAST_SUPPORT["wick_min"], FORECAST_SUPPORT["wick_max"],
        ))
        is_positive = jax.random.bernoulli(hurdle_key, jax.nn.sigmoid(logits))
        return sample.at[..., 2:5].set(jnp.where(is_positive, positive, 0.0))

    def checkpoint(self) -> dict[str, Any]:
        if self.params is None:
            raise RuntimeError("cannot checkpoint an uninitialized learner")
        return {"params": self.params, "optimizer_state": self.opt_state,
                "step": self.step, "seed": self.config.seed}

    def restore(self, checkpoint: dict[str, Any]) -> None:
        if set(checkpoint) != {"params", "optimizer_state", "step", "seed"}:
            raise ValueError("checkpoint does not match the Route B learner schema")
        if type(checkpoint["step"]) is not int or checkpoint["step"] < 0:
            raise ValueError("checkpoint training step is invalid")
        self.params = checkpoint["params"]
        self.opt_state = checkpoint["optimizer_state"]
        self.step = checkpoint["step"]

    def _validate_batch(self, observations, targets, reset, hourly_targets=None, hourly_mask=None):
        obs = self.jnp.asarray(observations, self.jnp.float32)
        target = self.jnp.asarray(targets, self.jnp.float32)
        resets = self.jnp.asarray(reset, self.jnp.bool_)
        hour_target = (self.jnp.zeros((*obs.shape[:2], 5), self.jnp.float32)
                       if hourly_targets is None else
                       self.jnp.asarray(hourly_targets, self.jnp.float32))
        hour_mask = (self.jnp.zeros(obs.shape[:2], self.jnp.float32)
                     if hourly_mask is None else
                     self.jnp.asarray(hourly_mask, self.jnp.float32))
        if (obs.ndim != 3 or target.shape != (*obs.shape[:2], 5)
                or resets.shape != obs.shape[:2] or hour_target.shape != (*obs.shape[:2], 5)
                or hour_mask.shape != obs.shape[:2]
                or bool(self.jax.device_get(((hour_mask != 0) & (hour_mask != 1)).any()))
                or obs.shape[0] < 1 or obs.shape[1] < 1
                or obs.shape[-1] != self.config.input_features
                or not bool(self.jax.device_get(self.jnp.isfinite(obs).all()))
                or not bool(self.jax.device_get(self.jnp.isfinite(target).all()))
                or not bool(self.jax.device_get(self.jnp.isfinite(hour_target).all()))
                or bool(self.jax.device_get((target[..., 2:] < 0).any()))):
            raise ValueError("training arrays need finite 79-feature inputs, minute/hour [batch,time,5] targets, and aligned masks")
        return obs, target, resets, hour_target, hour_mask
