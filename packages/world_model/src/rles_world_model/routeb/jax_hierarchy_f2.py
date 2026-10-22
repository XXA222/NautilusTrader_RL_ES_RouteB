"""JAX RSSM adapters for RouteB's synthetic F2 event loop.

This is an engineering scenario rollout. The caller supplies already-authorized
model parameters, an observed prefix, and hierarchy features built with frozen
Train scalers. It never reads validation/test data or emits a skill score.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from .clock import MultiScaleClock
from .codec import OHLCVCodec
from .hierarchy import build_plan
from .hierarchy_dataset import HierarchyScalers, HierarchyTensors, _bucket_candles
from .jax_hierarchy import make_hierarchical_model
from .jax_hierarchy_train import (
    HierarchyTrainingConfig,
    _as_model_inputs,
    _validated_arrays,
    _window_arrays,
    load_hierarchical_checkpoint_parameters,
)
from .jax_model import RSSMConfig
from .market import Candle, MarketSequence
from .randomness import MAX_JAX_SEED_EXCLUSIVE, validate_jax_seed
from .synthetic_feedback import F2Result, SyntheticFeedback, rollout_hierarchical_f2


def _assert_prefix_tensors(prefix: MarketSequence, tensors: HierarchyTensors,
                           scalers: HierarchyScalers) -> None:
    """Require the model context to be an exact causal projection of the prefix."""
    if prefix.scope in {"test", "holdout"}:
        raise ValueError("F2 scenario rollouts cannot consume Test or Holdout prefixes")
    if any(s.dataset_id != prefix.dataset_id or s.train_scope != "train"
           for s in (scalers.minute, scalers.hour, scalers.day)):
        raise ValueError("F2 requires same-dataset frozen Train scalers")
    fingerprints = (scalers.minute.fingerprint, scalers.hour.fingerprint,
                    scalers.day.fingerprint)
    if (tensors.dataset_id != prefix.dataset_id
            or tensors.scaler_fingerprints != fingerprints):
        raise ValueError("F2 context tensors are not bound to the supplied Train scalers")
    bars = prefix.candles
    anchor = bars[-1].ts_close_ns
    if bars[-1].available_at_ns != anchor:
        raise ValueError("F2 anchor candle must be available at its close")
    minute_coords = OHLCVCodec.encode(bars[1:], previous_close=bars[0].close)
    clock = MultiScaleClock(bars)
    hour_bars = _bucket_candles(clock.buckets(60, as_of_ns=anchor))
    day_bars = _bucket_candles(clock.buckets(1440, as_of_ns=anchor))
    if len(hour_bars) < 2 or len(day_bars) < 2:
        raise ValueError("F2 needs at least two fully observed bars at every scale")
    hour_coords = OHLCVCodec.encode(hour_bars, previous_close=hour_bars[0].open)
    day_coords = OHLCVCodec.encode(day_bars, previous_close=day_bars[0].open)
    plan = build_plan(bars)
    expected_hour_day = (np.searchsorted(plan.day_available_at,
                                         plan.hour_available_at, side="right") - 1)
    expected = {
        "minute_x": scalers.minute.transform(minute_coords)[None],
        "hour_x": scalers.hour.transform(hour_coords)[None],
        "day_x": scalers.day.transform(day_coords)[None],
        "hour_index": np.asarray(plan.hour_last_visible[1:], dtype=np.int32)[None],
        "day_index": np.asarray(plan.day_last_visible[1:], dtype=np.int32)[None],
        "hour_day_index": np.asarray(expected_hour_day, dtype=np.int32)[None],
        "minute_available_at": np.asarray(
            [bar.available_at_ns for bar in bars[1:]], dtype=np.int64),
        "hour_available_at": np.asarray(plan.hour_available_at, dtype=np.int64),
        "day_available_at": np.asarray(plan.day_available_at, dtype=np.int64),
    }
    for name, value in expected.items():
        actual = np.asarray(getattr(tensors, name))
        if actual.shape != value.shape:
            raise ValueError(f"F2 prefix tensor {name} does not match observed history")
        if np.issubdtype(value.dtype, np.floating):
            matches = np.allclose(actual, value, rtol=1e-6, atol=1e-6)
        else:
            matches = np.array_equal(actual, value)
        if not matches:
            raise ValueError(f"F2 prefix tensor {name} is not the frozen causal projection")


def rollout_hierarchical_f2_jax(*, prefix: MarketSequence,
        context_tensors: HierarchyTensors, scalers: HierarchyScalers,
        checkout: str, params, steps: int,
        mode: Literal["F2_SLOW_ONLY", "F2_FULL_FILTER"],
        minute_config: RSSMConfig | None = None,
        hour_config: RSSMConfig | None = None,
        day_config: RSSMConfig | None = None,
        training_config: HierarchyTrainingConfig | None = None,
        seed: int = 0, record_minute_states: bool = False) -> F2Result:
    """Connect generated OHLCV bars and closed slow bars to actual JAX RSSMs.

    F2_SLOW_ONLY advances the minute carry with the learned prior. F2_FULL_FILTER
    re-filters each generated minute observation from the same pre-step carry.
    Both modes feed only complete generated hourly/daily bars into slow RSSMs;
    a simultaneous daily/hourly close applies the day update first.
    """
    import jax
    import ninjax as nj

    validate_jax_seed(seed, name="F2 seed")
    if (type(steps) is not int or steps < 1
            or seed + steps - 1 + (1_000_003 if mode == "F2_FULL_FILTER" else 0)
            >= MAX_JAX_SEED_EXCLUSIVE):
        raise ValueError("F2 step-derived random seed exceeds the JAX seed domain")
    _assert_prefix_tensors(prefix, context_tensors, scalers)
    cfg = training_config or HierarchyTrainingConfig()
    minute_config = minute_config or RSSMConfig()
    hour_config = hour_config or RSSMConfig(
        deter=128, hidden=96, stoch=8, classes=8, blocks=8)
    day_config = day_config or RSSMConfig(
        deter=96, hidden=64, stoch=8, classes=8, blocks=8)
    arrays = _validated_arrays(context_tensors)
    if arrays["minute_x"].shape[1] < cfg.window_length:
        raise ValueError("F2 observed prefix is shorter than the configured RSSM context")
    start = arrays["minute_x"].shape[1] - cfg.window_length
    window = _window_arrays(arrays, start=start, config=cfg)
    inputs = _as_model_inputs(window)
    model = make_hierarchical_model(
        checkout, minute=minute_config, hour=hour_config, day=day_config,
        variant=cfg.hierarchy_variant, risk_horizons=cfg.risk_horizons,
        risk_horizon_weights=cfg.risk_horizon_weights,
        risk_auxiliary_scale=cfg.risk_auxiliary_scale)
    initialize_step = jax.jit(nj.pure(model.f2_initialize))
    minute_prior_step = jax.jit(nj.pure(model.f2_prior_step))
    day_posterior_step = jax.jit(nj.pure(model.f2_update_day))
    hour_posterior_step = jax.jit(nj.pure(model.f2_update_hour))
    minute_posterior_step = jax.jit(nj.pure(model.f2_filter_minute))
    _, initialized = initialize_step(
        params, *inputs[:6], inputs[7], inputs[8], seed=seed)
    minute_state, hour_state, day_state, hour_day_context = initialized

    clock = MultiScaleClock(prefix.candles)
    latest_hour = clock.latest(60, prefix.candles[-1].ts_close_ns)
    latest_day = clock.latest(1440, prefix.candles[-1].ts_close_ns)
    if latest_hour is None or latest_day is None:
        raise ValueError("F2 needs a fully observed latest hour and day anchor")
    last_minute_close = prefix.candles[-1].close
    last_hour_close = latest_hour.close
    last_day_close = latest_day.close
    pending: dict[str, object] = {}
    posterior_deltas: list[float] = []

    def slow_feature(bucket, scaler, previous_close: float) -> np.ndarray:
        candle = Candle(bucket.close_ns, bucket.available_at_ns,
                        bucket.instrument, bucket.open, bucket.high,
                        bucket.low, bucket.close, bucket.volume)
        coords = OHLCVCodec.encode((candle,), previous_close=previous_close)
        return scaler.transform(coords)[0]

    def minute_prior(current, hour, day, close_ns, step_seed):
        nonlocal last_minute_close
        _, (prior_carry, sample, _, scale, prior_logits) = minute_prior_step(
            params, current, hour, day, seed=step_seed)
        sample_np = np.asarray(jax.device_get(sample), dtype=np.float64)
        scale_np = np.asarray(jax.device_get(scale), dtype=np.float64)
        if (sample_np.shape != (1, 5) or scale_np.shape != (1, 5)
                or not np.isfinite(sample_np).all() or not np.isfinite(scale_np).all()
                or np.any(scale_np <= 0)):
            raise RuntimeError("JAX F2 prior returned an invalid minute distribution")
        raw_coords = scalers.minute.inverse(sample_np)
        candle, = OHLCVCodec.decode(
            raw_coords, previous_close=last_minute_close,
            start_close_ns=close_ns, instrument=prefix.candles[-1].instrument)
        pending.clear()
        pending.update({"minute_carry": current, "hour_state": hour,
                        "day_state": day, "previous_close": last_minute_close,
                        "close_ns": close_ns, "step_seed": step_seed,
                        "prior_logits": np.asarray(
                            jax.device_get(prior_logits), dtype=np.float32)})
        last_minute_close = candle.close
        return prior_carry, candle

    def day_posterior(current, bucket, step_seed):
        nonlocal last_day_close
        feature = slow_feature(bucket, scalers.day, last_day_close)
        if cfg.hierarchy_variant == "e3_features":
            updated = feature[None]
        else:
            _, updated = day_posterior_step(
                params, current, feature, seed=step_seed)
        last_day_close = bucket.close
        return updated

    def hour_posterior(current, bucket, day, step_seed):
        nonlocal last_hour_close, hour_day_context
        feature = slow_feature(bucket, scalers.hour, last_hour_close)
        _, (updated, visible_day_context) = hour_posterior_step(
            params, current, feature, hour_day_context, day, seed=step_seed)
        hour_day_context = visible_day_context
        last_hour_close = bucket.close
        return updated

    def minute_posterior(prior_state, synthetic, hour, day):
        del prior_state
        pending_ns = pending.get("close_ns")
        if pending_ns != synthetic.ts_close_ns:
            raise RuntimeError("JAX F2 minute posterior lost its matching prior state")
        coords = OHLCVCodec.encode(
            (synthetic,), previous_close=float(pending["previous_close"]))
        feature = scalers.minute.transform(coords)[0]
        _, (updated, posterior_logits) = minute_posterior_step(
            params, pending["minute_carry"], feature,
            pending["hour_state"], pending["day_state"], hour, day,
            seed=int(pending["step_seed"]) + 1_000_003)
        prior_logits = np.asarray(pending["prior_logits"], dtype=np.float64)
        posterior_logits = np.asarray(
            jax.device_get(posterior_logits), dtype=np.float64)
        posterior_deltas.append(float(np.max(np.abs(
            prior_logits - posterior_logits))))
        return updated

    result = rollout_hierarchical_f2(
        feedback=SyntheticFeedback(
            prefix.candles, anchor_ns=prefix.candles[-1].ts_close_ns),
        steps=steps, minute_state=minute_state, hour_state=hour_state,
        day_state=day_state, minute_prior=minute_prior,
        hour_posterior=hour_posterior, day_posterior=day_posterior,
        mode=mode,
        minute_posterior=(minute_posterior if mode == "F2_FULL_FILTER" else None),
        seed=seed, record_minute_states=record_minute_states)
    return replace(
        result,
        minute_posterior_updates=len(posterior_deltas),
        minute_posterior_max_abs_logit_delta=(
            max(posterior_deltas) if posterior_deltas else None))


def rollout_hierarchical_f2_checkpoint(*, training_tensors: HierarchyTensors,
        train_scope_id: str, dataset_id: str, checkpoint: str,
        prefix: MarketSequence, context_tensors: HierarchyTensors,
        scalers: HierarchyScalers, checkout: str, steps: int,
        mode: Literal["F2_SLOW_ONLY", "F2_FULL_FILTER"],
        minute_config: RSSMConfig | None = None,
        hour_config: RSSMConfig | None = None,
        day_config: RSSMConfig | None = None,
        training_config: HierarchyTrainingConfig | None = None,
        seed: int = 0, record_minute_states: bool = False) -> F2Result:
    """Run F2 from a checkpoint bound to the exact Train tensor identity."""
    expected_scalers = (scalers.minute.fingerprint, scalers.hour.fingerprint,
                        scalers.day.fingerprint)
    if (training_tensors.dataset_id != dataset_id
            or training_tensors.scaler_fingerprints != expected_scalers):
        raise ValueError("F2 model checkpoint and supplied scalers have different Train provenance")
    params = load_hierarchical_checkpoint_parameters(
        tensors=training_tensors, train_scope_id=train_scope_id,
        dataset_id=dataset_id, checkout=checkout, checkpoint=checkpoint,
        minute_config=minute_config, hour_config=hour_config,
        day_config=day_config, training_config=training_config)
    return rollout_hierarchical_f2_jax(
        prefix=prefix, context_tensors=context_tensors, scalers=scalers,
        checkout=checkout, params=params, steps=steps, mode=mode,
        minute_config=minute_config, hour_config=hour_config,
        day_config=day_config, training_config=training_config,
        seed=seed, record_minute_states=record_minute_states)
