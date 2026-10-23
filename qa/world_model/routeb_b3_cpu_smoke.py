"""Synthetic-only B3 three-scale gradient/checkpoint smoke on locked CPU JAX."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchicalModelConfig, HierarchicalTrainingConfig,
    forecast_hierarchical_f2, train_hierarchical,
)
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import SNAPSHOT_SHA
from rles_world_model.routeb.clock import MultiScaleClock
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS


def synthetic_bars(n: int = 4_317) -> tuple[Candle, ...]:
    price = 2_000.0
    records = []
    for i in range(1, n + 1):
        open_ = price * math.exp(0.00002 * math.sin(i / 5))
        price = open_ * math.exp(0.0006 * math.sin(i / 13))
        high = max(open_, price) * math.exp(0.0002 + 0.0001 * math.sin(i / 7) ** 2)
        low = min(open_, price) * math.exp(-0.0002 - 0.0001 * math.cos(i / 11) ** 2)
        records.append(Candle(i * MINUTE_NS, i * MINUTE_NS, "SYNTHETIC.SIM",
                              open_, high, low, price, 100 + i % 30))
    return tuple(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    replay = MarketSequence(synthetic_bars(), "synthetic-b3-NOT-REAL-MARKET", "train")
    tensors, scalers = build_train_hierarchy(replay, burn_in=8)
    model = RSSMConfig(deter=8, hidden=8, stoch=1, classes=2, blocks=1)
    model_cfg = HierarchicalModelConfig(model, model, model)
    one = HierarchicalTrainingConfig(seed=47, steps=1, batch_size=1,
                                     window_length=16, burn_in=3, stride=8,
                                     report_interval=1)
    two = HierarchicalTrainingConfig(seed=47, steps=2, batch_size=1,
                                     window_length=16, burn_in=3, stride=8,
                                     report_interval=1)
    resumed = root / "b3_resumed.npz"
    uninterrupted = root / "b3_uninterrupted.npz"
    first = train_hierarchical(tensors=tensors, train_scope_id="synthetic-train-only",
        dataset_id=replay.dataset_id, scaler_sha256=hashlib.sha256(
            (scalers.minute.fingerprint + scalers.hour.fingerprint +
             scalers.day.fingerprint).encode()).hexdigest(), checkout=args.checkout,
        output_checkpoint=resumed, model_config=model_cfg, training_config=one)
    rest = train_hierarchical(tensors=tensors, train_scope_id="synthetic-train-only",
        dataset_id=replay.dataset_id, scaler_sha256=hashlib.sha256(
            (scalers.minute.fingerprint + scalers.hour.fingerprint +
             scalers.day.fingerprint).encode()).hexdigest(), checkout=args.checkout,
        output_checkpoint=resumed, model_config=model_cfg, training_config=two, resume=True)
    full = train_hierarchical(tensors=tensors, train_scope_id="synthetic-train-only",
        dataset_id=replay.dataset_id, scaler_sha256=hashlib.sha256(
            (scalers.minute.fingerprint + scalers.hour.fingerprint +
             scalers.day.fingerprint).encode()).hexdigest(), checkout=args.checkout,
        output_checkpoint=uninterrupted, model_config=model_cfg, training_config=two)
    with np.load(resumed, allow_pickle=False) as left, np.load(uninterrupted, allow_pickle=False) as right:
        if set(left.files) != set(right.files) or not all(
                np.array_equal(left[key], right[key]) for key in left.files):
            raise AssertionError("B3 interrupted/resumed state differs from uninterrupted run")
    scaler_hash = hashlib.sha256((scalers.minute.fingerprint + scalers.hour.fingerprint +
                                  scalers.day.fingerprint).encode()).hexdigest()
    decisions = np.arange(4_318, 4_322, dtype=np.int64) * MINUTE_NS
    expected_hour_events = MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decisions, timeframe_minutes=60,
        availability_delay_ns=0, observed_close_through_ns=int(tensors.hour_close_at[-1]))
    expected_day_events = MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decisions, timeframe_minutes=1_440,
        availability_delay_ns=0, observed_close_through_ns=int(tensors.day_close_at[-1]))
    f2 = forecast_hierarchical_f2(tensors=tensors, train_scope_id="synthetic-train-only",
        dataset_id=replay.dataset_id, scaler_sha256=scaler_hash, checkout=args.checkout,
        checkpoint=resumed, model_config=model_cfg, training_config=two,
        future_minute_decisions_ns=decisions, hour_availability_delay_ns=0,
        day_availability_delay_ns=0, forecast_seed=811)
    f2_future_event_perturbed = forecast_hierarchical_f2(
        tensors=tensors, train_scope_id="synthetic-train-only", dataset_id=replay.dataset_id,
        scaler_sha256=scaler_hash, checkout=args.checkout, checkpoint=resumed,
        model_config=model_cfg, training_config=two,
        future_minute_decisions_ns=decisions,
        hour_availability_delay_ns=60 * MINUTE_NS,
        day_availability_delay_ns=60 * MINUTE_NS, forecast_seed=811)
    f2_delayed_event = forecast_hierarchical_f2(
        tensors=tensors, train_scope_id="synthetic-train-only", dataset_id=replay.dataset_id,
        scaler_sha256=scaler_hash, checkout=args.checkout, checkpoint=resumed,
        model_config=model_cfg, training_config=two,
        future_minute_decisions_ns=decisions,
        hour_availability_delay_ns=MINUTE_NS,
        day_availability_delay_ns=MINUTE_NS, forecast_seed=811)
    if f2["minute_mean"].shape != (1, 4, 5) or not np.isfinite(f2["minute_mean"]).all():
        raise AssertionError("F2 did not produce finite recursive minute forecasts")
    if not np.array_equal(f2["minute_mean"][:, :2], f2_future_event_perturbed["minute_mean"][:, :2]):
        raise AssertionError("a future slow-scale event changed an earlier F2 forecast")
    if np.array_equal(f2["minute_mean"][:, 2], f2_future_event_perturbed["minute_mean"][:, 2]):
        raise AssertionError("F2 failed to condition the decision-time forecast on an available slow event")
    if not np.array_equal(f2_delayed_event["minute_mean"][:, :3],
                          f2_future_event_perturbed["minute_mean"][:, :3]):
        raise AssertionError("a delayed slow-scale event changed a forecast before it became available")
    if np.array_equal(f2_delayed_event["minute_mean"][:, 3],
                      f2_future_event_perturbed["minute_mean"][:, 3]):
        raise AssertionError("F2 did not apply a delayed slow-scale event at its availability barrier")
    if (np.any(f2["hour_generated"][:, 0] != 0)
            or np.any(f2["day_generated"][:, 0] != 0)
            or not np.any(f2["hour_generated"][:, 2] != 0)
            or not np.any(f2["day_generated"][:, 2] != 0)):
        raise AssertionError("F2 generated slow features outside/without availability events")
    last = full[-1]
    gradient_keys = ("grad_minute_dyn_norm", "grad_hour_dyn_norm", "grad_day_dyn_norm")
    if not all(math.isfinite(float(last[key])) and float(last[key]) > 0
               for key in gradient_keys):
        raise AssertionError("a B3 time scale received no finite nonzero gradient")
    if not all(math.isfinite(float(last[key])) for key in ("loss", "minute_nll", "slow_aux", "grad_norm")):
        raise AssertionError("B3 final metrics are not finite")
    result = {"status": "PASS", "scope": "synthetic_mechanism_only",
        "time_scales": ["minute", "hour", "day"], "optimizer_updates": len(first) + len(rest),
        "final_step": last["step"], "train_windows": last["train_windows"],
        "parameter_count": last["parameter_count"], "resume_parity": True,
        "finite_metrics": True, "all_scale_gradients_nonzero": True,
        "scale_gradient_norms": {key: last[key] for key in gradient_keys},
        "f2_minute_forecast_shape": list(f2["minute_mean"].shape),
        "f2_generated_hour_shape": list(f2["hour_generated"].shape),
        "f2_generated_day_shape": list(f2["day_generated"].shape),
        "f2_event_masks_respected": True,
        "f2_future_event_perturbation_causal": True,
        "f2_delayed_event_barrier_alignment": True,
        "f2_event_schedule_from_multiscale_clock": True,
        "future_decision_ns": decisions.tolist(),
        "future_hour_available_at_ns": list(expected_hour_events),
        "future_day_available_at_ns": list(expected_day_events),
        "forecast_delay_ns": {"hour": 0, "day": 0},
        "delayed_hour_available_at_ns": [4_321 * MINUTE_NS],
        "delayed_day_available_at_ns": [4_321 * MINUTE_NS],
        "delayed_forecast_delay_ns": {"hour": MINUTE_NS, "day": MINUTE_NS},
        "real_market_validation": False, "economic_claim": False,
        "snapshot_sha": SNAPSHOT_SHA, "upstream_sha": PINNED_UPSTREAM}
    (root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
