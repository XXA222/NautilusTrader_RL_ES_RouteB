"""Synthetic end-to-end RSSM train, prior forecast, and bundle-resume smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from rles_world_model.bundle import load_bundle, save_bundle
from rles_world_model.dataset import build_train_sequences
from rles_world_model.features import MarketFeatureSpec
from rles_world_model.forecast import decode_prior_paths, generated_scale_state_updater
from rles_world_model.learner import RSSMConfig, create_learner
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_world_model.runtime import WorldModelRuntime
from rles_data.core import DAY_NS, MINUTE_NS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames = []
    previous = 100.0
    for index in range(1, 31 * 1440 + 15):
        ts = index * MINUTE_NS
        close = 100.0 + index * 1e-5
        frames.append(MarketFrame(
            ts, ts, "SYNTHETIC", previous, max(previous, close) + 0.01,
            min(previous, close) - 0.01, close, float(index % 100),
        ))
        previous = close
    replay = MarketReplay(tuple(frames), "b" * 64)
    train_start = 31 * DAY_NS
    sequences = build_train_sequences(
        replay, train_start_ns=train_start,
        train_end_ns=train_start + 10 * MINUTE_NS,
    )
    observations, targets, resets = sequences.observations, sequences.targets, sequences.reset
    config = RSSMConfig(deter=32, hidden=32, stoch=2, classes=4, blocks=4,
                        learning_rate=3e-4, kl_scale=1.0, free_nats=0.0, seed=12018)
    result: dict[str, object] = {"status": "FAILED", "train_loss": None, "forecast_shape": None,
              "hourly_aux_loss": None, "hourly_aux_count": None,
              "ohlcv_paths": None, "slow_state_paths": None,
              "bundle_resume": False,
              "feature_spec_hash": MarketFeatureSpec().hash}
    learner = create_learner(args.checkout, config)
    if any(device.platform != "cpu" for device in learner.jax.devices()):
        raise RuntimeError("wm1_cpu_smoke must run only on CPU devices")
    metrics = learner.train_step(
        observations, targets, resets, hourly_targets=sequences.hourly_targets,
        hourly_mask=sequences.hourly_mask,
    )
    # Exercise the independent auxiliary objective even though this short
    # synthetic score window contains no complete future-hour label.
    auxiliary_targets = sequences.hourly_targets.copy()
    auxiliary_mask = sequences.hourly_mask.copy()
    auxiliary_mask[0, 0] = True
    auxiliary_targets[0, 0] = targets[0, 0]
    auxiliary_metrics = learner.train_step(
        observations, targets, resets, hourly_targets=auxiliary_targets,
        hourly_mask=auxiliary_mask,
    )
    if auxiliary_metrics["hourly_aux_count"] != 1:
        raise RuntimeError("hourly auxiliary objective did not consume its masked target")
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("RSSM learner returned a non-finite training metric")
    carry, _ = learner.infer_posterior(observations, seed=12019)
    root_key = learner.jax.random.PRNGKey(12019)
    timestep_keys = [learner.jax.random.fold_in(root_key, index)
                     for index in range(observations.shape[1])]
    incremental = learner.initial_carry(1)
    for index in range(observations.shape[1]):
        incremental, _ = learner.infer_step(
            incremental, observations[:, index], reset=bool(resets[0, index]),
            seed=timestep_keys[index],
        )
    if not learner.jax.tree_util.tree_all(learner.jax.tree.map(
            lambda left, right: learner.jnp.array_equal(left, right), carry, incremental)):
        differences = learner.jax.tree.map(
            lambda left, right: float(np.max(np.abs(np.asarray(left) - np.asarray(right)))),
            carry, incremental,
        )
        raise RuntimeError(f"incremental and batch posterior state differ: {differences}")
    origin_index = sequences.timestamps_ns[-1] // MINUTE_NS - 1
    origin_frame = replay.frames[origin_index]
    history = tuple(replay.frames[:origin_index + 1])
    scale_update = generated_scale_state_updater(
        history, origin=origin_frame, initial_hour_state=observations[0, -1, 39:71],
        initial_day_state=observations[0, -1, 71:79], paths=128, tick_size=0.01,
    )
    initial_scale_state = (observations[:, -1, 39:71], observations[:, -1, 71:79])
    samples, rollout_state = learner.prior_forecast(
        carry, steps=120, paths=128, seed=12020,
        initial_scale_state=initial_scale_state, scale_state_updater=scale_update,
    )
    samples_np = np.asarray(samples)
    if samples_np.shape != (128, 120, 5) or not np.isfinite(samples_np).all():
        raise RuntimeError(f"invalid prior output shape or values: {samples_np.shape}")
    final_hour_state = np.asarray(rollout_state["hour_state"])
    if (final_hour_state.shape != (128, 32)
            or np.unique(np.round(final_hour_state, decimals=6), axis=0).shape[0] < 2):
        raise RuntimeError("generated complete-hour updates were not kept independent by path")
    # A single context uses separate recurrent carries and independent stochastic
    # prior paths; same explicit seed is reproducible for deployment parity.
    repeat_update = generated_scale_state_updater(
        history, origin=origin_frame, initial_hour_state=observations[0, -1, 39:71],
        initial_day_state=observations[0, -1, 71:79], paths=128, tick_size=0.01,
    )
    repeat, _ = learner.prior_forecast(
        carry, steps=120, paths=128, seed=12020,
        initial_scale_state=initial_scale_state, scale_state_updater=repeat_update,
    )
    if not np.array_equal(samples_np, np.asarray(repeat)):
        raise RuntimeError("fixed-seed prior forecast is not deterministic")
    paths = decode_prior_paths(
        samples_np, previous_close=origin_frame.close,
        last_close_ns=sequences.timestamps_ns[-1],
        available_from_ns=sequences.timestamps_ns[-1], instrument="SYNTHETIC",
        tick_size=0.01,
    )
    if len(paths) != 128 or any(len(path.minutes) != 120 for path in paths):
        raise RuntimeError("prior paths did not decode to 128 independent minute trajectories")
    if any(len(path.hours) != 1 for path in paths):
        raise RuntimeError("forecast aggregation did not complete one generated UTC hour")

    with tempfile.TemporaryDirectory(prefix="routeb-wm-bundle-") as temp:
        bundle_path = Path(temp) / "model"
        scaler_manifest = {
            key: {"hash": scaler.hash, "fit_start_ns": scaler.fit_start_ns,
                  "fit_end_ns": scaler.fit_end_ns, "fit_count": scaler.fit_count,
                  "center": scaler.center, "spread": scaler.spread}
            for key, scaler in sequences.scalers.items()
        }
        save_bundle(bundle_path, learner, feature_spec_hash=MarketFeatureSpec().hash,
                    scaler_manifest=scaler_manifest, instrument="SYNTHETIC",
                    source_code_sha256="a" * 64)
        restored = create_learner(args.checkout, config)
        restored.initialize(observations, targets, resets,
                            sequences.hourly_targets, sequences.hourly_mask)
        manifest = load_bundle(bundle_path, restored,
                               expected_feature_spec_hash=MarketFeatureSpec().hash,
                               expected_source_code_sha256="a" * 64)
        runtime = WorldModelRuntime(restored, bundle_manifest=manifest)
        live = runtime.forecast(observations, origin=origin_frame,
                                history=history,
                                steps=60, paths=128, seed=12019, tick_size=0.01)
        if live.forecast_type != "prior_only" or len(live.paths) != 128:
            raise RuntimeError("deployment inference did not return the expected prior-only ensemble")
        restored_carry, _ = restored.infer_posterior(observations, seed=12019)
        restored_update = generated_scale_state_updater(
            history, origin=origin_frame, initial_hour_state=observations[0, -1, 39:71],
            initial_day_state=observations[0, -1, 71:79], paths=128, tick_size=0.01,
        )
        resumed, _ = restored.prior_forecast(
            restored_carry, steps=120, paths=128, seed=12020,
            initial_scale_state=initial_scale_state,
            scale_state_updater=restored_update,
        )
        if manifest["training_step"] != learner.step or not np.array_equal(samples_np, np.asarray(resumed)):
            raise RuntimeError("resumed model changed its forecast")
    result.update(status="PASS", train_loss=metrics["loss"], forecast_shape=list(samples_np.shape),
                  hourly_aux_loss=auxiliary_metrics["hourly_aux_loss"],
                  hourly_aux_count=auxiliary_metrics["hourly_aux_count"],
                  ohlcv_paths=len(paths),
                  slow_state_paths=int(np.unique(np.round(final_hour_state, 6), axis=0).shape[0]),
                  bundle_resume=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
