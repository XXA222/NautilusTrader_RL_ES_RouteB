"""Synthetic-only B3 three-scale gradient/checkpoint smoke on locked CPU JAX."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
from rles_world_model.routeb.hierarchical_paths import (
    align_bucket_feedback, sample_hierarchical_ohlcv_paths,
)
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchicalModelConfig, HierarchicalTrainingConfig,
    create_stateful_hierarchical_f2_predictor, forecast_hierarchical_f2,
    generate_recursive_hierarchical_f2_paths,
    hierarchy_tensor_sha256, train_hierarchical,
)
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import SNAPSHOT_SHA
from rles_world_model.routeb.clock import MultiScaleClock
from rles_world_model.routeb.hierarchy import derive_future_scale_events
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS


def synthetic_bars(n: int = 5_757) -> tuple[Candle, ...]:
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
    score_start_ns = replay.candles[8].ts_close_ns
    score_end_ns = replay.candles[-1].ts_close_ns + MINUTE_NS
    tensors, scalers = build_train_hierarchy(
        replay, burn_in=8, score_start_ns=score_start_ns,
        score_end_ns=score_end_ns)
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
    tensor_hash = hierarchy_tensor_sha256(tensors)
    first_future_minute = int(tensors.minute_available_at[-1] // MINUTE_NS) + 1
    decisions = np.arange(first_future_minute, first_future_minute + 5,
                          dtype=np.int64) * MINUTE_NS
    expected_hour_events = MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decisions, timeframe_minutes=60,
        availability_delay_ns=0, observed_close_through_ns=int(tensors.hour_close_at[-1]))
    expected_day_events = MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decisions, timeframe_minutes=1_440,
        availability_delay_ns=0, observed_close_through_ns=int(tensors.day_close_at[-1]))
    f2 = forecast_hierarchical_f2(tensors=tensors, train_scope_id="synthetic-train-only",
        dataset_id=replay.dataset_id, scaler_sha256=scaler_hash, checkout=args.checkout,
        checkpoint=resumed, model_config=model_cfg, training_config=two,
        training_tensor_sha256=tensor_hash, inference_context_sha256=tensor_hash,
        future_minute_decisions_ns=decisions, hour_availability_delay_ns=0,
        day_availability_delay_ns=0, forecast_seed=811)
    e3 = forecast_hierarchical_f2(
        tensors=tensors, train_scope_id="synthetic-train-only", dataset_id=replay.dataset_id,
        scaler_sha256=scaler_hash, checkout=args.checkout, checkpoint=resumed,
        model_config=model_cfg, training_config=two,
        training_tensor_sha256=tensor_hash, inference_context_sha256=tensor_hash,
        future_minute_decisions_ns=decisions, hour_availability_delay_ns=0,
        day_availability_delay_ns=0, forecast_seed=811, advance_slow_priors=False)
    e3_future_event_perturbed = forecast_hierarchical_f2(
        tensors=tensors, train_scope_id="synthetic-train-only", dataset_id=replay.dataset_id,
        scaler_sha256=scaler_hash, checkout=args.checkout, checkpoint=resumed,
        model_config=model_cfg, training_config=two,
        training_tensor_sha256=tensor_hash, inference_context_sha256=tensor_hash,
        future_minute_decisions_ns=decisions,
        hour_availability_delay_ns=60 * MINUTE_NS,
        day_availability_delay_ns=60 * MINUTE_NS, forecast_seed=811,
        advance_slow_priors=False)
    f2_future_event_perturbed = forecast_hierarchical_f2(
        tensors=tensors, train_scope_id="synthetic-train-only", dataset_id=replay.dataset_id,
        scaler_sha256=scaler_hash, checkout=args.checkout, checkpoint=resumed,
        model_config=model_cfg, training_config=two,
        training_tensor_sha256=tensor_hash, inference_context_sha256=tensor_hash,
        future_minute_decisions_ns=decisions,
        hour_availability_delay_ns=60 * MINUTE_NS,
        day_availability_delay_ns=60 * MINUTE_NS, forecast_seed=811)
    f2_delayed_event = forecast_hierarchical_f2(
        tensors=tensors, train_scope_id="synthetic-train-only", dataset_id=replay.dataset_id,
        scaler_sha256=scaler_hash, checkout=args.checkout, checkpoint=resumed,
        model_config=model_cfg, training_config=two,
        training_tensor_sha256=tensor_hash, inference_context_sha256=tensor_hash,
        future_minute_decisions_ns=decisions,
        hour_availability_delay_ns=MINUTE_NS,
        day_availability_delay_ns=MINUTE_NS, forecast_seed=811)
    inference_minute_x = tensors.minute_x.copy()
    inference_minute_x[:, -1, 0] += np.float32(2.0)
    inference_tensors = replace(tensors, minute_x=inference_minute_x)
    inference_hash = hierarchy_tensor_sha256(inference_tensors)
    f2_independent_context = forecast_hierarchical_f2(
        tensors=inference_tensors, train_scope_id="synthetic-train-only",
        dataset_id=replay.dataset_id, scaler_sha256=scaler_hash,
        checkout=args.checkout, checkpoint=resumed, model_config=model_cfg,
        training_config=two, training_tensor_sha256=tensor_hash,
        inference_context_sha256=inference_hash,
        future_minute_decisions_ns=decisions,
        hour_availability_delay_ns=0, day_availability_delay_ns=0,
        forecast_seed=811)
    if f2["minute_mean"].shape != (1, 5, 5) or not np.isfinite(f2["minute_mean"]).all():
        raise AssertionError("F2 did not produce finite recursive minute forecasts")
    if not np.array_equal(f2["minute_mean"][:, 0], f2["anchor_minute_mean"]):
        raise AssertionError("F2 first step is not the trained posterior next-minute head")
    if not np.array_equal(e3["minute_mean"][:, 0], f2["minute_mean"][:, 0]):
        raise AssertionError("matched E3/E4 ablation changed the shared posterior first-step head")
    if not np.array_equal(e3["minute_mean"], e3_future_event_perturbed["minute_mean"]):
        raise AssertionError("E3 no-F2 control changed under a future slow-event schedule perturbation")
    if np.array_equal(e3["minute_mean"][:, 3], f2["minute_mean"][:, 3]):
        raise AssertionError("E3/E4 ablation did not diverge after an available slow-prior event")
    if np.array_equal(f2["minute_mean"][:, 0], f2["first_prior_minute_mean"]):
        raise AssertionError("F2 posterior and advanced prior head states were not distinguished")
    if inference_hash == tensor_hash:
        raise AssertionError("independent inference context did not receive a distinct identity")
    if f2_independent_context["minute_mean"].shape != f2["minute_mean"].shape:
        raise AssertionError("independent inference context produced a malformed forecast")
    if not np.array_equal(f2["minute_mean"][:, :3], f2_future_event_perturbed["minute_mean"][:, :3]):
        raise AssertionError("a future slow-scale event changed an earlier F2 forecast")
    if np.array_equal(f2["minute_mean"][:, 3], f2_future_event_perturbed["minute_mean"][:, 3]):
        raise AssertionError("F2 failed to condition the decision-time forecast on an available slow event")
    if not np.array_equal(f2_delayed_event["minute_mean"][:, :4],
                          f2_future_event_perturbed["minute_mean"][:, :4]):
        raise AssertionError("a delayed slow-scale event changed a forecast before it became available")
    if np.array_equal(f2_delayed_event["minute_mean"][:, 4],
                      f2_future_event_perturbed["minute_mean"][:, 4]):
        raise AssertionError("F2 did not apply a delayed slow-scale event at its availability barrier")
    if (np.any(f2["hour_latent_diagnostic"][:, 0] != 0)
            or np.any(f2["day_latent_diagnostic"][:, 0] != 0)
            or not np.any(f2["hour_latent_diagnostic"][:, 2] != 0)
            or not np.any(f2["day_latent_diagnostic"][:, 2] != 0)):
        raise AssertionError("F2 generated slow features outside/without availability events")
    e3_paths = sample_hierarchical_ohlcv_paths(
        minute_mean=e3["minute_mean"][0], minute_scale=e3["minute_scale"][0],
        minute_scaler=scalers.minute, hour_scaler=scalers.hour,
        day_scaler=scalers.day, observed_prefix=replay.candles,
        paths=4, seed=991)
    e4_paths = sample_hierarchical_ohlcv_paths(
        minute_mean=f2["minute_mean"][0], minute_scale=f2["minute_scale"][0],
        minute_scaler=scalers.minute, hour_scaler=scalers.hour,
        day_scaler=scalers.day, observed_prefix=replay.candles,
        paths=4, seed=991)
    if (e3_paths.minute.shape != (4, 5, 5) or e4_paths.minute.shape != (4, 5, 5)
            or e3_paths.hour.shape != (4, 1, 5) or e4_paths.hour.shape != (4, 1, 5)
            or e3_paths.day.shape != (4, 1, 5) or e4_paths.day.shape != (4, 1, 5)
            or not np.array_equal(e3_paths.minute[:, :3], e4_paths.minute[:, :3])
            or np.array_equal(e3_paths.minute[:, 3], e4_paths.minute[:, 3])):
        raise AssertionError("E3/E4 sampled OHLCV paths failed event-aligned divergence checks")
    hour_feedback_events, day_feedback_events = derive_future_scale_events(
        minute_decisions_ns=decisions,
        hour_available_at_ns=np.asarray(expected_hour_events, dtype=np.int64),
        day_available_at_ns=np.asarray(expected_day_events, dtype=np.int64))
    aligned_hour_feedback = align_bucket_feedback(
        close_ns=e4_paths.hour_close_ns, feedback=e4_paths.hour_feedback,
        minute_decisions_ns=decisions, availability_delay_ns=0,
        expected_events=hour_feedback_events)
    aligned_day_feedback = align_bucket_feedback(
        close_ns=e4_paths.day_close_ns, feedback=e4_paths.day_feedback,
        minute_decisions_ns=decisions, availability_delay_ns=0,
        expected_events=day_feedback_events)
    e3_feedback_replay = forecast_hierarchical_f2(
        tensors=tensors, train_scope_id="synthetic-train-only", dataset_id=replay.dataset_id,
        scaler_sha256=scaler_hash, checkout=args.checkout, checkpoint=resumed,
        model_config=model_cfg, training_config=two,
        training_tensor_sha256=tensor_hash, inference_context_sha256=tensor_hash,
        future_minute_decisions_ns=decisions, hour_availability_delay_ns=0,
        day_availability_delay_ns=0, forecast_seed=811, advance_slow_priors=False,
        minute_feedback_features=e4_paths.minute_features,
        hour_feedback_features=aligned_hour_feedback,
        day_feedback_features=aligned_day_feedback)
    e4_feedback_replay = forecast_hierarchical_f2(
        tensors=tensors, train_scope_id="synthetic-train-only", dataset_id=replay.dataset_id,
        scaler_sha256=scaler_hash, checkout=args.checkout, checkpoint=resumed,
        model_config=model_cfg, training_config=two,
        training_tensor_sha256=tensor_hash, inference_context_sha256=tensor_hash,
        future_minute_decisions_ns=decisions, hour_availability_delay_ns=0,
        day_availability_delay_ns=0, forecast_seed=811,
        minute_feedback_features=e4_paths.minute_features,
        hour_feedback_features=aligned_hour_feedback,
        day_feedback_features=aligned_day_feedback)
    stateful_means = {}
    for label, advance_slow_priors, reference in (
            ("E3", False, e3_feedback_replay),
            ("E4", True, e4_feedback_replay)):
        predictor = create_stateful_hierarchical_f2_predictor(
            tensors=tensors, train_scope_id="synthetic-train-only",
            dataset_id=replay.dataset_id, scaler_sha256=scaler_hash,
            checkout=args.checkout, checkpoint=resumed, model_config=model_cfg,
            training_config=two, training_tensor_sha256=tensor_hash,
            inference_context_sha256=tensor_hash, paths=4,
            advance_slow_priors=advance_slow_priors)
        means = []
        scales_out = []
        for step in range(len(decisions)):
            if step:
                minute_feedback = e4_paths.minute_features[:, step - 1]
                hour_feedback = aligned_hour_feedback[:, step - 1]
                day_feedback = aligned_day_feedback[:, step - 1]
                hour_event = bool(hour_feedback_events[step - 1])
                day_event = bool(day_feedback_events[step - 1])
            else:
                minute_feedback = np.zeros((4, 5), dtype=np.float32)
                hour_feedback = np.zeros_like(minute_feedback)
                day_feedback = np.zeros_like(minute_feedback)
                hour_event = day_event = False
            mean, scale = predictor.predict_step(
                step=step, minute_feedback=minute_feedback,
                hour_feedback=hour_feedback, day_feedback=day_feedback,
                hour_event=hour_event, day_event=day_event)
            means.append(mean)
            scales_out.append(scale)
        stateful_mean = np.stack(means, axis=1)
        stateful_scale = np.stack(scales_out, axis=1)
        if (not np.allclose(stateful_mean[:, 0], reference["minute_mean"][:, 0],
                            rtol=1e-6, atol=1e-7)
                or not np.allclose(stateful_scale[:, 0], reference["minute_scale"][:, 0],
                                   rtol=1e-6, atol=1e-7)
                or not np.isfinite(stateful_mean).all()
                or not np.isfinite(stateful_scale).all()):
            raise AssertionError(f"stateful {label} initial head differs from replay reference")
        stateful_means[label] = stateful_mean
    if (np.array_equal(stateful_means["E3"][:, 3], stateful_means["E4"][:, 3])
            or not np.array_equal(stateful_means["E3"][:, 0],
                                  stateful_means["E4"][:, 0])):
        raise AssertionError("stateful E3/E4 did not diverge after the available slow event")
    recursive_e4_paths = generate_recursive_hierarchical_f2_paths(
        tensors=tensors, train_scope_id="synthetic-train-only",
        dataset_id=replay.dataset_id, scaler_sha256=scaler_hash,
        checkout=args.checkout, checkpoint=resumed,
        model_config=model_cfg, training_config=two,
        training_tensor_sha256=tensor_hash,
        inference_context_sha256=tensor_hash,
        future_minute_decisions_ns=decisions,
        hour_availability_delay_ns=0, day_availability_delay_ns=0,
        minute_scaler=scalers.minute, hour_scaler=scalers.hour,
        day_scaler=scalers.day, observed_prefix=replay.candles,
        paths=2, forecast_seed=811, advance_slow_priors=True)
    if (e3_feedback_replay["minute_mean"].shape != (4, 5, 5)
            or e4_feedback_replay["minute_mean"].shape != (4, 5, 5)
            or recursive_e4_paths.ensemble.minute.shape != (2, 5, 5)
            or recursive_e4_paths.ensemble.minute_features.shape != (2, 5, 5)
            or recursive_e4_paths.ensemble.hour_close_ns.tolist() != [60 * 96 * MINUTE_NS]
            or recursive_e4_paths.ensemble.day_close_ns.tolist() != [60 * 96 * MINUTE_NS]
            or not np.any(recursive_e4_paths.ensemble.hour_feedback != 0)
            or not np.any(recursive_e4_paths.ensemble.day_feedback != 0)
            or not np.array_equal(e3_feedback_replay["minute_mean"][:, 0],
                                  e4_feedback_replay["minute_mean"][:, 0])
            or np.array_equal(e3_feedback_replay["minute_mean"][:, 3],
                              e4_feedback_replay["minute_mean"][:, 3])
            or not np.any(e4_feedback_replay["hour_latent_diagnostic"][:, 2] != 0)
            or not np.any(e4_feedback_replay["day_latent_diagnostic"][:, 2] != 0)):
        raise AssertionError("path feedback replay failed causal E3/E4 slow-update checks")
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
        "f2_hour_latent_diagnostic_shape": list(f2["hour_latent_diagnostic"].shape),
        "f2_day_latent_diagnostic_shape": list(f2["day_latent_diagnostic"].shape),
        "f2_slow_outputs_are_market_ohlcv": False,
        "f2_event_masks_respected": True,
        "f2_first_step_matches_trained_posterior_head": True,
        "f2_first_prior_state_distinguished": True,
        "f2_training_and_inference_context_identities_separated": True,
        "f2_independent_context_loaded": True,
        "f2_future_event_perturbation_causal": True,
        "f2_delayed_event_barrier_alignment": True,
        "e3_e4_minute_ohlcv_path_shapes": {
            "minute": list(e4_paths.minute.shape),
            "hour": list(e4_paths.hour.shape),
            "day": list(e4_paths.day.shape),
        },
        "e3_e4_path_rejection_counts": {
            "E3": e3_paths.rejected_paths, "E4": e4_paths.rejected_paths,
        },
        "e3_e4_paths_conserve_completed_slow_ohlcv": True,
        "e3_e4_path_divergence_after_slow_event": True,
        "feedback_replay_minute_shape": list(e4_feedback_replay["minute_mean"].shape),
        "feedback_replay_conditions_on_sampled_minute_features": True,
        "feedback_replay_slow_features_aligned_to_availability": True,
        "feedback_replay_e3_e4_diverge_after_slow_feedback": True,
        "stateful_initial_head_matches_feedback_replay": True,
        "stateful_e3_e4_diverge_after_slow_event": True,
        "recursive_e4_ohlcv_paths": list(recursive_e4_paths.ensemble.minute.shape),
        "recursive_e4_hour_day_feedback_reconciled": True,
        "recursive_e4_bar_attempts": recursive_e4_paths.attempted_bar_samples,
        "recursive_e4_rejected_bar_samples": recursive_e4_paths.rejected_bar_samples,
        "f2_event_schedule_from_multiscale_clock": True,
        "future_decision_ns": decisions.tolist(),
        "future_hour_available_at_ns": list(expected_hour_events),
        "future_day_available_at_ns": list(expected_day_events),
        "forecast_delay_ns": {"hour": 0, "day": 0},
        "delayed_hour_available_at_ns": [int(decisions[3])],
        "delayed_day_available_at_ns": [int(decisions[3])],
        "delayed_forecast_delay_ns": {"hour": MINUTE_NS, "day": MINUTE_NS},
        "real_market_validation": False, "economic_claim": False,
        "snapshot_sha": SNAPSHOT_SHA, "upstream_sha": PINNED_UPSTREAM}
    (root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
