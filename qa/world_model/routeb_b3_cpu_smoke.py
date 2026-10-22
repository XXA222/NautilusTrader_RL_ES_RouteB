"""Synthetic end-to-end gradient/resume smoke for the experimental B3 trainer.

Uses only an in-memory three-day Train sequence. It makes no exchange-data,
Validation/Test/Holdout, account, or credential access and is not skill evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

from rles_world_model.routeb.hierarchy_dataset import (
    build_train_hierarchy,
    build_validation_hierarchy,
)
from rles_world_model.routeb.jax_hierarchy_f2 import rollout_hierarchical_f2_checkpoint
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchyTrainingConfig,
    forecast_hierarchical_validation_static_context_batch_standardized,
    forecast_hierarchical_static_context_standardized,
    forecast_hierarchical_validation_static_context_standardized,
    train_hierarchical,
)
from rles_world_model.routeb.jax_model import RSSMConfig
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS


def synthetic_train_bars(count: int = 6_000) -> tuple[Candle, ...]:
    price = 2_000.0
    result = []
    for minute in range(1, count + 1):
        opening = price * math.exp(0.00003 * math.sin(minute / 13.0))
        price = opening * math.exp(0.0005 * math.sin(minute / 37.0))
        high = max(opening, price) * (1.0002 + 0.00005 * math.sin(minute / 17.0) ** 2)
        low = min(opening, price) * (0.9998 - 0.00005 * math.cos(minute / 19.0) ** 2)
        result.append(Candle(minute * MINUTE_NS, minute * MINUTE_NS,
                             "ETHUSDT-PERP.SIM", opening, high, low, price,
                             100.0 + minute % 23))
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-platform", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    import jax
    platforms = {device.platform for device in jax.devices()}
    required = {"cpu"} if args.expected_platform == "cpu" else {"gpu", "cuda"}
    if not platforms.intersection(required):
        raise RuntimeError(f"required {args.expected_platform} device missing: {sorted(platforms)}")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    replay = MarketSequence(synthetic_train_bars(),
                            "synthetic-b3-smoke-not-market-data", "train")
    tensors, scalers = build_train_hierarchy(replay, burn_in=32)
    model = RSSMConfig(deter=8, hidden=8, stoch=1, classes=2, blocks=1)
    cfg_one = HierarchyTrainingConfig(seed=31, steps=1, hierarchy_variant="e4_conditional",
                                      window_length=256,
                                      hour_context_steps=16, day_context_steps=8,
                                      report_interval=1)
    cfg_two = HierarchyTrainingConfig(seed=31, steps=2, hierarchy_variant="e4_conditional",
                                      window_length=256,
                                      hour_context_steps=16, day_context_steps=8,
                                      report_interval=1)
    resumed_path = output / "resumed.npz"
    direct_path = output / "direct.npz"
    initial = train_hierarchical(
        tensors=tensors, train_scope_id="train-synthetic-only",
        dataset_id=replay.dataset_id, checkout=args.checkout,
        output_checkpoint=resumed_path, minute_config=model,
        hour_config=model, day_config=model, training_config=cfg_one)
    continuation = train_hierarchical(
        tensors=tensors, train_scope_id="train-synthetic-only",
        dataset_id=replay.dataset_id, checkout=args.checkout,
        output_checkpoint=resumed_path, minute_config=model,
        hour_config=model, day_config=model, training_config=cfg_two, resume=True)
    train_hierarchical(
        tensors=tensors, train_scope_id="train-synthetic-only",
        dataset_id=replay.dataset_id, checkout=args.checkout,
        output_checkpoint=direct_path, minute_config=model,
        hour_config=model, day_config=model, training_config=cfg_two)
    if initial[-1]["step"] != 1 or continuation[-1]["step"] != 2:
        raise AssertionError("hierarchical checkpoint did not resume at the next update")
    if (not np.isfinite(continuation[-1]["loss"])
            or continuation[-1]["grad_norm"] <= 0
            or continuation[-1]["parameters"] <= 0):
        raise AssertionError("hierarchical smoke did not produce a finite nonzero gradient")
    with np.load(resumed_path, allow_pickle=False) as resumed, \
            np.load(direct_path, allow_pickle=False) as uninterrupted:
        if set(resumed.files) != set(uninterrupted.files) or not all(
                np.array_equal(resumed[name], uninterrupted[name]) for name in resumed.files):
            raise AssertionError("resumed and uninterrupted B3 training states differ")
    # Exercise all three explicitly named candidates with identical data,
    # seed, windows, and update budget before any real E3/E4 registration.
    variant_losses = {}
    variant_parameters = {"e4_conditional": continuation[-1]["parameters"]}
    variant_configs = {}
    for variant in ("e3_features", "e3b_fusion"):
        variant_cfg = HierarchyTrainingConfig(
            seed=31, steps=1, hierarchy_variant=variant, window_length=256,
            hour_context_steps=16, day_context_steps=8, report_interval=1,
            risk_auxiliary_scale=1.0 if variant == "e3_features" else 0.0)
        variant_reports = train_hierarchical(
            tensors=tensors, train_scope_id="train-synthetic-only",
            dataset_id=replay.dataset_id, checkout=args.checkout,
            output_checkpoint=output / f"{variant}.npz", minute_config=model,
            hour_config=model, day_config=model, training_config=variant_cfg)
        if not np.isfinite(variant_reports[-1]["loss"]):
            raise AssertionError(f"{variant} produced a nonfinite loss")
        if variant == "e3_features" and (
                not np.isfinite(variant_reports[-1]["risk_aux_nll"])
                or variant_reports[-1]["risk_aux_nll"] <= 0):
            raise AssertionError("matched multi-horizon risk objective did not activate")
        variant_configs[variant] = variant_cfg
        variant_losses[variant] = variant_reports[-1]["loss"]
        variant_parameters[variant] = variant_reports[-1]["parameters"]
    if not (variant_parameters["e3_features"] < variant_parameters["e3b_fusion"]
            < variant_parameters["e4_conditional"]):
        raise AssertionError("E3/E3b/E4 did not instantiate distinct model graphs")

    # Select a forecast window whose one-step horizon does not cross a UTC
    # hour/day event; the static-context diagnostic refuses any such crossing.
    forecast_cfg = cfg_two
    forecast_start = None
    eligible = continuation[-1]["eligible_window_count"]
    for candidate_start in range(eligible):
        try:
            forecast = forecast_hierarchical_static_context_standardized(
                tensors=tensors, start=candidate_start, steps=1,
                train_scope_id="train-synthetic-only", dataset_id=replay.dataset_id,
                checkout=args.checkout, checkpoint=resumed_path,
                minute_config=model, hour_config=model, day_config=model,
                training_config=forecast_cfg, forecast_seed=901)
        except ValueError as exc:
            if "crosses a future hour/day event" in str(exc):
                continue
            raise
        forecast_start = candidate_start
        repeated = forecast_hierarchical_static_context_standardized(
            tensors=tensors, start=candidate_start, steps=1,
            train_scope_id="train-synthetic-only", dataset_id=replay.dataset_id,
            checkout=args.checkout, checkpoint=resumed_path,
            minute_config=model, hour_config=model, day_config=model,
            training_config=forecast_cfg, forecast_seed=901)
        if any(not np.array_equal(forecast[key], repeated[key]) for key in forecast):
            raise AssertionError("fixed-seed prior forecast is not reproducible")
        break
    if forecast_start is None:
        raise AssertionError("no synthetic forecast anchor remained inside the slow-event boundary")
    if (forecast["samples"].shape != (1, 1, 5)
            or np.any(forecast["scales"] <= 0)):
        raise AssertionError("invalid E4 static-context prior forecast")

    validation_candles = tuple(Candle(
        bar.ts_close_ns + len(replay.candles) * MINUTE_NS,
        bar.available_at_ns + len(replay.candles) * MINUTE_NS,
        bar.instrument, bar.open, bar.high, bar.low, bar.close, bar.volume)
        for bar in replay.candles)
    validation = MarketSequence(validation_candles, replay.dataset_id, "validation")
    score_start = validation_candles[3_000].ts_close_ns
    score_end = validation_candles[-1].ts_close_ns + MINUTE_NS
    validation_tensors = build_validation_hierarchy(
        validation, scalers=scalers, score_start_ns=score_start,
        score_end_ns=score_end)
    validation_anchor_index = 3_255 - 1
    validation_forecast = forecast_hierarchical_validation_static_context_standardized(
        train_tensors=tensors, validation_tensors=validation_tensors,
        start=validation_anchor_index - forecast_cfg.window_length + 1,
        steps=15, train_scope_id="train-synthetic-only",
        dataset_id=replay.dataset_id, checkout=args.checkout,
        checkpoint=resumed_path, minute_config=model, hour_config=model,
        day_config=model, training_config=forecast_cfg,
        forecast_seeds=(1901, 1902))
    if (validation_forecast["samples"].shape != (2, 1, 15, 5)
            or np.any(validation_forecast["scales"] <= 0)):
        raise AssertionError("Train-bound Validation-context forecast failed")
    batch_validation_forecast = forecast_hierarchical_validation_static_context_batch_standardized(
        train_tensors=tensors, validation_tensors=validation_tensors,
        starts=(validation_anchor_index - forecast_cfg.window_length + 1,
                validation_anchor_index + 360 - forecast_cfg.window_length + 1),
        steps=15, forecast_seeds=(1911, 1912),
        train_scope_id="train-synthetic-only", dataset_id=replay.dataset_id,
        checkout=args.checkout, checkpoint=resumed_path, minute_config=model,
        hour_config=model, day_config=model, training_config=forecast_cfg,
        forecast_batch_size=1)
    if (batch_validation_forecast["samples"].shape != (2, 2, 15, 5)
            or np.any(batch_validation_forecast["scales"] <= 0)):
        raise AssertionError("batched Train-bound Validation forecast failed")

    # Load the Train-identity-bound E4 checkpoint, then drive its actual
    # recurrent carries through a coincident generated hour/day boundary.
    # End one minute before a UTC day close, so the next synthetic bar closes
    # both its daily and hourly buckets at exactly the same event time.
    prefix = MarketSequence(
        replay.candles[:5_759], replay.dataset_id, "train")
    prefix_tensors, _ = build_train_hierarchy(
        prefix, burn_in=32, frozen_scalers=scalers)
    f2_results = {}
    F2Mode = Literal["F2_SLOW_ONLY", "F2_FULL_FILTER"]
    f2_cases: list[tuple[str, HierarchyTrainingConfig, Path, tuple[F2Mode, ...]]] = [
        ("e3_features", variant_configs["e3_features"], output / "e3_features.npz",
         ("F2_FULL_FILTER",)),
        ("e3b_fusion", variant_configs["e3b_fusion"], output / "e3b_fusion.npz",
         ("F2_FULL_FILTER",)),
        ("e4_conditional", cfg_two, resumed_path,
         ("F2_SLOW_ONLY", "F2_FULL_FILTER")),
    ]
    for variant, variant_cfg, variant_checkpoint, modes in f2_cases:
        for mode in modes:
            key = f"{variant}:{mode}"
            f2_results[key] = rollout_hierarchical_f2_checkpoint(
                training_tensors=tensors, train_scope_id="train-synthetic-only",
                dataset_id=replay.dataset_id, checkpoint=str(variant_checkpoint),
                prefix=prefix, context_tensors=prefix_tensors, scalers=scalers,
                checkout=args.checkout, steps=2, mode=mode,
                minute_config=model, hour_config=model, day_config=model,
                training_config=variant_cfg, seed=1201, record_minute_states=False)
            result = f2_results[key]
            if (result.hour_updates != 1 or result.day_updates != 1
                    or result.score_capability != "SCENARIO_ONLY"
                    or result.minute_states):
                raise AssertionError(
                    f"integrated {key} failed its generated day/hour boundary")
            if mode == "F2_FULL_FILTER" and (
                    result.minute_posterior_updates != 2
                    or not result.minute_posterior_max_abs_logit_delta
                    or result.minute_posterior_max_abs_logit_delta <= 0):
                raise AssertionError(
                    f"{key} posterior evidence failed: "
                    f"updates={result.minute_posterior_updates}, "
                    f"max_abs_logit_delta={result.minute_posterior_max_abs_logit_delta}")
    full_filter = f2_results["e4_conditional:F2_FULL_FILTER"]
    report = {
        "status": "PASS_SYNTHETIC_ENGINEERING_ONLY",
        "device": str(jax.devices()[0]),
        "minute_steps": int(tensors.minute_x.shape[1]),
        "hour_steps": int(tensors.hour_x.shape[1]),
        "day_steps": int(tensors.day_x.shape[1]),
        "optimizer_updates": 2,
        "hierarchy_variant": cfg_two.hierarchy_variant,
        "variant_losses": variant_losses,
        "variant_parameters": variant_parameters,
        "validation_forecast_shape": list(validation_forecast["samples"].shape),
        "batched_validation_forecast_shape": list(
            batch_validation_forecast["samples"].shape),
        "f2_full_filter_max_abs_logit_delta": full_filter.minute_posterior_max_abs_logit_delta,
        "jax_f2_modes": {
            mode: {"hour_updates": result.hour_updates,
                   "day_updates": result.day_updates,
                   "score_capability": result.score_capability,
                   "minute_posterior_updates": result.minute_posterior_updates,
                   "minute_posterior_max_abs_logit_delta":
                       result.minute_posterior_max_abs_logit_delta,
                   "generated_bars": len(result.bars)}
            for mode, result in f2_results.items()},
        "window_length": cfg_two.window_length,
        "eligible_window_count": continuation[-1]["eligible_window_count"],
        "window_start_sha256": continuation[-1]["window_start_sha256"],
        "static_context_forecast_steps": 1,
        "static_context_forecast_start": forecast_start,
        "loss": continuation[-1]["loss"],
        "grad_norm": continuation[-1]["grad_norm"],
        "parameters": continuation[-1]["parameters"],
        "train_tensor_sha256": continuation[-1]["train_tensor_sha256"],
        "test_or_holdout_accessed": False,
        "predictive_or_economic_claim": False,
    }
    (output / "smoke_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
