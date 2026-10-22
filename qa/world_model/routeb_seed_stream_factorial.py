"""Synthetic-only factorial probe of initialization/sampling/model RNG streams.

The report describes optimizer behavior on a deterministic synthetic process.
It is not a market-data result, Validation score, or explanation of any sealed
or consumed Fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rles_world_model.cli import _read_json
from rles_world_model.routeb.codec import OHLCVCodec, TrainOnlyStandardizer
from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchyTrainingConfig,
    forecast_hierarchical_static_context_standardized,
    train_hierarchical,
)
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import (
    SNAPSHOT_SHA,
    TrainingConfig,
    forecast_standardized,
    train,
)
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS
from rles_world_model.routeb.randomness import SeedStreams
from rles_world_model.routeb.sequences import export_train_sequences, make_windows


def _synthetic_bars(count: int = 6_000) -> tuple[Candle, ...]:
    close = 100.0
    bars: list[Candle] = []
    for minute in range(1, count + 1):
        regime = 1.0 + 0.65 * math.sin(minute / 700.0)
        opening = close * math.exp(0.00005 * math.sin(minute / 17.0))
        move = regime * (0.00055 * math.sin(minute / 41.0)
                         + 0.00022 * math.sin(minute / 9.0))
        close = opening * math.exp(move)
        spread = 0.00025 + 0.00012 * (1.0 + math.sin(minute / 113.0))
        high = max(opening, close) * (1.0 + spread)
        low = min(opening, close) * (1.0 - spread)
        timestamp = minute * MINUTE_NS
        bars.append(Candle(timestamp, timestamp, "SYNTHETIC-PERP", opening,
                           high, low, close, 20.0 + minute % 19))
    return tuple(bars)


def _array_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as arrays:
        for name in sorted(arrays.files):
            value = np.asarray(arrays[name])
            digest.update(name.encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _same_arrays(left: Path, right: Path) -> bool:
    with np.load(left, allow_pickle=False) as lhs, np.load(right, allow_pickle=False) as rhs:
        return (set(lhs.files) == set(rhs.files)
                and all(np.array_equal(lhs[name], rhs[name]) for name in lhs.files))


def _finite_metrics(report: dict[str, Any]) -> dict[str, float]:
    result = {key: float(value) for key, value in report.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isfinite(value)}
    if "loss" not in result or "grad_norm" not in result or result["grad_norm"] <= 0:
        raise RuntimeError("synthetic training did not report finite loss and gradient")
    return result


def _factorial_contrasts(results: dict[str, dict[str, Any]],
                        metrics: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Return raw first-, second-, and third-order contrasts on the full 2^3 design."""
    roots = {
        "tied_31": (0, 0, 0),
        "initialization_only_102": (1, 0, 0),
        "sampling_only_102": (0, 1, 0),
        "model_noise_only_102": (0, 0, 1),
        "initialization_sampling_102": (1, 1, 0),
        "initialization_model_noise_102": (1, 0, 1),
        "sampling_model_noise_102": (0, 1, 1),
        "tied_102": (1, 1, 1),
    }
    if not set(roots).issubset(results):
        raise RuntimeError("factorial contrast requires all eight seed-stream combinations")
    cells = {roots[label]: results[label]["metrics"] for label in roots}
    if len(cells) != 8:
        raise RuntimeError("factorial contrast requires all eight seed-stream combinations")
    output: dict[str, dict[str, Any]] = {}
    factor_names = ("initialization", "sampling", "model_noise")
    for metric in metrics:
        values = {case: float(values[metric]) for case, values in cells.items()}
        main = {}
        for index, factor in enumerate(factor_names):
            changed = [value for case, value in values.items() if case[index] == 1]
            baseline = [value for case, value in values.items() if case[index] == 0]
            main[factor] = sum(changed) / 4.0 - sum(baseline) / 4.0
        pairs: dict[str, float] = {}
        for left, right in ((0, 1), (0, 2), (1, 2)):
            remaining = next(index for index in range(3) if index not in (left, right))
            contrasts = []
            for fixed in (0, 1):
                case = [0, 0, 0]
                case[remaining] = fixed
                c11 = case.copy(); c11[left] = c11[right] = 1
                c10 = case.copy(); c10[left] = 1
                c01 = case.copy(); c01[right] = 1
                key_11 = (c11[0], c11[1], c11[2])
                key_10 = (c10[0], c10[1], c10[2])
                key_01 = (c01[0], c01[1], c01[2])
                key_00 = (case[0], case[1], case[2])
                contrasts.append(values[key_11] - values[key_10]
                                 - values[key_01] + values[key_00])
            pairs[f"{factor_names[left]}:{factor_names[right]}"] = sum(contrasts) / 2.0
        triple = (values[(1, 1, 1)] - values[(1, 1, 0)] - values[(1, 0, 1)]
                  - values[(0, 1, 1)] + values[(1, 0, 0)] + values[(0, 1, 0)]
                  + values[(0, 0, 1)] - values[(0, 0, 0)])
        output[metric] = {"main_effects_mean_difference": main,
                          "pairwise_difference_in_differences": pairs,
                          "three_way_difference": triple}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if type(args.steps) is not int or not 2 <= args.steps <= 100:
        parser.error("--steps must be an integer in [2,100]")

    import jax

    if not any(device.platform == "cpu" for device in jax.devices()):
        raise RuntimeError(f"synthetic factor probe requires a CPU JAX device: {jax.devices()}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    dataset_id = "routeb-synthetic-rng-factorial-v1"
    replay = MarketSequence(_synthetic_bars(), dataset_id, "train")
    coordinates = OHLCVCodec.encode(
        replay.candles[1:], previous_close=replay.candles[0].close)
    scaler = TrainOnlyStandardizer.fit(
        coordinates, dataset_id=dataset_id, scope="train")
    batch = make_windows(replay, scaler=scaler, length=256, burn_in=32, stride=128)
    train_npz = output / "e2_train.npz"
    export_train_sequences(
        train_npz, batch, scaler, snapshot_sha=SNAPSHOT_SHA,
        upstream_sha=PINNED_UPSTREAM,
        train_scope_id="synthetic-rng-factorial-only", fold_sha256="d" * 64)
    hierarchy, _ = build_train_hierarchy(replay, burn_in=32)

    base_seed, changed_seed = 31, 102
    plans: dict[str, SeedStreams | None] = {
        "tied_31": None,
        "explicit_tied_31": SeedStreams(base_seed, base_seed, base_seed + 1),
        "tied_102": SeedStreams(changed_seed, changed_seed, changed_seed + 1),
        "initialization_only_102": SeedStreams(changed_seed, base_seed, base_seed + 1),
        "sampling_only_102": SeedStreams(base_seed, changed_seed, base_seed + 1),
        "model_noise_only_102": SeedStreams(base_seed, base_seed, changed_seed + 1),
        "initialization_sampling_102": SeedStreams(
            changed_seed, changed_seed, base_seed + 1),
        "initialization_model_noise_102": SeedStreams(
            changed_seed, base_seed, changed_seed + 1),
        "sampling_model_noise_102": SeedStreams(
            base_seed, changed_seed, changed_seed + 1),
    }
    model_config = RSSMConfig(
        deter=16, hidden=16, stoch=2, classes=4, blocks=2)
    e2_results: dict[str, dict[str, Any]] = {}
    e3_results: dict[str, dict[str, Any]] = {}
    e2_checkpoints: dict[str, Path] = {}
    e3_checkpoints: dict[str, Path] = {}

    for label, plan in plans.items():
        seed = changed_seed if label == "tied_102" else base_seed
        e2_checkpoint = output / f"e2-{label}.npz"
        e2_report = train(
            dataset_npz=train_npz,
            train_scope_id="synthetic-rng-factorial-only",
            checkout=args.checkout,
            output_checkpoint=e2_checkpoint,
            model_config=model_config,
            training_config=TrainingConfig(
                seed=seed, steps=args.steps, batch_size=2, lr=3e-4,
                report_interval=args.steps),
            seed_streams=plan)
        e2_metrics = _finite_metrics(e2_report[-1])
        e2_results[label] = {
            "metrics": e2_metrics,
            "parameter_optimizer_sha256": _array_digest(e2_checkpoint),
            "resolved_seed_streams": {
                "initialization_seed": e2_report[-1]["initialization_seed"],
                "sampler_seed_base": e2_report[-1]["sampler_seed_base"],
                "model_seed_base": e2_report[-1]["model_seed_base"],
            },
        }
        e2_checkpoints[label] = e2_checkpoint

        hierarchy_config = HierarchyTrainingConfig(
            seed=seed, steps=args.steps, hierarchy_variant="e3_features",
            window_length=256, burn_in=32, hour_context_steps=16,
            day_context_steps=8, learning_rate=3e-4,
            report_interval=args.steps, risk_auxiliary_scale=1.0)
        e3_checkpoint = output / f"e3-{label}.npz"
        e3_report = train_hierarchical(
            tensors=hierarchy, train_scope_id="synthetic-rng-factorial-only",
            dataset_id=dataset_id, checkout=args.checkout,
            output_checkpoint=e3_checkpoint, minute_config=model_config,
            hour_config=model_config, day_config=model_config,
            training_config=hierarchy_config, seed_streams=plan)
        e3_metrics = _finite_metrics(e3_report[-1])
        e3_results[label] = {
            "metrics": e3_metrics,
            "parameter_optimizer_sha256": _array_digest(e3_checkpoint),
            "resolved_seed_streams": {
                "initialization_seed": e3_report[-1]["initialization_seed"],
                "sampler_seed_base": e3_report[-1]["sampler_seed_base"],
                "model_seed_base": e3_report[-1]["model_seed_base"],
            },
        }
        e3_checkpoints[label] = e3_checkpoint

    # A fixed one-factor E2/E3 plan must be bitwise reproducible when repeated.
    split_plan = plans["initialization_only_102"]
    assert split_plan is not None
    e2_resume = output / "e2-explicit-resume.npz"
    e2_resume_cfg = TrainingConfig(seed=base_seed, steps=args.steps // 2,
                                   batch_size=2, report_interval=args.steps // 2)
    train(dataset_npz=train_npz, train_scope_id="synthetic-rng-factorial-only",
          checkout=args.checkout, output_checkpoint=e2_resume,
          model_config=model_config, training_config=e2_resume_cfg,
          seed_streams=split_plan)
    e2_resume_cfg = TrainingConfig(seed=base_seed, steps=args.steps, batch_size=2,
                                   report_interval=args.steps)
    train(dataset_npz=train_npz, train_scope_id="synthetic-rng-factorial-only",
          checkout=args.checkout, output_checkpoint=e2_resume,
          model_config=model_config, training_config=e2_resume_cfg,
          seed_streams=split_plan, resume=True)
    e3_resume = output / "e3-explicit-resume.npz"
    e3_resume_cfg = HierarchyTrainingConfig(
        seed=base_seed, steps=args.steps // 2, hierarchy_variant="e3_features",
        window_length=256, burn_in=32, hour_context_steps=16,
        day_context_steps=8, report_interval=args.steps // 2,
        risk_auxiliary_scale=1.0)
    train_hierarchical(
        tensors=hierarchy, train_scope_id="synthetic-rng-factorial-only",
        dataset_id=dataset_id, checkout=args.checkout,
        output_checkpoint=e3_resume, minute_config=model_config,
        hour_config=model_config, day_config=model_config,
        training_config=e3_resume_cfg, seed_streams=split_plan)
    e3_resume_cfg = HierarchyTrainingConfig(
        seed=base_seed, steps=args.steps, hierarchy_variant="e3_features",
        window_length=256, burn_in=32, hour_context_steps=16,
        day_context_steps=8, report_interval=args.steps,
        risk_auxiliary_scale=1.0)
    train_hierarchical(
        tensors=hierarchy, train_scope_id="synthetic-rng-factorial-only",
        dataset_id=dataset_id, checkout=args.checkout,
        output_checkpoint=e3_resume, minute_config=model_config,
        hour_config=model_config, day_config=model_config,
        training_config=e3_resume_cfg, seed_streams=split_plan, resume=True)
    resume_exact = (
        _same_arrays(e2_resume, e2_checkpoints["initialization_only_102"])
        and _same_arrays(e3_resume, e3_checkpoints["initialization_only_102"]))
    if not resume_exact:
        raise RuntimeError("explicit seed-stream resume differs from uninterrupted training")
    legacy_defaults_exact = (
        _same_arrays(e2_checkpoints["tied_31"], e2_checkpoints["explicit_tied_31"])
        and _same_arrays(e3_checkpoints["tied_31"], e3_checkpoints["explicit_tied_31"]))
    if not legacy_defaults_exact:
        raise RuntimeError("explicit legacy seed streams changed default training arrays")

    initialization_sweep: dict[str, dict[str, Any]] = {}
    initialization_seeds = (7, 19, 31, 64, 102, 20261018)
    for initialization_seed in initialization_seeds:
        checkpoint_path = output / f"e3-init-sweep-{initialization_seed}.npz"
        sweep_report = train_hierarchical(
            tensors=hierarchy, train_scope_id="synthetic-rng-factorial-only",
            dataset_id=dataset_id, checkout=args.checkout,
            output_checkpoint=checkpoint_path, minute_config=model_config,
            hour_config=model_config, day_config=model_config,
            training_config=HierarchyTrainingConfig(
                seed=base_seed, steps=args.steps, hierarchy_variant="e3_features",
                window_length=256, burn_in=32, hour_context_steps=16,
                day_context_steps=8, learning_rate=3e-4,
                report_interval=args.steps, risk_auxiliary_scale=1.0),
            seed_streams=SeedStreams(initialization_seed, base_seed, base_seed + 1))
        initialization_sweep[str(initialization_seed)] = {
            "metrics": _finite_metrics(sweep_report[-1]),
            "parameter_optimizer_sha256": _array_digest(checkpoint_path),
        }
    sweep_31_exact = _same_arrays(
        output / "e3-init-sweep-31.npz", e3_checkpoints["explicit_tied_31"])
    sweep_102_exact = _same_arrays(
        output / "e3-init-sweep-102.npz", e3_checkpoints["initialization_only_102"])
    if not sweep_31_exact or not sweep_102_exact:
        raise RuntimeError("initialization sweep does not reproduce its matched factorial arms")
    sweep_losses = np.asarray(
        [item["metrics"]["loss"] for item in initialization_sweep.values()],
        dtype=np.float64)
    sweep_grad_norms = np.asarray(
        [item["metrics"]["grad_norm"] for item in initialization_sweep.values()],
        dtype=np.float64)

    train_metadata = _read_json(str(train_npz.with_suffix(".json")))
    with np.load(train_npz, allow_pickle=False) as train_arrays:
        history = np.asarray(train_arrays["features"][:1], dtype=np.float32)
    e2_forecast = forecast_standardized(
        checkout=args.checkout, checkpoint=e2_checkpoints["initialization_only_102"],
        model_config=model_config,
        training_config=TrainingConfig(
            seed=base_seed, steps=args.steps, batch_size=2, lr=3e-4,
            report_interval=args.steps),
        train_scope_id="synthetic-rng-factorial-only", dataset_id=dataset_id,
        train_npz_sha256=train_metadata["artifact_sha256"],
        scaler_sha256=train_metadata["scaler_sha256"], history=history,
        steps=3, forecast_seed=707)
    e3_forecast = forecast_hierarchical_static_context_standardized(
        tensors=hierarchy, start=0, steps=3,
        train_scope_id="synthetic-rng-factorial-only", dataset_id=dataset_id,
        checkout=args.checkout, checkpoint=e3_checkpoints["initialization_only_102"],
        minute_config=model_config, hour_config=model_config, day_config=model_config,
        training_config=HierarchyTrainingConfig(
            seed=base_seed, steps=args.steps, hierarchy_variant="e3_features",
            window_length=256, burn_in=32, hour_context_steps=16,
            day_context_steps=8, learning_rate=3e-4,
            report_interval=args.steps, risk_auxiliary_scale=1.0),
        forecast_seed=707)
    explicit_stream_forecasts_valid = bool(
        e2_forecast.shape == (1, 3, 5)
        and np.isfinite(e2_forecast).all()
        and set(e3_forecast) == {"samples", "means", "scales"}
        and all(value.shape == (1, 3, 5) and np.isfinite(value).all()
                for value in e3_forecast.values())
        and np.all(e3_forecast["scales"] > 0))
    if not explicit_stream_forecasts_valid:
        raise RuntimeError("explicit-stream E2/E3 checkpoint forecasts are invalid")

    baseline_e2 = e2_results["tied_31"]["metrics"]
    baseline_e3 = e3_results["tied_31"]["metrics"]
    for result in (e2_results, e3_results):
        for item in result.values():
            item["loss_delta_vs_tied_31"] = (
                item["metrics"]["loss"]
                - (baseline_e2["loss"] if result is e2_results else baseline_e3["loss"]))

    report = {
        "schema": "routeb.synthetic-seed-stream-factorial.v1",
        "status": "PASS_SYNTHETIC_ENGINEERING_ONLY",
        "device": str(jax.devices()[0]),
        "dataset_id": dataset_id,
        "train_bars": len(replay.candles),
        "steps_per_arm": args.steps,
        "case_definitions": {
            "tied_31": "legacy roots init=31, sampler=31, model=32",
            "explicit_tied_31": "explicit roots init=31, sampler=31, model=32",
            "tied_102": "legacy roots init=102, sampler=102, model=103",
            "initialization_only_102": "init=102, sampler=31, model=32",
            "sampling_only_102": "init=31, sampler=102, model=32",
            "model_noise_only_102": "init=31, sampler=31, model=103",
            "initialization_sampling_102": "init=102, sampler=102, model=32",
            "initialization_model_noise_102": "init=102, sampler=31, model=103",
            "sampling_model_noise_102": "init=31, sampler=102, model=103",
        },
        "e2": e2_results,
        "e2_factorial_contrasts": _factorial_contrasts(
            e2_results, ("loss", "grad_norm", "pred_nll", "prior_risk_nll",
                         "rec_nll", "posterior_aux_nll")),
        "e3_features": e3_results,
        "e3_factorial_contrasts": _factorial_contrasts(
            e3_results, ("loss", "grad_norm", "minute_nll", "risk_aux_nll",
                         "posterior_aux_nll")),
        "e3_initialization_sweep_fixed_sampler31_model32": {
            "cases": initialization_sweep,
            "loss_mean": float(sweep_losses.mean()),
            "loss_std": float(sweep_losses.std()),
            "loss_min": float(sweep_losses.min()),
            "loss_max": float(sweep_losses.max()),
            "grad_norm_mean": float(sweep_grad_norms.mean()),
            "grad_norm_std": float(sweep_grad_norms.std()),
            "seed31_matches_factorial_arm_array_exact": sweep_31_exact,
            "seed102_matches_factorial_arm_array_exact": sweep_102_exact,
        },
        "explicit_legacy_seed_streams_array_exact": legacy_defaults_exact,
        "explicit_stream_checkpoint_resume_array_exact": resume_exact,
        "explicit_stream_e2_e3_forecast_smoke_valid": explicit_stream_forecasts_valid,
        "market_data_used": False,
        "research_ledger_read": False,
        "validation_test_holdout_used": False,
        "predictive_skill_claim": False,
        "economic_claim": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
