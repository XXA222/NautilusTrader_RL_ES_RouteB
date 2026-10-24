"""Fold-bound E0/E1/E2 and E0/E3/E4 validation scoring helpers.

This module consumes only the already-authorized Train/Validation replay. It
does not open market files or grant research access. Call it after candidate
checkpoints and forecast seeds are frozen.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from rles_data.core import FoldManifest, MINUTE_NS

from ..contracts import MarketReplay
from .codec import OHLCVCodec, TrainOnlyStandardizer
from .evaluation import (build_predictive_comparison_report,
                         diagonal_gaussian_return_nll, return_paths)
from .evaluation import _numeric_sha256
from .fold_protocol import validation_anchor_timestamps, validate_fold_experiment_protocol
from .jax_model import PINNED_UPSTREAM
from .market import Candle


_E1_E2 = ("E1", "E2")
_E3_E4 = ("E3", "E4")


def _close_returns(closes: np.ndarray) -> np.ndarray:
    values = np.asarray(closes, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("at least two finite positive closes are required")
    return np.log(values[1:] / values[:-1])


def _rolling_rv(returns: np.ndarray, window: int) -> np.ndarray:
    return np.asarray([
        np.sqrt(np.mean(np.square(returns[max(0, end - window + 1):end + 1])))
        for end in range(len(returns))
    ], dtype=np.float64)


def build_fold_predictive_report(
    replay: MarketReplay,
    fold: FoldManifest,
    candidate_ohlcv_paths: Mapping[str, np.ndarray],
    candidate_training_runs: Mapping[str, Mapping[str, object]],
    candidate_anchor_timestamps_ns: Mapping[str, np.ndarray],
    *,
    train_scope_id: str,
    snapshot_sha256: str,
    forecast_seed: int,
    horizons: tuple[int, ...] = (1, 5, 15),
    anchor_interval_minutes: int = 15,
    context_lookback: int = 128,
    volatility_window: int = 15,
    block_length: int = 8,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
    candidate_next_distributions: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    frozen_scaler: TrainOnlyStandardizer | None = None,
    candidate_pair: tuple[str, str] = _E1_E2,
) -> dict[str, object]:
    """Build a common-anchor, Train-fitted baseline/candidate report.

    ``candidate_ohlcv_paths`` must contain the selected matched pair with shape
    ``[anchors, paths, max_horizon, 5]``. Their CLI training run reports must
    bind both candidates to the same Train NPZ, Fold, budget and device, and
    their ordered anchor timestamps must equal the generated Validation
    anchors. E0's S0/S1/S2 paths are generated from Train score returns and
    only the visible context at each anchor. Labels are sliced strictly inside
    Validation score; Test is outside the accepted replay boundary.
    """
    if not isinstance(replay, MarketReplay) or not isinstance(fold, FoldManifest):
        raise ValueError("an authorized Fold replay and frozen FoldManifest are required")
    if (replay.dataset_hash != fold.dataset_hash or not replay.is_contiguous
            or not train_scope_id or len(snapshot_sha256) != 40
            or any(char not in "0123456789abcdef" for char in snapshot_sha256)
            or type(forecast_seed) is not int or forecast_seed < 0):
        raise ValueError("Fold identity, continuity, scope or forecast seed is invalid")
    if (replay.frames[0].ts_close_ns != fold.train.warmup_start_ns
            or replay.frames[-1].ts_close_ns != fold.validation.score_end_ns - MINUTE_NS
            or any(frame.ts_close_ns >= fold.validation.score_end_ns
                   or frame.available_at_ns > fold.validation.score_end_ns
                   for frame in replay.frames)):
        raise ValueError("evaluation replay must end at Validation score end and exclude Test")
    if candidate_pair not in (_E1_E2, _E3_E4):
        raise ValueError("registered comparison supports only the matched E1/E2 or E3/E4 pair")
    candidate_names = set(candidate_pair)
    if (set(candidate_ohlcv_paths) != candidate_names
            or set(candidate_training_runs) != candidate_names
            or set(candidate_anchor_timestamps_ns) != candidate_names):
        raise ValueError(f"the registered comparison requires exactly frozen {candidate_pair} candidates")
    validate_fold_experiment_protocol(
        horizons=horizons, anchor_interval_minutes=anchor_interval_minutes,
        context_lookback=context_lookback, volatility_window=volatility_window,
        block_length=block_length, bootstrap_replicates=bootstrap_replicates,
        seed=seed)

    frames = replay.frames
    by_time = {frame.ts_close_ns: frame for frame in frames}
    all_times = np.asarray([frame.ts_close_ns for frame in frames], dtype=np.int64)
    all_closes = np.asarray([frame.close for frame in frames], dtype=np.float64)
    all_returns = np.log(all_closes[1:] / all_closes[:-1])
    return_times = all_times[1:]
    train_mask = ((return_times >= fold.train.score_start_ns)
                  & (return_times < fold.train.score_end_ns))
    train_returns = all_returns[train_mask]
    if train_returns.size < max(2, volatility_window + 1):
        raise ValueError("Fold Train score has too few returns for fitting and volatility strata")
    train_volatility = _rolling_rv(train_returns, volatility_window)

    max_horizon = horizons[-1]
    anchors = validation_anchor_timestamps(
        fold=fold, max_horizon=max_horizon,
        anchor_interval_minutes=anchor_interval_minutes)
    anchor_stride = anchor_interval_minutes * MINUTE_NS
    if anchors.size < 4:
        raise ValueError("Validation score is too short for common-anchor block comparison")
    for name, values in candidate_anchor_timestamps_ns.items():
        candidate_anchors = np.asarray(values)
        if (candidate_anchors.shape != anchors.shape
                or not np.issubdtype(candidate_anchors.dtype, np.integer)
                or not np.array_equal(candidate_anchors, anchors)):
            raise ValueError(f"{name} forecasts are not bound to the common ordered Validation anchors")

    required_run_fields = {
        "schema", "status", "candidate", "dataset_hash", "fold_hash",
        "train_scope_id", "train_npz_sha256", "training_config",
        "optimizer_updates", "train_examples", "parameter_count",
        "training_seconds", "device", "configuration_sha256",
        "checkpoint_sha256", "snapshot_sha256", "upstream_sha256", "resume",
    }
    normalized_runs = {name: dict(candidate_training_runs[name]) for name in candidate_pair}
    if any(not required_run_fields.issubset(run) for run in normalized_runs.values()):
        raise ValueError(f"{candidate_pair} run reports are missing frozen training provenance")
    expected_run_schema = ("routeb.train-run.v1" if candidate_pair == _E1_E2
                           else "routeb.b3-rollout-run.v1")
    for name, run in normalized_runs.items():
        for key, expected in (
            ("schema", expected_run_schema), ("status", "COMPLETED"),
            ("dataset_hash", fold.dataset_hash),
            ("fold_hash", fold.hash), ("train_scope_id", train_scope_id),
            ("snapshot_sha256", snapshot_sha256), ("upstream_sha256", PINNED_UPSTREAM),
            ("resume", False),
        ):
            if run.get(key) != expected:
                raise ValueError(f"{name} training report has mismatched {key}")
        if str(run.get("candidate", "")).upper() != name:
            raise ValueError(f"{name} training report has mismatched candidate")
        for key in ("train_npz_sha256", "configuration_sha256", "checkpoint_sha256"):
            value = run.get(key)
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError(f"{name} training report has invalid {key}")
        if (not isinstance(run["training_config"], dict)
                or type(run["optimizer_updates"]) is not int
                or run["optimizer_updates"] != run["training_config"].get("steps")
                or type(run["train_examples"]) is not int or run["train_examples"] < 1
                or type(run["parameter_count"]) is not int or run["parameter_count"] < 1
                or not isinstance(run["device"], str) or not run["device"].strip()
                or not isinstance(run["training_seconds"], (int, float))
                or isinstance(run["training_seconds"], bool)
                or not np.isfinite(run["training_seconds"])
                or run["training_seconds"] < 0):
            raise ValueError(f"{name} training report has invalid run measurements")
    if (normalized_runs[candidate_pair[0]]["train_npz_sha256"]
            != normalized_runs[candidate_pair[1]]["train_npz_sha256"]
            or normalized_runs[candidate_pair[0]]["training_config"] != normalized_runs[candidate_pair[1]]["training_config"]
            or normalized_runs[candidate_pair[0]]["optimizer_updates"] != normalized_runs[candidate_pair[1]]["optimizer_updates"]
            or normalized_runs[candidate_pair[0]]["train_examples"] != normalized_runs[candidate_pair[1]]["train_examples"]
            or normalized_runs[candidate_pair[0]]["device"] != normalized_runs[candidate_pair[1]]["device"]):
        raise ValueError(f"{candidate_pair} must share one Train artifact, training budget, examples and device")
    expected_anchor_shape = (len(anchors), None, max_horizon, 5)
    n_paths: int | None = None
    for name, values in candidate_ohlcv_paths.items():
        paths = np.asarray(values, dtype=np.float64)
        if (paths.ndim != 4 or paths.shape[0] != expected_anchor_shape[0]
                or paths.shape[2:] != expected_anchor_shape[2:]
                or paths.shape[1] < 2 or not np.isfinite(paths).all()
                or (paths[..., 0:4] <= 0).any()
                or (paths[..., 4] < 0).any()
                or np.any(paths[..., 1] < np.maximum(paths[..., 0], paths[..., 3]))
                or np.any(paths[..., 2] > np.minimum(paths[..., 0], paths[..., 3]))):
            raise ValueError(f"{name} must contain finite valid OHLCV paths on every common anchor")
        if n_paths is None:
            n_paths = paths.shape[1]
        elif paths.shape[1] != n_paths:
            raise ValueError(f"{candidate_pair} must use the same path ensemble size")
    if n_paths is None:
        raise ValueError("candidate forecast ensembles are empty")

    anchor_closes = np.empty(len(anchors), dtype=np.float64)
    realized_ohlcv = np.empty((len(anchors), max_horizon, 5), dtype=np.float64)
    outcomes = np.empty((len(anchors), max_horizon), dtype=np.int64)
    visible_contexts = np.empty((len(anchors), context_lookback), dtype=np.float64)
    anchor_volatility = np.empty(len(anchors), dtype=np.float64)
    for row, anchor in enumerate(anchors):
        anchor_frame = by_time.get(int(anchor))
        if anchor_frame is None or anchor_frame.available_at_ns > anchor:
            raise ValueError("Validation anchor is missing or unavailable at decision time")
        anchor_closes[row] = anchor_frame.close
        context_returns = all_returns[return_times <= anchor]
        if context_returns.size < context_lookback:
            raise ValueError("replay lacks the requested causal context before a Validation anchor")
        visible_contexts[row] = context_returns[-context_lookback:]
        anchor_volatility[row] = float(np.sqrt(np.mean(
            np.square(visible_contexts[row, -volatility_window:]))))
        for step in range(1, max_horizon + 1):
            timestamp = int(anchor) + step * MINUTE_NS
            frame = by_time.get(timestamp)
            if (frame is None or not fold.validation.score_start_ns <= timestamp
                    < fold.validation.score_end_ns):
                raise ValueError("forecast target crosses the Validation score boundary")
            realized_ohlcv[row, step - 1] = (
                frame.open, frame.high, frame.low, frame.close, frame.volume,
            )
            outcomes[row, step - 1] = timestamp

    realized_returns = return_paths(
        realized_ohlcv[:, None, :, :], previous_close=anchor_closes,
    )[:, 0, :]
    forecast_returns = {
        name: return_paths(values, previous_close=anchor_closes)
        for name, values in candidate_ohlcv_paths.items()
    }
    nll_by_model: dict[str, dict[int, np.ndarray]] = {}
    if (candidate_next_distributions is None) != (frozen_scaler is None):
        raise ValueError("native next-step distributions require the frozen Train scaler")
    if candidate_next_distributions is not None:
        if set(candidate_next_distributions) != candidate_names:
            raise ValueError(f"native next-step distributions must cover exactly {candidate_pair}")
        assert frozen_scaler is not None
        standardized_target = np.empty((len(anchors), 5), dtype=np.float32)
        for row, anchor in enumerate(anchors):
            frame = by_time[int(anchor) + MINUTE_NS]
            encoded = OHLCVCodec.encode((Candle(
                frame.ts_close_ns, frame.available_at_ns, frame.instrument,
                frame.open, frame.high, frame.low, frame.close, frame.volume,
            ),), previous_close=anchor_closes[row])
            standardized_target[row] = frozen_scaler.transform(encoded)[0]
        for name, (mean, scale) in candidate_next_distributions.items():
            nll_by_model[name] = {1: diagonal_gaussian_return_nll(
                mean, scale, standardized_target,
                scaler_mean=frozen_scaler.mean, scaler_std=frozen_scaler.std,
            )}
    from .evaluation import sample_e0_paths
    e0_samples = sample_e0_paths(
        train_returns, visible_contexts, paths=int(n_paths), horizon=max_horizon,
        seed=forecast_seed,
    )
    e0 = {f"E0_{name}": values for name, values in e0_samples.items()}
    forecasts: dict[str, np.ndarray] = {**e0, **forecast_returns}
    provenance = {
        name: {
            "model_family": name,
            "model_config_sha256": run["configuration_sha256"],
            "artifact_sha256": run["checkpoint_sha256"],
            "train_examples": run["train_examples"],
            "optimizer_updates": run["optimizer_updates"],
            "training_seconds": run["training_seconds"],
            "parameter_count": run["parameter_count"],
            "device": run["device"],
            "forecast_seed": forecast_seed,
        }
        for name, run in normalized_runs.items()
    }
    e0_config = {
        "sampler": "sample_e0_paths", "seed": forecast_seed,
        "paths": int(n_paths), "horizon": max_horizon,
        "ewma_decay": 0.94, "mean_block_length": 16,
    }
    for name in e0:
        provenance[name] = {
            "model_family": name,
            "model_config_sha256": hashlib.sha256(json.dumps(
                {**e0_config, "model": name}, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
            "artifact_sha256": _numeric_sha256(train_returns, dtype="<f8"),
            "train_examples": int(train_returns.size),
            "optimizer_updates": 0,
            "training_seconds": 0.0,
            "parameter_count": 0,
            "device": "numpy-cpu",
            "forecast_seed": forecast_seed,
        }
    report = build_predictive_comparison_report(
        forecasts, realized_returns,
        anchor_timestamps_ns=anchors, outcome_timestamps_ns=outcomes,
        anchor_interval_ns=anchor_stride, block_length=block_length,
        train_volatility=train_volatility, anchor_volatility=anchor_volatility,
        metadata={
            "dataset_identity": fold.dataset_hash,
            "fold_identity": fold.hash,
            "train_scope_id": train_scope_id,
            "snapshot_sha256": snapshot_sha256,
        },
        model_provenance=provenance,
        horizons=horizons,
        nll_by_model=nll_by_model,
        generated_ohlcv_by_model={
            name: np.asarray(candidate_ohlcv_paths[name], dtype=np.float64)
            for name in candidate_pair
        },
        realized_ohlcv=realized_ohlcv,
        previous_close=anchor_closes,
        rolling_window=volatility_window,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    report["authorization_verified"] = False
    report["authorization_note"] = (
        "Fold boundaries are validated here; verify the registration marker and reader receipt separately."
    )
    report["fold_protocol"] = {
        "train_score_start_ns": fold.train.score_start_ns,
        "train_score_end_ns": fold.train.score_end_ns,
        "validation_score_start_ns": fold.validation.score_start_ns,
        "validation_score_end_ns": fold.validation.score_end_ns,
        "test_payload_decoded": False,
        "anchor_interval_minutes": anchor_interval_minutes,
        "context_lookback_minutes": context_lookback,
    }
    report.pop("report_sha256", None)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def build_fold_b3_predictive_report(
    replay: MarketReplay,
    fold: FoldManifest,
    candidate_ohlcv_paths: Mapping[str, np.ndarray],
    b3_training_report: Mapping[str, object],
    candidate_rollout_runs: Mapping[str, Mapping[str, object]],
    candidate_anchor_timestamps_ns: Mapping[str, np.ndarray],
    *,
    train_scope_id: str,
    snapshot_sha256: str,
    forecast_seed: int,
    inference_context_sha256: str,
    horizons: tuple[int, ...] = (1, 5, 15),
    anchor_interval_minutes: int = 15,
    context_lookback: int = 128,
    volatility_window: int = 15,
    block_length: int = 8,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Score matched B3 E3/E4 paths on common registered Validation anchors.

    This is a scorer, not a data reader or worker launcher. ``replay`` must be
    the already-authorized Train + Validation replay; the caller supplies the
    frozen Train-only B3 report and completed E3/E4 path artifacts. Both arms
    must share one trained checkpoint, Train artifact, seed, anchor list,
    inference context, path count, horizon, device and training budget. E3 and
    E4 differ only in the frozen-vs-advanced slow-prior rollout mode.
    """
    if not isinstance(b3_training_report, Mapping):
        raise ValueError("a provenance-bound B3 Train report is required")
    train_report = dict(b3_training_report)
    supplied_digest = train_report.get("report_sha256")
    unsigned_train_report = {
        key: value for key, value in train_report.items() if key != "report_sha256"
    }
    try:
        canonical_train_report = json.dumps(
            unsigned_train_report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("B3 Train report is not canonical finite JSON") from exc
    expected_digest = hashlib.sha256(canonical_train_report.encode("utf-8")).hexdigest()
    if supplied_digest != expected_digest:
        raise ValueError("B3 Train report digest is invalid")

    def valid_sha(value: object, length: int = 64) -> bool:
        return (isinstance(value, str) and len(value) == length
                and all(char in "0123456789abcdef" for char in value))

    required_train_fields = {
        "schema", "status", "candidate", "dataset_hash", "fold_hash",
        "train_scope_id", "reservation_label", "registration_receipt_sha256",
        "reservation_key", "reservation_token_sha256", "repository_commit",
        "source_code_sha256", "train_artifact_sha256", "scaler_sha256",
        "training_config", "optimizer_updates", "parameter_count", "train_windows",
        "training_seconds", "device", "configuration_sha256", "checkpoint_sha256",
        "snapshot_sha256", "upstream_sha256", "validation_scored", "test_scored",
        "economic_certification", "report_sha256",
    }
    if not required_train_fields.issubset(train_report):
        raise ValueError("B3 Train report is missing required provenance fields")
    for key, expected in (
        ("schema", "routeb.b3-train-run.v1"), ("status", "COMPLETED"),
        ("candidate", "B3"), ("dataset_hash", fold.dataset_hash),
        ("fold_hash", fold.hash), ("train_scope_id", train_scope_id),
        ("snapshot_sha256", snapshot_sha256), ("upstream_sha256", PINNED_UPSTREAM),
        ("validation_scored", False), ("test_scored", False),
        ("economic_certification", False),
    ):
        if train_report.get(key) != expected:
            raise ValueError(f"B3 Train report has mismatched {key}")
    if (not isinstance(train_report["reservation_label"], str)
            or not train_report["reservation_label"].startswith("routeb-fold-")
            or not valid_sha(train_report["repository_commit"], 40)
            or not valid_sha(train_report["source_code_sha256"])
            or not isinstance(train_report["training_config"], dict)
            or type(train_report["optimizer_updates"]) is not int
            or train_report["optimizer_updates"] < 1
            or train_report["optimizer_updates"] != train_report["training_config"].get("steps")
            or type(train_report["parameter_count"]) is not int
            or train_report["parameter_count"] < 1
            or type(train_report["train_windows"]) is not int
            or train_report["train_windows"] < 1
            or not isinstance(train_report["training_seconds"], (int, float))
            or isinstance(train_report["training_seconds"], bool)
            or not np.isfinite(train_report["training_seconds"])
            or train_report["training_seconds"] < 0
            or not isinstance(train_report["device"], str)
            or not train_report["device"].strip()):
        raise ValueError("B3 Train report has invalid source, budget or measurements")
    for key in ("train_artifact_sha256", "scaler_sha256", "configuration_sha256",
                "checkpoint_sha256", "registration_receipt_sha256", "reservation_key",
                "reservation_token_sha256", "snapshot_sha256", "upstream_sha256"):
        if not valid_sha(train_report[key],
                         40 if key in ("snapshot_sha256", "upstream_sha256") else 64):
            raise ValueError(f"B3 Train report has invalid {key}")
    if (not isinstance(candidate_rollout_runs, Mapping)
            or set(candidate_rollout_runs) != set(_E3_E4)
            or not isinstance(candidate_ohlcv_paths, Mapping)
            or set(candidate_ohlcv_paths) != set(_E3_E4)
            or not isinstance(candidate_anchor_timestamps_ns, Mapping)
            or set(candidate_anchor_timestamps_ns) != set(_E3_E4)):
        raise ValueError("registered B3 scoring requires exactly the matched E3/E4 pair")
    if (not valid_sha(inference_context_sha256)
            or type(forecast_seed) is not int or forecast_seed < 0
            or not horizons or any(type(horizon) is not int or horizon < 1
                                   for horizon in horizons)):
        raise ValueError("B3 rollout context, seed or horizon identity is invalid")

    anchors = np.asarray(candidate_anchor_timestamps_ns["E3"])
    if (anchors.ndim != 1 or not np.issubdtype(anchors.dtype, np.integer)
            or len(anchors) < 1):
        raise ValueError("E3/E4 rollout anchors must be a nonempty integer vector")
    anchor_sha256 = _numeric_sha256(anchors, dtype="<i8")
    max_horizon = horizons[-1]
    path_count: int | None = None
    normalized_runs: dict[str, dict[str, object]] = {}
    path_sampling: dict[str, dict[str, int | float]] = {}
    for name in _E3_E4:
        run = candidate_rollout_runs[name]
        paths = np.asarray(candidate_ohlcv_paths[name], dtype=np.float64)
        if (not isinstance(run, Mapping)
                or paths.ndim != 4 or paths.shape[0] != len(anchors)
                or paths.shape[2:] != (max_horizon, 5)):
            raise ValueError(f"{name} path artifact does not match registered anchors/horizon")
        if path_count is None:
            path_count = int(paths.shape[1])
        elif paths.shape[1] != path_count:
            raise ValueError("E3 and E4 must use the same path count")
        unsigned_run = {key: value for key, value in run.items()
                        if key != "report_sha256"}
        try:
            canonical_run = json.dumps(
                unsigned_run, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} rollout manifest is not canonical finite JSON") from exc
        if run.get("report_sha256") != hashlib.sha256(
                canonical_run.encode("utf-8")).hexdigest():
            raise ValueError(f"{name} rollout manifest digest is invalid")
        expected_bars = len(anchors) * int(paths.shape[1]) * max_horizon
        attempted = run.get("attempted_bar_samples")
        rejected = run.get("rejected_bar_samples")
        max_attempts = run.get("max_attempts_per_bar")
        rejection_rate = run.get("rejection_rate")
        if (type(attempted) is not int or type(rejected) is not int
                or type(max_attempts) is not int or not 1 <= max_attempts <= 100
                or attempted < expected_bars or rejected < 0
                or attempted - rejected != expected_bars
                or attempted > expected_bars * max_attempts
                or not isinstance(rejection_rate, (int, float))
                or isinstance(rejection_rate, bool)
                or not np.isfinite(rejection_rate)
                or not np.isclose(rejection_rate, rejected / attempted,
                                  rtol=0.0, atol=1e-12)):
            raise ValueError(f"{name} rollout has invalid path-sampling rejection accounting")
        path_sampling[name] = {
            "attempted_bar_samples": attempted,
            "rejected_bar_samples": rejected,
            "accepted_bar_samples": expected_bars,
            "max_attempts_per_bar": max_attempts,
            "rejection_rate": float(rejection_rate),
        }
        expected_mode = "FROZEN_OBSERVED_SLOW_POSTERIOR" if name == "E3" else "ADVANCE_PATH_CONDITIONED_SLOW_POSTERIOR"
        for key, expected in (
            ("schema", "routeb.b3-rollout-run.v1"), ("status", "COMPLETED"),
            ("candidate", name), ("rollout_mode", expected_mode),
            ("b3_train_report_sha256", supplied_digest),
            ("dataset_hash", fold.dataset_hash), ("fold_hash", fold.hash),
            ("train_scope_id", train_scope_id),
            ("registration_receipt_sha256", train_report["registration_receipt_sha256"]),
            ("reservation_key", train_report["reservation_key"]),
            ("reservation_token_sha256", train_report["reservation_token_sha256"]),
            ("train_artifact_sha256", train_report["train_artifact_sha256"]),
            ("checkpoint_sha256", train_report["checkpoint_sha256"]),
            ("configuration_sha256", train_report["configuration_sha256"]),
            ("training_config", train_report["training_config"]),
            ("optimizer_updates", train_report["optimizer_updates"]),
            ("train_windows", train_report["train_windows"]),
            ("parameter_count", train_report["parameter_count"]),
            ("device", train_report["device"]),
            ("snapshot_sha256", train_report["snapshot_sha256"]),
            ("upstream_sha256", train_report["upstream_sha256"]),
            ("forecast_seed", forecast_seed), ("anchors", len(anchors)),
            ("paths", path_count), ("horizon", max_horizon),
            ("inference_context_sha256", inference_context_sha256),
            ("anchor_timestamps_sha256", anchor_sha256),
            ("forecast_ohlcv_sha256", _numeric_sha256(paths, dtype="<f8")),
        ):
            if run.get(key) != expected:
                raise ValueError(f"{name} rollout report has mismatched {key}")
        # Adapt only after the B3-specific shared-checkpoint contract is proven;
        # the generic scorer then checks OHLCV validity and common Fold targets.
        normalized_runs[name] = {
            "schema": "routeb.b3-rollout-run.v1", "status": "COMPLETED",
            "candidate": name, "dataset_hash": fold.dataset_hash,
            "fold_hash": fold.hash, "train_scope_id": train_scope_id,
            "train_npz_sha256": train_report["train_artifact_sha256"],
            "training_config": train_report["training_config"],
            "optimizer_updates": train_report["optimizer_updates"],
            "train_examples": train_report["train_windows"],
            "parameter_count": train_report["parameter_count"],
            "training_seconds": train_report["training_seconds"],
            "device": train_report["device"],
            "configuration_sha256": train_report["configuration_sha256"],
            "checkpoint_sha256": train_report["checkpoint_sha256"],
            "snapshot_sha256": train_report["snapshot_sha256"],
            "upstream_sha256": train_report["upstream_sha256"],
            "resume": False,
        }
    if path_count is None or path_count < 2:
        raise ValueError("E3/E4 forecast ensembles require at least two paths")
    if (path_sampling["E3"]["max_attempts_per_bar"]
            != path_sampling["E4"]["max_attempts_per_bar"]):
        raise ValueError("E3 and E4 must use the same per-bar rejection budget")

    report = build_fold_predictive_report(
        replay, fold, candidate_ohlcv_paths, normalized_runs,
        candidate_anchor_timestamps_ns, train_scope_id=train_scope_id,
        snapshot_sha256=snapshot_sha256, forecast_seed=forecast_seed,
        horizons=horizons, anchor_interval_minutes=anchor_interval_minutes,
        context_lookback=context_lookback, volatility_window=volatility_window,
        block_length=block_length, bootstrap_replicates=bootstrap_replicates,
        seed=seed, candidate_pair=_E3_E4,
    )
    report["hierarchical_rollout"] = {
        "schema": "routeb.b3-e3-e4-validation.v1",
        "b3_train_report_sha256": supplied_digest,
        "registration_receipt_sha256": train_report["registration_receipt_sha256"],
        "reservation_key": train_report["reservation_key"],
        "reservation_token_sha256": train_report["reservation_token_sha256"],
        "train_artifact_sha256": train_report["train_artifact_sha256"],
        "scaler_sha256": train_report["scaler_sha256"],
        "checkpoint_sha256": train_report["checkpoint_sha256"],
        "inference_context_sha256": inference_context_sha256,
        "anchor_timestamps_sha256": anchor_sha256,
        "forecast_seed": forecast_seed,
        "paths_per_anchor": path_count,
        "max_horizon": max_horizon,
        "matched_checkpoint": True,
        "rollout_modes": {
            "E3": "FROZEN_OBSERVED_SLOW_POSTERIOR",
            "E4": "ADVANCE_PATH_CONDITIONED_SLOW_POSTERIOR",
        },
        "path_sampling": path_sampling,
        "validation_only": True,
        "test_payload_decoded": False,
        "economic_certification": False,
    }
    report.pop("report_sha256", None)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report
