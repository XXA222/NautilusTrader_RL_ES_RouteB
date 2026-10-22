"""Fold-bound E0/E1/E2 validation scoring helpers.

This module consumes only the already-authorized Train/Validation replay. It
does not open market files or grant research access. Call it after candidate
checkpoints and forecast seeds are frozen.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, cast

import numpy as np

from rles_data.core import FoldManifest, MINUTE_NS  # type: ignore[import-untyped]

from ..contracts import MarketReplay
from .evaluation import build_predictive_comparison_report, return_paths
from .evaluation import (_crps_per_anchor, _numeric_sha256,
                         paired_block_bootstrap_score_differences)
from .jax_model import PINNED_UPSTREAM
from .risk_targets import (build_risk_targets, estimate_train_barrier_fractions,
                           score_native_risk_distribution)


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
    candidate_risk_distribution: Mapping[str, np.ndarray] | None = None,
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
) -> dict[str, Any]:
    """Build a common-anchor, Train-fitted E0/E1/E2 comparison report.

    ``candidate_ohlcv_paths`` must contain frozen E1 and E2 outputs with shape
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
    if (set(candidate_ohlcv_paths) != {"E1", "E2"}
            or set(candidate_training_runs) != {"E1", "E2"}
            or set(candidate_anchor_timestamps_ns) != {"E1", "E2"}):
        raise ValueError("the registered comparison requires exactly frozen E1 and E2 candidates")
    if candidate_risk_distribution is not None and set(candidate_risk_distribution)!=\
            {"risk_mean","risk_scale","barrier_logits"}:
        raise ValueError("native risk distribution must contain mean, scale, and barrier logits")
    if (not horizons or tuple(sorted(set(horizons))) != horizons
            or any(type(horizon) is not int or horizon < 1 for horizon in horizons)
            or type(anchor_interval_minutes) is not int or anchor_interval_minutes < 1
            or type(context_lookback) is not int or context_lookback < 2
            or type(volatility_window) is not int or volatility_window < 2
            or type(block_length) is not int or block_length < 1
            or type(bootstrap_replicates) is not int or bootstrap_replicates < 1
            or type(seed) is not int or seed < 0):
        raise ValueError("invalid fold evaluation protocol")

    frames = replay.frames
    by_time = {frame.ts_close_ns: frame for frame in frames}
    all_times = np.asarray([frame.ts_close_ns for frame in frames], dtype=np.int64)
    all_availability = np.asarray([frame.available_at_ns for frame in frames], dtype=np.int64)
    all_closes = np.asarray([frame.close for frame in frames], dtype=np.float64)
    all_returns = np.log(all_closes[1:] / all_closes[:-1])
    return_times = all_times[1:]
    train_mask = ((return_times >= fold.train.score_start_ns)
                  & (return_times < fold.train.score_end_ns))
    train_returns = all_returns[train_mask]
    if train_returns.size < max(2, volatility_window + 1):
        raise ValueError("Fold Train score has too few returns for fitting and volatility strata")
    train_volatility = _rolling_rv(train_returns, volatility_window)
    train_score_mask=((all_times>=fold.train.score_start_ns)
                      &(all_times<fold.train.score_end_ns))
    train_barriers=estimate_train_barrier_fractions(
        all_closes[train_score_mask],train_timestamps_ns=all_times[train_score_mask],
        horizons=horizons,volatility_quantile=0.75)

    max_horizon = horizons[-1]
    first_anchor = fold.validation.score_start_ns - MINUTE_NS
    last_anchor = fold.validation.score_end_ns - max_horizon * MINUTE_NS
    anchor_stride = anchor_interval_minutes * MINUTE_NS
    anchors = np.arange(first_anchor, last_anchor + 1, anchor_stride, dtype=np.int64)
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
    normalized_runs: dict[str, dict[str, Any]] = {
        name: dict(candidate_training_runs[name]) for name in ("E1", "E2")
    }
    if any(not required_run_fields.issubset(run) for run in normalized_runs.values()):
        raise ValueError("E1/E2 run reports are missing frozen training provenance")
    for name, run in normalized_runs.items():
        for key, expected in (
            ("schema", "routeb.train-run.v1"), ("status", "COMPLETED"),
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
    if (normalized_runs["E1"]["train_npz_sha256"] != normalized_runs["E2"]["train_npz_sha256"]
            or normalized_runs["E1"]["training_config"] != normalized_runs["E2"]["training_config"]
            or normalized_runs["E1"]["optimizer_updates"] != normalized_runs["E2"]["optimizer_updates"]
            or normalized_runs["E1"]["train_examples"] != normalized_runs["E2"]["train_examples"]
            or normalized_runs["E1"]["device"] != normalized_runs["E2"]["device"]):
        raise ValueError("E1 and E2 must share one Train artifact, training budget, examples and device")
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
            raise ValueError("E1 and E2 must use the same path ensemble size")
    if n_paths is None:
        raise ValueError("candidate forecast ensembles are empty")

    anchor_closes = np.empty(len(anchors), dtype=np.float64)
    decision_times = np.empty(len(anchors), dtype=np.int64)
    realized_ohlcv = np.empty((len(anchors), max_horizon, 5), dtype=np.float64)
    outcomes = np.empty((len(anchors), max_horizon), dtype=np.int64)
    visible_contexts = np.empty((len(anchors), context_lookback), dtype=np.float64)
    anchor_volatility = np.empty(len(anchors), dtype=np.float64)
    for row, anchor in enumerate(anchors):
        anchor_frame = by_time.get(int(anchor))
        if anchor_frame is None:
            raise ValueError("Validation anchor is missing from the authorized replay")
        decision_ns = anchor_frame.available_at_ns
        first_target = by_time.get(int(anchor) + MINUTE_NS)
        if (anchor_frame.ts_close_ns != int(anchor) or decision_ns < anchor
                or first_target is None or decision_ns >= first_target.available_at_ns):
            raise ValueError("Validation decision must follow anchor availability and precede the first target")
        decision_times[row] = decision_ns
        anchor_closes[row] = anchor_frame.close
        visible_return_mask = ((return_times <= anchor)
                               & (all_availability[1:] <= decision_ns))
        context_returns = all_returns[visible_return_mask]
        if context_returns.size < context_lookback:
            raise ValueError("replay lacks the requested causal context before a Validation anchor")
        visible_contexts[row] = context_returns[-context_lookback:]
        anchor_volatility[row] = float(np.sqrt(np.mean(
            np.square(visible_contexts[row, -volatility_window:]))))
        for step in range(1, max_horizon + 1):
            timestamp = int(anchor) + step * MINUTE_NS
            frame = by_time.get(timestamp)
            if (frame is None or not fold.validation.score_start_ns <= timestamp
                    < fold.validation.score_end_ns or frame.available_at_ns <= decision_ns):
                raise ValueError("forecast target crosses the Validation boundary or was visible at decision time")
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
    from .evaluation import estimate_student_t_df, sample_e0_paths
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
        "s0_student_t_df_train_only": estimate_student_t_df(train_returns),
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
        generated_ohlcv_by_model={
            name: np.asarray(candidate_ohlcv_paths[name], dtype=np.float64)
            for name in ("E1", "E2")
        },
        realized_ohlcv=realized_ohlcv,
        previous_close=anchor_closes,
        barrier_fraction=tuple(float(value) for value in train_barriers),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    if candidate_risk_distribution is not None:
        risk_mean=np.asarray(candidate_risk_distribution["risk_mean"],dtype=np.float64)
        risk_horizons=tuple(normalized_runs["E2"].get("model_config",{}).get(
            "risk_horizons",(1,5,15,30,60)))
        if (risk_mean.shape!=(len(anchors),len(risk_horizons),4)
                or tuple(horizons[:len(risk_horizons)])!=risk_horizons
                or len(risk_horizons)>len(horizons)):
            raise ValueError("native risk forecasts do not match the frozen Train risk horizons")
        native=score_native_risk_distribution(
            risk_mean,candidate_risk_distribution["risk_scale"],
            candidate_risk_distribution["barrier_logits"],realized_ohlcv,
            anchor_close=anchor_closes,anchor_timestamps_ns=anchors,
            future_timestamps_ns=outcomes,horizons=risk_horizons,
            barrier_fraction=train_barriers[:len(risk_horizons)])
        per_anchor=cast(dict[str, dict[str, np.ndarray]],
                        native.pop("_per_anchor_scores"))
        target_risk=build_risk_targets(
            realized_ohlcv,anchor_close=anchor_closes,
            anchor_timestamps_ns=anchors,future_timestamps_ns=outcomes,
            horizons=risk_horizons,
            barrier_fraction=train_barriers[:len(risk_horizons)])
        risk_pairs: list[dict[str, Any]]=[]
        metric_targets=(
            ("return_crps","return",target_risk.log_return),
            ("log_realized_volatility_crps","log_realized_volatility",
             np.log(target_risk.realized_volatility+1e-8)),
        )
        for horizon_index,risk_horizon in enumerate(risk_horizons):
            valid=target_risk.valid[:,horizon_index]
            if not valid.any():
                continue
            for metric_name,target_name,observed in metric_targets:
                candidate_score=per_anchor[str(risk_horizon)][metric_name]
                baseline_score_by_name={}
                for baseline_name,baseline_paths in e0.items():
                    step_returns=np.asarray(baseline_paths,dtype=np.float64)
                    if target_name=="return":
                        samples=np.cumsum(step_returns[:,:,:risk_horizon],axis=-1)[:,:,-1]
                    else:
                        samples=np.log(np.sqrt(
                            np.square(step_returns[:,:,:risk_horizon]).sum(axis=-1))+1e-8)
                    baseline_score_by_name[baseline_name]=_crps_per_anchor(
                        samples[valid],observed[valid,horizon_index])
                strongest=min(baseline_score_by_name,
                    key=lambda name:float(baseline_score_by_name[name].mean()))
                result=paired_block_bootstrap_score_differences(
                    candidate_score,baseline_score_by_name[strongest],
                    anchor_timestamps_ns=anchors[valid],
                    anchor_interval_ns=anchor_stride,horizon=risk_horizon,
                    block_length=min(block_length,int(valid.sum())),
                    replicates=bootstrap_replicates,
                    seed=seed+risk_horizon+sum(map(ord,metric_name+strongest)),
                    baseline_name=strongest,candidate_name="E2_NATIVE_RISK")
                item={**result.__dict__,"target":target_name,
                      "strongest_baseline_selected_by_validation_crps":strongest}
                risk_pairs.append(item)
        path_comparison: dict[str, Any] | None=None
        raw_path_comparison=report.get("path_log_realized_volatility_comparison")
        if raw_path_comparison is not None:
            if not isinstance(raw_path_comparison,dict):
                raise ValueError("path log-RV comparison must be an object")
            path_comparison=raw_path_comparison
            path_pair=path_comparison.get("paired_block_bootstrap")
            if not isinstance(path_pair,dict):
                raise ValueError("path log-RV paired bootstrap evidence is missing")
            risk_pairs.append({
                **path_pair,
                "target":"path_log_realized_volatility",
                "strongest_baseline_selected_by_validation_crps":
                    path_comparison["baseline_selected_by_validation_crps"],
            })
        ordered=sorted(range(len(risk_pairs)),key=lambda i:risk_pairs[i]["p_value"])
        adjusted=[0.0]*len(risk_pairs)
        running=0.0
        for rank,index in enumerate(ordered):
            running=max(running,min(1.0,(len(ordered)-rank)*risk_pairs[index]["p_value"]))
            adjusted[index]=running
        for index,value in enumerate(adjusted):
            risk_pairs[index]["holm_p_value"]=value
            if (risk_pairs[index]["target"]=="path_log_realized_volatility"
                    and path_comparison is not None):
                path_comparison["paired_block_bootstrap"]["holm_p_value"]=value
        report["native_risk_scores"]=native
        report["native_risk_comparison"]={
            "schema":"routeb.native-risk-comparison.v1",
            "pairwise_paired_block_bootstrap":risk_pairs,
            "multiple_testing":("Holm family-wise adjustment across native risk target x horizon pairs "
                                "and preregistered path-derived 15m log-RV"),
            "block_length_anchors":block_length,
            "bootstrap_replicates":bootstrap_replicates,
            "score_capability":"REGISTERED_VALIDATION",
        }
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
        "decision_clock": "anchor_bar_available_at_ns",
        "decision_timestamps_sha256": _numeric_sha256(decision_times, dtype="<i8"),
        "context_lookback_minutes": context_lookback,
        "barrier_volatility_quantile": 0.75,
        "barrier_fraction_by_horizon": train_barriers.tolist(),
    }
    report.pop("report_sha256", None)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report
