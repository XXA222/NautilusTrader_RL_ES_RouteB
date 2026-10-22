"""Paired, validation-only scoring for the preregistered R3 lambda arms."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .evaluation import _crps_per_anchor, paired_block_bootstrap_score_differences
from .evaluation import return_paths

R3_LAMBDA_ARMS = (0.0, 0.01, 0.05)


def score_r2_ablation_ensembles(
    forecasts: Mapping[float, Mapping[int, np.ndarray]],
    realized_ohlcv: np.ndarray,
    previous_close: np.ndarray,
    anchor_timestamps_ns: np.ndarray,
    *,
    horizon: int = 15,
    anchor_interval_ns: int,
    block_length: int = 12,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
) -> dict:
    """Compare λ=.01/.05 against λ=0 using paired path and return CRPS.

    Arrays must already be restricted to authorized Validation anchors. This
    scorer does not open a source file or inspect any Fold Test rows.
    """
    if set(forecasts) != set(R3_LAMBDA_ARMS):
        raise ValueError("R3 requires the frozen lambda arms 0, 0.01, and 0.05")
    anchors = np.asarray(anchor_timestamps_ns)
    realized = np.asarray(realized_ohlcv, dtype=np.float64)
    previous = np.asarray(previous_close, dtype=np.float64)
    if (anchors.ndim != 1 or not np.issubdtype(anchors.dtype, np.integer)
            or len(anchors) < block_length or np.any(np.diff(anchors) != anchor_interval_ns)
            or realized.ndim != 3 or realized.shape[0] != len(anchors)
            or realized.shape[1] < horizon or realized.shape[2] != 5
            or previous.shape != (len(anchors),) or not np.isfinite(previous).all()
            or np.any(previous <= 0)):
        raise ValueError("R3 realized labels/anchors do not match the frozen validation geometry")
    if (type(horizon) is not int or horizon < 1 or type(block_length) is not int
            or type(bootstrap_replicates) is not int or bootstrap_replicates < 100
            or type(seed) is not int or seed < 0):
        raise ValueError("invalid R3 score protocol")

    seeds = tuple(sorted(forecasts[0.0]))
    if len(seeds) < 3 or any(type(value) is not int or value < 0 for value in seeds):
        raise ValueError("R3 needs the same three or more frozen model seeds")
    arrays: dict[float, dict[int, np.ndarray]] = {}
    expected_shape = None
    for arm in R3_LAMBDA_ARMS:
        if tuple(sorted(forecasts[arm])) != seeds:
            raise ValueError("all R3 lambda arms must use exactly the same seeds")
        arrays[arm] = {}
        for model_seed in seeds:
            values = np.asarray(forecasts[arm][model_seed], dtype=np.float64)
            if (values.ndim != 4 or values.shape[0] != len(anchors)
                    or values.shape[2] < horizon or values.shape[3] != 5
                    or values.shape[1] < 2 or not np.isfinite(values).all()
                    or (values[..., :4] <= 0).any() or (values[..., 4] < 0).any()
                    or np.any(values[..., 1] < np.maximum(values[..., 0], values[..., 3]))
                    or np.any(values[..., 2] > np.minimum(values[..., 0], values[..., 3]))):
                raise ValueError("R3 candidate contains invalid OHLCV paths")
            if expected_shape is None:
                expected_shape = values.shape
            if values.shape != expected_shape:
                raise ValueError("R3 arms must share anchors, paths, horizon and feature shape")
            arrays[arm][model_seed] = values

    if expected_shape is None:
        raise ValueError("R3 forecast arrays are empty")

    realized_returns = return_paths(
        realized[:, None, :horizon, :], previous_close=previous)[:, 0]
    observed_log_rv = np.log(np.sqrt(np.square(realized_returns).sum(axis=-1)) + 1e-8)
    observed_return = realized_returns.sum(axis=-1)

    def score(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        returns = return_paths(paths[:, :, :horizon, :], previous_close=previous)
        simulated_log_rv = np.log(np.sqrt(np.square(returns).sum(axis=-1)) + 1e-8)
        simulated_return = returns.sum(axis=-1)
        return (_crps_per_anchor(simulated_log_rv, observed_log_rv),
                _crps_per_anchor(simulated_return, observed_return))

    report: dict[str, object] = {
        "schema": "routeb.r3-r2-regularizer-comparison.v1",
        "lambda_arms": list(R3_LAMBDA_ARMS),
        "training_seeds": list(seeds),
        "horizon": horizon,
        "anchor_count": len(anchors),
        "paths_per_seed": int(next(iter(arrays[0.0].values())).shape[1]),
        "test_payload_decoded": False,
        "comparisons": {},
    }
    comparisons: dict[str, object] = {}
    for arm in R3_LAMBDA_ARMS[1:]:
        arm_result: dict[str, object] = {"per_seed": {}}
        per_seed: dict[str, object] = {}
        for model_seed in seeds:
            baseline_scores = score(arrays[0.0][model_seed])
            candidate_scores = score(arrays[arm][model_seed])
            metrics: dict[str, object] = {}
            for metric_index, name in enumerate(("path_log_rv_crps", "return_crps")):
                paired = paired_block_bootstrap_score_differences(
                    candidate_scores[metric_index], baseline_scores[metric_index],
                    anchor_timestamps_ns=anchors, anchor_interval_ns=anchor_interval_ns,
                    horizon=horizon, block_length=block_length,
                    replicates=bootstrap_replicates,
                    seed=seed + model_seed + int(arm * 1000) + metric_index,
                    baseline_name="lambda_0", candidate_name=f"lambda_{arm:g}")
                metrics[name] = paired.__dict__
            per_seed[str(model_seed)] = metrics
        pooled: dict[str, object] = {}
        baseline_ensemble = np.concatenate(
            [arrays[0.0][model_seed] for model_seed in seeds], axis=1)
        candidate_ensemble = np.concatenate(
            [arrays[arm][model_seed] for model_seed in seeds], axis=1)
        baseline_scores = score(baseline_ensemble)
        candidate_scores = score(candidate_ensemble)
        for metric_index, name in enumerate(("path_log_rv_crps", "return_crps")):
            paired = paired_block_bootstrap_score_differences(
                candidate_scores[metric_index], baseline_scores[metric_index],
                anchor_timestamps_ns=anchors, anchor_interval_ns=anchor_interval_ns,
                horizon=horizon, block_length=block_length,
                replicates=bootstrap_replicates,
                seed=seed + int(arm * 1000) + metric_index + 97,
                baseline_name="lambda_0_pooled", candidate_name=f"lambda_{arm:g}_pooled")
            pooled[name] = paired.__dict__
        arm_result["per_seed"] = per_seed
        arm_result["pooled"] = pooled
        comparisons[f"lambda_{arm:g}_vs_0"] = arm_result
    report["comparisons"] = comparisons
    return report
