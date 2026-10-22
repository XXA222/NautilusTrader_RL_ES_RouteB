"""Contract tests for RouteB shared-anchor probabilistic model evaluation."""
from __future__ import annotations

import json
import numpy as np
import pytest

from rles_world_model.routeb.evaluation import (
    build_predictive_comparison_report,
    bucket_aggregation_consistency,
    ohlcv_generation_diagnostics,
    compare_path_models,
    diagonal_gaussian_return_nll,
    paired_block_bootstrap_crps,
    predictive_horizon_scores,
    return_paths,
    verify_predictive_report_digest,
)


def sample_data():
    rng = np.random.default_rng(1701)
    truth = rng.normal(0.0, 0.01, size=(48, 8))
    baseline = rng.normal(0.0, 0.015, size=(48, 80, 8))
    candidate = rng.normal(0.0, 0.01, size=(48, 80, 8))
    return truth, baseline, candidate


def provenance(*names):
    return {
        name: {
            "model_family": name,
            "model_config_sha256": str(index + 1) * 64,
            "artifact_sha256": str(index + 4) * 64,
            "train_examples": 100,
            "optimizer_updates": index * 10,
            "training_seconds": float(index),
            "parameter_count": index * 100,
            "device": "cpu",
            "forecast_seed": 31,
        }
        for index, name in enumerate(names)
    }


def test_predictive_metrics_cover_calibration_direction_and_tail():
    truth, _, paths = sample_data()
    nll = np.full(len(truth), 1.25)
    score = predictive_horizon_scores(
        paths, truth, horizons=(1, 5, 8), nll_by_horizon={5: nll}, pit_seed=9)
    five = score[1]
    assert five.horizon == 5
    assert five.nll == pytest.approx(1.25)
    assert len(five.pit_values) == len(truth)
    assert all(0 <= value <= 1 for value in five.pit_values)
    assert 0 <= five.coverage_90 <= 1 and 0 <= five.coverage_95 <= 1
    assert five.width_95 >= five.width_90 >= 0
    assert 0 <= five.var_breach_01 <= 1 and 0 <= five.var_breach_05 <= 1
    assert np.isfinite(five.direction_brier) and five.cvar_05 <= five.var_05
    assert five.realized_breach_loss_05 is None or np.isfinite(five.realized_breach_loss_05)
    assert five.direction_auc is None or 0 <= five.direction_auc <= 1
    assert five.direction_log_loss >= 0 and 0 <= five.sign_accuracy <= 1


def test_scores_reject_invalid_nll_and_horizons():
    truth, _, paths = sample_data()
    with pytest.raises(ValueError, match="native NLL"):
        predictive_horizon_scores(paths, truth, horizons=(5,),
                                  nll_by_horizon={5: np.ones(2)})
    with pytest.raises(ValueError, match="horizons"):
        predictive_horizon_scores(paths, truth, horizons=(5, 4))


def test_one_step_return_nll_is_exact_under_diagonal_gaussian_head():
    center = np.array([0.01, -0.002, 0.0, 0.0, 0.0])
    spread = np.array([0.1, 0.2, 1.0, 1.0, 1.0])
    mean = np.zeros((1, 5))
    scale = np.array([[0.5, 0.25, 0.2, 0.2, 0.3]])
    target = np.zeros((1, 5))
    target[0, 0] = 1.0
    nll = diagonal_gaussian_return_nll(
        mean, scale, target, scaler_mean=center, scaler_std=spread)
    return_scale = np.sqrt((0.5 * 0.1) ** 2 + (0.25 * 0.2) ** 2)
    expected = 0.5 * (spread[0] / return_scale) ** 2
    expected += np.log(return_scale) + 0.5 * np.log(2 * np.pi)
    assert nll.shape == (1,)
    assert nll[0] == pytest.approx(expected)
    with pytest.raises(ValueError, match="finite aligned"):
        diagonal_gaussian_return_nll(
            mean, -scale, target, scaler_mean=center, scaler_std=spread)


def test_model_comparison_uses_same_anchors_and_future_labels():
    truth, baseline, candidate = sample_data()
    anchors = np.arange(48, dtype=np.int64) * 60
    outcomes = anchors[:, None] + np.arange(1, 9, dtype=np.int64)[None, :] * 60
    scores = compare_path_models(
        {"E0": baseline, "E1": candidate}, truth,
        anchor_timestamps_ns=anchors, outcome_timestamps_ns=outcomes,
        horizons=(1, 5, 8))
    assert list(scores) == ["E0", "E1"]
    assert scores["E1"][1].samples == 48
    bad_outcomes = outcomes.copy()
    bad_outcomes[0, 0] = anchors[0]
    with pytest.raises(ValueError, match="strictly after"):
        compare_path_models({"E0": baseline}, truth, anchor_timestamps_ns=anchors,
                            outcome_timestamps_ns=bad_outcomes, horizons=(1,))
    with pytest.raises(ValueError, match="common anchor"):
        compare_path_models({"E0": baseline, "E1": candidate[:-1]}, truth,
                            anchor_timestamps_ns=anchors, outcome_timestamps_ns=outcomes,
                            horizons=(1,))


def test_paired_block_bootstrap_is_deterministic_and_reports_improvement():
    truth, baseline, candidate = sample_data()
    anchors = np.arange(len(truth), dtype=np.int64) * 60_000_000_000
    result = paired_block_bootstrap_crps(
        candidate, baseline, truth, anchor_timestamps_ns=anchors,
        anchor_interval_ns=60_000_000_000, horizon=5, block_length=6,
        replicates=500, seed=42, baseline_name="E0", candidate_name="E1")
    replay = paired_block_bootstrap_crps(
        candidate, baseline, truth, anchor_timestamps_ns=anchors,
        anchor_interval_ns=60_000_000_000, horizon=5, block_length=6,
        replicates=500, seed=42, baseline_name="E0", candidate_name="E1")
    assert result == replay
    assert result.mean_difference < 0
    assert result.ci_95_low <= result.ci_95_high
    assert result.baseline == "E0" and result.candidate == "E1"


def test_block_bootstrap_does_not_bridge_missing_time_segment():
    truth, baseline, candidate = sample_data()
    anchors = np.arange(len(truth), dtype=np.int64) * 60
    anchors[24:] += 3_600
    with pytest.raises(ValueError, match="continuous timestamp segment"):
        paired_block_bootstrap_crps(
            candidate, baseline, truth, anchor_timestamps_ns=anchors,
            anchor_interval_ns=60, horizon=5, block_length=25, replicates=100)


def test_complete_report_binds_protocol_strata_and_adjusts_comparisons():
    truth, e0, e1 = sample_data()
    anchors = np.arange(len(truth), dtype=np.int64) * 60_000_000_000
    outcomes = anchors[:, None] + np.arange(1, 9, dtype=np.int64)[None, :] * 60_000_000_000
    anchor_vol = np.resize(np.array([25.0, 75.0]), len(truth))
    report = build_predictive_comparison_report(
        {"E0": e0, "E1": e1}, truth, anchor_timestamps_ns=anchors,
        outcome_timestamps_ns=outcomes, anchor_interval_ns=60_000_000_000,
        block_length=6, train_volatility=np.arange(1.0, 102.0),
        anchor_volatility=anchor_vol, volatility_quantiles=(0.5,),
        metadata={"dataset_identity": "d" * 64, "fold_identity": "f" * 64,
                  "train_scope_id": "registered-scope", "snapshot_sha256": "a" * 40},
        model_provenance=provenance("E0", "E1"),
        nll_by_model={"E1": {1: np.full(len(truth), 0.75)}},
        horizons=(1, 5), bootstrap_replicates=200, seed=31)
    assert report["authorization_verified"] is False
    assert report["model_selection"] == "not_performed"
    assert set(report["volatility_strata"]) == {"0", "1"}
    assert report["overall"]["E1"][0]["nll"] == pytest.approx(0.75)
    assert report["overall"]["E0"][0]["nll"] is None
    differences = report["paired_crps_differences"]
    assert len(differences) == 2
    assert all("holm_p_value" in item for item in differences)
    assert all(0 <= item["holm_p_value"] <= 1 for item in differences)
    serialized = json.dumps(report, allow_nan=False)
    assert report["report_sha256"] in serialized
    assert verify_predictive_report_digest(report)
    report["economic_claim"] = True
    assert not verify_predictive_report_digest(report)
    mismatched_train_provenance = provenance("E0", "E1")
    mismatched_train_provenance["E1"]["train_examples"] = 99
    with pytest.raises(ValueError, match="same number"):
        build_predictive_comparison_report(
            {"E0": e0, "E1": e1}, truth, anchor_timestamps_ns=anchors,
            outcome_timestamps_ns=outcomes, anchor_interval_ns=60_000_000_000,
            block_length=6, train_volatility=np.arange(1.0, 102.0),
            anchor_volatility=anchor_vol, volatility_quantiles=(0.5,),
            metadata={"dataset_identity": "d" * 64, "fold_identity": "f" * 64,
                      "train_scope_id": "registered-scope", "snapshot_sha256": "a" * 40},
            model_provenance=mismatched_train_provenance,
            horizons=(1,), bootstrap_replicates=100)
    mismatched_seed_provenance = provenance("E0", "E1")
    mismatched_seed_provenance["E1"]["forecast_seed"] = 32
    with pytest.raises(ValueError, match="forecast seed"):
        build_predictive_comparison_report(
            {"E0": e0, "E1": e1}, truth, anchor_timestamps_ns=anchors,
            outcome_timestamps_ns=outcomes, anchor_interval_ns=60_000_000_000,
            block_length=6, train_volatility=np.arange(1.0, 102.0),
            anchor_volatility=anchor_vol, volatility_quantiles=(0.5,),
            metadata={"dataset_identity": "d" * 64, "fold_identity": "f" * 64,
                      "train_scope_id": "registered-scope", "snapshot_sha256": "a" * 40},
            model_provenance=mismatched_seed_provenance,
            horizons=(1,), bootstrap_replicates=100)


def valid_ohlcv_paths():
    generated = np.empty((2, 3, 6, 5), dtype=np.float64)
    anchors = np.array([100.0, 110.0])
    for n in range(2):
        for p in range(3):
            previous = anchors[n]
            for t in range(6):
                close = previous * (1 + 0.001 * (-1) ** (t + p))
                generated[n, p, t] = (previous, max(previous, close) * 1.002,
                                      min(previous, close) * 0.998, close, 4 + p)
                previous = close
    realized = generated[:, 0].copy()
    return generated, realized, anchors


def test_generated_ohlcv_path_diagnostics_compare_structure_and_touch():
    generated, realized, anchors = valid_ohlcv_paths()
    result = ohlcv_generation_diagnostics(
        generated, realized, previous_close=anchors, rolling_window=3)
    assert result["generated_bar_valid_rate"] == 1.0
    assert result["generated_path_valid_rate"] == 1.0
    assert result["realized_bar_valid_rate"] == 1.0
    assert result["generated_rolling_rv_mean"] is not None
    assert "0.005" in result["touch_reliability"]
    generated[0, 0, 2, 1] = generated[0, 0, 2, 0] * 0.9
    result = ohlcv_generation_diagnostics(
        generated, realized, previous_close=anchors, rolling_window=3)
    assert result["generated_bar_valid_rate"] < 1.0
    assert result["generated_path_valid_rate"] < 1.0


def test_f2_bucket_ohlcv_must_match_synthetic_minute_aggregation():
    generated, _, _ = valid_ohlcv_paths()
    minutes = generated[:, :, :4]
    ranges = ((0, 2), (2, 4))
    expected = []
    for start, end in ranges:
        section = minutes[:, :, start:end]
        expected.append(np.stack((section[:, :, 0, 0], section[:, :, :, 1].max(axis=2),
                                  section[:, :, :, 2].min(axis=2), section[:, :, -1, 3],
                                  section[:, :, :, 4].sum(axis=2)), axis=-1))
    buckets = np.stack(expected, axis=2)
    result = bucket_aggregation_consistency(minutes, buckets, bucket_ranges=ranges)
    assert result["bucket_consistency_rate"] == 1.0
    buckets[0, 0, 0, 1] *= 1.01
    result = bucket_aggregation_consistency(minutes, buckets, bucket_ranges=ranges)
    assert result["bucket_consistency_rate"] < 1.0


def test_comparison_report_integrates_generation_and_f2_checks():
    generated, realized_bars, anchors_close = valid_ohlcv_paths()
    e0 = return_paths(generated, previous_close=anchors_close)
    e1_bars = generated[:, ::-1].copy()
    e1 = return_paths(e1_bars, previous_close=anchors_close)
    realized = return_paths(realized_bars[:, None], previous_close=anchors_close)[:, 0]
    timestamps = np.arange(2, dtype=np.int64) * 60_000_000_000
    outcomes = timestamps[:, None] + np.arange(1, 7, dtype=np.int64)[None, :] * 60_000_000_000
    ranges = ((0, 2), (2, 4), (4, 6))
    buckets = []
    for start, end in ranges:
        section = generated[:, :, start:end]
        buckets.append(np.stack((section[:, :, 0, 0], section[:, :, :, 1].max(axis=2),
                                 section[:, :, :, 2].min(axis=2), section[:, :, -1, 3],
                                 section[:, :, :, 4].sum(axis=2)), axis=-1))
    bucket_bars = np.stack(buckets, axis=2)
    report = build_predictive_comparison_report(
        {"E0": e0, "E1": e1}, realized, anchor_timestamps_ns=timestamps,
        outcome_timestamps_ns=outcomes, anchor_interval_ns=60_000_000_000,
        block_length=1, train_volatility=np.arange(1.0, 10.0),
        anchor_volatility=np.array([2.0, 8.0]),
        metadata={"dataset_identity": "d" * 64, "fold_identity": "f" * 64,
                  "train_scope_id": "registered-scope", "snapshot_sha256": "a" * 40},
        model_provenance=provenance("E0", "E1"),
        horizons=(1, 2), generated_ohlcv_by_model={"E0": generated, "E1": e1_bars},
        realized_ohlcv=realized_bars, previous_close=anchors_close,
        f2_bucket_ohlcv_by_model={"E0": bucket_bars}, f2_bucket_ranges=ranges,
        rolling_window=3, bootstrap_replicates=100)
    assert set(report["generation_diagnostics"]) == {"E0", "E1"}
    assert report["f2_bucket_consistency"]["E0"]["bucket_consistency_rate"] == 1.0
    assert "generated_ohlcv" in report["input_sha256"]
    assert verify_predictive_report_digest(report)
