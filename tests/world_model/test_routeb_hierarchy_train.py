from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from rles_world_model.routeb.hierarchy_dataset import HierarchyTensors
from rles_world_model.routeb.hierarchy_dataset import _train_risk_targets
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchyTrainingConfig, _require_static_context_horizon,
    _hierarchy_source_identity, _validated_arrays, _window_arrays,
    _validate_risk_objective,
    forecast_hierarchical_validation_static_context_standardized,
    load_hierarchical_checkpoint_parameters,
)
from rles_world_model.routeb.market import MINUTE_NS


def _tensors(*, hour_index: np.ndarray | None = None,
             hour_available_at: np.ndarray | None = None,
             hour_day_index: np.ndarray | None = None) -> HierarchyTensors:
    return HierarchyTensors(
        minute_x=np.zeros((1, 3, 5), dtype=np.float32),
        hour_x=np.zeros((1, 2, 5), dtype=np.float32),
        day_x=np.zeros((1, 2, 5), dtype=np.float32),
        hour_index=(np.asarray([[0, 0, 1]], dtype=np.int32)
                    if hour_index is None else hour_index),
        day_index=np.asarray([[0, 0, 1]], dtype=np.int32),
        minute_mask=np.asarray([[0, 1, 1]], dtype=np.float32),
        hour_mask=np.ones((1, 2), dtype=np.float32),
        day_mask=np.ones((1, 2), dtype=np.float32),
        minute_available_at=np.asarray([1, 2, 3], dtype=np.int64),
        hour_available_at=(np.asarray([1, 3], dtype=np.int64)
                           if hour_available_at is None else hour_available_at),
        day_available_at=np.asarray([1, 3], dtype=np.int64),
        hour_day_index=hour_day_index,
    )


def test_hierarchical_training_accepts_causal_train_tensors() -> None:
    arrays = _validated_arrays(_tensors())
    assert arrays["minute_x"].shape == (1, 3, 5)
    assert arrays["hour_index"].tolist() == [[0, 0, 1]]
    assert arrays["risk_targets"].shape == (1, 3, 5, 6)
    assert not arrays["risk_mask"].any()


def test_train_risk_targets_match_e2_fields_and_stop_at_train_boundary() -> None:
    from rles_world_model.routeb.market import Candle

    candles = tuple(Candle(
        ts_close_ns=(index + 1) * MINUTE_NS,
        available_at_ns=(index + 1) * MINUTE_NS,
        instrument="SYNTH.SIM", open=100.0 + index, high=101.1 + index,
        low=99.9 + index, close=101.0 + index, volume=10.0)
        for index in range(80))
    score_end = candles[-1].ts_close_ns + MINUTE_NS
    targets, masks, barriers = _train_risk_targets(
        candles, score_start_ns=candles[0].ts_close_ns,
        score_end_ns=score_end, horizons=(1, 5))
    assert targets.shape == (1, 79, 2, 6)
    assert masks.shape == (1, 79, 2)
    assert barriers[0] > 0 and barriers[1] > 0
    # First feature anchor is candle 1; the first target is candle 2's close.
    assert targets[0, 0, 0, 0] == pytest.approx(np.log(103.0 / 102.0))
    assert targets[0, 0, 0, 1] == pytest.approx(np.log(abs(np.log(103.0 / 102.0)) + 1e-8))
    assert masks[0, 0, 0] == 1
    from rles_world_model.routeb.risk_targets import build_risk_targets
    path = np.asarray([[[bar.open, bar.high, bar.low, bar.close, bar.volume]
                        for bar in candles[2:7]]], dtype=np.float64)
    reference = build_risk_targets(
        path, anchor_close=np.asarray([candles[1].close]),
        anchor_timestamps_ns=np.asarray([candles[1].ts_close_ns], dtype=np.int64),
        future_timestamps_ns=np.asarray(
            [[bar.ts_close_ns for bar in candles[2:7]]], dtype=np.int64),
        horizons=(1, 5), barrier_fraction=barriers)
    reference_fields = np.stack((reference.log_return, np.log(
        reference.realized_volatility + 1e-8), reference.max_high_return,
        reference.min_low_return, reference.upper_barrier_hit,
        reference.lower_barrier_hit), axis=-1)
    np.testing.assert_allclose(targets[0, 0], reference_fields[0], rtol=0, atol=2e-7)
    # The final anchors lack future Train rows and cannot receive labels.
    assert masks[0, -1].tolist() == [0, 0]
    assert masks[0, -5, 0] == 1 and masks[0, -5, 1] == 0


def test_hierarchical_training_rejects_future_hour_state() -> None:
    # The hour state at availability 3 is not observable at minute time 2.
    with pytest.raises(ValueError, match="leaks a future slow state"):
        _validated_arrays(_tensors(hour_index=np.asarray([[0, 1, 1]], dtype=np.int32)))


def test_hierarchical_training_rejects_future_day_condition_for_hour() -> None:
    with pytest.raises(ValueError, match="hour-scale transition leaks a future daily state"):
        _validated_arrays(_tensors(
            hour_available_at=np.asarray([1, 2], dtype=np.int64),
            hour_day_index=np.asarray([[0, 1]], dtype=np.int32)))


def test_hierarchical_training_rejects_unscored_sequence() -> None:
    tensors = _tensors()
    empty = HierarchyTensors(
        tensors.minute_x, tensors.hour_x, tensors.day_x, tensors.hour_index,
        tensors.day_index, np.zeros_like(tensors.minute_mask), tensors.hour_mask,
        tensors.day_mask, tensors.minute_available_at, tensors.hour_available_at,
        tensors.day_available_at)
    with pytest.raises(ValueError, match="no scored minute targets"):
        _validated_arrays(empty)


def test_hierarchical_window_keeps_slow_states_causal_and_fixed_shape() -> None:
    minutes = 20
    tensors = HierarchyTensors(
        minute_x=np.zeros((1, minutes, 5), dtype=np.float32),
        hour_x=np.zeros((1, 3, 5), dtype=np.float32),
        day_x=np.zeros((1, 3, 5), dtype=np.float32),
        hour_index=np.asarray([[0] * 8 + [1] * 8 + [2] * 4], dtype=np.int32),
        day_index=np.asarray([[0] * 8 + [1] * 8 + [2] * 4], dtype=np.int32),
        minute_mask=np.ones((1, minutes), dtype=np.float32),
        hour_mask=np.ones((1, 3), dtype=np.float32),
        day_mask=np.ones((1, 3), dtype=np.float32),
        minute_available_at=np.arange(1, minutes + 1, dtype=np.int64),
        hour_available_at=np.asarray([1, 9, 17], dtype=np.int64),
        day_available_at=np.asarray([1, 9, 17], dtype=np.int64),
    )
    tensors = replace(
        tensors,
        risk_targets=np.ones((1, minutes, 5, 6), dtype=np.float32),
        risk_masks=np.ones((1, minutes, 5), dtype=np.float32),
        risk_barrier_fractions=(0.01,) * 5)
    arrays = _validated_arrays(tensors)
    config = HierarchyTrainingConfig(window_length=16, burn_in=4,
                                    hour_context_steps=3, day_context_steps=3)
    window = _window_arrays(arrays, start=2, config=config)
    assert window["minute_x"].shape == (1, 16, 5)
    assert window["hour_x"].shape == (1, 3, 5)
    assert window["day_x"].shape == (1, 3, 5)
    assert np.all(window["minute_mask"][0, :4] == 0)
    assert np.all(window["minute_mask"][0, 4:] == 1)
    assert np.all(tensors.minute_mask == 1)
    assert window["risk_targets"].shape == (1, 16, 5, 6)
    assert np.all(window["risk_mask"][0, :4] == 0)
    assert np.all(window["risk_mask"][0, 4:] == 1)
    for key, clock in (("hour_index", "hour_available_at"),
                       ("day_index", "day_available_at")):
        index = window[key][0]
        selected = index >= 0
        assert np.all(window[clock][index[selected]] <= window["minute_available_at"][selected])


def test_risk_objective_fails_closed_without_train_only_targets() -> None:
    disabled = HierarchyTrainingConfig()
    disabled_tensors = replace(_tensors(), risk_barrier_quantile=0.8)
    _validate_risk_objective(disabled_tensors,
                             _validated_arrays(disabled_tensors), disabled)
    config = HierarchyTrainingConfig(risk_auxiliary_scale=1.0)
    arrays = _validated_arrays(_tensors())
    with pytest.raises(ValueError, match="Train-only targets and fitted barriers"):
        _validate_risk_objective(_tensors(), arrays, config)
    targets = np.zeros((1, 3, 5, 6), dtype=np.float32)
    masks = np.ones((1, 3, 5), dtype=np.float32)
    tensors = replace(_tensors(), risk_targets=targets, risk_masks=masks,
                      risk_barrier_fractions=(0.01,) * 5)
    _validate_risk_objective(tensors, _validated_arrays(tensors), config)
    with pytest.raises(ValueError, match="barrier quantile"):
        _validate_risk_objective(
            replace(tensors, risk_barrier_quantile=0.8),
            _validated_arrays(replace(tensors, risk_barrier_quantile=0.8)), config)
    with pytest.raises(ValueError, match="Train-only targets and fitted barriers"):
        validation = replace(tensors, scope="validation")
        _validate_risk_objective(validation, _validated_arrays(validation), config)


def test_static_context_rollout_refuses_crossing_hour_or_day_event() -> None:
    _require_static_context_horizon(58 * MINUTE_NS, 1)
    with pytest.raises(ValueError, match="crosses a future hour/day event"):
        _require_static_context_horizon(59 * MINUTE_NS, 1)
    with pytest.raises(ValueError, match="crosses a future hour/day event"):
        _require_static_context_horizon(1439 * MINUTE_NS, 1)


def test_hierarchical_checkpoint_identity_binds_scale_and_dataset_provenance() -> None:
    fingerprints = ("a" * 64, "b" * 64, "c" * 64)
    tensors = replace(_tensors(), dataset_id="dataset-one",
                      scaler_fingerprints=fingerprints)
    assert _hierarchy_source_identity(tensors, "dataset-one") == {
        "hierarchy_dataset_id": "dataset-one",
        "scaler_fingerprints": list(fingerprints),
        "risk_horizons": [1, 5, 15, 30, 60],
        "risk_barrier_quantile": 0.75,
        "risk_barrier_fractions": None,
    }
    with pytest.raises(ValueError, match="dataset identity"):
        _hierarchy_source_identity(tensors, "dataset-two")
    with pytest.raises(ValueError, match="scaler fingerprints"):
        _hierarchy_source_identity(
            replace(tensors, scaler_fingerprints=("not-a-hash",)), "dataset-one")
    with pytest.raises(ValueError, match="Train tensors only"):
        _hierarchy_source_identity(replace(tensors, scope="validation"), "dataset-one")


def test_validation_forecast_rejects_validation_tensors_as_checkpoint_identity() -> None:
    validation = replace(_tensors(), scope="validation",
                         score_start_ns=1, score_end_ns=100 * MINUTE_NS)
    with pytest.raises(ValueError, match="separate Train and Validation"):
        forecast_hierarchical_validation_static_context_standardized(
            train_tensors=validation, validation_tensors=validation,
            start=0, steps=1, train_scope_id="train-scope", dataset_id="dataset-one",
            checkout="unused", checkpoint="unused")
    with pytest.raises(ValueError, match="Train tensors"):
        load_hierarchical_checkpoint_parameters(
            tensors=validation, train_scope_id="train-scope", dataset_id="dataset-one",
            checkout="unused", checkpoint="unused")
