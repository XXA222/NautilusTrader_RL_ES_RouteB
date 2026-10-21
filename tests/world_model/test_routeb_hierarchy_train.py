from __future__ import annotations

import numpy as np
import pytest

from rles_world_model.routeb.hierarchy_dataset import HierarchyTensors
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchyTrainingConfig, _validated_arrays, _window_arrays,
)


def _tensors(*, hour_index: np.ndarray | None = None) -> HierarchyTensors:
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
        hour_available_at=np.asarray([1, 3], dtype=np.int64),
        day_available_at=np.asarray([1, 3], dtype=np.int64),
    )


def test_hierarchical_training_accepts_causal_train_tensors() -> None:
    arrays = _validated_arrays(_tensors())
    assert arrays["minute_x"].shape == (1, 3, 5)
    assert arrays["hour_index"].tolist() == [[0, 0, 1]]


def test_hierarchical_training_rejects_future_hour_state() -> None:
    # The hour state at availability 3 is not observable at minute time 2.
    with pytest.raises(ValueError, match="leaks a future slow state"):
        _validated_arrays(_tensors(hour_index=np.asarray([[0, 1, 1]], dtype=np.int32)))


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
    arrays = _validated_arrays(tensors)
    config = HierarchyTrainingConfig(window_length=16, hour_context_steps=3,
                                    day_context_steps=3)
    window = _window_arrays(arrays, start=2, config=config)
    assert window["minute_x"].shape == (1, 16, 5)
    assert window["hour_x"].shape == (1, 3, 5)
    assert window["day_x"].shape == (1, 3, 5)
    for key, clock in (("hour_index", "hour_available_at"),
                       ("day_index", "day_available_at")):
        index = window[key][0]
        selected = index >= 0
        assert np.all(window[clock][index[selected]] <= window["minute_available_at"][selected])
