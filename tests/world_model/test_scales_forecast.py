from __future__ import annotations

import numpy as np
import pytest

from rles_data.core import DataError, MINUTE_NS
from rles_world_model.forecast import decode_prior_paths
from rles_world_model.scales import (
    DailyD0State, FusionParameters, GRUParameters, HourlyGRUState,
    MultiScaleState, ScaleContext, fuse_scales,
)
from rles_world_model.temporal import AggregatedCandle, CompletedPeriods, HOUR_NS


def _hour(end_ns: int, close: float = 102.0) -> AggregatedCandle:
    return AggregatedCandle(end_ns-HOUR_NS, end_ns, end_ns, "ETHUSDT", 100.0,
                            max(100.0, close), min(100.0, close), close, 42.0, 60)


def test_hourly_gru_only_advances_for_complete_contiguous_hours() -> None:
    hourly = HourlyGRUState(GRUParameters.initialize(3, 6, seed=8))
    state = MultiScaleState(hourly)
    initial = state.context()
    assert initial.hour is None
    no_period = state.update(CompletedPeriods())
    assert no_period.hour is None
    first = state.update(CompletedPeriods(hour=_hour(HOUR_NS)), hour_features=np.ones(3))
    assert first.hour is not None and first.hour_end_ns == HOUR_NS
    with pytest.raises(DataError, match="contiguous"):
        state.update(CompletedPeriods(hour=_hour(3*HOUR_NS)), hour_features=np.ones(3))


def test_daily_d0_updates_only_on_complete_contiguous_utc_days() -> None:
    day_ns = 24 * HOUR_NS
    d0 = DailyD0State().update(AggregatedCandle(
        0, day_ns, day_ns, "ETHUSDT", 100, 110, 99, 105, 1200, 1440,
    ))
    assert d0.last_end_ns == day_ns
    assert d0.features().shape == (8,)
    assert np.isfinite(d0.features()).all()
    with pytest.raises(DataError, match="contiguous"):
        d0.update(AggregatedCandle(
            2*day_ns, 3*day_ns, 3*day_ns, "ETHUSDT", 105, 115, 104, 112, 1400, 1440,
        ))


def test_cross_scale_fusion_masks_missing_slow_states() -> None:
    params = FusionParameters.initialize(4, 6, 8, 5, seed=7)
    context = ScaleContext(None, np.zeros(8, np.float32), None, None)
    output, gates = fuse_scales(np.ones(4), context, params)
    assert output.shape == (5,)
    assert gates == (0.0, 0.0)
    context = ScaleContext(np.ones(6), np.ones(8), HOUR_NS, 24*HOUR_NS)
    _, gates = fuse_scales(np.ones(4), context, params)
    assert all(0 < gate < 1 for gate in gates)


def test_prior_samples_decode_to_independent_legal_minute_paths() -> None:
    samples = np.zeros((3, 61, 5), dtype=np.float32)
    samples[0, :, 0] = 0.001
    samples[1, :, 0] = -0.001
    paths = decode_prior_paths(
        samples, previous_close=100.0, last_close_ns=HOUR_NS,
        available_from_ns=HOUR_NS, instrument="ETHUSDT", tick_size=0.01,
    )
    assert len(paths) == 3
    assert len(paths[0].minutes) == 61
    assert len(paths[0].hours) == 1
    assert paths[0].minutes[0].close != paths[1].minutes[0].close
    for path in paths:
        for frame in path.minutes:
            assert frame.low <= min(frame.open, frame.close) <= max(frame.open, frame.close) <= frame.high
            assert frame.volume == 0
    assert paths[0].hours[0].minute_count == 60
    assert paths[0].hours[0].end_ns == 2*HOUR_NS


def test_prior_decoder_rejects_bad_shape_and_noncausal_origin() -> None:
    with pytest.raises(DataError, match="tensor"):
        decode_prior_paths(np.zeros((4, 5)), previous_close=100,
                           last_close_ns=HOUR_NS, available_from_ns=HOUR_NS,
                           instrument="ETHUSDT")
    with pytest.raises(DataError, match="available"):
        decode_prior_paths(np.zeros((1, 1, 5)), previous_close=100,
                           last_close_ns=HOUR_NS, available_from_ns=HOUR_NS-MINUTE_NS,
                           instrument="ETHUSDT")
