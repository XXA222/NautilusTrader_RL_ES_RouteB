from __future__ import annotations

import numpy as np
import pytest

from rles_world_model.routeb.clock import MINUTE_NS, MultiScaleClock
from rles_world_model.routeb.hierarchy import derive_future_scale_events


def test_future_scale_events_follow_actual_availability_barriers() -> None:
    hour, day = derive_future_scale_events(
        minute_decisions_ns=np.asarray([100, 200, 300, 400], dtype=np.int64),
        hour_available_at_ns=np.asarray([200, 400], dtype=np.int64),
        day_available_at_ns=np.asarray([200], dtype=np.int64),
    )
    assert hour == (False, True, False, True)
    assert day == (False, True, False, False)


@pytest.mark.parametrize(
    "decisions,hours,days",
    [
        ([100, 100], [], []),
        ([100, 200], [200, 200], []),
        ([100, 200], [], [-1]),
        ([100, 200], [300], []),
    ],
)
def test_future_scale_events_reject_invalid_or_unaligned_schedules(
    decisions: list[int], hours: list[int], days: list[int],
) -> None:
    with pytest.raises(ValueError):
        derive_future_scale_events(
            minute_decisions_ns=np.asarray(decisions, dtype=np.int64),
            hour_available_at_ns=np.asarray(hours, dtype=np.int64),
            day_available_at_ns=np.asarray(days, dtype=np.int64),
        )


def test_future_scale_events_allow_independent_slow_scale_delays() -> None:
    hour, day = derive_future_scale_events(
        minute_decisions_ns=np.asarray([100, 200], dtype=np.int64),
        hour_available_at_ns=np.asarray([200], dtype=np.int64),
        day_available_at_ns=np.asarray([100], dtype=np.int64),
    )
    assert hour == (False, True)
    assert day == (True, False)


def test_future_scale_events_reject_multiple_updates_at_one_barrier() -> None:
    with pytest.raises(ValueError, match="multiple future hour states"):
        derive_future_scale_events(
            minute_decisions_ns=np.asarray([100, 200], dtype=np.int64),
            hour_available_at_ns=np.asarray([200, 200], dtype=np.int64),
            day_available_at_ns=np.asarray([], dtype=np.int64),
        )


def test_clock_projects_future_slow_availability_without_future_bars() -> None:
    decisions = np.asarray([4_321, 4_322, 4_323], dtype=np.int64) * MINUTE_NS
    delayed_hour = MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decisions,
        timeframe_minutes=60,
        availability_delay_ns=2 * MINUTE_NS,
        observed_close_through_ns=4_260 * MINUTE_NS,
    )
    daily = MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decisions,
        timeframe_minutes=1_440,
        availability_delay_ns=0,
        observed_close_through_ns=2_880 * MINUTE_NS,
    )
    assert delayed_hour == (4_322 * MINUTE_NS,)
    assert daily == (4_320 * MINUTE_NS,)
    hour_events, day_events = derive_future_scale_events(
        minute_decisions_ns=decisions,
        hour_available_at_ns=np.asarray(delayed_hour, dtype=np.int64),
        day_available_at_ns=np.asarray(daily, dtype=np.int64),
    )
    assert hour_events == (False, True, False)
    assert day_events == (True, False, False)
