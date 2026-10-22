"""Causal UTC market streams and complete multi-timeframe candle aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Generic, TypeVar

from rles_data.core import DataError, MINUTE_NS, require_ns

from .contracts import MarketFrame, MarketReplay

HOUR_NS = 60 * MINUTE_NS
DAY_NS = 24 * HOUR_NS


@dataclass(frozen=True, slots=True)
class AggregatedCandle:
    """A complete UTC aligned candle derived only from available 1m bars."""

    start_ns: int
    end_ns: int
    available_at_ns: int
    instrument: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    minute_count: int

    def __post_init__(self) -> None:
        for name in ("start_ns", "end_ns", "available_at_ns"):
            require_ns(getattr(self, name))
        if self.end_ns <= self.start_ns or self.available_at_ns < self.end_ns:
            raise DataError("aggregated candle has invalid event or availability time")
        duration = self.end_ns - self.start_ns
        expected = duration // MINUTE_NS
        if duration % MINUTE_NS or self.minute_count != expected:
            raise DataError("aggregated candle must contain every expected minute")
        if duration not in (HOUR_NS, DAY_NS):
            raise DataError("only UTC hour and day aggregates are supported")
        if self.end_ns % duration:
            raise DataError("aggregate end must align to its UTC period")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if any(type(value) not in (int, float) for value in values):
            raise DataError("aggregate OHLCV values must be numeric")
        if any(not isfinite(value) for value in values):
            raise DataError("aggregate OHLCV values must be finite")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise DataError("aggregate prices must be positive and volume nonnegative")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise DataError("aggregate has invalid OHLC bounds")


@dataclass(frozen=True, slots=True)
class CompletedPeriods:
    hour: AggregatedCandle | None = None
    day: AggregatedCandle | None = None


class CausalMarketProvider:
    """Read-only stream over one frozen replay with an explicit availability barrier."""

    def __init__(self, replay: MarketReplay, *, feature_spec_hash: str) -> None:
        if not isinstance(replay, MarketReplay):
            raise DataError("causal provider requires a frozen MarketReplay")
        if not isinstance(feature_spec_hash, str) or len(feature_spec_hash) != 64:
            raise DataError("causal provider requires a feature-spec SHA-256")
        if any(right.ts_close_ns <= left.ts_close_ns
               for left, right in zip(replay.frames, replay.frames[1:])):
            raise DataError("causal provider rejects duplicate or unordered bars")
        self._replay = replay
        self._feature_spec_hash = feature_spec_hash

    @property
    def dataset_hash(self) -> str:
        return self._replay.dataset_hash

    def feature_spec_hash(self) -> str:
        return self._feature_spec_hash

    def stream(self, start_ns: int, end_ns: int) -> tuple[MarketFrame, ...]:
        require_ns(start_ns)
        require_ns(end_ns)
        if end_ns <= start_ns:
            raise DataError("market stream requires a positive time window")
        return tuple(frame for frame in self._replay.frames
                     if start_ns <= frame.ts_close_ns < end_ns)

    def visible_as_of(self, decision_ns: int, *, start_ns: int = 0,
                      end_ns: int | None = None) -> tuple[MarketFrame, ...]:
        require_ns(decision_ns)
        finish = decision_ns + 1 if end_ns is None else end_ns
        visible = tuple(frame for frame in self.stream(start_ns, finish)
                        if frame.available_at_ns <= decision_ns)
        if any(frame.ts_close_ns > decision_ns for frame in visible):
            raise DataError("market provider exposed a candle before its close")
        return visible


class UtcCandleAggregator:
    """Aggregate contiguous available 1m candles into complete UTC hours/days."""

    def __init__(self) -> None:
        self._last: MarketFrame | None = None
        self._instrument: str | None = None
        self._hour_end: int | None = None
        self._day_end: int | None = None
        self._hour_frames: list[MarketFrame] = []
        self._day_frames: list[MarketFrame] = []

    @property
    def last_close_ns(self) -> int | None:
        return None if self._last is None else self._last.ts_close_ns

    @property
    def partial_hour_count(self) -> int:
        return len(self._hour_frames)

    @property
    def partial_day_count(self) -> int:
        return len(self._day_frames)

    def push(self, frame: MarketFrame, *, decision_ns: int) -> CompletedPeriods:
        require_ns(decision_ns)
        if not isinstance(frame, MarketFrame):
            raise DataError("UTC aggregation requires a validated MarketFrame")
        if frame.available_at_ns > decision_ns or frame.ts_close_ns > decision_ns:
            raise DataError("market candle is not available at this decision barrier")
        if self._instrument is not None and frame.instrument != self._instrument:
            raise DataError("UTC aggregator cannot mix instruments")
        if self._last is not None:
            delta = frame.ts_close_ns - self._last.ts_close_ns
            if delta != MINUTE_NS:
                label = "duplicate" if delta <= 0 else "missing minute"
                raise DataError(f"UTC aggregator rejected {label}; explicit warmup reset required")

        self._instrument = frame.instrument
        hour_end = _period_end(frame.ts_close_ns, HOUR_NS)
        day_end = _period_end(frame.ts_close_ns, DAY_NS)
        if hour_end != self._hour_end:
            self._hour_end, self._hour_frames = hour_end, []
        if day_end != self._day_end:
            self._day_end, self._day_frames = day_end, []
        self._hour_frames.append(frame)
        self._day_frames.append(frame)
        self._last = frame

        hour = None
        if len(self._hour_frames) == 60:
            hour = _aggregate(self._hour_frames, self._hour_end or 0, HOUR_NS)
            self._hour_frames = []
        day = None
        if len(self._day_frames) == 1440:
            day = _aggregate(self._day_frames, self._day_end or 0, DAY_NS)
            self._day_frames = []
        return CompletedPeriods(hour=hour, day=day)

    def reset_with_warmup(self, frames: tuple[MarketFrame, ...], *, decision_ns: int,
                          min_warmup_minutes: int) -> None:
        """Start a new market segment from explicitly supplied contiguous history."""
        if type(min_warmup_minutes) is not int or min_warmup_minutes < 1:
            raise DataError("verified warmup length must be positive")
        if type(frames) is not tuple or len(frames) < min_warmup_minutes:
            raise DataError("restart requires the configured amount of warmup history")
        candidate = UtcCandleAggregator()
        for frame in frames:
            candidate.push(frame, decision_ns=decision_ns)
        self._last = candidate._last
        self._instrument = candidate._instrument
        self._hour_end = candidate._hour_end
        self._day_end = candidate._day_end
        self._hour_frames = candidate._hour_frames
        self._day_frames = candidate._day_frames


def _period_end(close_ns: int, period_ns: int) -> int:
    if close_ns <= 0:
        raise DataError("market candle close must be after the UTC epoch")
    return ((close_ns - 1) // period_ns + 1) * period_ns


def _aggregate(frames: list[MarketFrame], end_ns: int, period_ns: int) -> AggregatedCandle:
    start_ns = end_ns - period_ns
    expected_first = start_ns + MINUTE_NS
    expected_count = period_ns // MINUTE_NS
    if (len(frames) != expected_count or frames[0].ts_close_ns != expected_first
            or frames[-1].ts_close_ns != end_ns):
        raise DataError("incomplete UTC period cannot update a slow state")
    return AggregatedCandle(
        start_ns=start_ns, end_ns=end_ns,
        available_at_ns=max(frame.available_at_ns for frame in frames),
        instrument=frames[0].instrument, open=frames[0].open,
        high=max(frame.high for frame in frames), low=min(frame.low for frame in frames),
        close=frames[-1].close, volume=sum(frame.volume for frame in frames),
        minute_count=len(frames),
    )


State = TypeVar("State")
Prior = TypeVar("Prior")


@dataclass(frozen=True, slots=True)
class TemporalStep(Generic[State, Prior]):
    prior: Prior
    state: State
    completed: CompletedPeriods


class TemporalScheduler(Generic[State, Prior]):
    """Enforce prior -> current observation -> completed slow-period updates."""

    def __init__(self, state: State, aggregator: UtcCandleAggregator | None = None) -> None:
        self.state = state
        self.aggregator = aggregator or UtcCandleAggregator()

    def advance_one_minute(
        self,
        frame: MarketFrame,
        *,
        decision_ns: int,
        predict_prior: Callable[[State, int], Prior],
        observe: Callable[[State, MarketFrame], State],
        update_hour: Callable[[State, AggregatedCandle], State],
        update_day: Callable[[State, AggregatedCandle], State],
    ) -> TemporalStep[State, Prior]:
        if frame.available_at_ns > decision_ns or frame.ts_close_ns > decision_ns:
            raise DataError("scheduler cannot observe a candle before it is available")
        # The prior callback sees only the previous state and known clock time.
        prior = predict_prior(self.state, frame.ts_close_ns)
        next_state = observe(self.state, frame)
        completed = self.aggregator.push(frame, decision_ns=decision_ns)
        if completed.day is not None:
            next_state = update_day(next_state, completed.day)
        if completed.hour is not None:
            next_state = update_hour(next_state, completed.hour)
        self.state = next_state
        return TemporalStep(prior=prior, state=next_state, completed=completed)
