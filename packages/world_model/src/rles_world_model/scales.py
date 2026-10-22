"""Causal low-frequency state encoders and bounded cross-scale fusion."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from rles_data.core import DataError, require_ns

from .temporal import AggregatedCandle, CompletedPeriods


@dataclass(frozen=True, slots=True)
class GRUParameters:
    input_weights: np.ndarray
    recurrent_weights: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        if (self.input_weights.ndim != 2 or self.recurrent_weights.ndim != 2
                or self.bias.ndim != 1 or self.input_weights.shape[1] != self.bias.size
                or self.recurrent_weights.shape != (self.bias.size // 3, self.bias.size)
                or self.bias.size % 3):
            raise DataError("GRU parameter dimensions are inconsistent")
        if not (np.isfinite(self.input_weights).all() and np.isfinite(self.recurrent_weights).all()
                and np.isfinite(self.bias).all()):
            raise DataError("GRU parameters must be finite")

    @classmethod
    def initialize(cls, input_size: int, hidden_size: int, *, seed: int) -> GRUParameters:
        if min(input_size, hidden_size) < 1:
            raise DataError("GRU input and hidden dimensions must be positive")
        rng = np.random.default_rng(seed)
        return cls(
            rng.normal(0, math.sqrt(2 / (input_size + hidden_size)),
                      (input_size, 3 * hidden_size)).astype(np.float32),
            rng.normal(0, math.sqrt(1 / hidden_size),
                      (hidden_size, 3 * hidden_size)).astype(np.float32),
            np.zeros(3 * hidden_size, dtype=np.float32),
        )


class HourlyGRUState:
    """One GRU transition per completed UTC hour, never per partial hour."""

    def __init__(self, parameters: GRUParameters, *, state: np.ndarray | None = None,
                 last_end_ns: int | None = None) -> None:
        self.parameters = parameters
        self.state = np.zeros(parameters.bias.size // 3, dtype=np.float32) if state is None else np.asarray(state, np.float32).copy()
        if self.state.shape != (parameters.bias.size // 3,) or not np.isfinite(self.state).all():
            raise DataError("hour GRU initial state has invalid shape or values")
        if last_end_ns is not None:
            require_ns(last_end_ns)
        self.last_end_ns = last_end_ns

    def update(self, candle: AggregatedCandle, features: np.ndarray) -> np.ndarray:
        if candle.end_ns % (60 * 60 * 1_000_000_000):
            raise DataError("hour GRU only accepts complete UTC hourly candles")
        if self.last_end_ns is not None and candle.end_ns - self.last_end_ns != 60 * 60 * 1_000_000_000:
            raise DataError("hour GRU requires contiguous complete hourly updates")
        x = np.asarray(features, dtype=np.float32)
        if x.shape != (self.parameters.input_weights.shape[0],) or not np.isfinite(x).all():
            raise DataError("hour GRU feature vector does not match its frozen input shape")
        h = self.state
        xg = x @ self.parameters.input_weights + self.parameters.bias
        hg = h @ self.parameters.recurrent_weights
        xr, xz, xn = np.split(xg, 3)
        hr, hz, hn = np.split(hg, 3)
        reset = _sigmoid(xr + hr)
        update = _sigmoid(xz + hz)
        candidate = np.tanh(xn + reset * hn)
        self.state = ((1 - update) * candidate + update * h).astype(np.float32)
        self.last_end_ns = candle.end_ns
        return self.state.copy()


@dataclass(frozen=True, slots=True)
class DailyD0State:
    """Fixed-capacity rolling daily statistics; D1 remains explicitly optional."""

    closes: tuple[float, ...] = ()
    volumes: tuple[float, ...] = ()
    returns: tuple[float, ...] = ()
    last_end_ns: int | None = None
    capacity: int = 30

    def __post_init__(self) -> None:
        if self.capacity != 30 or not (len(self.closes) == len(self.volumes) == len(self.returns)):
            raise DataError("D0 uses frozen 30-day rolling statistics")
        if len(self.closes) > self.capacity:
            raise DataError("D0 state exceeds its fixed capacity")
        if any(not math.isfinite(value) for value in (*self.closes, *self.volumes, *self.returns)):
            raise DataError("D0 state values must be finite")
        if self.last_end_ns is not None:
            require_ns(self.last_end_ns)

    def update(self, candle: AggregatedCandle) -> DailyD0State:
        day_ns = 24 * 60 * 60 * 1_000_000_000
        if candle.end_ns % day_ns:
            raise DataError("D0 only accepts a complete UTC daily candle")
        if self.last_end_ns is not None and candle.end_ns - self.last_end_ns != day_ns:
            raise DataError("D0 requires contiguous complete daily updates")
        previous = self.closes[-1] if self.closes else candle.open
        result = DailyD0State(
            (*self.closes, candle.close)[-self.capacity:],
            (*self.volumes, candle.volume)[-self.capacity:],
            (*self.returns, math.log(candle.close / previous))[-self.capacity:],
            candle.end_ns, self.capacity,
        )
        return result

    def features(self) -> np.ndarray:
        if not self.closes:
            return np.zeros(8, dtype=np.float32)
        returns = np.asarray(self.returns, dtype=np.float64)
        volumes = np.asarray(self.volumes, dtype=np.float64)
        def summary(window: int) -> tuple[float, float, float]:
            tail = returns[-window:]
            return float(tail.sum()), float(np.sqrt(np.mean(np.square(tail)))), float(abs(tail.sum()) / (np.abs(tail).sum() + 1e-12))
        ret7, vol7, trend7 = summary(7)
        ret30, vol30, trend30 = summary(30)
        vol_ratio = math.log1p(volumes[-1]) - math.log1p(float(volumes.mean()))
        values = np.asarray([ret7, ret30, vol7, vol30, trend7, trend30, vol_ratio,
                             min(len(self.closes) / 30, 1.0)], dtype=np.float32)
        return values


@dataclass(frozen=True, slots=True)
class ScaleContext:
    hour: np.ndarray | None
    day: np.ndarray
    hour_end_ns: int | None
    day_end_ns: int | None


class MultiScaleState:
    """Update low-frequency context only from aggregator completion events."""

    def __init__(self, hour: HourlyGRUState, day: DailyD0State | None = None) -> None:
        self.hour = hour
        self.day = day or DailyD0State()

    def update(self, periods: CompletedPeriods, *, hour_features: np.ndarray | None = None) -> ScaleContext:
        if periods.hour is not None:
            if hour_features is None:
                raise DataError("completed hour requires its causal hourly feature vector")
            self.hour.update(periods.hour, hour_features)
        elif hour_features is not None:
            raise DataError("partial hour must not update the hourly GRU")
        if periods.day is not None:
            self.day = self.day.update(periods.day)
        return self.context()

    def context(self) -> ScaleContext:
        return ScaleContext(
            None if self.hour.last_end_ns is None else self.hour.state.copy(),
            self.day.features(), self.hour.last_end_ns, self.day.last_end_ns,
        )


@dataclass(frozen=True, slots=True)
class FusionParameters:
    minute_weights: np.ndarray
    hour_weights: np.ndarray
    day_weights: np.ndarray
    bias: np.ndarray
    gate_weights: np.ndarray
    gate_bias: np.ndarray

    @classmethod
    def initialize(cls, minute_size: int, hour_size: int, day_size: int,
                   output_size: int, *, seed: int) -> FusionParameters:
        rng = np.random.default_rng(seed)
        scale = 1 / math.sqrt(max(minute_size, hour_size, day_size, 1))
        return cls(
            rng.normal(0, scale, (minute_size, output_size)).astype(np.float32),
            rng.normal(0, scale, (hour_size, output_size)).astype(np.float32),
            rng.normal(0, scale, (day_size, output_size)).astype(np.float32),
            np.zeros(output_size, np.float32),
            rng.normal(0, scale, (minute_size + 2, 2)).astype(np.float32),
            np.zeros(2, np.float32),
        )


def fuse_scales(minute: np.ndarray, context: ScaleContext,
                parameters: FusionParameters) -> tuple[np.ndarray, tuple[float, float]]:
    """Bound each slow-scale contribution through availability-masked sigmoid gates."""
    m = np.asarray(minute, np.float32)
    if m.shape != (parameters.minute_weights.shape[0],) or not np.isfinite(m).all():
        raise DataError("minute vector does not match fusion parameters")
    h = np.zeros(parameters.hour_weights.shape[0], np.float32) if context.hour is None else np.asarray(context.hour, np.float32)
    d = np.asarray(context.day, np.float32)
    if h.shape != (parameters.hour_weights.shape[0],) or d.shape != (parameters.day_weights.shape[0],):
        raise DataError("slow state does not match fusion parameter shapes")
    raw = np.concatenate([m, np.asarray([float(context.hour is not None), float(context.day_end_ns is not None)], np.float32)])
    gates = _sigmoid(raw @ parameters.gate_weights + parameters.gate_bias)
    gates[0] *= float(context.hour is not None)
    gates[1] *= float(context.day_end_ns is not None)
    result = m @ parameters.minute_weights + parameters.bias
    result += gates[0] * (h @ parameters.hour_weights)
    result += gates[1] * (d @ parameters.day_weights)
    return np.asarray(result, np.float32), (float(gates[0]), float(gates[1]))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))
