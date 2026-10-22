"""Versioned causal OHLCV features, train-only scaling, and invertible targets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import math
from typing import Literal, Sequence

from rles_data.core import DataError, MINUTE_NS, require_ns

from .contracts import MarketFrame
from .temporal import AggregatedCandle, DAY_NS, HOUR_NS

MINUTE_FEATURE_NAMES = (
    "log_close_return", "log_open_gap", "log_upper_wick", "log_lower_wick",
    "log1p_volume", "rv_5m", "rv_15m", "trend_efficiency_5m",
    "trend_efficiency_15m", "relative_volume_15m",
)
HOUR_FEATURE_NAMES = (
    "log_close_return", "log_open_gap", "log_upper_wick", "log_lower_wick",
    "log1p_volume", "log_return_4h", "log_return_8h", "log_return_24h",
    "rv_24h", "trend_efficiency_24h", "relative_volume_24h",
)
DAY_FEATURE_NAMES = (
    "log_close_return", "log_open_gap", "log_upper_wick", "log_lower_wick",
    "log1p_volume", "log_return_7d", "log_return_30d", "rv_7d", "rv_30d",
    "trend_efficiency_7d", "trend_efficiency_30d", "relative_volume_30d",
)
MODEL_INPUT_SIZE = 79
CALENDAR_FEATURE_NAMES = (
    "utc_minute_sin", "utc_minute_cos", "utc_weekday_sin", "utc_weekday_cos",
    "is_complete_hour", "is_complete_day",
)
FeatureScale = Literal["1m", "1h", "1d", "calendar"]


@dataclass(frozen=True, slots=True)
class MarketFeatureSpec:
    """The input feature schema; target variables remain a separate contract."""

    version: str = "market_features_v1"
    minute_names: tuple[str, ...] = MINUTE_FEATURE_NAMES
    hour_names: tuple[str, ...] = HOUR_FEATURE_NAMES
    day_names: tuple[str, ...] = DAY_FEATURE_NAMES
    calendar_names: tuple[str, ...] = CALENDAR_FEATURE_NAMES
    minute_windows: tuple[int, ...] = (5, 15)
    hour_windows: tuple[int, ...] = (4, 8, 24)
    day_windows: tuple[int, ...] = (7, 30)

    def __post_init__(self) -> None:
        groups = (self.minute_names, self.hour_names, self.day_names, self.calendar_names)
        if self.version != "market_features_v1" or any(
            type(names) is not tuple or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            for names in groups
        ):
            raise DataError("unsupported or malformed market feature schema")
        if (self.minute_windows != (5, 15) or self.hour_windows != (4, 8, 24)
                or self.day_windows != (7, 30)):
            raise DataError("market_features_v1 calculation windows are frozen")

    @property
    def hash(self) -> str:
        payload = {
            "version": self.version,
            "minute": self.minute_names,
            "hour": self.hour_names,
            "day": self.day_names,
            "calendar": self.calendar_names,
            "windows": {
                "minute": self.minute_windows,
                "hour": self.hour_windows,
                "day": self.day_windows,
            },
            "formulas": (
                "log-ratios;realized-rms;absolute-net-over-total-path;"
                "volume-vs-prior-mean;utc-v1;hour-gru-input11-hidden32-seed20261018;"
                "daily-d0-rolling30-output8;minute-hour-day-calendar-hourstate-d0"
            ),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=True).encode("ascii")).hexdigest()

    def names(self, scale: FeatureScale) -> tuple[str, ...]:
        if scale == "1m":
            return self.minute_names
        if scale == "1h":
            return self.hour_names
        if scale == "1d":
            return self.day_names
        if scale == "calendar":
            return self.calendar_names
        raise DataError(f"unsupported market feature scale: {scale}")


@dataclass(frozen=True, slots=True)
class MarketFeatureVector:
    scale: FeatureScale
    asof_ns: int
    names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        require_ns(self.asof_ns)
        if (self.scale not in ("1m", "1h", "1d", "calendar")
                or type(self.names) is not tuple or type(self.values) is not tuple
                or len(self.names) != len(self.values) or not self.names
                or len(set(self.names)) != len(self.names)):
            raise DataError("market feature vector names and values do not align")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in self.values):
            raise DataError("market features must be finite numeric values")

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


@dataclass(frozen=True, slots=True)
class OHLCVTarget:
    """Invertible next-bar target: open gap, body, upper/lower wick, log volume."""

    open_gap: float
    body: float
    upper_wick: float
    lower_wick: float
    log1p_volume: float

    def __post_init__(self) -> None:
        values = (self.open_gap, self.body, self.upper_wick,
                  self.lower_wick, self.log1p_volume)
        if any(not math.isfinite(value) for value in values):
            raise DataError("OHLCV target values must be finite")
        if min(self.upper_wick, self.lower_wick, self.log1p_volume) < 0:
            raise DataError("wick and log-volume target components must be nonnegative")


@dataclass(frozen=True, slots=True)
class ScaledMarketFeatureVector:
    scale: FeatureScale
    asof_ns: int
    names: tuple[str, ...]
    values: tuple[float, ...]
    clipped_count: int


@dataclass(frozen=True, slots=True)
class TrainOnlyScaler:
    """Robust scaler fitted only on explicitly bounded Train score timestamps."""

    scale: FeatureScale
    names: tuple[str, ...]
    center: tuple[float, ...]
    spread: tuple[float, ...]
    fit_start_ns: int
    fit_end_ns: int
    fit_count: int
    clip_sigma: float = 8.0

    def __post_init__(self) -> None:
        require_ns(self.fit_start_ns)
        require_ns(self.fit_end_ns)
        if (self.fit_end_ns <= self.fit_start_ns or self.fit_count < 1
                or len(self.names) != len(self.center) or len(self.names) != len(self.spread)
                or not math.isfinite(self.clip_sigma) or self.clip_sigma <= 0
                or any(not math.isfinite(value) for value in (*self.center, *self.spread))
                or any(value <= 0 for value in self.spread)):
            raise DataError("invalid frozen Train-only scaler")

    @classmethod
    def fit(cls, vectors: Sequence[MarketFeatureVector], *, train_start_ns: int,
            train_end_ns: int, clip_sigma: float = 8.0) -> TrainOnlyScaler:
        require_ns(train_start_ns)
        require_ns(train_end_ns)
        if train_end_ns <= train_start_ns or not vectors:
            raise DataError("scaler requires nonempty Train score data")
        first = vectors[0]
        if any(vector.scale != first.scale or vector.names != first.names for vector in vectors):
            raise DataError("scaler input feature schemas differ")
        if any(not train_start_ns <= vector.asof_ns < train_end_ns for vector in vectors):
            raise DataError("scaler fit timestamps must stay inside Train score interval")
        columns = tuple(tuple(vector.values[index] for vector in vectors)
                        for index in range(len(first.names)))
        centers = tuple(_quantile(column, 0.5) for column in columns)
        spreads = tuple(max((_quantile(column, 0.75) - _quantile(column, 0.25)) / 1.349,
                            1e-8) for column in columns)
        return cls(first.scale, first.names, centers, spreads, train_start_ns,
                   train_end_ns, len(vectors), clip_sigma)

    @property
    def hash(self) -> str:
        payload = {
            "scale": self.scale, "names": self.names, "center": self.center,
            "spread": self.spread, "fit_start_ns": self.fit_start_ns,
            "fit_end_ns": self.fit_end_ns, "fit_count": self.fit_count,
            "clip_sigma": self.clip_sigma,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode("ascii")).hexdigest()

    def transform(self, vector: MarketFeatureVector) -> ScaledMarketFeatureVector:
        if vector.scale != self.scale or vector.names != self.names:
            raise DataError("scaler feature schema mismatch")
        raw = tuple((value - center) / spread
                    for value, center, spread in zip(vector.values, self.center,
                                                     self.spread, strict=True))
        clipped = tuple(min(self.clip_sigma, max(-self.clip_sigma, value)) for value in raw)
        return ScaledMarketFeatureVector(
            vector.scale, vector.asof_ns, vector.names, clipped,
            sum(before != after for before, after in zip(raw, clipped, strict=True)),
        )


def encode_ohlcv_target(frame: MarketFrame | AggregatedCandle, *, previous_close: float) -> OHLCVTarget:
    if (not isinstance(frame, (MarketFrame, AggregatedCandle))
            or not math.isfinite(previous_close) or previous_close <= 0):
        raise DataError("OHLCV target needs a valid frame and positive previous close")
    return OHLCVTarget(
        open_gap=math.log(frame.open / previous_close),
        body=math.log(frame.close / frame.open),
        upper_wick=math.log(frame.high / max(frame.open, frame.close)),
        lower_wick=math.log(min(frame.open, frame.close) / frame.low),
        log1p_volume=math.log1p(frame.volume),
    )


def decode_ohlcv_target(target: OHLCVTarget, *, previous_close: float, ts_close_ns: int,
                        available_at_ns: int, instrument: str,
                        tick_size: Decimal | None = None) -> MarketFrame:
    if (not isinstance(target, OHLCVTarget) or not math.isfinite(previous_close)
            or previous_close <= 0 or not instrument):
        raise DataError("OHLCV target decode needs a valid anchor and instrument")
    require_ns(ts_close_ns)
    require_ns(available_at_ns)
    try:
        opening = previous_close * math.exp(target.open_gap)
        closing = opening * math.exp(target.body)
        high = max(opening, closing) * math.exp(target.upper_wick)
        low = min(opening, closing) * math.exp(-target.lower_wick)
        volume = math.expm1(target.log1p_volume)
    except OverflowError as exc:
        raise DataError("OHLCV target decode overflowed") from exc
    if tick_size is not None:
        if not tick_size.is_finite() or tick_size <= 0:
            raise DataError("OHLCV tick size must be finite and positive")
        opening, closing, high, low = (
            _quantize_tick(value, tick_size) for value in (opening, closing, high, low)
        )
        high = max(high, opening, closing)
        low = min(low, opening, closing)
    try:
        return MarketFrame(ts_close_ns, available_at_ns, instrument, opening, high,
                           low, closing, volume)
    except (OverflowError, ValueError) as exc:
        raise DataError("decoded OHLCV target is outside the valid market domain") from exc


def minute_features(frames: Sequence[MarketFrame], *, spec: MarketFeatureSpec | None = None
                    ) -> MarketFeatureVector:
    spec = spec or MarketFeatureSpec()
    if len(frames) < 16:
        raise DataError("minute features require 15 prior bars of warmup")
    _validate_history(frames, MINUTE_NS)
    frame = frames[-1]
    previous = frames[-2].close
    returns = _log_returns([item.close for item in frames[-16:]])
    volumes = [item.volume for item in frames]
    values = _shape(frame, previous) + (
        math.log1p(frame.volume), _realized_vol(returns[-5:]),
        _realized_vol(returns[-15:]), _trend_efficiency(returns[-5:]),
        _trend_efficiency(returns[-15:]), _relative_volume(frame.volume, volumes[-16:-1]),
    )
    return MarketFeatureVector("1m", frame.ts_close_ns, spec.minute_names, values)


def hourly_features(bars: Sequence[AggregatedCandle], *, spec: MarketFeatureSpec | None = None
                    ) -> MarketFeatureVector:
    spec = spec or MarketFeatureSpec()
    if len(bars) < 25:
        raise DataError("hour features require 24 prior complete hourly bars")
    _validate_history(bars, HOUR_NS)
    bar = bars[-1]
    returns = _log_returns([item.close for item in bars[-25:]])
    values = _shape(bar, bars[-2].close) + (math.log1p(bar.volume),
        _log_ratio(bar.close, bars[-5].close), _log_ratio(bar.close, bars[-9].close),
        _log_ratio(bar.close, bars[-25].close), _realized_vol(returns[-24:]),
        _trend_efficiency(returns[-24:]),
        _relative_volume(bar.volume, [item.volume for item in bars[-25:-1]]))
    return MarketFeatureVector("1h", bar.end_ns, spec.hour_names, values)


def daily_features(bars: Sequence[AggregatedCandle], *, spec: MarketFeatureSpec | None = None
                   ) -> MarketFeatureVector:
    spec = spec or MarketFeatureSpec()
    if len(bars) < 31:
        raise DataError("daily features require 30 prior complete daily bars")
    _validate_history(bars, DAY_NS)
    bar = bars[-1]
    returns = _log_returns([item.close for item in bars[-31:]])
    values = _shape(bar, bars[-2].close) + (math.log1p(bar.volume),
        _log_ratio(bar.close, bars[-8].close), _log_ratio(bar.close, bars[-31].close),
        _realized_vol(returns[-7:]), _realized_vol(returns[-30:]),
        _trend_efficiency(returns[-7:]), _trend_efficiency(returns[-30:]),
        _relative_volume(bar.volume, [item.volume for item in bars[-31:-1]]))
    return MarketFeatureVector("1d", bar.end_ns, spec.day_names, values)


def calendar_features(ts_ns: int, *, spec: MarketFeatureSpec | None = None
                      ) -> MarketFeatureVector:
    require_ns(ts_ns)
    spec = spec or MarketFeatureSpec()
    stamp = datetime.fromtimestamp(ts_ns // 1_000_000_000, tz=timezone.utc)
    minute_of_day = stamp.hour * 60 + stamp.minute
    weekday = stamp.weekday()
    values = (
        math.sin(2 * math.pi * minute_of_day / 1440),
        math.cos(2 * math.pi * minute_of_day / 1440),
        math.sin(2 * math.pi * weekday / 7), math.cos(2 * math.pi * weekday / 7),
        float(ts_ns % HOUR_NS == 0), float(ts_ns % DAY_NS == 0),
    )
    return MarketFeatureVector("calendar", ts_ns, spec.calendar_names, values)


def _shape(bar: MarketFrame | AggregatedCandle, previous_close: float) -> tuple[float, ...]:
    return (
        _log_ratio(bar.close, previous_close), _log_ratio(bar.open, previous_close),
        math.log(bar.high / max(bar.open, bar.close)),
        math.log(min(bar.open, bar.close) / bar.low),
    )


def _validate_history(bars: Sequence[MarketFrame] | Sequence[AggregatedCandle],
                      period_ns: int) -> None:
    if any(not isinstance(bar, (MarketFrame, AggregatedCandle)) for bar in bars):
        raise DataError("feature history contains an unsupported bar type")
    if any(_instrument(right) != _instrument(left)
           for left, right in zip(bars, bars[1:])):
        raise DataError("feature history cannot mix instruments")
    timestamps = [bar.ts_close_ns if isinstance(bar, MarketFrame) else bar.end_ns
                  for bar in bars]
    if any(right - left != period_ns for left, right in zip(timestamps, timestamps[1:])):
        raise DataError("feature history must be contiguous at its declared scale")
    if any(isinstance(bar, AggregatedCandle) and bar.end_ns - bar.start_ns != period_ns
           for bar in bars):
        raise DataError("aggregate feature history period differs from its scale")


def _instrument(bar: object) -> str:
    if isinstance(bar, (MarketFrame, AggregatedCandle)):
        return bar.instrument
    raise DataError("feature history contains an unsupported bar type")


def _log_returns(closes: Sequence[float]) -> tuple[float, ...]:
    if len(closes) < 2 or any(not math.isfinite(value) or value <= 0 for value in closes):
        raise DataError("feature returns require at least two positive closes")
    return tuple(_log_ratio(right, left) for left, right in zip(closes, closes[1:]))


def _log_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or min(numerator, denominator) <= 0:
        raise DataError("log-ratio feature requires positive finite prices")
    return math.log(numerator / denominator)


def _realized_vol(returns: Sequence[float]) -> float:
    if not returns:
        raise DataError("realized volatility requires returns")
    return math.sqrt(sum(value * value for value in returns) / len(returns))


def _trend_efficiency(returns: Sequence[float]) -> float:
    path = sum(abs(value) for value in returns)
    return 0.0 if path == 0 else min(1.0, abs(sum(returns)) / path)


def _relative_volume(current: float, prior: Sequence[float]) -> float:
    if not prior or any(not math.isfinite(value) or value < 0 for value in prior):
        raise DataError("relative volume requires nonnegative prior volume history")
    return math.log((current + 1e-12) / (sum(prior) / len(prior) + 1e-12))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered or not 0 <= probability <= 1:
        raise DataError("quantile requires nonempty values and probability in [0,1]")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _quantize_tick(value: float, tick_size: Decimal) -> float:
    if not math.isfinite(value) or value <= 0:
        raise DataError("decoded price must be finite and positive")
    try:
        units = (Decimal(str(value)) / tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise DataError("tick quantization exceeded the supported numeric range") from exc
    quantized = units * tick_size
    if quantized <= 0:
        quantized = tick_size
    return float(quantized)
