"""Action-free prior forecasting API for an already prepared causal feature stream."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rles_data.core import DataError

from .contracts import MarketFrame
from .features import MarketFeatureSpec
from .forecast import MarketForecastPath, decode_prior_paths, generated_scale_state_updater
from .learner import FORECAST_SUPPORT


@dataclass(frozen=True, slots=True)
class PriorForecast:
    status: str
    forecast_type: str
    instrument: str
    origin_close_ns: int
    feature_spec_hash: str
    seed: int
    paths: tuple[MarketForecastPath, ...]


class WorldModelRuntime:
    """Run posterior warmup then prior-only futures from one closed market context."""

    def __init__(self, learner: Any, *, bundle_manifest: dict[str, Any],
                 spec: MarketFeatureSpec | None = None) -> None:
        self.learner = learner
        self.spec = spec or MarketFeatureSpec()
        if (not isinstance(bundle_manifest, dict)
                or bundle_manifest.get("schema_version") != "rles.world_model.bundle.v1"
                or bundle_manifest.get("feature_spec_hash") != self.spec.hash
                or bundle_manifest.get("forecast_support") != FORECAST_SUPPORT):
            raise DataError("world-model bundle and feature schema are incompatible")
        self.bundle_manifest = bundle_manifest

    def forecast(self, observations: np.ndarray, *, origin: MarketFrame,
                 history: tuple[MarketFrame, ...], steps: int,
                 paths: int = 128, seed: int = 0, tick_size: float | None = None) -> PriorForecast:
        if not isinstance(origin, MarketFrame):
            raise DataError("prior forecast requires one closed, available market origin")
        if origin.instrument != self.bundle_manifest.get("instrument"):
            raise DataError("forecast instrument differs from immutable model bundle")
        values = np.asarray(observations, dtype=np.float32)
        if (values.ndim != 3 or values.shape[0] != 1 or values.shape[1] < 1
                or not np.isfinite(values).all()):
            raise DataError("causal features must have shape [1, time, feature] and be finite")
        if self.bundle_manifest.get("feature_count") != values.shape[-1]:
            raise DataError("feature count differs from the trained model bundle")
        if type(history) is not tuple or not history or history[-1] != origin:
            raise DataError("deployment forecast requires causal market history ending at its origin")
        carry, _ = self.learner.infer_posterior(values, seed=seed)
        hour_state = values[:, -1, 39:71]
        day_state = values[:, -1, 71:79]
        update_slow_state = generated_scale_state_updater(
            history, origin=origin, initial_hour_state=hour_state[0],
            initial_day_state=day_state[0], paths=paths, tick_size=tick_size,
        )
        samples, _ = self.learner.prior_forecast(
            carry, steps=steps, paths=paths, seed=seed ^ 0x4A17,
            initial_scale_state=(hour_state, day_state),
            scale_state_updater=update_slow_state,
        )
        decoded = decode_prior_paths(
            np.asarray(samples), previous_close=origin.close,
            last_close_ns=origin.ts_close_ns,
            available_from_ns=origin.available_at_ns,
            instrument=origin.instrument, tick_size=tick_size,
        )
        return PriorForecast(
            status="COMPLETED", forecast_type="prior_only",
            instrument=origin.instrument, origin_close_ns=origin.ts_close_ns,
            feature_spec_hash=self.spec.hash, seed=seed, paths=decoded,
        )
