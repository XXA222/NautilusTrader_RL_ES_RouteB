"""Causal multi-horizon market risk targets and path-derived forecast scores.

The helpers accept in-memory bars only. Callers remain responsible for obtaining
the bars through the authorized Train or frozen Validation reader. All predicted
return, volatility, extrema, and barrier metrics are derived from the same OHLCV
path ensemble so their probabilities remain mutually consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .market import MINUTE_NS


def estimate_train_barrier_fractions(
    train_closes: np.ndarray,
    *,
    train_timestamps_ns: np.ndarray,
    horizons: tuple[int, ...] = (1, 5, 15, 30, 60),
    volatility_quantile: float = 0.75,
) -> np.ndarray:
    """Fit one barrier per horizon from the Train-only realized-volatility quantile."""
    closes=np.asarray(train_closes,dtype=np.float64)
    raw_times=np.asarray(train_timestamps_ns)
    if (closes.ndim!=1 or closes.size<2 or not np.isfinite(closes).all()
            or (closes<=0).any() or raw_times.shape!=closes.shape
            or not np.issubdtype(raw_times.dtype,np.integer)):
        raise ValueError("Train closes and integer timestamps must be aligned")
    times=raw_times.astype(np.int64,copy=False)
    if not np.all(np.diff(times)==MINUTE_NS):
        raise ValueError("Train barrier fitting requires a contiguous one-minute score window")
    if (not horizons or tuple(sorted(set(horizons)))!=horizons
            or any(type(h) is not int or h<1 for h in horizons)
            or not np.isfinite(volatility_quantile)
            or not 0.5<volatility_quantile<1):
        raise ValueError("invalid horizon or pre-registered volatility quantile")
    returns=np.log(closes[1:]/closes[:-1])
    thresholds=np.empty(len(horizons),dtype=np.float64)
    for index,horizon in enumerate(horizons):
        if returns.size<horizon:
            raise ValueError(f"Train score lacks {horizon} returns for barrier fitting")
        rv=np.asarray([np.sqrt(np.square(returns[end-horizon+1:end+1]).sum())
                       for end in range(horizon-1,returns.size)],dtype=np.float64)
        threshold=float(np.expm1(np.quantile(rv,volatility_quantile)))
        thresholds[index]=np.clip(threshold,1e-6,0.95)
    return thresholds


def _barriers_by_anchor_horizon(value: float | tuple[float, ...] | np.ndarray, anchors: int,
                                horizons: int) -> np.ndarray:
    barrier=np.asarray(value,dtype=np.float64)
    if barrier.ndim==0:
        barrier=np.full((anchors,horizons),float(barrier))
    elif barrier.shape==(horizons,):
        barrier=np.broadcast_to(barrier[None,:],(anchors,horizons)).copy()
    elif barrier.shape==(anchors,horizons):
        barrier=barrier.copy()
    elif barrier.shape==(anchors,):
        barrier=np.broadcast_to(barrier[:,None],(anchors,horizons)).copy()
    else:
        raise ValueError("barrier fraction must be scalar, per-anchor, per-horizon, or [N,H]")
    if not np.isfinite(barrier).all() or (barrier<=0).any() or (barrier>=1).any():
        raise ValueError("barrier fraction must be finite and in (0,1)")
    return barrier


@dataclass(frozen=True)
class RiskTargets:
    """Targets for each anchor/horizon; invalid horizons are zero-masked."""

    horizons: tuple[int, ...]
    valid: np.ndarray  # [N,H], no gaps and complete future path
    log_return: np.ndarray  # [N,H]
    realized_volatility: np.ndarray  # [N,H], sqrt(sum(close returns ** 2))
    max_high_return: np.ndarray  # [N,H], relative to anchor close
    min_low_return: np.ndarray  # [N,H], relative to anchor close
    upper_barrier_hit: np.ndarray  # [N,H] bool
    lower_barrier_hit: np.ndarray  # [N,H] bool


def build_risk_targets(
    future_ohlcv: np.ndarray,
    *,
    anchor_close: np.ndarray,
    anchor_timestamps_ns: np.ndarray,
    future_timestamps_ns: np.ndarray,
    horizons: tuple[int, ...] = (1, 5, 15, 30, 60),
    barrier_fraction: float | tuple[float, ...] | np.ndarray = 0.01,
) -> RiskTargets:
    """Build per-anchor targets without filling over missing bars.

    `future_ohlcv` is [N,T,5] ordered as open/high/low/close/volume. Timestamp
    arrays contain close times. A target is valid only if every minute through
    that horizon is present and follows its anchor exactly.
    """
    bars = np.asarray(future_ohlcv, dtype=np.float64)
    anchors = np.asarray(anchor_close, dtype=np.float64)
    raw_anchor_ts = np.asarray(anchor_timestamps_ns)
    raw_future_ts = np.asarray(future_timestamps_ns)
    if (not np.issubdtype(raw_anchor_ts.dtype, np.integer)
            or not np.issubdtype(raw_future_ts.dtype, np.integer)):
        raise ValueError("market timestamps must be integer nanoseconds")
    anchor_ts = raw_anchor_ts.astype(np.int64, copy=False)
    future_ts = raw_future_ts.astype(np.int64, copy=False)
    if (bars.ndim != 3 or bars.shape[0] < 1 or bars.shape[1] < 1
            or bars.shape[2] != 5 or anchors.shape != (bars.shape[0],)
            or anchor_ts.shape != anchors.shape or future_ts.shape != bars.shape[:2]):
        raise ValueError("expected aligned OHLCV [N,T,5], anchors [N], and timestamps [N,T]")
    if (not np.isfinite(bars).all() or not np.isfinite(anchors).all()
            or (bars[..., :4] <= 0).any() or (bars[..., 4] < 0).any()
            or (anchors <= 0).any()
            or (bars[..., 1] < np.maximum(bars[..., 0], bars[..., 3])).any()
            or (bars[..., 2] > np.minimum(bars[..., 0], bars[..., 3])).any()):
        raise ValueError("risk targets require finite geometrically valid OHLCV")
    if (not horizons or any(type(h) is not int or h < 1 for h in horizons)
            or tuple(sorted(set(horizons))) != horizons):
        raise ValueError("horizons must be strictly increasing positive integers")
    barrier = _barriers_by_anchor_horizon(barrier_fraction,bars.shape[0],len(horizons))

    n, width, _ = bars.shape
    # `valid_prefix[h-1]` is false once a missing/delayed/out-of-order bar occurs.
    expected = anchor_ts[:, None] + np.arange(1, width + 1, dtype=np.int64)[None, :] * MINUTE_NS
    contiguous = future_ts == expected
    valid_prefix = np.logical_and.accumulate(contiguous, axis=1)
    close_returns = np.empty((n, width), dtype=np.float64)
    previous = np.concatenate([anchors[:, None], bars[:, :-1, 3]], axis=1)
    close_returns[:] = np.log(bars[..., 3] / previous)
    cumulative_return = np.cumsum(close_returns, axis=1)
    cumulative_rv = np.sqrt(np.cumsum(np.square(close_returns), axis=1))
    previous_return = np.concatenate([np.zeros((n, 1)), cumulative_return[:, :-1]], axis=1)
    high_return = previous_return + np.log(bars[..., 1] / previous)
    low_return = previous_return + np.log(bars[..., 2] / previous)
    running_high = np.maximum.accumulate(high_return, axis=1)
    running_low = np.minimum.accumulate(low_return, axis=1)

    valid = np.zeros((n, len(horizons)), dtype=bool)
    returns = np.zeros_like(valid, dtype=np.float64)
    rv = np.zeros_like(returns)
    max_high = np.zeros_like(returns)
    min_low = np.zeros_like(returns)
    up_hit = np.zeros_like(valid)
    down_hit = np.zeros_like(valid)
    for index, horizon in enumerate(horizons):
        if horizon > width:
            continue
        ok = valid_prefix[:, horizon - 1]
        valid[:, index] = ok
        returns[:, index] = np.where(ok, cumulative_return[:, horizon - 1], 0.0)
        rv[:, index] = np.where(ok, cumulative_rv[:, horizon - 1], 0.0)
        max_high[:, index] = np.where(ok, running_high[:, horizon - 1], 0.0)
        min_low[:, index] = np.where(ok, running_low[:, horizon - 1], 0.0)
        up_hit[:, index] = ok & (running_high[:, horizon - 1] >= np.log1p(barrier[:,index]))
        down_hit[:, index] = ok & (running_low[:, horizon - 1] <= np.log1p(-barrier[:,index]))
    return RiskTargets(horizons, valid, returns, rv, max_high, min_low, up_hit, down_hit)


def path_log_realized_volatility_crps_per_anchor(
    forecast_log_returns: np.ndarray,
    realized_log_returns: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    """Return per-anchor CRPS for log-RV derived from complete return paths."""
    from .evaluation import _crps_per_anchor

    forecast = np.asarray(forecast_log_returns, dtype=np.float64)
    realized = np.asarray(realized_log_returns, dtype=np.float64)
    if (type(horizon) is not int or horizon < 1
            or forecast.ndim != 3 or forecast.shape[0] < 1 or forecast.shape[1] < 2
            or forecast.shape[2] < 1 or realized.ndim != 2
            or realized.shape[0] != forecast.shape[0]
            or realized.shape[1] < horizon
            or forecast.shape[2] < horizon
            or not np.isfinite(forecast).all() or not np.isfinite(realized).all()):
        raise ValueError("aligned finite [N,P>=2,T] forecasts and [N,T] outcomes required")
    predicted = np.log(np.sqrt(np.square(forecast[:, :, :horizon]).sum(axis=-1)) + 1e-8)
    observed = np.log(np.sqrt(np.square(realized[:, :horizon]).sum(axis=-1)) + 1e-8)
    return _crps_per_anchor(predicted, observed)


def score_path_risk(
    forecast_ohlcv: np.ndarray,
    realized_ohlcv: np.ndarray,
    *,
    anchor_close: np.ndarray,
    anchor_timestamps_ns: np.ndarray,
    future_timestamps_ns: np.ndarray,
    horizons: tuple[int, ...] = (1, 5, 15, 30, 60),
    barrier_fraction: float | tuple[float, ...] | np.ndarray = 0.01,
) -> dict[str, object]:
    """Score return/RV/extreme CRPS and barrier Brier from common path draws."""
    from .evaluation import _crps_per_anchor  # local import avoids a module cycle

    forecast = np.asarray(forecast_ohlcv, dtype=np.float64)
    realized = np.asarray(realized_ohlcv, dtype=np.float64)
    anchors = np.asarray(anchor_close, dtype=np.float64)
    if (forecast.ndim != 4 or forecast.shape[0] < 1 or forecast.shape[1] < 2
            or forecast.shape[2] < 1 or forecast.shape[-1] != 5
            or realized.shape != (forecast.shape[0], forecast.shape[2], 5)):
        raise ValueError("expected forecasts [N,P,T,5] and realized paths [N,T,5]")
    if not np.isfinite(forecast).all():
        raise ValueError("forecast paths must be finite")
    n, paths, width, _ = forecast.shape
    target = build_risk_targets(
        realized, anchor_close=anchors,
        anchor_timestamps_ns=anchor_timestamps_ns,
        future_timestamps_ns=future_timestamps_ns,
        horizons=horizons, barrier_fraction=barrier_fraction)
    if (np.any(forecast[..., :4] <= 0)
            or np.any(forecast[..., 1] < np.maximum(forecast[..., 0], forecast[..., 3]))
            or np.any(forecast[..., 2] > np.minimum(forecast[..., 0], forecast[..., 3]))
            or np.any(forecast[..., 4] < 0)):
        raise ValueError("forecast paths contain invalid OHLCV geometry")
    barrier = _barriers_by_anchor_horizon(barrier_fraction,n,len(horizons))
    previous = np.concatenate([np.broadcast_to(anchors[:, None, None], (n, paths, 1)),
                               forecast[:, :, :-1, 3]], axis=2)
    step_returns = np.log(forecast[..., 3] / previous)
    cumulative = np.cumsum(step_returns, axis=2)
    rv = np.sqrt(np.cumsum(np.square(step_returns), axis=2))
    previous_cumulative = np.concatenate([np.zeros((n, paths, 1)), cumulative[:, :, :-1]], axis=2)
    previous_close = previous
    highs = previous_cumulative + np.log(forecast[..., 1] / previous_close)
    lows = previous_cumulative + np.log(forecast[..., 2] / previous_close)
    max_high = np.maximum.accumulate(highs, axis=2)
    min_low = np.minimum.accumulate(lows, axis=2)
    report: dict[str, object] = {}
    for i, horizon in enumerate(horizons):
        valid = target.valid[:, i]
        if not valid.any():
            report[str(horizon)] = {"samples": 0, "status": "NO_VALID_TARGETS"}
            continue
        if horizon > width:
            raise ValueError("forecast horizon exceeds path width")
        predicted_return = cumulative[:, :, horizon - 1]
        predicted_rv = rv[:, :, horizon - 1]
        predicted_high = max_high[:, :, :horizon].max(axis=2)
        predicted_low = min_low[:, :, :horizon].min(axis=2)
        predicted_up = (predicted_high >= np.log1p(barrier[:, i, None])).mean(axis=1)
        predicted_down = (predicted_low <= np.log1p(-barrier[:, i, None])).mean(axis=1)
        truth_return = target.log_return[:, i]
        truth_rv = target.realized_volatility[:, i]
        truth_up = target.upper_barrier_hit[:, i].astype(float)
        truth_down = target.lower_barrier_hit[:, i].astype(float)
        item = {
            "samples": int(valid.sum()),
            "return_crps": float(_crps_per_anchor(predicted_return[valid], truth_return[valid]).mean()),
            "realized_volatility_crps": float(_crps_per_anchor(predicted_rv[valid], truth_rv[valid]).mean()),
            "max_high_return_crps": float(_crps_per_anchor(predicted_high[valid], target.max_high_return[valid, i]).mean()),
            "min_low_return_crps": float(_crps_per_anchor(predicted_low[valid], target.min_low_return[valid, i]).mean()),
            "upper_barrier_brier": float(np.mean(np.square(predicted_up[valid] - truth_up[valid]))),
            "lower_barrier_brier": float(np.mean(np.square(predicted_down[valid] - truth_down[valid]))),
            "upper_barrier_observed_rate": float(truth_up[valid].mean()),
            "lower_barrier_observed_rate": float(truth_down[valid].mean()),
        }
        if not all(math.isfinite(value) for value in item.values() if isinstance(value, float)):
            raise RuntimeError("path risk scoring generated a nonfinite metric")
        report[str(horizon)] = item
    return {
        "schema": "routeb.path-risk-scores.v1",
        "barrier_fraction": barrier.tolist(),
        "barrier_fraction_by_anchor_and_horizon": barrier.tolist(),
        "metrics_from_shared_path_ensemble": report,
        "score_capability": "TRAIN_DIAGNOSTIC",
    }


def score_native_risk_distribution(
    mean: np.ndarray,
    scale: np.ndarray,
    barrier_logits: np.ndarray,
    realized_ohlcv: np.ndarray,
    *,
    anchor_close: np.ndarray,
    anchor_timestamps_ns: np.ndarray,
    future_timestamps_ns: np.ndarray,
    horizons: tuple[int, ...] = (1, 5, 15, 30, 60),
    barrier_fraction: float | tuple[float, ...] | np.ndarray = 0.01,
) -> dict[str, object]:
    """Score direct prior-native Gaussian risk heads against aligned observations.

    Returns aggregate proper scores plus private per-anchor score arrays for
    paired block-bootstrap comparisons. Channels are return, realized
    volatility, maximum high excursion and minimum low excursion; barriers are
    upper then lower. Nothing in this function reads raw market files.
    """
    predicted_mean=np.asarray(mean,dtype=np.float64)
    predicted_scale=np.asarray(scale,dtype=np.float64)
    logits=np.asarray(barrier_logits,dtype=np.float64)
    n=len(np.asarray(anchor_close))
    expected=(n,len(horizons),4)
    if (predicted_mean.shape!=expected or predicted_scale.shape!=expected
            or logits.shape!=(n,len(horizons),2)
            or not np.isfinite(predicted_mean).all()
            or not np.isfinite(predicted_scale).all() or np.any(predicted_scale<=0)
            or not np.isfinite(logits).all()):
        raise ValueError("native risk forecasts must be aligned finite [N,H,4]/[N,H,2]")
    targets=build_risk_targets(
        realized_ohlcv,anchor_close=anchor_close,
        anchor_timestamps_ns=anchor_timestamps_ns,
        future_timestamps_ns=future_timestamps_ns,horizons=horizons,
        barrier_fraction=barrier_fraction)
    # The second E2 training channel is log(realized_volatility + 1e-8),
    # matching the target construction in sequences.py. Keep validation in
    # that same space; scoring it against raw volatility is dimensionally wrong.
    truth=np.stack((targets.log_return,np.log(targets.realized_volatility+1e-8),
                    targets.max_high_return,targets.min_low_return),axis=-1)
    per_anchor:dict[str,dict[str,np.ndarray]]={}
    report:dict[str,dict[str,float|int]]={}
    metric_names=("return_crps","log_realized_volatility_crps",
                  "max_high_return_crps","min_low_return_crps")
    for index,horizon in enumerate(horizons):
        valid=targets.valid[:,index]
        if not valid.any():
            report[str(horizon)]={"samples":0}
            continue
        z=(truth[:,index,:]-predicted_mean[:,index,:])/predicted_scale[:,index,:]
        cdf=0.5*(1.0+np.fromiter(
            (math.erf(float(value)/math.sqrt(2.0)) for value in z.ravel()),
            dtype=np.float64,count=z.size).reshape(z.shape))
        density=np.exp(-0.5*np.square(z))/math.sqrt(2.0*math.pi)
        crps=predicted_scale[:,index,:]*(
            z*(2.0*cdf-1.0)+2.0*density-1.0/math.sqrt(math.pi))
        probabilities=1.0/(1.0+np.exp(-np.clip(logits[:,index,:],-60.0,60.0)))
        barrier_truth=np.stack((targets.upper_barrier_hit[:,index],
                                targets.lower_barrier_hit[:,index]),axis=-1).astype(np.float64)
        scores={name:crps[valid,channel].copy()
                for channel,name in enumerate(metric_names)}
        upper_brier=np.square(probabilities[valid,0]-barrier_truth[valid,0])
        lower_brier=np.square(probabilities[valid,1]-barrier_truth[valid,1])
        per_anchor[str(horizon)]={**scores,"upper_barrier_brier":upper_brier,
                                  "lower_barrier_brier":lower_brier,
                                  "valid_mask":valid.copy()}
        report[str(horizon)]={
            "samples":int(valid.sum()),
            **{name:float(values.mean()) for name,values in scores.items()},
            "upper_barrier_brier":float(upper_brier.mean()),
            "lower_barrier_brier":float(lower_brier.mean()),
            "return_coverage_90":float((np.abs(z[valid,0])<=1.6448536269514722).mean()),
            "return_coverage_95":float((np.abs(z[valid,0])<=1.959963984540054).mean()),
            "upper_barrier_observed_rate":float(barrier_truth[valid,0].mean()),
            "lower_barrier_observed_rate":float(barrier_truth[valid,1].mean()),
        }
    return {"schema":"routeb.native-risk-scores.v2","horizons":list(horizons),
            "continuous_target_space":["log_return","log_realized_volatility",
                                       "max_high_return","min_low_return"],
            "metrics":report,"_per_anchor_scores":per_anchor,
            "score_capability":"REGISTERED_VALIDATION"}
