import numpy as np
import pytest

from rles_world_model.routeb.market import MINUTE_NS
from rles_world_model.routeb.risk_targets import (
    build_risk_targets,
    estimate_train_barrier_fractions,
    path_log_realized_volatility_crps_per_anchor,
    score_native_risk_distribution,
    score_path_risk,
)
from rles_world_model.routeb.evaluation import paired_block_bootstrap_score_differences

pytestmark = pytest.mark.unit


def _bars(closes: np.ndarray) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([closes[:, :1] * 0 + 100.0, closes[:, :-1]], axis=1)
    highs = np.maximum(opens, closes) * 1.01
    lows = np.minimum(opens, closes) * 0.99
    volumes = np.ones_like(closes)
    return np.stack([opens, highs, lows, closes, volumes], axis=-1)


def test_risk_targets_horizons_and_gaps_are_masked_per_horizon():
    anchors = np.array([100.0, 100.0])
    anchor_ts = np.array([MINUTE_NS * 100, MINUTE_NS * 200], dtype=np.int64)
    times = anchor_ts[:, None] + np.arange(1, 5, dtype=np.int64)[None, :] * MINUTE_NS
    times[1, 2:] += MINUTE_NS
    bars = _bars(np.array([[101.0, 100.5, 102.0, 103.0],
                           [98.0, 97.0, 96.0, 95.0]]))
    targets = build_risk_targets(
        bars, anchor_close=anchors, anchor_timestamps_ns=anchor_ts,
        future_timestamps_ns=times, horizons=(1, 2, 3, 4), barrier_fraction=0.02)
    assert targets.valid.tolist() == [[True, True, True, True], [True, True, False, False]]
    assert targets.log_return[0, 0] > 0
    assert targets.realized_volatility[0, 1] > targets.realized_volatility[0, 0]
    assert targets.upper_barrier_hit[0, 0]
    assert targets.lower_barrier_hit[1, 0]
    assert targets.log_return[1, 2] == 0  # invalid labels have zero and false mask


def test_path_risk_metrics_derive_from_shared_valid_ohlcv_paths():
    anchor = np.array([100.0])
    anchor_ts = np.array([MINUTE_NS * 100], dtype=np.int64)
    times = anchor_ts[:, None] + np.arange(1, 4, dtype=np.int64)[None, :] * MINUTE_NS
    realized = _bars(np.array([[101.0, 99.0, 102.0]]))
    forecast = np.repeat(realized[:, None, :, :], 4, axis=1)
    report = score_path_risk(
        forecast, realized, anchor_close=anchor, anchor_timestamps_ns=anchor_ts,
        future_timestamps_ns=times, horizons=(1, 2, 3), barrier_fraction=0.01)
    scores = report["metrics_from_shared_path_ensemble"]
    assert report["score_capability"] == "TRAIN_DIAGNOSTIC"
    assert scores["1"]["return_crps"] == 0
    assert scores["1"]["realized_volatility_crps"] == 0
    assert scores["1"]["upper_barrier_brier"] == 0
    assert scores["3"]["samples"] == 1


def test_path_log_rv_crps_is_scored_per_anchor_in_log_target_space():
    realized=np.asarray([[.001,-.002,.003],[.002,.001,-.001]])
    forecasts=np.repeat(realized[:,None,:],4,axis=1)
    scores=path_log_realized_volatility_crps_per_anchor(
        forecasts,realized,horizon=3)
    assert scores.shape==(2,)
    assert np.allclose(scores,0.0,atol=1e-15)
    with pytest.raises(ValueError,match="aligned finite"):
        path_log_realized_volatility_crps_per_anchor(
            forecasts,realized,horizon=True)


def test_native_prior_risk_distribution_scores_each_target_and_bootstraps_by_blocks():
    n,width=12,3
    anchors=100*np.ones(n)
    anchor_ts=np.arange(n,dtype=np.int64)*120*MINUTE_NS+1000*MINUTE_NS
    times=anchor_ts[:,None]+np.arange(1,width+1,dtype=np.int64)[None,:]*MINUTE_NS
    closes=np.stack([100*np.exp((row+1)*np.asarray([.0002,-.0001,.0003]))
                     for row in range(n)])
    realized=_bars(closes)
    targets=build_risk_targets(realized,anchor_close=anchors,
        anchor_timestamps_ns=anchor_ts,future_timestamps_ns=times,
        horizons=(1,2,3),barrier_fraction=(.001,.002,.003))
    mean=np.stack((targets.log_return,np.log(targets.realized_volatility+1e-8),
                   targets.max_high_return,targets.min_low_return),axis=-1)
    scale=np.full_like(mean,.001)
    labels=np.stack((targets.upper_barrier_hit,targets.lower_barrier_hit),axis=-1)
    logits=np.where(labels,4.0,-4.0)
    result=score_native_risk_distribution(mean,scale,logits,realized,
        anchor_close=anchors,anchor_timestamps_ns=anchor_ts,
        future_timestamps_ns=times,horizons=(1,2,3),
        barrier_fraction=(.001,.002,.003))
    assert result["score_capability"]=="REGISTERED_VALIDATION"
    assert result["metrics"]["2"]["samples"]==n
    assert result["schema"]=="routeb.native-risk-scores.v2"
    assert result["continuous_target_space"][1]=="log_realized_volatility"
    expected_centered_normal_crps=.001*(2/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    assert result["metrics"]["2"]["log_realized_volatility_crps"]==pytest.approx(
        expected_centered_normal_crps,abs=1e-10)
    per_anchor=result["_per_anchor_scores"]["2"]["log_realized_volatility_crps"]
    baseline=per_anchor+.01
    difference=paired_block_bootstrap_score_differences(
        per_anchor,baseline,anchor_timestamps_ns=anchor_ts,
        anchor_interval_ns=120*MINUTE_NS,horizon=2,block_length=2,
        replicates=100,seed=7,candidate_name="E2",baseline_name="E0")
    assert difference.mean_difference==pytest.approx(-.01)
    assert difference.replicates==100


def test_invalid_ohlcv_and_nonpositive_barrier_are_rejected():
    anchor = np.array([100.0])
    anchor_ts = np.array([MINUTE_NS], dtype=np.int64)
    times = anchor_ts[:, None] + np.array([[MINUTE_NS]], dtype=np.int64)
    bad = _bars(np.array([[101.0]]))
    bad[0, 0, 1] = 99.0
    try:
        build_risk_targets(bad, anchor_close=anchor, anchor_timestamps_ns=anchor_ts,
                           future_timestamps_ns=times, horizons=(1,))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid OHLC geometry was accepted")
    good = _bars(np.array([[101.0]]))
    try:
        build_risk_targets(good, anchor_close=anchor, anchor_timestamps_ns=anchor_ts,
                           future_timestamps_ns=times, horizons=(1,), barrier_fraction=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero barrier fraction was accepted")


def test_barriers_are_train_volatility_quantiles_by_horizon():
    timestamps=np.arange(1,181,dtype=np.int64)*MINUTE_NS
    closes=100*np.exp(np.sin(np.arange(180)/5)*0.001)
    fitted=estimate_train_barrier_fractions(
        closes,train_timestamps_ns=timestamps,horizons=(1,5,15,30,60),
        volatility_quantile=0.75)
    assert fitted.shape==(5,)
    assert np.isfinite(fitted).all() and np.all((fitted>0)&(fitted<1))
    assert np.all(np.diff(fitted)>=0)
    with pytest.raises(ValueError,match="contiguous"):
        estimate_train_barrier_fractions(
            closes,train_timestamps_ns=timestamps+np.arange(180,dtype=np.int64),
            horizons=(1,))


def test_horizon_specific_barriers_are_applied_to_matching_targets():
    anchors=np.asarray([100.0])
    anchor_ts=np.asarray([100*MINUTE_NS],dtype=np.int64)
    times=anchor_ts[:,None]+np.arange(1,3,dtype=np.int64)[None,:]*MINUTE_NS
    bars=_bars(np.asarray([[101.0,102.0]]))
    target=build_risk_targets(
        bars,anchor_close=anchors,anchor_timestamps_ns=anchor_ts,
        future_timestamps_ns=times,horizons=(1,2),barrier_fraction=np.asarray([.01,.20]))
    assert target.upper_barrier_hit.tolist()==[[True,False]]
