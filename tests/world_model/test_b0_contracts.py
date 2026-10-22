from __future__ import annotations

import math
import random

import pytest

from rles_data.core import DataError, FoldManifest, MINUTE_NS, Window
from rles_world_model.baselines import score_baselines, stationary_block_bootstrap
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_world_model.data_audit import audit_replay
from rles_world_model.train_dataset import causal_frames_as_of


def frame(index: int, *, available_delay: int = 0, close: float | None = None) -> MarketFrame:
    value = float(close if close is not None else 100 + index * 0.1 + math.sin(index / 4))
    ts = index * MINUTE_NS
    return MarketFrame(ts, ts + available_delay, 'ETHUSDT-PERP.SIM',
                       value, value * 1.001, value * 0.999, value, 1.0)


def fold() -> FoldManifest:
    return FoldManifest(
        '0.4.0', 'a' * 64, 'b' * 64,
        Window(MINUTE_NS, MINUTE_NS, 70 * MINUTE_NS),
        Window(70 * MINUTE_NS, 130 * MINUTE_NS, 140 * MINUTE_NS),
        Window(140 * MINUTE_NS, 140 * MINUTE_NS, 150 * MINUTE_NS),
        0, 0, None,
    )


def test_causal_history_hides_future_and_not_yet_available_bars() -> None:
    frames = (frame(1), frame(2, available_delay=MINUTE_NS), frame(3))
    visible = causal_frames_as_of(frames, decision_ts_ns=2 * MINUTE_NS)
    assert tuple(item.ts_close_ns for item in visible) == (MINUTE_NS,)


def test_replay_audit_rejects_gap_and_duplicate_timestamps() -> None:
    gap_report = audit_replay(MarketReplay((frame(1), frame(3)), 'd' * 64))
    assert gap_report.missing_minutes == 1
    assert not gap_report.market_model_eligible
    duplicate_report = audit_replay(MarketReplay((frame(1), frame(1)), 'd' * 64))
    assert duplicate_report.duplicate_minutes == 1
    assert not duplicate_report.market_model_eligible


def test_market_frame_rejects_unavailable_or_invalid_ohlc() -> None:
    with pytest.raises(DataError, match='before its close'):
        frame(1, available_delay=-1)
    with pytest.raises(DataError, match='OHLC bounds'):
        MarketFrame(MINUTE_NS, MINUTE_NS, 'ETHUSDT-PERP.SIM', 100, 100, 101, 102, 1)


def test_baselines_use_frozen_fold_and_seeded_sampling() -> None:
    replay = MarketReplay(tuple(frame(index) for index in range(1, 150)), 'a' * 64)
    first = score_baselines(replay, fold(), seed=19, ensemble_size=64,
                            volatility_lookback=10, bootstrap_block=8)
    second = score_baselines(replay, fold(), seed=19, ensemble_size=64,
                             volatility_lookback=10, bootstrap_block=8)
    assert first == second
    assert {result.name for result in first} == {
        'S0_ZERO_DRIFT', 'S1_CONDITIONAL_VOL', 'S2_BLOCK_BOOTSTRAP',
    }
    assert all(result.observations == 10 and result.mean_crps >= 0 for result in first)


def test_stationary_bootstrap_is_seeded_and_preserves_source_values() -> None:
    returns = tuple(float(index) for index in range(16))
    first = stationary_block_bootstrap(returns, paths=8, horizon=5,
                                       mean_block_length=4, rng=random.Random(99))
    second = stationary_block_bootstrap(returns, paths=8, horizon=5,
                                        mean_block_length=4, rng=random.Random(99))
    assert first == second
    assert len(first) == 8 and all(len(path) == 5 for path in first)
    assert all(value in returns for path in first for value in path)
