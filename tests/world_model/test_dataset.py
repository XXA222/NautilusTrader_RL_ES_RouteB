from __future__ import annotations

from hashlib import sha256

import numpy as np

from rles_data.core import DataError, DAY_NS, MINUTE_NS
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_world_model.dataset import build_train_sequences


def _replay(extra_after_train: float = 0.0, extra_minutes: int = 15) -> MarketReplay:
    # Thirty-one complete UTC days provide warmup only. The small score interval
    # is deliberately followed by Validation-like rows that must be ignored.
    count = 31 * 1440 + extra_minutes
    frames = []
    previous = 100.0
    for index in range(1, count + 1):
        ts = index * MINUTE_NS
        close = 100.0 + index * 1e-5
        if ts >= 31 * DAY_NS + 10 * MINUTE_NS:
            close += extra_after_train
        opening = previous
        frames.append(MarketFrame(
            ts, ts, "SYNTHETIC", opening, max(opening, close) + 0.01,
            min(opening, close) - 0.01, close, float(index % 100),
        ))
        previous = close
    return MarketReplay(tuple(frames), sha256(b"synthetic-dataset-v1").hexdigest())


def test_train_sequences_use_multiscale_warmup_and_train_only_scalers() -> None:
    train_start = 31 * DAY_NS
    train_end = train_start + 10 * MINUTE_NS
    first = build_train_sequences(
        _replay(), train_start_ns=train_start, train_end_ns=train_end,
    )
    changed_validation = build_train_sequences(
        _replay(extra_after_train=5.0), train_start_ns=train_start, train_end_ns=train_end,
    )
    assert first.observations.shape == (1, 9, 79)
    assert first.targets.shape == (1, 9, 5)
    assert first.reset.shape == (1, 9)
    assert first.reset[0, 0]
    assert first.clipped_values >= 0
    assert {key: scaler.fit_count for key, scaler in first.scalers.items()} == {
        "1m": 9, "1h": 9, "1d": 9, "calendar": 9,
    }
    assert np.array_equal(first.observations, changed_validation.observations)
    assert np.array_equal(first.targets, changed_validation.targets)
    assert {key: value.hash for key, value in first.scalers.items()} == {
        key: value.hash for key, value in changed_validation.scalers.items()
    }
    assert not first.hourly_mask.any()


def test_train_hour_auxiliary_targets_use_only_next_complete_train_hour() -> None:
    train_start = 31 * DAY_NS
    train_end = train_start + 4 * 60 * MINUTE_NS + 10 * MINUTE_NS
    sequences = build_train_sequences(
        _replay(extra_minutes=4 * 60 + 15),
        train_start_ns=train_start, train_end_ns=train_end,
    )
    assert int(sequences.hourly_mask.sum()) == 4
    masked_times = tuple(
        timestamp for timestamp, mask in zip(
            sequences.timestamps_ns, sequences.hourly_mask[0], strict=True,
        ) if mask
    )
    assert masked_times == tuple(train_start + index * 60 * MINUTE_NS for index in range(4))


def test_train_sequence_builder_rejects_incomplete_multiscale_warmup() -> None:
    replay = MarketReplay(_replay().frames[:1440], sha256(b"short").hexdigest())
    try:
        build_train_sequences(replay, train_start_ns=MINUTE_NS,
                              train_end_ns=1440 * MINUTE_NS)
    except DataError as exc:
        assert "no samples" in str(exc)
    else:
        raise AssertionError("insufficient hourly/daily warmup was accepted")
