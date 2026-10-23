import pytest

from rles_data.core import (
    Availability,
    DataError,
    Event,
    FeatureField,
    FeatureSpec,
)
from rles_data.features import FeatureRow, Scaler


def _spec() -> FeatureSpec:
    return FeatureSpec("0.4.0", (
        FeatureField("return_1m", "float64", "ratio", "ohlcv_1m", 2, 1, "return"),
    ))


def _event() -> Event:
    return Event("bar-1", "ohlcv_1m", 60_000_000_000, 60_000_000_000,
                 60_001_000_000, Availability.DERIVED, "a" * 64,
                 {"instrument": "BTC/USD", "open": "100", "high": "101",
                  "low": "99", "close": "100", "volume": "1"}, "unit-test")


def test_event_hash_is_bound_to_fields_and_nested_payload():
    event = _event()
    initial = event.hash
    assert event.hash == initial
    object.__setattr__(event, "ts_event_ns", event.ts_event_ns + 1)
    with pytest.raises(DataError, match="event mutated after construction"):
        _ = event.hash

    payload_event = _event()
    dict.__setitem__(payload_event.payload, "close", "200")
    with pytest.raises(DataError, match="event mutated after construction"):
        _ = payload_event.hash


def test_feature_spec_and_feature_row_hashes_fail_closed_after_mutation():
    spec = _spec()
    object.__setattr__(spec, "fields", (FeatureField(
        "other_return", "float64", "ratio", "ohlcv_1m", 2, 1, "return"),))
    with pytest.raises(DataError, match="feature spec mutated after construction"):
        _ = spec.hash

    row = FeatureRow("b" * 64, 120_000_000_000, 119_000_000_000,
                     "BTC/USD", (0.01,))
    assert not hasattr(row, "__dict__")
    import pickle
    restored = pickle.loads(pickle.dumps(row))
    assert restored.hash == row.hash
    object.__setattr__(row, "values", (0.02,))
    with pytest.raises(DataError, match="feature row mutated after construction"):
        _ = row.hash


def test_scaler_hash_binding_survives_serialization_and_detects_mutation():
    import pickle

    scaler = Scaler("b" * 64, (0.0,), (1.0,), "c" * 64, 10)
    restored = pickle.loads(pickle.dumps(scaler))
    assert restored.hash == scaler.hash
    object.__setattr__(restored, "mean", (2.0,))
    with pytest.raises(DataError, match="scaler mutated after construction"):
        _ = restored.hash

