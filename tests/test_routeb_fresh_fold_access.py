import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event, MINUTE_NS
from rles_data.io import write_curated
from rles_orchestrator import research_access
from rles_research.access import ProjectResearchRegistry
from rles_research.ledger import ResearchLockError

pytestmark = pytest.mark.contracts
def test_fresh_development_reservation_is_one_shot_and_rejects_any_overlap(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "fresh.sqlite")
    args = dict(
        key="routeb-fold-a", owner="routeb-owner", market="ETHUSDT",
        family="perpetual-market-v1", start_ns=100, end_ns=200,
        identity={"fold_sha256": "a" * 64, "source_sha256": "b" * 64},
    )
    token = registry.reserve_fresh_development(**args)
    assert isinstance(token, str) and len(token) == 32
    with pytest.raises(ResearchLockError, match="key was already used"):
        registry.reserve_fresh_development(**args)
    with pytest.raises(ResearchLockError, match="overlaps previously consumed interval"):
        registry.reserve_fresh_development(**{
            **args, "key": "routeb-fold-overlap", "start_ns": 199, "end_ns": 300,
        })
    adjacent = registry.reserve_fresh_development(**{
        **args, "key": "routeb-fold-adjacent", "start_ns": 200, "end_ns": 300,
    })
    assert isinstance(adjacent, str)


def test_development_result_is_append_only_and_bound_to_exact_access(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "fresh.sqlite")
    identity = {"fold_sha256": "a" * 64, "source_sha256": "b" * 64}
    args = dict(
        key="routeb-fold-result", owner="routeb-owner", market="ETHUSDT",
        family="perpetual-market-v1", start_ns=100, end_ns=200,
        identity=identity,
    )
    registry.reserve_fresh_development(**args)
    registry.record_development_result(
        key=args["key"], identity=identity, status="completed",
        report_sha="c" * 64, detail={"test_payload_decoded": False},
    )
    assert registry.development_result(args["key"]) == {
        "status": "completed", "report_sha256": "c" * 64,
        "detail": {"test_payload_decoded": False},
    }
    registry.record_development_result(
        key=args["key"], identity=identity, status="completed",
        report_sha="c" * 64, detail={"test_payload_decoded": False},
    )
    with pytest.raises(ResearchLockError, match="receipt changed"):
        registry.record_development_result(
            key=args["key"], identity=identity, status="completed",
            report_sha="d" * 64, detail={"test_payload_decoded": False},
        )
    with pytest.raises(ResearchLockError, match="exact recorded access"):
        registry.record_development_result(
            key=args["key"], identity={**identity, "extra": True},
            status="completed", report_sha="c" * 64,
        )


def test_development_result_requires_existing_development_access(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "fresh.sqlite")
    with pytest.raises(ResearchLockError, match="exact recorded access"):
        registry.record_development_result(
            key="missing", identity={"fold_sha256": "a" * 64},
            status="failed", report_sha="b" * 64,
        )


def test_routeb_fresh_fold_completion_and_candidate_lifecycle(tmp_path, monkeypatch):
    registry = ProjectResearchRegistry(tmp_path / "fresh.sqlite")
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    config = {"environment": {
        "instrument": "TRXUSDT-PERP.BINANCE",
        "market_path": str(tmp_path / "market.parquet"),
    }}
    window = {"start_ns": 100, "end_ns": 200}
    source_identity = {
        "schema_version": "0.4.1", "snapshot_hash": "a" * 64,
        "file_sha256": "b" * 64, "marker_sha256": "c" * 64,
        "events": 42,
    }
    reservation = research_access.reserve_routeb_fresh_fold(
        config, window, label="routeb-fold-trxusdt-lifecycle-v1",
        fold_sha256="d" * 64, repository_commit="e" * 40,
        source_code_sha256="f" * 64, source_identity=source_identity,
    )
    assert reservation["access_key"] == research_access.routeb_fresh_fold_access_key(
        config, window, label="routeb-fold-trxusdt-lifecycle-v1",
        fold_sha256="d" * 64, repository_commit="e" * 40,
        source_code_sha256="f" * 64, source_identity=source_identity,
    )
    attempt_key = research_access.register_routeb_candidate_attempt(
        label="routeb-fold-trxusdt-lifecycle-v1", candidate_id="E2",
        configuration={"model": {"hidden": 8}, "training": {"steps": 2}},
        fold_sha256="d" * 64, feature_sha256="1" * 64, seed=31,
    )
    research_access.record_routeb_candidate_event(
        attempt_key, event_key="started", status="started",
        detail={"access_key": reservation["access_key"]},
    )
    research_access.record_routeb_candidate_event(
        attempt_key, event_key="completed", status="completed",
        detail={"report_sha256": "2" * 64},
    )
    completion = research_access.complete_routeb_fresh_fold(
        config, window, label="routeb-fold-trxusdt-lifecycle-v1",
        fold_sha256="d" * 64, repository_commit="e" * 40,
        source_code_sha256="f" * 64, source_identity=source_identity,
        report_sha256="3" * 64, detail={"test_payload_decoded": False},
    )
    assert completion["access_key"] == reservation["access_key"]
    assert registry.development_result(reservation["access_key"])["report_sha256"] == "3" * 64
    summary = registry.trial_attempt_summary("routeb-r2:routeb-fold-trxusdt-lifecycle-v1")
    assert summary["protocol_registered_attempt_count"] == 1
    assert [event["status"] for event in summary["attempts"][0]["events"]] == [
        "registered", "started", "completed",
    ]


def test_fresh_development_reservation_rejects_existing_query_intent(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "intent.sqlite")
    registry.reserve_query_intent(
        protocol_id="wfo", intent_id="test-1", owner="wfo-owner",
        market="ETHUSDT", family="perpetual-market-v1", start_ns=100, end_ns=200,
        purpose="test", identity={"protocol_id": "wfo", "query_id": "test-1",
                                 "source_sha256": "a" * 64,
                                 "candidate_ids": ["candidate-a"]},
    )
    with pytest.raises(ResearchLockError, match="frozen formal query intent"):
        registry.reserve_fresh_development(
            key="routeb-fold", owner="routeb-owner", market="ETHUSDT",
            family="perpetual-market-v1", start_ns=150, end_ns=250,
            identity={"fold_sha256": "b" * 64},
        )


def test_registered_fresh_fold_marker_and_reservation_precede_payload_decode(
    tmp_path, monkeypatch,
):
    source_identity = {
        "schema_version": "0.4.1", "snapshot_hash": "a" * 64,
        "file_sha256": "b" * 64, "marker_sha256": "c" * 64, "events": 42,
    }
    config = {"environment": {
        "instrument": "ETHUSDT-PERP.BINANCE",
        "market_path": str(tmp_path / "market.parquet"),
    }}
    calls = []
    marker = lambda path: calls.append(("marker", path)) or source_identity
    reserve_fresh = lambda *args, **kwargs: calls.append(("reserve", kwargs))
    decode = lambda *args, **kwargs: calls.append(("decode", kwargs)) or "store"
    monkeypatch.setattr(research_access, "curated_source_identity", marker)
    monkeypatch.setattr(research_access, "curated_event_time_bounds",
                        lambda _path, **_kwargs: {"min_ts_event_ns": 0, "max_ts_event_ns": 10**15})
    monkeypatch.setattr(research_access, "reserve_routeb_fresh_fold", reserve_fresh)
    monkeypatch.setattr(research_access, "read_curated_window", decode)

    result = research_access.read_registered_fold_window(
        config, {"start_ns": 100, "end_ns": 200}, marker_sha256="c" * 64,
        dataset_hash="a" * 64, required_start_ns=100, required_end_ns=200,
        label="routeb-fold-001", fold_sha256="d" * 64, repository_commit="c" * 40,
        source_code_sha256="e" * 64,
    )

    assert result == "store"
    assert [call[0] for call in calls] == ["marker", "reserve", "decode"]
    assert calls[1][1]["source_identity"] == source_identity
    assert calls[2][1]["expected_marker_sha256"] == "c" * 64


def test_registered_fresh_fold_rejects_dataset_mismatch_before_reservation(
    tmp_path, monkeypatch,
):
    source_identity = {
        "schema_version": "0.4.1", "snapshot_hash": "a" * 64,
        "file_sha256": "b" * 64, "marker_sha256": "c" * 64, "events": 42,
    }
    config = {"environment": {"instrument": "ETHUSDT-PERP.BINANCE",
                               "market_path": str(tmp_path / "market.parquet")}}
    monkeypatch.setattr(research_access, "curated_source_identity", lambda _path: source_identity)
    monkeypatch.setattr(research_access, "reserve_routeb_fresh_fold",
                        lambda *_args, **_kwargs: pytest.fail("must fail before reserving"))
    with pytest.raises(DataError, match="dataset hash differs"):
        research_access.read_registered_fold_window(
            config, {"start_ns": 100, "end_ns": 200}, marker_sha256="c" * 64,
            dataset_hash="e" * 64, required_start_ns=100, required_end_ns=200,
            label="routeb-fold-002", fold_sha256="d" * 64, repository_commit="c" * 40,
            source_code_sha256="e" * 64,
        )


def test_fresh_fold_reader_reads_only_row_groups_after_atomic_scope_record(tmp_path, monkeypatch):
    import json
    import sqlite3

    market_path = tmp_path / "market.parquet"
    events = []
    for minute in range(1, 11):
        timestamp = minute * MINUTE_NS
        price = str(100 + minute)
        events.append(Event(
            event_id=f"bar-{minute}", kind="ohlcv_1m",
            ts_event_ns=timestamp, ts_init_ns=timestamp, available_at_ns=timestamp,
            availability_source=Availability.NATIVE, source_hash="a" * 64,
            payload={"instrument": "ETHUSDT-PERP.BINANCE", "open": price,
                     "high": price, "low": price, "close": price, "volume": "1"},
            provenance="synthetic:routeb-fresh-fold-test",
        ))
    source = CuratedStore(events)
    write_curated(market_path, source, row_group_size=4)
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    original_reader = research_access.read_curated_window
    observed = {}

    def verify_reserved_then_decode(path, start_ns, end_ns, **kwargs):
        with sqlite3.connect(registry.path) as connection:
            row = connection.execute(
                "SELECT start_ns,end_ns,purpose,state,identity FROM access"
            ).fetchone()
        observed["reservation"] = row
        assert row[:4] == (start_ns, end_ns, "development", "recorded")
        observed["identity"] = json.loads(row[4])
        return original_reader(path, start_ns, end_ns, **kwargs)

    monkeypatch.setattr(research_access, "read_curated_window", verify_reserved_then_decode)
    from rles_data.io import curated_source_identity
    source_identity = curated_source_identity(market_path)
    config = {"environment": {"instrument": "ETHUSDT-PERP.BINANCE",
                               "market_path": str(market_path)}}
    window = {"start_ns": MINUTE_NS, "end_ns": 9 * MINUTE_NS}

    with pytest.raises(DataError, match="time bounds"):
        research_access.read_registered_fold_window(
            config, window, marker_sha256=source_identity["marker_sha256"],
            dataset_hash=source.snapshot_hash, required_start_ns=MINUTE_NS,
            required_end_ns=12 * MINUTE_NS, label="routeb-fold-uncovered",
            fold_sha256="e" * 64, repository_commit="c" * 40,
            source_code_sha256="d" * 64,
        )
    with sqlite3.connect(registry.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM access").fetchone()[0] == 0

    store = research_access.read_registered_fold_window(
        config, window, marker_sha256=source_identity["marker_sha256"],
        dataset_hash=source.snapshot_hash, required_start_ns=MINUTE_NS,
        required_end_ns=11 * MINUTE_NS, label="routeb-fold-synthetic-01",
        fold_sha256="f" * 64, repository_commit="c" * 40,
        source_code_sha256="d" * 64,
    )

    assert len(store.events) == 8
    assert observed["identity"]["fold_sha256"] == "f" * 64
    assert observed["identity"]["repository_commit"] == "c" * 40
    assert observed["identity"]["source_code_sha256"] == "d" * 64
    assert observed["identity"]["source_identity"] == source_identity
    with pytest.raises(ResearchLockError, match="key was already used"):
        research_access.read_registered_fold_window(
            config, window, marker_sha256=source_identity["marker_sha256"],
            dataset_hash=source.snapshot_hash, required_start_ns=MINUTE_NS,
            required_end_ns=11 * MINUTE_NS, label="routeb-fold-synthetic-01",
            fold_sha256="f" * 64, repository_commit="c" * 40,
            source_code_sha256="d" * 64,
        )
