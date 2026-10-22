from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rles_data.core import digest
from rles_orchestrator import application, research_access
from rles_research.access import ProjectResearchRegistry, authorize_evaluation
from rles_research.ledger import ResearchLedger, ResearchLockError

pytestmark = pytest.mark.contracts


def reserve(registry, *, key="one", owner="run-a", start=100, end=200,
            purpose="holdout", source="a" * 64):
    return registry.reserve(key=key, owner=owner, market="ETHUSDT", family="perpetual-market-v1",
                            start_ns=start, end_ns=end, purpose=purpose, identity={"source": source})


@pytest.mark.parametrize("start,end", [(100, 200), (50, 201), (150, 250), (50, 101)])
@pytest.mark.parametrize("purpose", ["holdout", "test", "development"])
def test_project_overlap_survives_output_protocol_seed_and_snapshot_changes(tmp_path, start, end, purpose):
    path = tmp_path / "project.sqlite"
    reserve(ProjectResearchRegistry(path))
    with pytest.raises(ResearchLockError, match="interval already"):
        reserve(ProjectResearchRegistry(path), key="different-output-protocol-seed", owner="run-b",
                start=start, end=end, purpose=purpose, source="b" * 64)


@pytest.mark.parametrize("purpose", [[], {}, None, 1, "unknown"])
def test_project_registry_rejects_malformed_scope_purpose(tmp_path, purpose):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="valid research scope"):
        registry.reserve(
            key="scope", owner="run", market="ETHUSDT", family="perpetual-market-v1",
            start_ns=100, end_ns=200, purpose=purpose, identity={"source": "a" * 64},
        )


@pytest.mark.parametrize(("field", "value"), [
    ("market", "ETHUSDT "),
    ("market", "ethusdt"),
    ("family", "perpetual-market-v1 "),
    ("family", "Perpetual-Market-V1"),
])
def test_project_registry_rejects_noncanonical_overlap_namespace(tmp_path, field, value):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    reserve(registry, purpose="development")
    scope = dict(key="alias", owner="another-owner", market="ETHUSDT",
                 family="perpetual-market-v1", start_ns=100, end_ns=200,
                 purpose="development", identity={})
    scope[field] = value
    with pytest.raises(ValueError, match="valid research scope"):
        registry.reserve(**scope)


def test_project_registry_enables_sqlite_foreign_keys_for_every_transaction(tmp_path):
    import sqlite3

    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with registry._tx() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            connection.execute(
                "INSERT INTO development_result VALUES(?,?,?,?,?)",
                ("missing-access", "completed", "a" * 64, "{}", 0),
            )


def test_wfo_development_reservation_freezes_curated_source_identity_before_payload_read(
    tmp_path, monkeypatch,
):
    import json
    import sqlite3

    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    source_identity = {
        "schema_version": "0.4.1", "snapshot_hash": "a" * 64,
        "file_sha256": "b" * 64, "marker_sha256": "c" * 64, "events": 42,
    }
    config = {"output": str(tmp_path / "wfo"),
              "environment": {"instrument": "ETHUSDT-PERP.BINANCE",
                              "market_path": str(tmp_path / "market.parquet")}}

    research_access.record_development(
        config, {"start_ns": 100, "end_ns": 200}, label="wfo-development",
        source_identity=source_identity,
    )

    with sqlite3.connect(registry.path) as connection:
        row = connection.execute("SELECT identity FROM access").fetchone()
    assert json.loads(row[0])["source_identity"] == source_identity
    changed_source = {**source_identity, "snapshot_hash": "d" * 64}
    with pytest.raises(ResearchLockError, match="reservation identity changed"):
        research_access.record_development(
            config, {"start_ns": 100, "end_ns": 200}, label="wfo-development",
            source_identity=changed_source,
        )


@pytest.mark.parametrize("field,value,message", [
    ("snapshot_hash", "not-a-sha", "SHA-256"),
    ("events", 0, "source identity"),
    ("schema_version", "0.5.0", "source identity"),
    ("schema_version", [], "source identity"),
    ("schema_version", {}, "source identity"),
])
def test_development_source_identity_rejects_unverified_metadata(
    tmp_path, monkeypatch, field, value, message,
):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    identity = {"schema_version": "0.4.1", "snapshot_hash": "a" * 64,
                "file_sha256": "b" * 64, "marker_sha256": "c" * 64, "events": 42}
    identity[field] = value
    config = {"output": str(tmp_path / "wfo"),
              "environment": {"instrument": "ETHUSDT-PERP.BINANCE"}}
    with pytest.raises(ValueError, match=message):
        research_access.record_development(
            config, {"start_ns": 100, "end_ns": 200}, label="wfo-development",
            source_identity=identity,
        )


def test_half_open_intervals_and_prior_development_cannot_be_rebranded_holdout(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    reserve(registry, purpose="development")
    reserve(registry, key="adjacent", owner="new", start=200, end=300)
    with pytest.raises(ResearchLockError, match="interval already"):
        reserve(registry, key="promoted", owner="new", purpose="holdout")


def test_project_reservations_are_atomic_across_connections(tmp_path):
    path = tmp_path / "project.sqlite"
    ProjectResearchRegistry(path)

    def attempt(index):
        try:
            reserve(ProjectResearchRegistry(path), key=str(index), owner=str(index))
            return True
        except ResearchLockError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(attempt, (1, 2))) == 1


def test_wfo_query_intent_is_frozen_before_read_and_later_binds_selected_candidate(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    identity = {
        "protocol_id": "wfo-protocol", "query_id": "fold-test-0001",
        "source_sha256": "a" * 64,
        "candidate_ids": ["candidate-a", "candidate-b"],
        "fold_id": "60-15-15-0001",
    }
    args = dict(
        protocol_id="wfo-protocol", intent_id="fold-test-0001", owner="wfo-owner",
        market="ETHUSDT", family="perpetual-market-v1", start_ns=300, end_ns=400,
        purpose="test", identity=identity,
    )
    registry.reserve_query_intent(**args)
    registry.reserve_query_intent(**args)
    registry.start_query_intents(
        protocol_id="wfo-protocol", owner="wfo-owner", intent_ids=("fold-test-0001",),
    )
    registry.start_query_intents(
        protocol_id="wfo-protocol", owner="wfo-owner", intent_ids=("fold-test-0001",),
    )
    registry.verify_query_intent(
        protocol_id="wfo-protocol", intent_id="fold-test-0001", owner="wfo-owner",
        market="ETHUSDT", family="perpetual-market-v1", start_ns=300, end_ns=400,
        purpose="test", source_sha256="a" * 64, candidate_id="candidate-b",
    )
    with pytest.raises(ResearchLockError, match="candidate"):
        registry.verify_query_intent(
            protocol_id="wfo-protocol", intent_id="fold-test-0001", owner="wfo-owner",
            market="ETHUSDT", family="perpetual-market-v1", start_ns=300, end_ns=400,
            purpose="test", source_sha256="a" * 64, candidate_id="candidate-c",
        )
    changed = {**identity, "source_sha256": "b" * 64}
    with pytest.raises(ResearchLockError, match="identity changed"):
        registry.reserve_query_intent(**{**args, "identity": changed})
    with pytest.raises(ResearchLockError, match="interval already reserved"):
        registry.reserve_query_intent(**{
            **args, "protocol_id": "other-protocol", "owner": "other-owner",
            "intent_id": "other-test", "identity": {
                **identity, "protocol_id": "other-protocol", "query_id": "other-test",
            },
        })
    with pytest.raises(ResearchLockError, match="formal query intent blocks development"):
        registry.reserve(
            key="new-development", owner="other-owner", market="ETHUSDT",
            family="perpetual-market-v1", start_ns=320, end_ns=330,
            purpose="development", identity={"window": "reused"},
        )
    with pytest.raises(ResearchLockError, match="formal query intent blocks evaluation"):
        registry.reserve(
            key="unintended-evaluation", owner="other-owner", market="ETHUSDT",
            family="perpetual-market-v1", start_ns=320, end_ns=330,
            purpose="test", identity={"protocol_id": "other-protocol", "query_id": "other-test"},
        )


def test_start_query_intents_rejects_string_instead_of_sequence_of_ids(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    identity = {
        "protocol_id": "wfo-protocol", "query_id": "x",
        "source_sha256": "a" * 64, "candidate_ids": ["candidate-a"],
    }
    registry.reserve_query_intent(
        protocol_id="wfo-protocol", intent_id="x", owner="wfo-owner",
        market="ETHUSDT", family="perpetual-market-v1", start_ns=1, end_ns=2,
        purpose="test", identity=identity,
    )

    with pytest.raises(ResearchLockError, match="invalid project query intent start batch"):
        registry.start_query_intents(
            protocol_id="wfo-protocol", owner="wfo-owner", intent_ids="x",
        )
    registry.start_query_intents(
        protocol_id="wfo-protocol", owner="wfo-owner", intent_ids=("x",),
    )


def test_formal_query_intent_rejects_duplicate_candidate_ids(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    identity = {
        "protocol_id": "wfo-protocol", "query_id": "query-one",
        "source_sha256": "a" * 64,
        "candidate_ids": ["candidate-a", "candidate-a"],
    }
    with pytest.raises(ResearchLockError, match="invalid pre-read formal query intent"):
        registry.reserve_query_intent(
            protocol_id="wfo-protocol", intent_id="query-one", owner="wfo-owner",
            market="ETHUSDT", family="perpetual-market-v1", start_ns=1, end_ns=2,
            purpose="test", identity=identity,
        )


def test_formal_query_intent_rejects_padded_candidate_aliases(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    identity = {
        "protocol_id": "wfo-protocol", "query_id": "query-one",
        "source_sha256": "a" * 64,
        "candidate_ids": ["candidate-a", "candidate-a "],
    }
    with pytest.raises(ResearchLockError, match="invalid pre-read formal query intent"):
        registry.reserve_query_intent(
            protocol_id="wfo-protocol", intent_id="query-one", owner="wfo-owner",
            market="ETHUSDT", family="perpetual-market-v1", start_ns=1, end_ns=2,
            purpose="test", identity=identity,
        )


def test_verify_query_intent_rejects_bool_timestamp_alias(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    identity = {
        "protocol_id": "wfo-protocol", "query_id": "query-one",
        "source_sha256": "a" * 64, "candidate_ids": ["candidate-a"],
    }
    registry.reserve_query_intent(
        protocol_id="wfo-protocol", intent_id="query-one", owner="wfo-owner",
        market="ETHUSDT", family="perpetual-market-v1", start_ns=1, end_ns=2,
        purpose="test", identity=identity,
    )
    registry.start_query_intents(
        protocol_id="wfo-protocol", owner="wfo-owner", intent_ids=("query-one",),
    )

    with pytest.raises(ResearchLockError, match="invalid formal query intent verification"):
        registry.verify_query_intent(
            protocol_id="wfo-protocol", intent_id="query-one", owner="wfo-owner",
            market="ETHUSDT", family="perpetual-market-v1", start_ns=True, end_ns=2,
            purpose="test", source_sha256="a" * 64, candidate_id="candidate-a",
        )


@pytest.mark.parametrize("identifier", [" ", " candidate-a "])
def test_trial_attempt_rejects_blank_or_padded_identity_strings(tmp_path, identifier):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="invalid project trial attempt identity"):
        registry.register_trial_attempt(
            attempt_key="attempt", protocol_id="protocol", candidate_id=identifier,
            recipe={"steps": 64}, feature_sha="a" * 64, seed=1,
        )


def test_trial_attempt_event_rejects_padded_event_key(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    registry.register_trial_attempt(
        attempt_key="attempt", protocol_id="protocol", candidate_id="candidate",
        recipe={"steps": 64}, feature_sha="a" * 64, seed=1,
    )
    with pytest.raises(ValueError, match="invalid project trial attempt event"):
        registry.record_trial_attempt_event(
            "attempt", event_key=" started ", status="started",
        )


def test_trial_attempt_summary_rejects_padded_protocol_id(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="protocol ID is required"):
        registry.trial_attempt_summary(" protocol ")


def test_project_trial_attempts_are_append_only_idempotent_and_report_seed_feature_recipe(tmp_path):
    import sqlite3

    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    identity = dict(attempt_key="attempt-a", protocol_id="protocol-a",
                    candidate_id="candidate-a", recipe={"lr": "0.001"},
                    feature_sha="a" * 64, seed=17)
    registry.register_trial_attempt(**identity)
    registry.register_trial_attempt(**identity)
    registry.record_trial_attempt_event("attempt-a", event_key="started-1", status="started")
    registry.record_trial_attempt_event("attempt-a", event_key="started-1", status="started")
    registry.record_trial_attempt_event("attempt-a", event_key="selection", status="selected",
                                        detail={"selection": "median-fold-sharpe"})
    summary = registry.trial_attempt_summary("protocol-a")
    assert summary["project_registered_attempt_count"] == 1
    assert summary["protocol_registered_attempt_count"] == 1
    attempt = summary["attempts"][0]
    assert attempt["candidate_sha256"] == digest({"lr": "0.001"})
    assert attempt["feature_sha256"] == "a" * 64
    assert attempt["seed"] == 17
    assert [event["status"] for event in attempt["events"]] == ["registered", "started", "selected"]
    with pytest.raises(ResearchLockError, match="identity changed"):
        registry.register_trial_attempt(**{**identity, "seed": 18})
    with pytest.raises(ResearchLockError, match="event identity changed"):
        registry.record_trial_attempt_event("attempt-a", event_key="selection", status="not_selected")
    with sqlite3.connect(registry.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE trial_attempt SET seed=18 WHERE attempt_key='attempt-a'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM trial_attempt_event WHERE attempt_key='attempt-a'")


def test_project_trial_attempt_summary_orders_same_candidate_seeds_stably(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    common = dict(
        protocol_id="protocol-a", candidate_id="candidate-a",
        recipe={"lr": "0.001"}, feature_sha="a" * 64,
    )
    registry.register_trial_attempt(attempt_key="seed-18", seed=18, **common)
    registry.register_trial_attempt(attempt_key="seed-17", seed=17, **common)
    summary = registry.trial_attempt_summary("protocol-a")
    assert [(attempt["seed"], attempt["attempt_key"])
            for attempt in summary["attempts"]] == [(17, "seed-17"), (18, "seed-18")]


@pytest.mark.parametrize("status", [[], {}, None, 7, "unknown"])
def test_project_trial_attempt_event_rejects_non_string_or_unknown_status(tmp_path, status):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="invalid project trial attempt event"):
        registry.record_trial_attempt_event(
            "not-registered", event_key="event-a", status=status,
        )


@pytest.mark.parametrize("statuses", [
    ["completed"],
    ["started", "started"],
    ["started", "selected", "failed"],
    ["started", "not_selected", "selected"],
    ["started", "completed", "resumed"],
    ["started", "failed", "completed"],
])
def test_project_trial_attempt_rejects_invalid_lifecycle_transitions(tmp_path, statuses):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    registry.register_trial_attempt(
        attempt_key="attempt", protocol_id="standalone-application-training-v1",
        candidate_id="candidate",
        recipe={"steps": 64}, feature_sha="a" * 64, seed=1,
    )
    accepted = ["registered"]
    for index, status in enumerate(statuses):
        try:
            registry.record_trial_attempt_event(
                "attempt", event_key=f"event-{index}", status=status,
            )
        except ResearchLockError as exc:
            assert "invalid project trial lifecycle transition" in str(exc)
            break
        accepted.append(status)
    else:
        pytest.fail(f"invalid lifecycle was accepted: {statuses}")
    assert [event["status"] for event in registry.trial_attempt_events("attempt")] == accepted


def test_project_trial_lifecycle_retry_is_idempotent_after_terminal_event(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    registry.register_trial_attempt(
        attempt_key="attempt", protocol_id="protocol", candidate_id="candidate",
        recipe={"steps": 64}, feature_sha="a" * 64, seed=1,
    )
    registry.record_trial_attempt_event("attempt", event_key="started", status="started")
    registry.record_trial_attempt_event(
        "attempt", event_key="selection", status="selected", detail={"candidate": "candidate"},
    )
    registry.record_trial_attempt_event(
        "attempt", event_key="selection", status="selected", detail={"candidate": "candidate"},
    )
    assert [event["status"] for event in registry.trial_attempt_events("attempt")] == [
        "registered", "started", "selected",
    ]


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_registered_trial_can_be_terminalized_before_start(tmp_path, status):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    registry.register_trial_attempt(
        attempt_key="attempt", protocol_id="routeb-r4-baseline:fold",
        candidate_id="candidate", recipe={"steps": 64},
        feature_sha="a" * 64, seed=1,
    )
    registry.record_trial_attempt_event(
        "attempt", event_key=f"registration-{status}", status=status,
        detail={"test_payload_decoded": False},
    )
    assert [event["status"] for event in registry.trial_attempt_events("attempt")] == [
        "registered", status,
    ]


def test_reopening_registry_rebuilds_trial_transition_trigger(tmp_path):
    import sqlite3

    path = tmp_path / "project.sqlite"
    registry = ProjectResearchRegistry(path)
    registry.register_trial_attempt(
        attempt_key="attempt", protocol_id="routeb-r4-baseline:fold",
        candidate_id="candidate", recipe={"steps": 64},
        feature_sha="a" * 64, seed=1,
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER trial_event_transition_guard")
        connection.execute("""CREATE TRIGGER trial_event_transition_guard
            BEFORE INSERT ON trial_attempt_event WHEN NEW.status='failed' BEGIN
            SELECT RAISE(ABORT,'stale lifecycle trigger'); END""")

    reopened = ProjectResearchRegistry(path)
    reopened.record_trial_attempt_event(
        "attempt", event_key="registered-run-failed", status="failed",
        detail={"error": "preflight"},
    )
    assert [event["status"] for event in reopened.trial_attempt_events("attempt")] == [
        "registered", "failed",
    ]


def test_project_trial_lifecycle_transition_guard_also_protects_sqlite_inserts(tmp_path):
    import sqlite3

    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    registry.register_trial_attempt(
        attempt_key="standalone", protocol_id="standalone-application-training-v1",
        candidate_id="run", recipe={"steps": 64}, feature_sha="a" * 64, seed=1,
    )
    registry.register_trial_attempt(
        attempt_key="wfo", protocol_id="protocol-a", candidate_id="candidate",
        recipe={"steps": 64}, feature_sha="b" * 64, seed=2,
    )
    with sqlite3.connect(registry.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="invalid project trial lifecycle transition"):
            connection.execute(
                "INSERT INTO trial_attempt_event VALUES(?,?,?,?,?)",
                ("standalone", "finish-before-start", "completed", "{}", 1),
            )
        for attempt_key, event_key, status in (
            ("wfo", "start", "started"),
            ("wfo", "selection", "selected"),
            ("wfo", "resume", "resumed"),
        ):
            connection.execute(
                "INSERT INTO trial_attempt_event VALUES(?,?,?,?,?)",
                (attempt_key, event_key, status, "{}", 1),
            )


@pytest.mark.parametrize("recipe", [{"score": float("nan")}, {"value": object()}])
def test_project_trial_attempt_rejects_nonfinite_or_non_json_recipe(tmp_path, recipe):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="recipe must be finite JSON data"):
        registry.register_trial_attempt(
            attempt_key="attempt", protocol_id="protocol", candidate_id="candidate",
            recipe=recipe, feature_sha="a" * 64, seed=1,
        )


@pytest.mark.parametrize("seed", [-1, 2**32, 2**100, True])
def test_project_trial_attempt_requires_uint32_seed(tmp_path, seed):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="invalid project trial attempt identity"):
        registry.register_trial_attempt(
            attempt_key="attempt", protocol_id="protocol", candidate_id="candidate",
            recipe={"steps": 64}, feature_sha="a" * 64, seed=seed,
        )


@pytest.mark.parametrize("detail", [{"score": float("nan")}, {"value": object()}])
def test_project_trial_attempt_event_rejects_nonfinite_or_non_json_detail(tmp_path, detail):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    registry.register_trial_attempt(
        attempt_key="attempt", protocol_id="protocol", candidate_id="candidate",
        recipe={"lr": "0.001"}, feature_sha="a" * 64, seed=1,
    )
    with pytest.raises(ValueError, match="event detail must be finite JSON data"):
        registry.record_trial_attempt_event(
            "attempt", event_key="started", status="started", detail=detail,
        )


def test_wfo_trial_summary_exposes_project_attempts_from_other_protocols(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    common = {"recipe": {"steps": 64}, "feature_sha": "a" * 64}
    registry.register_trial_attempt(
        attempt_key="wfo-attempt", protocol_id="wfo-protocol", candidate_id="candidate-1",
        seed=7, **common,
    )
    registry.record_trial_attempt_event("wfo-attempt", event_key="wfo-started", status="started")
    registry.register_trial_attempt(
        attempt_key="standalone-attempt", protocol_id="standalone-application-training-v1",
        candidate_id="run-2", seed=11, **common,
    )
    registry.record_trial_attempt_event(
        "standalone-attempt", event_key="standalone-started", status="started",
    )
    registry.record_trial_attempt_event(
        "standalone-attempt", event_key="standalone-failed", status="failed",
    )

    summary = registry.trial_attempt_summary("wfo-protocol")

    assert summary["protocol_registered_attempt_count"] == 1
    assert summary["project_registered_attempt_count"] == 2
    by_protocol = {row["protocol_id"]: row for row in summary["project_protocol_inventory"]}
    assert by_protocol["wfo-protocol"]["lifecycle_event_counts"] == {
        "registered": 1, "started": 1,
    }
    assert by_protocol["standalone-application-training-v1"]["registered_attempt_count"] == 1
    assert by_protocol["standalone-application-training-v1"]["distinct_seed_count"] == 1
    assert by_protocol["standalone-application-training-v1"]["lifecycle_event_counts"] == {
        "failed": 1, "registered": 1, "started": 1,
    }
    assert summary["historical_coverage"].startswith("incomplete;")


def test_standalone_application_trial_records_seed_feature_and_lifecycle(tmp_path, monkeypatch):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    identity = {
        "run_id": "historical-run-a", "recipe": {"steps": 64, "dataset_sha256": "b" * 64},
        "feature_sha": "a" * 64, "seed": 23,
    }
    attempt_key = research_access.begin_application_trial(
        **identity, resumed=False, detail={"route": "A0"},
    )
    assert attempt_key is not None
    research_access.finish_application_trial(
        attempt_key, status="failed", detail={"error": "RuntimeError"},
    )
    resumed_key = research_access.begin_application_trial(
        **identity, resumed=True, detail={"route": "A0", "stage": "R1"},
    )
    assert resumed_key == attempt_key
    research_access.finish_application_trial(
        resumed_key, status="completed", detail={"bundle_sha256": "c" * 64},
    )
    summary = registry.trial_attempt_summary("standalone-application-training-v1")
    assert summary["protocol_registered_attempt_count"] == 1
    attempt = summary["attempts"][0]
    assert attempt["seed"] == 23
    assert attempt["feature_sha256"] == "a" * 64
    assert attempt["recipe"] == identity["recipe"]
    assert [event["status"] for event in attempt["events"]] == [
        "registered", "started", "failed", "resumed", "completed",
    ]

    token = research_access.ACTIVE_OWNER.set("wfo-owner")
    try:
        assert research_access.begin_application_trial(
            **{**identity, "run_id": "wfo-owned-run"}, resumed=False, detail={},
        ) is None
    finally:
        research_access.ACTIVE_OWNER.reset(token)
    assert registry.trial_attempt_summary(
        "standalone-application-training-v1",
    )["protocol_registered_attempt_count"] == 1

    hard_exit_identity = {**identity, "run_id": "historical-run-hard-exit"}
    interrupted_key = research_access.begin_application_trial(
        **hard_exit_identity, resumed=False, detail={"route": "A0"},
    )
    assert interrupted_key is not None
    resumed_after_exit = research_access.begin_application_trial(
        **hard_exit_identity, resumed=True, detail={"route": "A0", "stage": "R1"},
    )
    assert resumed_after_exit == interrupted_key
    hard_exit_events = registry.trial_attempt_events(interrupted_key)
    assert [event["status"] for event in hard_exit_events] == [
        "registered", "started", "interrupted", "resumed",
    ]
    inferred = hard_exit_events[-2]
    assert inferred["detail"] == {
        "prior_event_key": hard_exit_events[1]["event_key"],
        "resolution": "resume_observed_without_terminal_event",
    }


def test_started_access_is_never_refunded_and_receipt_detects_tampering(tmp_path):
    import sqlite3

    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    token = reserve(registry)
    registry.consume("one", token, {"source": "a" * 64})
    with pytest.raises(ResearchLockError, match="already started"):
        registry.consume("one", token, {"source": "a" * 64})
    registry.complete("one", token, "c" * 64)
    ProjectResearchRegistry(registry.path).verify_report("one", "c" * 64)
    with pytest.raises(ResearchLockError, match="completion receipt"):
        registry.verify_report("one", "d" * 64)
    with pytest.raises(ResearchLockError, match="identity changed"):
        reserve(registry, source="b" * 64)
    with sqlite3.connect(registry.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="identity is immutable"):
            connection.execute("UPDATE access SET purpose='development' WHERE access_key='one'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM access WHERE access_key='one'")
        with pytest.raises(sqlite3.IntegrityError, match="state transition"):
            connection.execute("UPDATE access SET state='reserved' WHERE access_key='one'")


@pytest.mark.parametrize("key,token,identity", [
    ([], "token", {}),
    ("key", [], {}),
    ("key", "token", []),
])
def test_consume_rejects_malformed_registry_arguments(tmp_path, key, token, identity):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="evaluation access identity required"):
        registry.consume(key, token, identity)


@pytest.mark.parametrize("key,report_sha", [
    ([], "a" * 64),
    ("key", []),
    ("key", "g" * 64),
])
def test_verify_report_rejects_malformed_registry_arguments(tmp_path, key, report_sha):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="access key and report SHA256"):
        registry.verify_report(key, report_sha)


@pytest.mark.parametrize("split", ["test", "holdout"])
@pytest.mark.parametrize("supplied_factory", [False, True])
def test_ordinary_api_rejects_formal_reads_before_bundle_or_environment(tmp_path, monkeypatch, split, supplied_factory):
    from types import SimpleNamespace
    monkeypatch.setattr(application, "_verify_file", lambda *_: pytest.fail("bundle must not be read"))
    monkeypatch.setattr(application, "HistoricalEnvironmentFactory",
                        lambda **_: pytest.fail("evaluation market must not be read"))
    config = {"evaluation": {"split": split}}
    factory = SimpleNamespace(split=split, mode="stress") if supplied_factory else None
    with pytest.raises(application.DataError, match="requires WFO research access"):
        application.backtest(config, tmp_path, factory=factory, allow_execution_stress=True)


def test_formal_holdout_authorization_requires_matching_retrain_and_single_use_scope(tmp_path):
    ledger = ResearchLedger(tmp_path / "run.sqlite")
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    ledger.create("p", "r", validation_budget=1, hpo_budget=0, es_evaluation_budget=0)
    args = dict(protocol_id="p", query_id="final-holdout", owner="owner", market="ETHUSDT",
                family="perpetual-market-v1", start_ns=100, end_ns=200, split="holdout",
                source_sha="a" * 64, bundle_sha="b" * 64)
    with pytest.raises(ResearchLockError, match="registered retrain"):
        authorize_evaluation(registry, ledger, **args)
    ledger.freeze_recipe("p", recipe_sha="c" * 64, code_sha="d" * 64, selection_spec_sha="e" * 64)
    ledger.register_retrain("p", "b" * 64, pre_holdout_end_ns=100)
    ledger.open_holdout("p", holdout_dataset_sha=digest(["a" * 64, 100, 200]),
                        reviewer_approval_sha="f" * 64, holdout_start_ns=100)
    with pytest.raises(ResearchLockError, match="matching one-shot"):
        authorize_evaluation(registry, ledger, **{**args, "end_ns": 201})
    access = authorize_evaluation(registry, ledger, **args)
    consume = dict(bundle_sha="b" * 64, split="holdout", start_ns=100, end_ns=200,
                   market="ETHUSDT", mode="base")
    with pytest.raises(ResearchLockError, match="frozen access scope"):
        access.consume(**{**consume, "mode": "stress"})
    access.consume(**consume)
    restored = authorize_evaluation(registry, ledger, **args)
    with pytest.raises(ResearchLockError, match="already started"):
        restored.consume(**consume)


def test_development_entry_cannot_relabel_previously_reserved_holdout(tmp_path, monkeypatch):
    registry = research_access.project_registry()
    reserve(registry)
    config = {"output": str(tmp_path / "new-output"), "environment": {
        "instrument": "ETHUSDT-PERP.SIM", "start_ns": 100, "end_ns": 200,
    }}
    monkeypatch.setattr(application, "HistoricalEnvironmentFactory",
                        lambda **_: pytest.fail("blocked before market/scaler access"))
    with pytest.raises(ResearchLockError, match="interval already"):
        application.prepare(config)


def test_project_registry_rejects_duplicate_unresolved_inventory_key(tmp_path, monkeypatch):
    import importlib.util
    from hashlib import sha256
    from types import SimpleNamespace

    from rles_data.core import DataError

    source_root = tmp_path / "repository"
    inventory_path = (source_root / "docs" / "audit_evidence" / "20261016"
                      / "legacy_research_access_v2.json")
    inventory_path.parent.mkdir(parents=True)
    (source_root / "pyproject.toml").write_text(
        "[tool.uv.workspace]\nmembers = []\n", encoding="utf-8",
    )
    content = (
        b'{"schema":"rles.legacy-research-access.v1","entries":[],'
        b'"unresolved":[{"market":"BTCUSDT"}],"unresolved":[]}'
    )
    inventory_path.write_bytes(content)
    inventory_path.with_suffix(".json.sha256").write_text(
        sha256(content).hexdigest(), encoding="ascii",
    )

    source_file = (source_root / "packages" / "orchestrator" / "src"
                   / "rles_orchestrator" / "research_access.py")
    module_path = Path(research_access.__file__).resolve()
    spec = importlib.util.spec_from_file_location("strict_inventory_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(source_file))
    registry = SimpleNamespace(imported=None)
    def import_inventory(entries, inventory_sha):
        registry.imported = (entries, inventory_sha)
    registry.import_inventory = import_inventory
    monkeypatch.setattr(module, "ProjectResearchRegistry", lambda _path: registry)

    with pytest.raises(DataError, match="duplicate JSON object key"):
        module.project_registry()
    assert registry.imported is None


def test_registry_path_does_not_depend_on_output_or_cwd(tmp_path, monkeypatch):
    # Resolve the real function (the autouse fixture isolates other tests).
    import importlib
    module_path = Path(research_access.__file__).resolve()
    spec = importlib.util.spec_from_file_location("registry_path_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from types import SimpleNamespace
    monkeypatch.setattr(module, "ProjectResearchRegistry",
                        lambda path: SimpleNamespace(path=path, import_inventory=lambda *_: None))
    workspace_root = Path(__file__).resolve().parents[1]
    # Model a Linux wheel installed inside the checkout: its package path has
    # lib/pythonX/site-packages between the workspace root and __file__.
    monkeypatch.setattr(
        module, "__file__",
        str(workspace_root / ".venv" / "lib" / "python3.14" / "site-packages"
            / "rles_orchestrator" / "research_access.py"),
    )
    monkeypatch.chdir(tmp_path)
    registry = module.project_registry()
    assert registry.path == workspace_root / ".artifacts" / "research-access.sqlite"


@pytest.mark.parametrize("prior,new", [("development", "holdout"), ("test", "holdout"),
                                     ("holdout", "holdout"), ("holdout", "test"),
                                     ("holdout", "development")])
def test_same_owner_cannot_rebrand_or_reopen_holdout_with_a_new_key(tmp_path, prior, new):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    token = reserve(registry, purpose=prior)
    # Identical reservation remains idempotent, but its consume state still
    # controls recovery. A second identity cannot escape overlap detection.
    assert reserve(registry, purpose=prior) == token
    with pytest.raises(ResearchLockError, match="interval already"):
        reserve(registry, key="new-key", purpose=new)


@pytest.mark.parametrize("new_purpose", ["development", "test", "holdout"])
def test_same_owner_cannot_relabel_or_reopen_test_with_a_new_key(tmp_path, new_purpose):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    reserve(registry, purpose="test")
    with pytest.raises(ResearchLockError, match="interval already"):
        reserve(registry, key="second-query", owner="run-a", purpose=new_purpose)


def test_wfo_owner_may_authorize_fold_test_after_declaring_development_window(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    reserve(registry, purpose="development", owner="wfo-run")
    reserve(registry, key="fold-test", purpose="test", owner="wfo-run")
    with pytest.raises(ResearchLockError, match="interval already"):
        reserve(registry, key="relabel-development", purpose="development", owner="wfo-run")


def test_one_frozen_test_query_may_share_only_its_policy_and_baseline_components(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")

    def reserve_component(key, component):
        return registry.reserve(
            key=key, owner="final-run", market="ETHUSDT", family="perpetual-market-v1",
            start_ns=100, end_ns=200, purpose="test",
            identity={"protocol_id": "protocol", "query_id": "final-test",
                      "source_sha": "a" * 64, "bundle_sha": "b" * 64,
                      "recipe_sha": "c" * 64, "code_sha": "d" * 64,
                      "split": "test", "mode": "base", "component": component},
        )

    reserve_component("policy-access", "policy")
    reserve_component("baseline-access", "baselines")
    with pytest.raises(ResearchLockError, match="query identity already reserved"):
        reserve_component("policy-second-key", "policy")
    with pytest.raises(ResearchLockError, match="query identity already reserved"):
        registry.reserve(
            key="baseline-stress", owner="final-run", market="ETHUSDT",
            family="perpetual-market-v1", start_ns=100, end_ns=200,
            purpose="test",
            identity={"protocol_id": "protocol", "query_id": "final-test",
                      "source_sha": "a" * 64, "bundle_sha": "b" * 64,
                      "recipe_sha": "c" * 64, "code_sha": "d" * 64,
                      "split": "test", "mode": "stress", "component": "baselines"},
        )
    with pytest.raises(ResearchLockError, match="query identity already reserved"):
        registry.reserve(
            key="baseline-wrong-split", owner="final-run", market="ETHUSDT",
            family="perpetual-market-v1", start_ns=100, end_ns=200,
            purpose="test",
            identity={"protocol_id": "protocol", "query_id": "final-test",
                      "source_sha": "a" * 64, "bundle_sha": "b" * 64,
                      "recipe_sha": "c" * 64, "code_sha": "d" * 64,
                      "split": "holdout", "mode": "base", "component": "baselines"},
        )
    with pytest.raises(ResearchLockError, match="interval already"):
        registry.reserve(
            key="development", owner="final-run", market="ETHUSDT",
            family="perpetual-market-v1", start_ns=100, end_ns=200,
            purpose="development", identity={"label": "same-window"},
        )


def test_frozen_test_query_cannot_be_reopened_for_another_window_or_output(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    identity = {
        "protocol_id": "protocol", "query_id": "fold-test-1", "component": "policy",
        "split": "test", "mode": "base", "source_sha": "a" * 64,
        "bundle_sha": "b" * 64, "recipe_sha": "c" * 64, "code_sha": "d" * 64,
        "start_ns": 100, "end_ns": 200,
    }
    registry.reserve(
        key="first-output-query", owner="output-a", market="ETHUSDT",
        family="perpetual-market-v1", start_ns=100, end_ns=200,
        purpose="test", identity=identity,
    )

    with pytest.raises(ResearchLockError, match="query identity already reserved"):
        registry.reserve(
            key="second-output-query", owner="output-b", market="ETHUSDT",
            family="perpetual-market-v1", start_ns=300, end_ns=400,
            purpose="test", identity={**identity, "start_ns": 300, "end_ns": 400},
        )


def test_frozen_test_query_rejects_malformed_component_without_type_error(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    shared = {
        "protocol_id": "protocol", "query_id": "query", "split": "test",
        "mode": "base", "source_sha": "a" * 64, "bundle_sha": "b" * 64,
        "recipe_sha": "c" * 64, "code_sha": "d" * 64,
    }
    registry.reserve(
        key="policy", owner="owner", market="ETHUSDT",
        family="perpetual-market-v1", start_ns=100, end_ns=200,
        purpose="test", identity={**shared, "component": "policy"},
    )
    with pytest.raises(ResearchLockError, match="query identity already reserved"):
        registry.reserve(
            key="malformed", owner="owner", market="ETHUSDT",
            family="perpetual-market-v1", start_ns=100, end_ns=200,
            purpose="test", identity={**shared, "component": ["baselines"]},
        )


def test_historical_inventory_records_overlap_atomically_without_authorizing_reads(tmp_path):
    import sqlite3

    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    entries = [dict(key="old-dev", market="ETHUSDT", family="perpetual-market-v1",
                    start_ns=100, end_ns=200, purpose="development", identity={}),
               dict(key="old-holdout", market="ETHUSDT", family="perpetual-market-v1",
                    start_ns=150, end_ns=250, purpose="holdout", identity={})]
    registry.import_inventory(entries, "a" * 64)
    registry.import_inventory(entries, "a" * 64)
    with sqlite3.connect(registry.path) as connection:
        rows = connection.execute("SELECT access_key,state FROM access ORDER BY access_key").fetchall()
    assert rows == [("old-dev", "recorded"), ("old-holdout", "recorded")]
    with pytest.raises(ResearchLockError, match="interval already"):
        reserve(registry, key="fresh", start=200, end=250)
    bad = [{**entries[0], "key": "would-be-partial"},
           {**entries[1], "key": "invalid", "end_ns": 0}]
    with pytest.raises(ValueError, match="scope"):
        registry.import_inventory(bad, "b" * 64)
    with sqlite3.connect(registry.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM access").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM inventory_import").fetchone()[0] == 1


@pytest.mark.parametrize("entries", [None, ["not-an-entry"], [{}], [
    {"key": "entry", "owner": "caller", "market": "ETHUSDT",
     "family": "perpetual-market-v1", "start_ns": 1, "end_ns": 2,
     "purpose": "development", "identity": {}},
], [
    {"key": [], "market": "ETHUSDT", "family": "perpetual-market-v1",
     "start_ns": 1, "end_ns": 2, "purpose": "development", "identity": {}},
], [
    {"key": "x", "market": "ETHUSDT ", "family": "perpetual-market-v1",
     "start_ns": 1, "end_ns": 2, "purpose": "development", "identity": {}},
], [
    {"key": "x", "market": "ETHUSDT", "family": "Perpetual-Market-V1",
     "start_ns": 1, "end_ns": 2, "purpose": "development", "identity": {}},
]])
def test_historical_inventory_import_rejects_malformed_entry_shapes_atomically(
    tmp_path, entries,
):
    import sqlite3

    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="legacy inventory"):
        registry.import_inventory(entries, "a" * 64)
    with sqlite3.connect(registry.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM access").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM inventory_import").fetchone()[0] == 0


def test_historical_inventory_import_requires_sha256_identity(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="legacy inventory"):
        registry.import_inventory([], "not-a-sha256")


def test_historical_inventory_import_rejects_duplicate_access_keys_atomically(tmp_path):
    import sqlite3

    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    entry = dict(key="duplicate", market="ETHUSDT", family="perpetual-market-v1",
                 start_ns=100, end_ns=200, purpose="development", identity={})
    with pytest.raises(ValueError, match="duplicate access keys"):
        registry.import_inventory([entry, dict(entry)], "a" * 64)
    with sqlite3.connect(registry.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM access").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM inventory_import").fetchone()[0] == 0


@pytest.mark.parametrize("change", [
    {"entry": None},
    {"entry": {"key": "x", "market": "ETHUSDT", "family": "perpetual-market-v1",
               "start_ns": 1, "end_ns": 2, "purpose": [], "identity": {}}},
    {"entry": {"key": "x", "market": "ETHUSDT", "family": "perpetual-market-v1",
               "start_ns": True, "end_ns": 2, "purpose": "test", "identity": {}}},
])
def test_legacy_inventory_parser_rejects_malformed_access_entry(change):
    import json

    from rles_data.core import DataError

    entry = {"key": "x", "market": "ETHUSDT", "family": "perpetual-market-v1",
             "start_ns": 1, "end_ns": 2, "purpose": "test", "identity": {}}
    inventory = {
        "schema": "rles.legacy-research-access.v1",
        "entries": [change.get("entry", entry)], "unresolved": [],
    }
    with pytest.raises(DataError, match="invalid access entry"):
        research_access._strict_inventory_json(json.dumps(inventory).encode("utf-8"))


def test_legacy_inventory_parser_rejects_duplicate_access_keys():
    import json

    from rles_data.core import DataError

    entry = {"key": "duplicate", "market": "ETHUSDT",
             "family": "perpetual-market-v1", "start_ns": 1, "end_ns": 2,
             "purpose": "development", "identity": {}}
    inventory = {"schema": "rles.legacy-research-access.v1",
                 "entries": [entry, dict(entry)], "unresolved": []}
    with pytest.raises(DataError, match="duplicate access keys"):
        research_access._strict_inventory_json(json.dumps(inventory).encode("utf-8"))


def test_development_result_rejects_unhashable_status(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="valid development result receipt"):
        registry.record_development_result(
            key="key", identity={}, status=[], report_sha="a" * 64,
        )


def test_evaluation_completion_rejects_non_string_report_digest(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="report SHA256 required"):
        registry.complete("key", "token", 7)


def test_trial_attempt_summary_rejects_non_string_protocol_id(tmp_path):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="protocol ID is required"):
        registry.trial_attempt_summary(7)


@pytest.mark.parametrize("recipe", [{1: "value"}, {"nested": {1: "value"}}])
def test_trial_attempt_recipe_rejects_non_string_json_object_keys(tmp_path, recipe):
    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    with pytest.raises(ValueError, match="finite JSON data"):
        registry.register_trial_attempt(
            attempt_key="attempt", protocol_id="protocol", candidate_id="candidate",
            recipe=recipe, feature_sha="a" * 64, seed=1,
        )


def test_project_registry_fails_closed_on_duplicate_persisted_formal_identity(tmp_path):
    import sqlite3

    registry = ProjectResearchRegistry(tmp_path / "project.sqlite")
    identity = {"protocol_id": "protocol-a", "query_id": "query-a", "component": "policy"}
    registry.reserve(
        key="first", owner="run-a", market="ETHUSDT", family="perpetual-market-v1",
        start_ns=100, end_ns=200, purpose="test", identity=identity,
    )
    with sqlite3.connect(registry.path) as connection:
        # Emulate on-disk corruption by bypassing the application immutability trigger.
        connection.execute("DROP TRIGGER access_identity_immutable")
        connection.execute(
            "UPDATE access SET identity=? WHERE access_key=?",
            ('{"protocol_id":"protocol-a","protocol_id":"protocol-b",'
             '"query_id":"query-a","component":"policy"}', "first"),
        )

    with pytest.raises(ResearchLockError, match="corrupted prior formal reservation identity"):
        registry.reserve(
            key="disjoint-query-reuse", owner="run-a", market="ETHUSDT",
            family="perpetual-market-v1", start_ns=300, end_ns=400,
            purpose="test", identity=identity,
        )



def test_legacy_inventory_parser_rejects_nonstandard_json_constant():
    from rles_data.core import DataError

    payload = (b'{"schema":"rles.legacy-research-access.v1",'
               b'"entries":[],"unresolved":[{"identity":{"value":NaN}}]}')
    with pytest.raises(DataError, match="invalid or ambiguous"):
        research_access._strict_inventory_json(payload)
