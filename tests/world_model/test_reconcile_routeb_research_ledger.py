from __future__ import annotations

from hashlib import sha256
import json
import sqlite3

import pytest

from qa.world_model.reconcile_routeb_research_ledger import (
    _read_artifact_bytes,
    _read_development_accesses,
    ReconciliationPlan,
    _report_paths,
    _unresolved_r4_accesses,
    apply_plan,
    build_plan,
    reconcile,
)
from rles_data.core import digest
from rles_orchestrator import research_access
from rles_research.access import ProjectResearchRegistry


def _seal(report: dict) -> None:
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = sha256(payload.encode("utf-8")).hexdigest()


def test_historical_report_reconciliation_is_exact_and_idempotent(tmp_path):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    label = "routeb-fold-trxusdt-reconcile-v1"
    fold_sha = "a" * 64
    feature_sha = "b" * 64
    source_identity = {
        "schema_version": "0.4.1", "snapshot_hash": "c" * 64,
        "file_sha256": "d" * 64, "marker_sha256": "e" * 64, "events": 42,
    }
    access_identity = {
        "protocol": "routeb-world-model-b0-fresh-fold-v0",
        "availability_policy": "closed_1m_bar;available_at_ns<=decision_ts_ns",
        "fold_sha256": fold_sha,
        "repository_commit": "f" * 40,
        "source_code_sha256": "1" * 64,
        "window": {"start_ns": 100, "end_ns": 200},
        "environment": {
            "instrument": "TRXUSDT-PERP.BINANCE", "market_path": "isolated.parquet",
            "curated_marker_sha256": source_identity["marker_sha256"],
            "curated_read_mode": "row_group_scoped_v1",
        },
        "label": label,
        "source_identity": source_identity,
    }
    access_key = digest(["test", label])
    registry.reserve_fresh_development(
        key=access_key, owner=digest(["owner"]), market="TRXUSDT",
        family="perpetual-market-v1", start_ns=100, end_ns=200,
        identity=access_identity,
    )
    candidates = {}
    for candidate_id in ("E1", "E2"):
        candidates[candidate_id] = {
            "status": "COMPLETED", "fold_hash": fold_sha,
            "model_config": {"candidate": candidate_id, "hidden": 8},
            "training_config": {"seed": 31, "steps": 2},
            "optimizer_updates": 2,
            "checkpoint_sha256": ("2" if candidate_id == "E1" else "3") * 64,
        }
    historical_configuration = {
        "model": candidates["E1"].pop("model_config"),
        "training": candidates["E1"]["training_config"],
    }
    configuration_path = tmp_path / "seed-31" / "e1_configuration.json"
    configuration_path.parent.mkdir()
    configuration_content = json.dumps(historical_configuration, indent=2).encode("utf-8")
    configuration_path.write_bytes(configuration_content)
    candidates["E1"]["configuration_sha256"] = sha256(configuration_content).hexdigest()
    report = {
        "schema": "routeb.multi-seed-fold-experiment.v1", "status": "COMPLETED",
        "reservation_label": label, "fold_hash": fold_sha,
        "repository_commit": "f" * 40, "source_code_sha256": "1" * 64,
        "source_identity": source_identity, "marker_sha256": source_identity["marker_sha256"],
        "dataset_hash": source_identity["snapshot_hash"],
        "fold_manifest": {"dataset_hash": source_identity["snapshot_hash"],
                          "feature_hash": feature_sha},
        "read_scope": {"start_ns": 100, "end_ns": 200, "includes_test": False},
        "training_seeds": [31],
        "candidate_training_runs_by_seed": {"31": candidates},
        "per_seed_report_sha256": {"31": "4" * 64},
        "test_payload_decoded": False, "economic_certification": False,
    }
    _seal(report)
    report_path = tmp_path / "multi_seed_fold_experiment_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    plan = build_plan(report_path, registry_path)
    assert plan.access_key == access_key
    assert len(plan.candidates) == 2
    apply_plan(plan, registry_path)
    apply_plan(plan, registry_path)

    assert registry.development_result(access_key) == {
        "status": "completed", "report_sha256": report["report_sha256"],
        "detail": {
            "report_schema": "routeb.multi-seed-fold-experiment.v1",
            "test_payload_decoded": False, "economic_certification": False,
        },
    }
    summary = registry.trial_attempt_summary(f"routeb-r2:{label}")
    assert summary["protocol_registered_attempt_count"] == 2
    assert all(
        [event["status"] for event in attempt["events"]]
        == ["registered", "started", "completed"]
        for attempt in summary["attempts"]
    )


def test_reconciliation_rejects_changed_report_payload(tmp_path):
    report = {
        "schema": "routeb.multi-seed-fold-experiment.v1", "status": "COMPLETED",
        "test_payload_decoded": False, "economic_certification": False,
    }
    _seal(report)
    report["status"] = "CORRUPTED"
    path = tmp_path / "multi_seed_fold_experiment_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    registry = ProjectResearchRegistry(tmp_path / "research.sqlite")

    try:
        build_plan(path, registry.path)
    except ValueError as exc:
        assert "completed Route B" in str(exc)
    else:
        raise AssertionError("changed report was accepted")


def test_reconciliation_rejects_duplicate_paths_for_same_access(tmp_path, monkeypatch):
    import qa.world_model.reconcile_routeb_research_ledger as reconciler

    source = tmp_path / "reports"
    first = source / "one" / "r4_e2_e3_comparison_report.json"
    duplicate = source / "two" / "r4_e2_e3_comparison_report.json"
    first.parent.mkdir(parents=True)
    duplicate.parent.mkdir(parents=True)
    first.write_text("{}", encoding="utf-8")
    duplicate.write_text("{}", encoding="utf-8")
    plan = ReconciliationPlan(
        report_path=first, access_key="a" * 64, access_identity={},
        report_sha256="b" * 64, report_schema="routeb.registered-r4-e2-e3-comparison.v1",
        candidates=(),
    )
    monkeypatch.setattr(reconciler, "build_plan", lambda _path, _registry: plan)

    result = reconcile(source, tmp_path / "research.sqlite", apply=False)
    assert result["reports_discovered"] == 2
    assert result["reports_accepted"] == 1
    assert result["reports_rejected"] == 1
    assert result["candidate_attempts"] == 0
    assert "duplicate report path" in result["rejected"][0]["error"]

    try:
        reconcile(source, tmp_path / "research.sqlite", apply=True)
    except ValueError as exc:
        assert "refusing partial reconciliation" in str(exc)
    else:
        raise AssertionError("apply accepted duplicate reports for one access")


def test_report_discovery_keeps_r4_failure_when_either_receipt_is_damaged(tmp_path):
    source = tmp_path / "reports"
    missing_registration = source / "missing-registration" / "failure_receipt.json"
    damaged_failure = source / "damaged-failure" / "failure_receipt.json"
    unrelated_failure = source / "unrelated" / "failure_receipt.json"
    for path in (missing_registration, damaged_failure, unrelated_failure):
        path.parent.mkdir(parents=True)
    missing_registration.write_text(json.dumps({
        "schema": "routeb.registered-r4-e2-e3-failure.v2", "status": "FAILED",
    }), encoding="utf-8")
    (damaged_failure.parent / "registration_receipt.json").write_text(json.dumps({
        "schema": "routeb.registered-r4-e2-e3-receipt.v1",
    }), encoding="utf-8")
    damaged_failure.write_text("{ damaged", encoding="utf-8")
    unrelated_failure.write_text(json.dumps({"schema": "legacy.failure.v1"}),
                                 encoding="utf-8")
    (unrelated_failure.parent / "registration_receipt.json").write_text(
        json.dumps({"schema": "legacy.registration.v1"}), encoding="utf-8")

    assert _report_paths(source) == sorted((missing_registration, damaged_failure))


def test_report_discovery_surfaces_unreadable_failure_receipt(tmp_path, monkeypatch):
    source = tmp_path / "reports"
    failure = source / "run" / "failure_receipt.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"schema":"legacy.failure.v1"}', encoding="utf-8")
    original_read_text = type(failure).read_text

    def deny_failure_read(path, *args, **kwargs):
        if path == failure:
            raise PermissionError(13, "permission denied", str(failure))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(failure), "read_text", deny_failure_read)

    assert _report_paths(source) == [failure]
    result = reconcile(source, tmp_path / "research.sqlite", apply=False)
    assert result["reports_discovered"] == 1
    assert result["reports_accepted"] == 0
    assert result["reports_rejected"] == 1
    with pytest.raises(ValueError, match="refusing partial reconciliation"):
        reconcile(source, tmp_path / "research.sqlite", apply=True)


def test_report_discovery_surfaces_unreadable_registration_receipt(tmp_path, monkeypatch):
    source = tmp_path / "reports"
    failure = source / "run" / "failure_receipt.json"
    registration = failure.with_name("registration_receipt.json")
    failure.parent.mkdir(parents=True)
    failure.write_text('{"schema":"legacy.failure.v1"}', encoding="utf-8")
    registration.write_text('{"schema":"legacy.registration.v1"}', encoding="utf-8")
    original_read_text = type(failure).read_text

    def deny_registration_read(path, *args, **kwargs):
        if path == registration:
            raise PermissionError(13, "permission denied", str(registration))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(failure), "read_text", deny_registration_read)

    assert _report_paths(source) == [failure]
    result = reconcile(source, tmp_path / "research.sqlite", apply=False)
    assert result["reports_discovered"] == 1
    assert result["reports_accepted"] == 0
    assert result["reports_rejected"] == 1
    with pytest.raises(ValueError, match="refusing partial reconciliation"):
        reconcile(source, tmp_path / "research.sqlite", apply=True)


def test_report_discovery_fails_closed_when_subdirectory_cannot_be_scanned(
        tmp_path, monkeypatch):
    import qa.world_model.reconcile_routeb_research_ledger as reconciler

    source = tmp_path / "reports"
    source.mkdir()
    (source / "r4_e2_e3_comparison_report.json").write_text("{}", encoding="utf-8")
    original_walk = reconciler.os.walk

    def walk_with_unreadable_directory(root, *, followlinks=False, onerror=None):
        yield from original_walk(root, followlinks=followlinks, onerror=onerror)
        assert onerror is not None
        onerror(PermissionError(13, "permission denied", str(source / "locked")))

    monkeypatch.setattr(reconciler.os, "walk", walk_with_unreadable_directory)

    with pytest.raises(ValueError, match="cannot completely scan report source"):
        reconcile(source, tmp_path / "research.sqlite", apply=False)

    with pytest.raises(ValueError, match="cannot completely scan report source"):
        reconcile(source, tmp_path / "research.sqlite", apply=True)


def test_report_discovery_fails_closed_on_untraversed_symlink_directory(
        tmp_path, monkeypatch):
    import qa.world_model.reconcile_routeb_research_ledger as reconciler

    source = tmp_path / "reports"
    source.mkdir()
    linked = source / "linked-reports"
    original_is_symlink = type(linked).is_symlink

    def walk_with_linked_directory(root, *, followlinks=False, onerror=None):
        assert followlinks is False
        yield str(source), [linked.name], []

    def is_symlink(path):
        return path == linked or original_is_symlink(path)

    monkeypatch.setattr(reconciler.os, "walk", walk_with_linked_directory)
    monkeypatch.setattr(type(linked), "is_symlink", is_symlink)

    with pytest.raises(ValueError, match="symlink directory is not traversed"):
        reconcile(source, tmp_path / "research.sqlite", apply=False)

    with pytest.raises(ValueError, match="symlink directory is not traversed"):
        reconcile(source, tmp_path / "research.sqlite", apply=True)


@pytest.mark.parametrize(
    ("failure_content", "registration_content"),
    [
        (
            '{"schema":"routeb.registered-r4-e2-e3-failure.v2",'
            '"schema":"legacy.failure.v1","status":"FAILED"}',
            None,
        ),
        (
            '{"schema":"\\u0072outeb.registered-r4-e2-e3-failure.v2",'
            '"schema":"legacy.failure.v1","status":"FAILED"}',
            None,
        ),
        (
            "{ damaged",
            '{"schema":"routeb.registered-r4-e2-e3-receipt.v1",'
            '"schema":"legacy.registration.v1"}',
        ),
    ],
)
def test_reconciliation_fails_closed_on_ambiguous_r4_failure_receipts(
        tmp_path, failure_content, registration_content):
    source = tmp_path / "reports"
    failure = source / "run" / "failure_receipt.json"
    failure.parent.mkdir(parents=True)
    failure.write_text(failure_content, encoding="utf-8")
    if registration_content is not None:
        (failure.parent / "registration_receipt.json").write_text(
            registration_content, encoding="utf-8",
        )

    assert _report_paths(source) == [failure]
    result = reconcile(source, tmp_path / "research.sqlite", apply=False)
    assert result["reports_discovered"] == 1
    assert result["reports_accepted"] == 0
    assert result["reports_rejected"] == 1
    assert "cannot read canonical JSON report" in result["rejected"][0]["error"]
    with pytest.raises(ValueError, match="refusing partial reconciliation"):
        reconcile(source, tmp_path / "research.sqlite", apply=True)


def test_reconciliation_rejects_duplicate_keys_in_canonical_report(tmp_path):
    source = tmp_path / "reports"
    source.mkdir()
    report = source / "r4_e2_e3_comparison_report.json"
    report.write_text(
        '{"schema":"routeb.registered-r4-e2-e3-comparison.v1",'
        '"status":"COMPLETED","status":"FAILED"}',
        encoding="utf-8",
    )

    result = reconcile(source, tmp_path / "research.sqlite", apply=False)
    assert result["reports_discovered"] == 1
    assert result["reports_accepted"] == 0
    assert result["reports_rejected"] == 1
    assert "cannot read canonical JSON report" in result["rejected"][0]["error"]


def test_sidecar_reader_rejects_resolved_path_outside_report_tree(tmp_path, monkeypatch):
    report_root = tmp_path / "report"
    report_root.mkdir()
    external = tmp_path / "external-run.json"
    external.write_text('{"status":"COMPLETED"}', encoding="utf-8")
    sidecar = report_root / "run.json"
    sidecar.write_text('{"status":"COMPLETED"}', encoding="utf-8")
    original_resolve = type(sidecar).resolve

    def resolve_with_external_target(path, strict=False):
        if path == sidecar:
            return original_resolve(external, strict=strict)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(type(sidecar), "resolve", resolve_with_external_target)

    with pytest.raises(ValueError, match="escapes its report directory"):
        _read_artifact_bytes(sidecar, report_root, "test sidecar")


def test_reconcile_rejects_report_resolved_outside_selected_directory(tmp_path, monkeypatch):
    import qa.world_model.reconcile_routeb_research_ledger as reconciler

    source = tmp_path / "reports"
    source.mkdir()
    report = source / "r4_e2_e3_comparison_report.json"
    report.write_text("{}", encoding="utf-8")
    outside = tmp_path / "external-report.json"
    outside.write_text("{}", encoding="utf-8")
    original_resolve = type(report).resolve
    external_path = original_resolve(outside)

    def resolve_with_external_target(path, strict=False):
        if path == report:
            return external_path
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(type(report), "resolve", resolve_with_external_target)
    monkeypatch.setattr(reconciler, "build_plan", lambda *_args: pytest.fail(
        "outside report must be rejected before parsing"))

    result = reconcile(source, tmp_path / "research.sqlite", apply=False)
    assert result["reports_discovered"] == 1
    assert result["reports_accepted"] == 0
    assert result["reports_rejected"] == 1
    assert "escapes the selected source directory" in result["rejected"][0]["error"]


def test_report_discovery_does_not_read_failure_symlink_target(tmp_path, monkeypatch):
    source = tmp_path / "reports"
    source.mkdir()
    failure = source / "failure_receipt.json"
    failure.write_text(json.dumps({
        "schema": "routeb.registered-r4-e2-e3-failure.v2", "status": "FAILED",
    }), encoding="utf-8")
    outside = tmp_path / "external-failure.json"
    outside.write_text(json.dumps({
        "schema": "routeb.registered-r4-e2-e3-failure.v2", "status": "FAILED",
    }), encoding="utf-8")
    original_resolve = type(failure).resolve
    external_path = original_resolve(outside)
    original_read_text = type(failure).read_text

    def resolve_with_external_target(path, strict=False):
        if path == failure:
            return external_path
        return original_resolve(path, strict=strict)

    def reject_external_read(path, *args, **kwargs):
        if path == external_path:
            raise AssertionError("failure discovery read outside source root")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(failure), "resolve", resolve_with_external_target)
    monkeypatch.setattr(type(failure), "read_text", reject_external_read)

    assert _report_paths(source) == [failure]


def test_completed_r4_comparison_reconciliation_is_exact_and_idempotent(tmp_path, monkeypatch):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    label = "routeb-fold-adausdt-r4-reconcile-v1"
    fold_sha = "a" * 64
    feature_sha = "b" * 64
    dataset_sha = "c" * 64
    source_identity = {
        "schema_version": "0.4.1", "snapshot_hash": dataset_sha,
        "file_sha256": "d" * 64, "marker_sha256": "e" * 64, "events": 42,
    }
    instrument = "ADAUSDT-PERP.BINANCE"
    window = {"start_ns": 100, "end_ns": 200}
    config = {"environment": {"instrument": instrument,
                               "market_path": str(tmp_path / "market.parquet")}}
    reservation = research_access.reserve_routeb_fresh_fold(
        config, window, label=label, fold_sha256=fold_sha,
        repository_commit="f" * 40, source_code_sha256="1" * 64,
        source_identity=source_identity)
    identity_sha = research_access.routeb_fresh_fold_access_identity_sha256(
        config, window, label=label, fold_sha256=fold_sha,
        repository_commit="f" * 40, source_code_sha256="1" * 64,
        source_identity=source_identity)

    protocol = {
        "schema": "routeb.r4-e2-e3-baseline-comparison.v1",
        "instrument": instrument, "training_seeds": [31, 32, 33],
        "e2_training_steps": 2, "e3_training_steps": 2,
    }
    report = {
        "schema": "routeb.registered-r4-e2-e3-comparison.v1",
        "status": "COMPLETED", "test_payload_decoded": False,
        "decision": {"economic_certification": False, "e3_promoted": False},
        "protocol": protocol, "reservation_label": label,
        "instrument": instrument, "fold_hash": fold_sha,
        "feature_hash": feature_sha, "dataset_hash": dataset_sha,
        "source_identity": source_identity,
        "repository_commit": "f" * 40, "source_code_sha256": "1" * 64,
        "access_key": reservation["access_key"], "access_window": window,
        "access_identity_sha256": identity_sha,
        "training_runs": {},
    }
    report_path = tmp_path / "r4_e2_e3_comparison_report.json"
    for candidate in ("R4_E2_SINGLE_SCALE", "R4_E3_HIERARCHY"):
        for seed in protocol["training_seeds"]:
            steps = protocol["e2_training_steps" if candidate == "R4_E2_SINGLE_SCALE"
                             else "e3_training_steps"]
            configuration = {"model": {"hidden": 8},
                              "training": {"seed": seed, "steps": steps}}
            run = {
                "status": "COMPLETED", "candidate": candidate, "seed": seed,
                "fold_hash": fold_sha, "dataset_hash": dataset_sha,
                "test_payload_decoded": False, "train_config": configuration,
                "training": {"optimizer_updates": steps},
                "checkpoint_sha256": "2" * 64,
                "forecast": {"forecast_sha256": "3" * 64},
            }
            run_path = (tmp_path / candidate.lower() / f"seed-{seed}" / "run.json")
            run_path.parent.mkdir(parents=True, exist_ok=True)
            contents = json.dumps(run, sort_keys=True, separators=(",", ":")).encode()
            run_path.write_bytes(contents)
            report["training_runs"][f"{candidate}:{seed}"] = {
                "run_sha256": sha256(contents).hexdigest(),
            }
    _seal(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    # Model a hard exit after the ledger completion receipt but before the
    # runner writes research_completion_receipt.json.
    research_access.complete_routeb_fresh_fold(
        config, window, label=label, fold_sha256=fold_sha,
        repository_commit="f" * 40, source_code_sha256="1" * 64,
        source_identity=source_identity, report_sha256=report["report_sha256"],
        detail={"test_payload_decoded": False, "e3_promoted": False})

    plan = build_plan(report_path, registry_path)
    assert plan.access_key == reservation["access_key"]
    assert len(plan.candidates) == 6
    apply_plan(plan, registry_path)
    apply_plan(plan, registry_path)
    assert registry.development_result(reservation["access_key"]) == {
        "status": "completed", "report_sha256": report["report_sha256"],
        "detail": {"test_payload_decoded": False, "e3_promoted": False},
    }
    summary = registry.trial_attempt_summary(f"routeb-r4-baseline:{label}")
    assert summary["protocol_registered_attempt_count"] == 6
    assert all([event["status"] for event in attempt["events"]]
               == ["registered", "started", "completed"]
               for attempt in summary["attempts"])


def test_legacy_completed_r4_report_is_verified_read_only(tmp_path, monkeypatch):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    label = "routeb-fold-bnbusdt-r4-legacy-v1"
    fold_sha = "a" * 64
    instrument = "BNBUSDT-PERP.BINANCE"
    window = {"start_ns": 100, "end_ns": 200}
    source_identity = {"schema_version": "0.4.1", "snapshot_hash": "b" * 64,
                       "file_sha256": "c" * 64, "marker_sha256": "d" * 64,
                       "events": 42}
    config = {"environment": {"instrument": instrument,
                               "market_path": str(tmp_path / "market.parquet")}}
    reservation = research_access.reserve_routeb_fresh_fold(
        config, window, label=label, fold_sha256=fold_sha,
        repository_commit="e" * 40, source_code_sha256="f" * 64,
        source_identity=source_identity)
    registration = {
        "schema": "routeb.registered-r4-e2-e3-receipt.v1", "label": label,
        "access_key": reservation["access_key"], "fold_hash": fold_sha,
        "dataset_hash": source_identity["snapshot_hash"],
        "marker_sha256": source_identity["marker_sha256"],
        "archive_receipt_sha256": "4" * 64,
        "preregistration_sha256": "5" * 64,
        "repository_commit": "e" * 40, "source_code_sha256": "f" * 64,
        "protocol_sha256": "1" * 64, "fold_config_sha256": "2" * 64,
        "source_preparation_sha256": "3" * 64,
        "read_scope": {**window, "includes_test_payload": False},
    }
    report = {
        "schema": "routeb.registered-r4-e2-e3-comparison.v1",
        "status": "COMPLETED", "test_payload_decoded": False,
        "economic_certification": False,
        "protocol": {"schema": "routeb.r4-e2-e3-baseline-comparison.v1",
                     "instrument": instrument},
        "access_key": reservation["access_key"], "access_window": window,
        "fold_hash": fold_sha, "dataset_hash": source_identity["snapshot_hash"],
        "feature_hash": "6" * 64, "archive_receipt_sha256": "4" * 64,
        "preregistration_sha256": "5" * 64,
        "source_identity": source_identity, "repository_commit": "e" * 40,
        "source_code_sha256": "f" * 64, "protocol_sha256": "1" * 64,
        "fold_config_sha256": "2" * 64, "source_preparation_sha256": "3" * 64,
        "decision": {"economic_certification": False, "e3_promoted": False},
        "training_runs": {},
    }
    _seal(report)
    output = tmp_path / "legacy-r4"
    output.mkdir()
    report_path = output / "r4_e2_e3_comparison_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (output / "registration_receipt.json").write_text(
        json.dumps(registration), encoding="utf-8")
    (output / "research_completion_receipt.json").write_text(json.dumps({
        "schema": "routeb.development-result-receipt.v1", "status": "completed",
        "access_key": reservation["access_key"],
        "report_sha256": report["report_sha256"],
    }), encoding="utf-8")
    research_access.complete_routeb_fresh_fold(
        config, window, label=label, fold_sha256=fold_sha,
        repository_commit="e" * 40, source_code_sha256="f" * 64,
        source_identity=source_identity, report_sha256=report["report_sha256"],
        detail={"test_payload_decoded": False, "e3_promoted": False})

    plan = build_plan(report_path, registry_path)
    assert plan.read_only is True
    assert plan.candidates == ()
    prior_result = registry.development_result(reservation["access_key"])
    apply_plan(plan, registry_path)
    assert registry.development_result(reservation["access_key"]) == prior_result

    completion_path = output / "research_completion_receipt.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["report_sha256"] = "9" * 64
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="completion receipt does not bind"):
        build_plan(report_path, registry_path)


def test_legacy_failed_r4_report_is_verified_from_terminal_ledger_only(tmp_path, monkeypatch):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    label = "routeb-fold-adausdt-r4-legacy-failure-v1"
    fold_sha = "a" * 64
    instrument = "ADAUSDT-PERP.BINANCE"
    window = {"start_ns": 100, "end_ns": 200}
    source_identity = {"schema_version": "0.4.1", "snapshot_hash": "b" * 64,
                       "file_sha256": "c" * 64, "marker_sha256": "d" * 64,
                       "events": 42}
    config = {"environment": {"instrument": instrument,
                               "market_path": str(tmp_path / "market.parquet")}}
    reservation = research_access.reserve_routeb_fresh_fold(
        config, window, label=label, fold_sha256=fold_sha,
        repository_commit="e" * 40, source_code_sha256="f" * 64,
        source_identity=source_identity)
    registration = {
        "schema": "routeb.registered-r4-e2-e3-receipt.v1", "label": label,
        "access_key": reservation["access_key"], "fold_hash": fold_sha,
        "dataset_hash": source_identity["snapshot_hash"],
        "marker_sha256": source_identity["marker_sha256"],
        "repository_commit": "e" * 40, "source_code_sha256": "f" * 64,
        "read_scope": {**window, "includes_test_payload": False},
    }
    failure = {
        "schema": "routeb.registered-r4-e2-e3-failure.v1", "status": "FAILED",
        "error": "WorkerFailure", "detail": "worker terminated",
        "fold_hash": fold_sha, "test_payload_decoded": False,
    }
    _seal(failure)
    output = tmp_path / "legacy-failed-r4"
    output.mkdir()
    failure_path = output / "failure_receipt.json"
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    (output / "registration_receipt.json").write_text(
        json.dumps(registration), encoding="utf-8")
    attempt_key = research_access.register_routeb_r4_baseline_comparison_attempt(
        label=label, candidate_id="R4_E2_SINGLE_SCALE",
        configuration={"training": {"seed": 31, "steps": 2}},
        fold_sha256=fold_sha, feature_sha256="6" * 64, seed=31)
    research_access.record_routeb_candidate_event(
        attempt_key, event_key="started", status="started",
        detail={"access_key": reservation["access_key"], "fold_hash": fold_sha,
                "test_payload_decoded": False})
    research_access.record_routeb_candidate_event(
        attempt_key, event_key="failed", status="failed",
        detail={"error": "WorkerFailure", "test_payload_decoded": False})
    registry.record_development_result(
        key=reservation["access_key"], identity=reservation["identity"],
        status="failed", report_sha=failure["report_sha256"],
        detail={"error": "WorkerFailure", "test_payload_decoded": False})

    plan = build_plan(failure_path, registry_path)
    assert plan.read_only is True
    assert plan.result_status == "failed"
    prior_result = registry.development_result(reservation["access_key"])
    apply_plan(plan, registry_path)
    assert registry.development_result(reservation["access_key"]) == prior_result

    failure["detail"] = "changed"
    _seal(failure)
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical payload"):
        build_plan(failure_path, registry_path)


def test_legacy_failed_r4_report_rejects_unbound_attempt_lifecycle(tmp_path, monkeypatch):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    label = "routeb-fold-adausdt-r4-unbound-legacy-failure-v1"
    fold_sha = "a" * 64
    instrument = "ADAUSDT-PERP.BINANCE"
    window = {"start_ns": 100, "end_ns": 200}
    source_identity = {"schema_version": "0.4.1", "snapshot_hash": "b" * 64,
                       "file_sha256": "c" * 64, "marker_sha256": "d" * 64,
                       "events": 42}
    config = {"environment": {"instrument": instrument,
                               "market_path": str(tmp_path / "market.parquet")}}
    reservation = research_access.reserve_routeb_fresh_fold(
        config, window, label=label, fold_sha256=fold_sha,
        repository_commit="e" * 40, source_code_sha256="f" * 64,
        source_identity=source_identity)
    output = tmp_path / "unbound-legacy-failure"
    output.mkdir()
    failure = {"schema": "routeb.registered-r4-e2-e3-failure.v1",
               "status": "FAILED", "error": "WorkerFailure",
               "detail": "worker terminated", "fold_hash": fold_sha,
               "test_payload_decoded": False}
    _seal(failure)
    failure_path = output / "failure_receipt.json"
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    (output / "registration_receipt.json").write_text(json.dumps({
        "schema": "routeb.registered-r4-e2-e3-receipt.v1", "label": label,
        "access_key": reservation["access_key"], "fold_hash": fold_sha,
        "dataset_hash": source_identity["snapshot_hash"],
        "marker_sha256": source_identity["marker_sha256"],
        "repository_commit": "e" * 40, "source_code_sha256": "f" * 64,
        "read_scope": {**window, "includes_test_payload": False},
    }), encoding="utf-8")
    attempt_key = research_access.register_routeb_r4_baseline_comparison_attempt(
        label=label, candidate_id="R4_E2_SINGLE_SCALE",
        configuration={"training": {"seed": 31, "steps": 2}},
        fold_sha256=fold_sha, feature_sha256="6" * 64, seed=31)
    research_access.record_routeb_candidate_event(
        attempt_key, event_key="started", status="started",
        detail={"access_key": "0" * 64, "fold_hash": fold_sha,
                "test_payload_decoded": False})
    research_access.record_routeb_candidate_event(
        attempt_key, event_key="failed", status="failed",
        detail={"error": "WorkerFailure", "test_payload_decoded": False})
    registry.record_development_result(
        key=reservation["access_key"], identity=reservation["identity"],
        status="failed", report_sha=failure["report_sha256"],
        detail={"error": "WorkerFailure", "test_payload_decoded": False})

    with pytest.raises(ValueError, match="not uniquely bound"):
        build_plan(failure_path, registry_path)


def test_unresolved_r4_access_audit_is_read_only_and_never_releases_scope(tmp_path, monkeypatch):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    label = "routeb-fold-adausdt-r4-interrupted-v1"
    config = {"environment": {"instrument": "ADAUSDT-PERP.BINANCE",
                               "market_path": str(tmp_path / "market.parquet")}}
    source_identity = {"schema_version": "0.4.1", "snapshot_hash": "a" * 64,
                       "file_sha256": "b" * 64, "marker_sha256": "c" * 64,
                       "events": 42}
    reservation = research_access.reserve_routeb_fresh_fold(
        config, {"start_ns": 100, "end_ns": 200}, label=label,
        fold_sha256="d" * 64, repository_commit="e" * 40,
        source_code_sha256="f" * 64, source_identity=source_identity)
    key = research_access.register_routeb_r4_baseline_comparison_attempt(
        label=label, candidate_id="R4_E2_SINGLE_SCALE",
        configuration={"training": {"steps": 2}}, fold_sha256="d" * 64,
        feature_sha256="1" * 64, seed=31)
    research_access.record_routeb_candidate_event(
        key, event_key="started", status="started", detail={"access_key": reservation["access_key"]})
    sibling_fold_key = research_access.register_routeb_r4_baseline_comparison_attempt(
        label=label, candidate_id="R4_E2_SINGLE_SCALE",
        configuration={"training": {"steps": 2}}, fold_sha256="9" * 64,
        feature_sha256="2" * 64, seed=31)
    research_access.record_routeb_candidate_event(
        sibling_fold_key, event_key="started", status="started",
        detail={"access_key": "0" * 64})
    before = registry.trial_attempt_events(key)

    unresolved = _unresolved_r4_accesses(registry_path)
    assert unresolved == [{
        "access_key": reservation["access_key"], "label": label,
        "start_ns": 100, "end_ns": 200, "access_state": "recorded",
        "state": "attempts_unresolved",
        "attempts": [{"attempt_key": key, "candidate_id": "R4_E2_SINGLE_SCALE",
                      "last_status": "started", "last_event_key": "started"}],
        "action": "manual_evidence_review_required; consumed scope is not reusable",
    }]
    assert unresolved[0]["attempts"] == [{
        "attempt_key": key, "candidate_id": "R4_E2_SINGLE_SCALE",
        "last_status": "started", "last_event_key": "started",
    }]
    assert registry.trial_attempt_events(key) == before


def test_unresolved_r4_access_fails_closed_on_corrupt_development_identity(tmp_path):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    label = "routeb-fold-solusdt-r4-corrupt-identity-v1"
    key = digest(["test", label])
    registry.reserve_fresh_development(
        key=key, owner=digest(["owner", label]), market="SOLUSDT",
        family="perpetual-market-v1", start_ns=100, end_ns=200,
        identity={"label": label, "fold_sha256": "a" * 64},
    )
    with sqlite3.connect(registry_path) as connection:
        # Simulate persisted corruption outside the registry's immutable-write API.
        connection.execute("DROP TRIGGER access_identity_immutable")
        connection.execute("UPDATE access SET identity=? WHERE access_key=?",
                           ("{malformed", key))

    with pytest.raises(ValueError, match="invalid development access identity"):
        _unresolved_r4_accesses(registry_path)


def test_development_access_reader_rejects_non_object_identity(tmp_path):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    label = "routeb-fold-solusdt-r4-scalar-identity-v1"
    key = digest(["test", label])
    registry.reserve_fresh_development(
        key=key, owner=digest(["owner", label]), market="SOLUSDT",
        family="perpetual-market-v1", start_ns=100, end_ns=200,
        identity={"label": label, "fold_sha256": "c" * 64},
    )
    with sqlite3.connect(registry_path) as connection:
        connection.execute("DROP TRIGGER access_identity_immutable")
        connection.execute("UPDATE access SET identity=? WHERE access_key=?", ("null", key))

    with pytest.raises(ValueError, match="invalid identity object"):
        _read_development_accesses(registry_path)


def test_unresolved_r4_access_supports_uri_reserved_registry_path_characters(tmp_path):
    registry_path = tmp_path / "research#archive.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    label = "routeb-fold-solusdt-r4-uri-path-v1"
    key = digest(["test", label])
    registry.reserve_fresh_development(
        key=key, owner=digest(["owner", label]), market="SOLUSDT",
        family="perpetual-market-v1", start_ns=100, end_ns=200,
        identity={"label": label, "fold_sha256": "b" * 64},
    )

    unresolved = _unresolved_r4_accesses(registry_path)

    assert len(unresolved) == 1
    assert unresolved[0]["access_key"] == key
    assert unresolved[0]["state"] == "reservation_without_attempts"


def test_unresolved_r4_access_marks_unparseable_attempt_identity_ambiguous(tmp_path, monkeypatch):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    label = "routeb-fold-solusdt-r4-ambiguous-v1"
    config = {"environment": {"instrument": "SOLUSDT-PERP.BINANCE",
                               "market_path": str(tmp_path / "market.parquet")}}
    source_identity = {"schema_version": "0.4.1", "snapshot_hash": "a" * 64,
                       "file_sha256": "b" * 64, "marker_sha256": "c" * 64,
                       "events": 42}
    reservation = research_access.reserve_routeb_fresh_fold(
        config, {"start_ns": 100, "end_ns": 200}, label=label,
        fold_sha256="d" * 64, repository_commit="e" * 40,
        source_code_sha256="f" * 64, source_identity=source_identity)
    attempt_key = "1" * 64
    registry.register_trial_attempt(
        attempt_key=attempt_key, protocol_id=f"routeb-r4-baseline:{label}",
        candidate_id="R4_E2_SINGLE_SCALE", recipe={"training": {"seed": 31}},
        feature_sha="2" * 64, seed=31)

    unresolved = _unresolved_r4_accesses(registry_path)
    assert unresolved[0]["access_key"] == reservation["access_key"]
    assert unresolved[0]["state"] == "attempt_identity_ambiguous"
    assert unresolved[0]["attempts"] == [{
        "attempt_key": attempt_key, "candidate_id": "R4_E2_SINGLE_SCALE",
        "last_status": "registered", "last_event_key": "registered",
    }]


def test_r4_failure_receipt_reconciliation_is_exact_and_idempotent(tmp_path, monkeypatch):
    registry_path = tmp_path / "research.sqlite"
    registry = ProjectResearchRegistry(registry_path)
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    label = "routeb-fold-xrpusdt-r4-failed-v1"
    fold_sha = "a" * 64
    feature_sha = "b" * 64
    dataset_sha = "c" * 64
    source_identity = {"schema_version": "0.4.1", "snapshot_hash": dataset_sha,
                       "file_sha256": "d" * 64, "marker_sha256": "e" * 64,
                       "events": 42}
    instrument = "XRPUSDT-PERP.BINANCE"
    window = {"start_ns": 100, "end_ns": 200}
    config = {"environment": {"instrument": instrument,
                               "market_path": str(tmp_path / "market.parquet")}}
    reservation = research_access.reserve_routeb_fresh_fold(
        config, window, label=label, fold_sha256=fold_sha,
        repository_commit="f" * 40, source_code_sha256="1" * 64,
        source_identity=source_identity)
    identity_sha = research_access.routeb_fresh_fold_access_identity_sha256(
        config, window, label=label, fold_sha256=fold_sha,
        repository_commit="f" * 40, source_code_sha256="1" * 64,
        source_identity=source_identity)
    started_key = research_access.register_routeb_r4_baseline_comparison_attempt(
        label=label, candidate_id="R4_E2_SINGLE_SCALE",
        configuration={"training": {"seed": 31, "steps": 2}},
        fold_sha256=fold_sha, feature_sha256=feature_sha, seed=31)
    research_access.record_routeb_candidate_event(
        started_key, event_key="started", status="started",
        detail={"access_key": reservation["access_key"]})
    completed_key = research_access.register_routeb_r4_baseline_comparison_attempt(
        label=label, candidate_id="R4_E3_HIERARCHY",
        configuration={"training": {"seed": 31, "steps": 2}},
        fold_sha256=fold_sha, feature_sha256=feature_sha, seed=31)
    research_access.record_routeb_candidate_event(
        completed_key, event_key="started", status="started", detail={})
    research_access.record_routeb_candidate_event(
        completed_key, event_key="completed", status="completed", detail={})

    output = tmp_path / "interrupted-run"
    output.mkdir()
    registration_path = output / "registration_receipt.json"
    registration_path.write_text(json.dumps({
        "schema": "routeb.registered-r4-e2-e3-receipt.v1", "label": label,
        "access_key": reservation["access_key"], "fold_hash": fold_sha,
        "dataset_hash": dataset_sha, "marker_sha256": source_identity["marker_sha256"],
        "instrument": instrument, "repository_commit": "f" * 40,
        "source_code_sha256": "1" * 64, "access_identity_sha256": identity_sha,
        "read_scope": {**window, "includes_test_payload": False},
    }), encoding="utf-8")
    failure = {"schema": "routeb.registered-r4-e2-e3-failure.v2", "status": "FAILED",
               "error": "RuntimeError", "detail": "worker interruption",
               "fold_hash": fold_sha, "reservation_label": label,
               "instrument": instrument, "access_key": reservation["access_key"],
               "access_identity_sha256": identity_sha,
               "test_payload_decoded": False}
    _seal(failure)
    failure_path = output / "failure_receipt.json"
    failure_path.write_text(json.dumps(failure), encoding="utf-8")

    mismatched_failure = {**failure, "access_identity_sha256": "9" * 64}
    mismatched_failure.pop("report_sha256")
    _seal(mismatched_failure)
    failure_path.write_text(json.dumps(mismatched_failure), encoding="utf-8")
    with pytest.raises(ValueError, match="failure and registration receipts disagree"):
        build_plan(failure_path, registry_path)
    failure_path.write_text(json.dumps(failure), encoding="utf-8")

    external_registration = tmp_path / "external-registration.json"
    external_registration.write_bytes(registration_path.read_bytes())
    original_resolve = type(registration_path).resolve
    external_path = original_resolve(external_registration)

    def resolve_with_external_registration(path, strict=False):
        if path == registration_path:
            return external_path
        return original_resolve(path, strict=strict)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(type(registration_path), "resolve",
                              resolve_with_external_registration)
        with pytest.raises(ValueError, match="escapes its report directory"):
            build_plan(failure_path, registry_path)

    plan = build_plan(failure_path, registry_path)
    assert plan.result_status == "failed"
    assert plan.failed_attempts == ((started_key, {
        "error": "RuntimeError", "test_payload_decoded": False,
    }),)
    apply_plan(plan, registry_path)
    apply_plan(plan, registry_path)
    assert registry.development_result(reservation["access_key"]) == {
        "status": "failed", "report_sha256": failure["report_sha256"],
        "detail": {"error": "RuntimeError", "test_payload_decoded": False},
    }
    assert [event["status"] for event in registry.trial_attempt_events(started_key)] == [
        "registered", "started", "failed",
    ]
    assert [event["status"] for event in registry.trial_attempt_events(completed_key)] == [
        "registered", "started", "completed",
    ]
    assert _unresolved_r4_accesses(registry_path) == []


def test_reconciliation_rejects_nonstandard_json_constant_in_report(tmp_path):
    source = tmp_path / "reports"
    source.mkdir()
    report = source / "r4_e2_e3_comparison_report.json"
    report.write_text(
        '{"schema":"routeb.registered-r4-e2-e3-comparison.v1",'
        '"status":"COMPLETED","detail":{"untrusted":NaN}}',
        encoding="utf-8",
    )

    result = reconcile(source, tmp_path / "research.sqlite", apply=False)
    assert result["reports_discovered"] == 1
    assert result["reports_accepted"] == 0
    assert result["reports_rejected"] == 1
    assert "cannot read canonical JSON report" in result["rejected"][0]["error"]
