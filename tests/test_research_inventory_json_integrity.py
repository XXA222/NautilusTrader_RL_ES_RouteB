import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from rles_data.core import MAX_NS

from qa import research_access_inventory, research_trial_inventory


@pytest.mark.parametrize("reader", [
    research_access_inventory._read_json_object,
    research_trial_inventory._read_json_object,
])
@pytest.mark.parametrize("payload", [
    b'{"scope":["first"],"scope":["second"]}',
    b'{"scope":\xff}',
    b'["not", "an", "object"]',
])
def test_legacy_inventory_json_reader_rejects_ambiguous_or_invalid_input(
        tmp_path: Path, reader, payload: bytes) -> None:
    source = tmp_path / "inventory.json"
    source.write_bytes(payload)

    with pytest.raises(ValueError, match="JSON"):
        reader(source)


def test_legacy_inventory_quarantines_unknown_holdout_without_reading_report(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path.resolve()
    scan_root = root / ".tmp/real-ethusdt-scenario"
    run_root = scan_root / "run"
    run_root.mkdir(parents=True)
    (root / ".tmp/wfo-validation-recovery-20261013").mkdir(parents=True)
    db = run_root / "research.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE protocol (id TEXT, state TEXT, holdout_sha TEXT, holdout_start_ns INTEGER)"
        )
        connection.execute(
            "INSERT INTO protocol VALUES ('p-1', 'HOLDOUT_COMPLETE', 'a' || printf('%064d', 1), 100)"
        )
        connection.execute(
            "CREATE TABLE query (query_id TEXT, protocol_id TEXT, kind TEXT, candidate TEXT, fold_id TEXT)"
        )
    manifest_path = run_root / "wfo-run.json"
    manifest_path.write_text(
        '{"dataset_hash":"' + "a" * 64 + '","fold_manifests":[]}', encoding="utf-8",
    )
    report_path = run_root / "holdout-report.json"
    report_path.write_text('{"account_rows":[{"ts_ns":999}]}', encoding="utf-8")
    monkeypatch.setattr(
        research_access_inventory.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=str(manifest_path) + "\n", returncode=0),
    )
    original_reader = research_access_inventory._read_json_object

    def reject_holdout_report(path: Path) -> dict:
        if path.name == "holdout-report.json":
            pytest.fail("inventory must not read Holdout account rows")
        return original_reader(path)

    monkeypatch.setattr(research_access_inventory, "_read_json_object", reject_holdout_report)
    result = research_access_inventory.inventory(root)

    assert result["unresolved"] == [{
        "database": db.relative_to(root).as_posix(),
        "reason": "unknown Holdout boundary; quarantined",
    }]
    holdout = next(row for row in result["entries"] if row["purpose"] == "holdout")
    assert holdout["start_ns"] == 100
    assert holdout["end_ns"] == MAX_NS


def test_legacy_trial_inventory_rejects_scope_traversal_before_scan(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    manifest_path = root / "docs/audit_evidence/20261016/legacy_research_access_v2.json"
    manifest_path.parent.mkdir(parents=True)
    payload = ('{"schema":"rles.legacy-research-access.v1",'
               '"scope":["../../outside"],"databases":[]}')
    manifest_path.write_text(payload, encoding="utf-8")
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
        research_trial_inventory._file_sha(manifest_path), encoding="ascii",
    )

    with pytest.raises(ValueError, match="scan scope"):
        research_trial_inventory.inventory(root)


@pytest.mark.parametrize("helper", [
    research_access_inventory._contained_path,
    research_trial_inventory._contained_path,
])
def test_legacy_inventory_path_guard_rejects_parent_traversal(
        tmp_path: Path, helper) -> None:
    with pytest.raises(ValueError, match="parent traversal"):
        helper(tmp_path, Path("../outside.json"), "fixture")


@pytest.mark.parametrize("helper", [
    research_access_inventory._contained_path,
    research_trial_inventory._contained_path,
])
def test_legacy_inventory_path_guard_rejects_symlink_before_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, helper) -> None:
    source = tmp_path / "linked.json"
    source.write_text('{"ignored":true}', encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == source or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(ValueError, match="symlink"):
        helper(tmp_path, Path("linked.json"), "fixture")


def test_legacy_trial_inventory_rejects_database_outside_frozen_roots(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    for relative_root in research_trial_inventory._EXPECTED_SCOPE:
        (root / relative_root).mkdir(parents=True)
    manifest_path = root / "docs/audit_evidence/20261016/legacy_research_access_v2.json"
    manifest_path.parent.mkdir(parents=True)
    payload = {
        "schema": "rles.legacy-research-access.v1",
        "scope": list(research_trial_inventory._EXPECTED_SCOPE),
        "databases": [{"path": "../../outside.sqlite", "sha256": "a" * 64}],
        "standalone_reports": [], "entries": [], "unresolved": [],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
        research_trial_inventory._file_sha(manifest_path), encoding="ascii",
    )

    with pytest.raises(ValueError, match="escapes frozen scan roots"):
        research_trial_inventory.inventory(root)


def test_legacy_trial_inventory_reports_missing_historical_root(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    manifest_path = root / "docs/audit_evidence/20261016/legacy_research_access_v2.json"
    manifest_path.parent.mkdir(parents=True)
    payload = {
        "schema": "rles.legacy-research-access.v1",
        "scope": list(research_trial_inventory._EXPECTED_SCOPE),
        "databases": [], "standalone_reports": [], "entries": [], "unresolved": [],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
        research_trial_inventory._file_sha(manifest_path), encoding="ascii",
    )

    with pytest.raises(ValueError, match="legacy research root missing"):
        research_trial_inventory.inventory(root)


def test_legacy_access_inventory_marks_missing_roots_and_discovery_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        research_access_inventory.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="", returncode=2),
    )
    result = research_access_inventory.inventory(tmp_path.resolve())

    missing_roots = {row["path"] for row in result["unresolved"] if "path" in row}
    assert missing_roots == {
        ".tmp/real-ethusdt-scenario",
        ".tmp/wfo-validation-recovery-20261013",
    }
    assert any(row.get("discovery_exit_code") == 2 for row in result["unresolved"])
