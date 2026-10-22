"""Reconcile completed Route B fold reports with the append-only research ledger.

The reconciler reads report metadata only.  It never opens the curated market
payload or any forecast array.  Dry-run is the default; ``--apply`` appends the
exact receipts and candidate lifecycle events already emitted by the current
registered-fold runner.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
import re
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

from rles_data.core import digest, require_hash
from rles_research.access import ProjectResearchRegistry


REPORT_SCHEMA = "routeb.multi-seed-fold-experiment.v1"
R4_REPORT_SCHEMA = "routeb.registered-r4-e2-e3-comparison.v1"
R4_FAILURE_SCHEMA = "routeb.registered-r4-e2-e3-failure.v2"
R4_LEGACY_FAILURE_SCHEMA = "routeb.registered-r4-e2-e3-failure.v1"
PROTOCOL_PREFIX = "routeb-r2:"
R4_PROTOCOL_PREFIX = "routeb-r4-baseline:"


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise json.JSONDecodeError(f"duplicate JSON object key: {key!r}", "", 0)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise json.JSONDecodeError(f"non-standard JSON constant: {value}", "", 0)


def _strict_json_loads(content: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicate keys and non-standard constants."""
    return json.loads(content, object_pairs_hook=_json_object_without_duplicate_keys,
                      parse_constant=_reject_json_constant)


def _read_only_sqlite_uri(path: Path) -> str:
    """Build an SQLite file URI without interpreting path characters as URI syntax."""
    return f"file:{quote(path.resolve().as_posix(), safe='/:')}?mode=ro"


def _contains_r4_receipt_schema(text: str) -> bool:
    """Find known R4 schema strings, including JSON-escaped spellings."""
    markers = {R4_FAILURE_SCHEMA, R4_LEGACY_FAILURE_SCHEMA,
               "routeb.registered-r4-e2-e3-receipt.v1"}
    for match in re.finditer(r'"(?:[^"\\]|\\.)*"', text):
        try:
            value = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if value in markers:
            return True
    return False


@dataclass(frozen=True)
class CandidateReceipt:
    attempt_key: str
    protocol_id: str
    candidate_id: str
    recipe: dict[str, Any]
    feature_sha256: str
    seed: int
    started_detail: dict[str, Any]
    completed_detail: dict[str, Any]


@dataclass(frozen=True)
class ReconciliationPlan:
    report_path: Path
    access_key: str
    access_identity: dict[str, Any]
    report_sha256: str
    report_schema: str
    candidates: tuple[CandidateReceipt, ...]
    result_status: str = "completed"
    result_detail: dict[str, Any] | None = None
    failed_attempts: tuple[tuple[str, dict[str, Any]], ...] = ()
    read_only: bool = False


def _canonical_report_sha256(report: dict[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return require_hash(value)


def _read_artifact_bytes(path: Path, report_root: Path, name: str) -> bytes:
    """Read a report sidecar only when its resolved path stays in the report tree."""
    root = report_root.resolve()
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{name} is missing or unreadable") from exc
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes its report directory")
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"{name} is missing or unreadable") from exc


def _load_report(path: Path) -> dict[str, Any]:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read canonical JSON report: {path}") from exc
    report = _require_dict(value, "report")
    supported = {REPORT_SCHEMA, R4_REPORT_SCHEMA, R4_FAILURE_SCHEMA,
                 R4_LEGACY_FAILURE_SCHEMA}
    expected_status = ("FAILED" if report.get("schema") in
                       {R4_FAILURE_SCHEMA, R4_LEGACY_FAILURE_SCHEMA} else "COMPLETED")
    if report.get("schema") not in supported or report.get("status") != expected_status:
        raise ValueError("report is not a completed Route B fold experiment with supported schema")
    if report.get("test_payload_decoded") is not False:
        raise ValueError("report does not prove that the test payload stayed sealed")
    if expected_status == "FAILED":
        if not isinstance(report.get("error"), str) or not report["error"]:
            raise ValueError("failure receipt lacks an error type")
        if not isinstance(report.get("detail"), str):
            raise ValueError("failure receipt lacks error detail")
    elif report.get("economic_certification") is not False:
        decision = report.get("decision")
        if (report.get("schema") != R4_REPORT_SCHEMA or not isinstance(decision, dict)
                or decision.get("economic_certification") is not False):
            raise ValueError("historical predictive evidence must not claim economic certification")
    recorded = _require_sha256(report.get("report_sha256"), "report_sha256")
    if recorded != _canonical_report_sha256(report):
        raise ValueError("aggregate report SHA256 does not match its canonical payload")
    return report


def _read_development_accesses(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.is_file():
        raise ValueError(f"research registry does not exist: {registry_path}")
    try:
        with sqlite3.connect(_read_only_sqlite_uri(registry_path), uri=True) as connection:
            rows = connection.execute(
                "SELECT access_key,start_ns,end_ns,purpose,identity,state "
                "FROM access WHERE purpose='development'"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError("cannot read the project research registry") from exc
    result = []
    for key, start_ns, end_ns, purpose, encoded, state in rows:
        try:
            identity = _strict_json_loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"access {key} has invalid identity JSON") from exc
        result.append({
            "access_key": key, "start_ns": start_ns, "end_ns": end_ns,
            "purpose": purpose, "identity": identity, "state": state,
        })
    return result


def _matching_access(report: dict[str, Any], accesses: list[dict[str, Any]]) -> dict[str, Any]:
    read_scope = _require_dict(report.get("read_scope"), "read_scope")
    if read_scope.get("includes_test") is not False:
        raise ValueError("historical development read scope includes the test interval")
    start_ns, end_ns = read_scope.get("start_ns"), read_scope.get("end_ns")
    if type(start_ns) is not int or type(end_ns) is not int or start_ns >= end_ns:
        raise ValueError("invalid historical development read scope")
    source_identity = _require_dict(report.get("source_identity"), "source_identity")
    fold_manifest = _require_dict(report.get("fold_manifest"), "fold_manifest")
    if (report.get("dataset_hash") != source_identity.get("snapshot_hash")
            or fold_manifest.get("dataset_hash") != report.get("dataset_hash")
            or report.get("marker_sha256") != source_identity.get("marker_sha256")):
        raise ValueError("report source, marker and Fold dataset identities disagree")
    expected = {
        "label": report.get("reservation_label"),
        "fold_sha256": report.get("fold_hash"),
        "repository_commit": report.get("repository_commit"),
        "source_code_sha256": report.get("source_code_sha256"),
        "source_identity": source_identity,
        "window": {"start_ns": start_ns, "end_ns": end_ns},
    }
    candidates = []
    for access in accesses:
        identity = access["identity"]
        if (access["state"] == "recorded"
                and access["start_ns"] == start_ns and access["end_ns"] == end_ns
                and identity.get("protocol") == "routeb-world-model-b0-fresh-fold-v0"
                and all(identity.get(name) == value for name, value in expected.items())):
            candidates.append(access)
    if len(candidates) != 1:
        raise ValueError(f"expected one exact recorded access, found {len(candidates)}")
    access = candidates[0]
    embedded = report.get("research_access")
    if embedded is not None:
        embedded = _require_dict(embedded, "research_access")
        if embedded.get("access_key") != access["access_key"]:
            raise ValueError("report embeds a different research access key")
    return access


def _matching_r4_access(report: dict[str, Any], accesses: list[dict[str, Any]]) -> dict[str, Any]:
    from rles_orchestrator.research_access import routeb_fresh_fold_access_key

    protocol = _require_dict(report.get("protocol"), "protocol")
    source_identity = _require_dict(report.get("source_identity"), "source_identity")
    window = _require_dict(report.get("access_window"), "access_window")
    label = report.get("reservation_label")
    if (protocol.get("schema") != "routeb.r4-e2-e3-baseline-comparison.v1"
            or not isinstance(label, str) or not label.startswith("routeb-fold-")
            or protocol.get("instrument") != report.get("instrument")
            or report.get("dataset_hash") != source_identity.get("snapshot_hash")
            or type(window.get("start_ns")) is not int
            or type(window.get("end_ns")) is not int
            or window["start_ns"] >= window["end_ns"]
            or report.get("test_payload_decoded") is not False):
        raise ValueError("R4 report source, Fold window, or sealed Test identity is invalid")
    access_key = _require_sha256(report.get("access_key"), "access_key")
    matches = [access for access in accesses if access["access_key"] == access_key]
    if len(matches) != 1 or matches[0]["state"] != "recorded":
        raise ValueError(f"expected one exact R4 development access, found {len(matches)}")
    access = matches[0]
    identity = access["identity"]
    environment = _require_dict(identity.get("environment"), "ledger environment")
    if (access["start_ns"] != window["start_ns"]
            or access["end_ns"] != window["end_ns"]
            or report.get("fold_hash") != identity.get("fold_sha256")
            or report.get("repository_commit") != identity.get("repository_commit")
            or report.get("source_code_sha256") != identity.get("source_code_sha256")
            or identity.get("label") != label
            or environment.get("instrument") != report.get("instrument")
            or identity.get("source_identity") != source_identity
            or identity.get("window") != window):
        raise ValueError("R4 report differs from its exact recorded access identity")
    config = {"environment": environment, "allow_legacy_isolated_snapshot": False}
    expected_key = routeb_fresh_fold_access_key(
        config, window, label=label, fold_sha256=report["fold_hash"],
        repository_commit=report["repository_commit"],
        source_code_sha256=report["source_code_sha256"],
        source_identity=source_identity,
    )
    if expected_key != access_key or identity.get("protocol") != "routeb-world-model-b0-fresh-fold-v0":
        raise ValueError("R4 report access key does not reproduce its frozen ledger identity")
    if digest(identity) != _require_sha256(
            report.get("access_identity_sha256"), "access_identity_sha256"):
        raise ValueError("R4 report does not bind the complete recorded access identity")
    return access


def _matching_completed_legacy_r4(report: dict[str, Any], accesses: list[dict[str, Any]],
                                  report_path: Path,
                                  registry_path: str | Path) -> dict[str, Any]:
    """Validate pre-identity-field R4 reports only when both completion records exist.

    Older R4 v1 reports omitted top-level label/instrument/identity digest. The
    registration receipt and ledger still bind these values. Reconciliation is
    deliberately audit-only: it cannot backfill attempts or write a result.
    """
    try:
        registration = _strict_json_loads(_read_artifact_bytes(
            report_path.parent / "registration_receipt.json", report_path.parent,
            "legacy R4 registration receipt",
        ))
        completion = _strict_json_loads(_read_artifact_bytes(
            report_path.parent / "research_completion_receipt.json", report_path.parent,
            "legacy R4 completion receipt",
        ))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("legacy R4 report receipts are invalid JSON") from exc
    registration = _require_dict(registration, "legacy R4 registration receipt")
    completion = _require_dict(completion, "legacy R4 completion receipt")
    protocol = _require_dict(report.get("protocol"), "protocol")
    source_identity = _require_dict(report.get("source_identity"), "source_identity")
    window = _require_dict(report.get("access_window"), "access_window")
    scope = _require_dict(registration.get("read_scope"), "registration read scope")
    label = registration.get("label")
    access_key = _require_sha256(report.get("access_key"), "access_key")
    if (registration.get("schema") != "routeb.registered-r4-e2-e3-receipt.v1"
            or not isinstance(label, str) or not label.startswith("routeb-fold-")
            or registration.get("access_key") != access_key
            or protocol.get("schema") != "routeb.r4-e2-e3-baseline-comparison.v1"
            or not isinstance(protocol.get("instrument"), str)
            or report.get("dataset_hash") != source_identity.get("snapshot_hash")
            or type(window.get("start_ns")) is not int
            or type(window.get("end_ns")) is not int
            or window["start_ns"] >= window["end_ns"]
            or scope.get("start_ns") != window["start_ns"]
            or scope.get("end_ns") != window["end_ns"]
            or scope.get("includes_test_payload") is not False
            or report.get("test_payload_decoded") is not False):
        raise ValueError("legacy R4 report and registration scope disagree")
    for report_name, receipt_name in (
        ("dataset_hash", "dataset_hash"), ("fold_hash", "fold_hash"),
        ("archive_receipt_sha256", "archive_receipt_sha256"),
        ("preregistration_sha256", "preregistration_sha256"),
        ("repository_commit", "repository_commit"),
        ("source_code_sha256", "source_code_sha256"),
        ("protocol_sha256", "protocol_sha256"),
        ("fold_config_sha256", "fold_config_sha256"),
        ("source_preparation_sha256", "source_preparation_sha256"),
    ):
        if report.get(report_name) != registration.get(receipt_name):
            raise ValueError(f"legacy R4 report differs from registration {receipt_name}")
    if (registration.get("marker_sha256") != source_identity.get("marker_sha256")
            or report.get("feature_hash") is None):
        raise ValueError("legacy R4 report source marker or feature identity is missing")
    if (completion.get("schema") != "routeb.development-result-receipt.v1"
            or completion.get("status") != "completed"
            or completion.get("access_key") != access_key
            or completion.get("report_sha256") != report.get("report_sha256")):
        raise ValueError("legacy R4 completion receipt does not bind this report")
    matches = [access for access in accesses if access["access_key"] == access_key]
    if len(matches) != 1 or matches[0]["state"] != "recorded":
        raise ValueError("legacy R4 report lacks one exact recorded development access")
    access = matches[0]
    identity = access["identity"]
    environment = _require_dict(identity.get("environment"), "ledger environment")
    if (access["start_ns"] != window["start_ns"]
            or access["end_ns"] != window["end_ns"]
            or identity.get("label") != label
            or identity.get("fold_sha256") != report.get("fold_hash")
            or identity.get("repository_commit") != report.get("repository_commit")
            or identity.get("source_code_sha256") != report.get("source_code_sha256")
            or environment.get("instrument") != protocol.get("instrument")
            or identity.get("source_identity") != source_identity
            or identity.get("window") != window):
        raise ValueError("legacy R4 report differs from its exact recorded access identity")
    from rles_orchestrator.research_access import routeb_fresh_fold_access_key
    if routeb_fresh_fold_access_key(
            {"environment": environment, "allow_legacy_isolated_snapshot": False}, window,
            label=label, fold_sha256=report["fold_hash"],
            repository_commit=report["repository_commit"],
            source_code_sha256=report["source_code_sha256"],
            source_identity=source_identity) != access_key:
        raise ValueError("legacy R4 access key does not reproduce its frozen ledger identity")
    try:
        registry_file = Path(registry_path).resolve()
        with sqlite3.connect(
                _read_only_sqlite_uri(registry_file), uri=True) as connection:
            result_row = connection.execute(
                "SELECT status,report_sha FROM development_result WHERE access_key=?",
                (access_key,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise ValueError("cannot read the legacy R4 completion from the project ledger") from exc
    recorded_result = ({"status": result_row[0], "report_sha256": result_row[1]}
                       if result_row is not None else None)
    if (not isinstance(recorded_result, dict)
            or recorded_result.get("status") != "completed"
            or recorded_result.get("report_sha256") != report.get("report_sha256")):
        raise ValueError("legacy R4 report is not already completed in the research ledger")
    return access


def _matching_failed_legacy_r4(report: dict[str, Any], accesses: list[dict[str, Any]],
                               report_path: Path,
                               registry_path: str | Path) -> dict[str, Any]:
    """Audit a v1 R4 failure only against its exact terminal ledger result."""
    try:
        registration = _strict_json_loads(_read_artifact_bytes(
            report_path.parent / "registration_receipt.json", report_path.parent,
            "legacy R4 failure registration receipt",
        ))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("legacy R4 failure registration receipt is invalid JSON") from exc
    registration = _require_dict(registration, "legacy R4 failure registration receipt")
    label = registration.get("label")
    access_key = _require_sha256(registration.get("access_key"), "registration access_key")
    fold_sha = _require_sha256(registration.get("fold_hash"), "registration fold_hash")
    scope = _require_dict(registration.get("read_scope"), "registration read_scope")
    scope_start, scope_end = scope.get("start_ns"), scope.get("end_ns")
    if (registration.get("schema") != "routeb.registered-r4-e2-e3-receipt.v1"
            or not isinstance(label, str) or "-r4-" not in label.lower()
            or report.get("fold_hash") != fold_sha
            or report.get("test_payload_decoded") is not False
            or type(scope_start) is not int
            or type(scope_end) is not int
            or scope.get("includes_test_payload") is not False):
        raise ValueError("legacy R4 failure and registration receipts disagree")
    if scope_start >= scope_end:
        raise ValueError("legacy R4 failure registration scope is empty or reversed")
    matches = [access for access in accesses if access["access_key"] == access_key]
    if len(matches) != 1 or matches[0]["state"] != "recorded":
        raise ValueError("legacy R4 failure lacks one exact recorded development access")
    access = matches[0]
    identity = access["identity"]
    environment = _require_dict(identity.get("environment"), "ledger environment")
    source_identity = _require_dict(identity.get("source_identity"), "ledger source identity")
    window = {"start_ns": scope_start, "end_ns": scope_end}
    if (access["start_ns"] != window["start_ns"]
            or access["end_ns"] != window["end_ns"]
            or identity.get("protocol") != "routeb-world-model-b0-fresh-fold-v0"
            or identity.get("availability_policy")
            != "closed_1m_bar;available_at_ns<=decision_ts_ns"
            or identity.get("window") != window
            or identity.get("label") != label
            or identity.get("fold_sha256") != fold_sha
            or identity.get("repository_commit") != registration.get("repository_commit")
            or identity.get("source_code_sha256") != registration.get("source_code_sha256")
            or not isinstance(environment.get("instrument"), str)
            or environment.get("curated_marker_sha256")
            != source_identity.get("marker_sha256")
            or environment.get("curated_read_mode") != "row_group_scoped_v1"
            or registration.get("dataset_hash") != source_identity.get("snapshot_hash")
            or registration.get("marker_sha256") != source_identity.get("marker_sha256")):
        raise ValueError("legacy R4 failure registration differs from its ledger access")
    from rles_orchestrator.research_access import routeb_fresh_fold_access_key
    if routeb_fresh_fold_access_key(
            {"environment": environment, "allow_legacy_isolated_snapshot": False}, window,
            label=label, fold_sha256=fold_sha,
            repository_commit=identity["repository_commit"],
            source_code_sha256=identity["source_code_sha256"],
            source_identity=source_identity) != access_key:
        raise ValueError("legacy R4 failure access key does not reproduce its ledger identity")
    try:
        registry_file = Path(registry_path).resolve()
        with sqlite3.connect(
                _read_only_sqlite_uri(registry_file), uri=True) as connection:
            row = connection.execute(
                "SELECT status,report_sha FROM development_result WHERE access_key=?",
                (access_key,),
            ).fetchone()
            attempts = connection.execute(
                "SELECT attempt_key,recipe_json FROM trial_attempt WHERE protocol_id=?",
                (f"{R4_PROTOCOL_PREFIX}{label}",),
            ).fetchall()
            bound_attempts: list[str] = []
            for attempt_key, recipe_encoded in attempts:
                try:
                    recipe = _strict_json_loads(recipe_encoded)
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(recipe, dict) or recipe.get("fold_sha256") != fold_sha:
                    continue
                events = connection.execute(
                    "SELECT event_key,status,detail_json FROM trial_attempt_event "
                    "WHERE attempt_key=? ORDER BY rowid", (attempt_key,),
                ).fetchall()
                started_positions = []
                failed_positions = []
                for position, (event_key, status, detail_encoded) in enumerate(events):
                    try:
                        detail = _strict_json_loads(detail_encoded)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if not isinstance(detail, dict):
                        continue
                    if (event_key == "started" and status == "started"
                            and detail.get("access_key") == access_key
                            and detail.get("fold_hash") == fold_sha
                            and detail.get("test_payload_decoded") is False):
                        started_positions.append(position)
                    if (event_key == "failed" and status == "failed"
                            and detail.get("error") == report.get("error")
                            and detail.get("test_payload_decoded") is False):
                        failed_positions.append(position)
                if any(started < failed for started in started_positions
                       for failed in failed_positions):
                    bound_attempts.append(attempt_key)
    except sqlite3.Error as exc:
        raise ValueError("cannot read the legacy R4 failure from the project ledger") from exc
    if row != ("failed", report.get("report_sha256")):
        raise ValueError("legacy R4 failure is not already terminal in the research ledger")
    if len(bound_attempts) != 1:
        raise ValueError("legacy R4 failure is not uniquely bound to a started ledger attempt")
    return access


def _matching_r4_failure_access(report: dict[str, Any], accesses: list[dict[str, Any]],
                                report_path: Path) -> dict[str, Any]:
    from rles_orchestrator.research_access import routeb_fresh_fold_access_key

    try:
        registration_bytes = _read_artifact_bytes(
            report_path.parent / "registration_receipt.json", report_path.parent,
            "R4 registration receipt",
        )
        registration = _strict_json_loads(registration_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("R4 failure receipt requires its registration receipt") from exc
    registration = _require_dict(registration, "registration receipt")
    scope = _require_dict(registration.get("read_scope"), "registration read scope")
    label = registration.get("label")
    access_key = _require_sha256(registration.get("access_key"), "registration access_key")
    fold_sha = _require_sha256(registration.get("fold_hash"), "registration fold_hash")
    commit = registration.get("repository_commit")
    source_code_sha = _require_sha256(registration.get("source_code_sha256"),
                                      "registration source_code_sha256")
    instrument = registration.get("instrument")
    if (registration.get("schema") != "routeb.registered-r4-e2-e3-receipt.v1"
            or not isinstance(label, str) or "-r4-" not in label.lower()
            or not isinstance(instrument, str)
            or report.get("access_key") != access_key
            or report.get("reservation_label") != label
            or report.get("instrument") != instrument
            or report.get("access_identity_sha256")
            != registration.get("access_identity_sha256")
            or type(scope.get("start_ns")) is not int
            or type(scope.get("end_ns")) is not int
            or scope.get("includes_test_payload") is not False
            or report.get("fold_hash") != fold_sha
            or report.get("test_payload_decoded") is not False):
        raise ValueError("R4 failure and registration receipts disagree")
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError("registration receipt has an invalid repository commit")
    matches = [access for access in accesses if access["access_key"] == access_key]
    if len(matches) != 1 or matches[0]["state"] != "recorded":
        raise ValueError("R4 failure receipt lacks one exact recorded development access")
    access = matches[0]
    identity = access["identity"]
    environment = _require_dict(identity.get("environment"), "ledger environment")
    source_identity = _require_dict(identity.get("source_identity"), "ledger source identity")
    window = {"start_ns": scope["start_ns"], "end_ns": scope["end_ns"]}
    if (access["start_ns"] != window["start_ns"] or access["end_ns"] != window["end_ns"]
            or identity.get("label") != label or identity.get("fold_sha256") != fold_sha
            or identity.get("repository_commit") != commit
            or identity.get("source_code_sha256") != source_code_sha
            or environment.get("instrument") != instrument
            or registration.get("dataset_hash") != source_identity.get("snapshot_hash")
            or registration.get("marker_sha256") != source_identity.get("marker_sha256")):
        raise ValueError("R4 registration receipt differs from its exact ledger access")
    config = {"environment": environment, "allow_legacy_isolated_snapshot": False}
    if routeb_fresh_fold_access_key(
            config, window, label=label, fold_sha256=fold_sha,
            repository_commit=commit, source_code_sha256=source_code_sha,
            source_identity=source_identity) != access_key:
        raise ValueError("registration receipt access key does not reproduce ledger identity")
    if digest(identity) != _require_sha256(
            registration.get("access_identity_sha256"), "access_identity_sha256"):
        raise ValueError("registration receipt does not bind the full ledger identity")
    return access


def _r4_failure_attempts(registry_path: str | Path, label: str, fold_sha: str,
                         error: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    path = Path(registry_path).resolve()
    protocol_id = f"{R4_PROTOCOL_PREFIX}{label}"
    try:
        connection = sqlite3.connect(_read_only_sqlite_uri(path), uri=True)
        try:
            rows = connection.execute(
                "SELECT a.attempt_key,a.candidate_id,a.feature_sha,a.seed,a.recipe_json "
                "FROM trial_attempt AS a WHERE a.protocol_id=? ORDER BY a.candidate_id,a.seed",
                (protocol_id,),
            ).fetchall()
            result = []
            for attempt_key, candidate_id, feature_sha, seed, recipe_json in rows:
                try:
                    recipe_value = _strict_json_loads(recipe_json)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError("R4 attempt recipe is invalid JSON") from exc
                recipe = _require_dict(recipe_value, "R4 attempt recipe")
                feature_sha = _require_sha256(feature_sha, "R4 attempt feature_sha")
                if (candidate_id not in {"R4_E2_SINGLE_SCALE", "R4_E3_HIERARCHY"}
                        or type(seed) is not int or not 0 <= seed < 2**32
                        or not isinstance(feature_sha, str)
                        or recipe.get("candidate") != candidate_id
                        or recipe.get("fold_sha256") != fold_sha):
                    raise ValueError("R4 failure receipt conflicts with registered attempt identity")
                expected_key = digest([
                    "routeb-world-model-r4-baseline-comparison-v1", protocol_id,
                    candidate_id, seed, feature_sha, recipe,
                ])
                if attempt_key != expected_key:
                    raise ValueError("R4 attempt key differs from its frozen recipe")
                events = connection.execute(
                    "SELECT event_key,status,detail_json FROM trial_attempt_event "
                    "WHERE attempt_key=? ORDER BY rowid", (attempt_key,),
                ).fetchall()
                if not events:
                    raise ValueError("R4 attempt is missing lifecycle events")
                _, last_status, last_detail_json = events[-1]
                if last_status in {"registered", "started", "resumed"}:
                    result.append((attempt_key, {"error": error,
                                                  "test_payload_decoded": False}))
                elif last_status == "failed":
                    try:
                        last_detail = _strict_json_loads(last_detail_json)
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ValueError("R4 failed attempt detail is invalid JSON") from exc
                    if last_detail != {"error": error, "test_payload_decoded": False}:
                        raise ValueError("R4 failure receipt conflicts with failed attempt detail")
                elif last_status == "completed":
                    continue
                else:
                    raise ValueError(f"unsupported R4 attempt terminal state: {last_status}")
            return tuple(result)
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ValueError("cannot inspect R4 attempts for failure reconciliation") from exc


def _historical_configuration(report_path: Path, seed: int, candidate_id: str,
                              run: dict[str, Any]) -> dict[str, Any]:
    """Recover an old runner's omitted model config from its sealed sidecar."""
    configuration_path = report_path.parent / f"seed-{seed}" / (
        f"{candidate_id.lower()}_configuration.json"
    )
    expected_file_sha = _require_sha256(
        run.get("configuration_sha256"), "configuration_sha256",
    )
    try:
        content = _read_artifact_bytes(
            configuration_path, report_path.parent, "historical configuration sidecar",
        )
        configuration = _strict_json_loads(content)
    except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"candidate {candidate_id} seed {seed} lacks a readable configuration sidecar"
        ) from exc
    if sha256(content).hexdigest() != expected_file_sha:
        raise ValueError(f"candidate {candidate_id} seed {seed} configuration sidecar changed")
    configuration = _require_dict(configuration, "historical candidate configuration")
    if set(configuration) != {"model", "training"}:
        raise ValueError("historical candidate configuration has an unexpected schema")
    _require_dict(configuration["model"], "historical model configuration")
    _require_dict(configuration["training"], "historical training configuration")
    return configuration


def _candidate_receipts(report: dict[str, Any], access_key: str,
                        report_path: Path) -> tuple[CandidateReceipt, ...]:
    label = report.get("reservation_label")
    if not isinstance(label, str) or not label.startswith("routeb-fold-"):
        raise ValueError("invalid Route B reservation label")
    fold_sha = _require_sha256(report.get("fold_hash"), "fold_hash")
    feature_sha = _require_sha256(
        _require_dict(report.get("fold_manifest"), "fold_manifest").get("feature_hash"),
        "feature_hash",
    )
    seeds = report.get("training_seeds")
    runs_by_seed = _require_dict(
        report.get("candidate_training_runs_by_seed"), "candidate_training_runs_by_seed",
    )
    report_by_seed = _require_dict(report.get("per_seed_report_sha256"), "per_seed_report_sha256")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or set(runs_by_seed) != {str(seed) for seed in seeds}
            or set(report_by_seed) != {str(seed) for seed in seeds}):
        raise ValueError("training seed evidence is incomplete or inconsistent")
    receipts: list[CandidateReceipt] = []
    protocol_id = f"{PROTOCOL_PREFIX}{label}"
    for seed in seeds:
        seed_key = str(seed)
        seed_report_sha = _require_sha256(report_by_seed[seed_key], "seed_report_sha256")
        runs = _require_dict(runs_by_seed[seed_key], f"candidate runs for seed {seed}")
        if set(runs) != {"E1", "E2"}:
            raise ValueError(f"seed {seed} lacks the paired E1/E2 candidates")
        for candidate_id in ("E1", "E2"):
            run = _require_dict(runs[candidate_id], f"{candidate_id} run for seed {seed}")
            training = _require_dict(run.get("training_config"), "training_config")
            if run.get("model_config") is None:
                configuration = _historical_configuration(
                    report_path, seed, candidate_id, run,
                )
                model = configuration["model"]
                if configuration["training"] != training:
                    raise ValueError(
                        f"candidate {candidate_id} seed {seed} training sidecar disagrees"
                    )
            else:
                model = _require_dict(run.get("model_config"), "model_config")
                configuration = {"model": model, "training": training}
            if (run.get("status") != "COMPLETED" or run.get("fold_hash") != fold_sha
                    or training.get("seed") != seed
                    or type(run.get("optimizer_updates")) is not int
                    or run["optimizer_updates"] <= 0):
                raise ValueError(f"candidate {candidate_id} seed {seed} is not a completed exact Fold run")
            checkpoint_sha = _require_sha256(run.get("checkpoint_sha256"), "checkpoint_sha256")
            recipe = {
                "candidate": candidate_id,
                "configuration": configuration,
                "fold_sha256": fold_sha,
            }
            attempt_key = digest([
                "routeb-world-model-candidate-v1", protocol_id, candidate_id,
                seed, feature_sha, recipe,
            ])
            receipts.append(CandidateReceipt(
                attempt_key=attempt_key, protocol_id=protocol_id,
                candidate_id=candidate_id, recipe=recipe,
                feature_sha256=feature_sha, seed=seed,
                started_detail={
                    "access_key": access_key, "seed": seed,
                    "candidate": candidate_id, "test_payload_decoded": False,
                },
                completed_detail={
                    "seed_report_sha256": seed_report_sha,
                    "checkpoint_sha256": checkpoint_sha,
                    "optimizer_updates": run["optimizer_updates"],
                    "test_payload_decoded": False,
                },
            ))
    return tuple(receipts)


def _r4_candidate_receipts(report: dict[str, Any], access_key: str,
                           report_path: Path) -> tuple[CandidateReceipt, ...]:
    label = report["reservation_label"]
    fold_sha = _require_sha256(report.get("fold_hash"), "fold_hash")
    feature_sha = _require_sha256(report.get("feature_hash"), "feature_hash")
    protocol = _require_dict(report.get("protocol"), "protocol")
    seeds = protocol.get("training_seeds")
    if (not isinstance(seeds, list) or len(seeds) != 3
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != 3):
        raise ValueError("R4 training seeds are invalid or incomplete")
    training_runs = _require_dict(report.get("training_runs"), "training_runs")
    expected_keys = {f"{candidate}:{seed}"
                     for candidate in ("R4_E2_SINGLE_SCALE", "R4_E3_HIERARCHY")
                     for seed in seeds}
    if set(training_runs) != expected_keys:
        raise ValueError("R4 report does not contain the complete candidate/seed matrix")
    receipts: list[CandidateReceipt] = []
    protocol_id = f"{R4_PROTOCOL_PREFIX}{label}"
    for candidate_id in ("R4_E2_SINGLE_SCALE", "R4_E3_HIERARCHY"):
        for seed in seeds:
            name = f"{candidate_id}:{seed}"
            summary = _require_dict(training_runs[name], f"training run summary {name}")
            run_path = report_path.parent / candidate_id.lower() / f"seed-{seed}" / "run.json"
            try:
                contents = _read_artifact_bytes(
                    run_path, report_path.parent, f"R4 candidate run receipt {name}",
                )
                run = _strict_json_loads(contents)
            except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"R4 candidate run receipt is missing or invalid: {name}") from exc
            if sha256(contents).hexdigest() != _require_sha256(
                    summary.get("run_sha256"), "run_sha256"):
                raise ValueError(f"R4 candidate run receipt changed: {name}")
            run = _require_dict(run, f"candidate run {name}")
            configuration = _require_dict(run.get("train_config"), "train_config")
            training = _require_dict(configuration.get("training"), "training config")
            training_report = _require_dict(run.get("training"), "training report")
            expected_updates = protocol[
                "e2_training_steps" if candidate_id == "R4_E2_SINGLE_SCALE"
                else "e3_training_steps"]
            if (type(expected_updates) is not int or expected_updates <= 0
                    or type(training.get("seed")) is not int
                    or type(training.get("steps")) is not int
                    or training.get("steps") != expected_updates):
                raise ValueError(f"R4 candidate configuration differs from frozen budget: {name}")
            checkpoint_sha = _require_sha256(run.get("checkpoint_sha256"), "checkpoint_sha256")
            forecast = _require_dict(run.get("forecast"), "forecast")
            forecast_sha = _require_sha256(forecast.get("forecast_sha256"), "forecast_sha256")
            if (run.get("status") != "COMPLETED" or run.get("candidate") != candidate_id
                    or run.get("seed") != seed or run.get("fold_hash") != fold_sha
                    or run.get("dataset_hash") != report.get("dataset_hash")
                    or run.get("test_payload_decoded") is not False
                    or training.get("seed") != seed
                    or type(training_report.get("optimizer_updates")) is not int
                    or training_report["optimizer_updates"] != expected_updates):
                raise ValueError(f"R4 candidate run is not a completed exact registered run: {name}")
            recipe = {"candidate": candidate_id, "configuration": configuration,
                      "fold_sha256": fold_sha}
            attempt_key = digest([
                "routeb-world-model-r4-baseline-comparison-v1", protocol_id,
                candidate_id, seed, feature_sha, recipe,
            ])
            receipts.append(CandidateReceipt(
                attempt_key=attempt_key, protocol_id=protocol_id,
                candidate_id=candidate_id, recipe=recipe,
                feature_sha256=feature_sha, seed=seed,
                started_detail={"seed": seed, "fold_hash": fold_sha,
                                "test_payload_decoded": False, "access_key": access_key},
                completed_detail={"checkpoint_sha256": checkpoint_sha,
                                  "optimizer_updates": expected_updates,
                                  "forecast_sha256": forecast_sha,
                                  "test_payload_decoded": False},
            ))
    return tuple(receipts)


def build_plan(report_path: str | Path, registry_path: str | Path) -> ReconciliationPlan:
    path = Path(report_path).resolve()
    report = _load_report(path)
    accesses = _read_development_accesses(Path(registry_path).resolve())
    if report["schema"] == R4_LEGACY_FAILURE_SCHEMA:
        access = _matching_failed_legacy_r4(report, accesses, path, registry_path)
        return ReconciliationPlan(
            report_path=path, access_key=access["access_key"],
            access_identity=access["identity"], report_sha256=report["report_sha256"],
            report_schema=report["schema"], candidates=(), result_status="failed",
            result_detail={"legacy_receipts_verified": True,
                           "error": report["error"], "test_payload_decoded": False},
            read_only=True,
        )
    if report["schema"] == R4_FAILURE_SCHEMA:
        access = _matching_r4_failure_access(report, accesses, path)
        failed_attempts = _r4_failure_attempts(
            registry_path, access["identity"]["label"], report["fold_hash"], report["error"])
        return ReconciliationPlan(
            report_path=path, access_key=access["access_key"],
            access_identity=access["identity"], report_sha256=report["report_sha256"],
            report_schema=report["schema"], candidates=(), result_status="failed",
            result_detail={"error": report["error"], "test_payload_decoded": False},
            failed_attempts=failed_attempts,
        )
    if report["schema"] == R4_REPORT_SCHEMA:
        legacy_identity_fields_missing = any(
            report.get(field) is None
            for field in ("reservation_label", "instrument", "access_identity_sha256")
        )
        if legacy_identity_fields_missing:
            access = _matching_completed_legacy_r4(report, accesses, path, registry_path)
            return ReconciliationPlan(
                report_path=path, access_key=access["access_key"],
                access_identity=access["identity"], report_sha256=report["report_sha256"],
                report_schema=report["schema"], candidates=(),
                result_detail={"legacy_receipts_verified": True,
                               "test_payload_decoded": False,
                               "economic_certification": False},
                read_only=True,
            )
        access = _matching_r4_access(report, accesses)
        candidates = _r4_candidate_receipts(report, access["access_key"], path)
    else:
        access = _matching_access(report, accesses)
        candidates = _candidate_receipts(report, access["access_key"], path)
    if report["schema"] == R4_REPORT_SCHEMA:
        decision = _require_dict(report.get("decision"), "decision")
        if type(decision.get("e3_promoted")) is not bool:
            raise ValueError("R4 report promotion decision must be Boolean")
        result_detail = {"test_payload_decoded": False,
                         "e3_promoted": decision["e3_promoted"]}
    else:
        result_detail = {"report_schema": report["schema"],
                         "test_payload_decoded": False,
                         "economic_certification": False}
    return ReconciliationPlan(
        report_path=path,
        access_key=access["access_key"],
        access_identity=access["identity"],
        report_sha256=report["report_sha256"],
        report_schema=report["schema"],
        candidates=candidates,
        result_detail=result_detail,
    )


def apply_plan(plan: ReconciliationPlan, registry_path: str | Path) -> None:
    if plan.read_only:
        return
    registry = ProjectResearchRegistry(registry_path)
    for candidate in plan.candidates:
        registry.register_trial_attempt(
            attempt_key=candidate.attempt_key, protocol_id=candidate.protocol_id,
            candidate_id=candidate.candidate_id, recipe=candidate.recipe,
            feature_sha=candidate.feature_sha256, seed=candidate.seed,
        )
        registry.record_trial_attempt_event(
            candidate.attempt_key, event_key="started", status="started",
            detail=candidate.started_detail,
        )
        registry.record_trial_attempt_event(
            candidate.attempt_key, event_key="completed", status="completed",
            detail=candidate.completed_detail,
        )
    for attempt_key, detail in plan.failed_attempts:
        registry.record_trial_attempt_event(
            attempt_key, event_key="failed", status="failed", detail=detail,
        )
    registry.record_development_result(
        key=plan.access_key, identity=plan.access_identity, status=plan.result_status,
        report_sha=plan.report_sha256,
        detail=plan.result_detail or {"report_schema": plan.report_schema,
                                     "test_payload_decoded": False,
                                     "economic_certification": False},
    )


def _report_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ValueError(f"report source does not exist: {source}")
    source_root = source.resolve()
    scan_errors: list[OSError] = []

    def record_scan_error(error: OSError) -> None:
        scan_errors.append(error)

    report_names = {"multi_seed_fold_experiment_report.json",
                    "r4_e2_e3_comparison_report.json"}
    paths: set[Path] = set()
    failure_paths: list[Path] = []
    symlink_directories: list[Path] = []
    for directory, dirnames, filenames in os.walk(
            source_root, followlinks=False, onerror=record_scan_error):
        dirnames.sort()
        symlink_directories.extend(
            Path(directory) / name for name in dirnames
            if (Path(directory) / name).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(directory) / name
            if name in report_names:
                paths.add(path)
            elif name == "failure_receipt.json":
                failure_paths.append(path)
    if scan_errors:
        error = scan_errors[0]
        failed_path = error.filename or str(source_root)
        raise ValueError(
            f"cannot completely scan report source {source_root}: "
            f"{failed_path}: {error.strerror or type(error).__name__}"
        ) from error
    if symlink_directories:
        link_path = min(symlink_directories)
        raise ValueError(
            f"cannot completely scan report source {source_root}: "
            f"symlink directory is not traversed: {link_path}"
        )

    for failure_path in failure_paths:
        try:
            resolved_failure = failure_path.resolve(strict=True)
        except (OSError, RuntimeError):
            paths.add(failure_path)
            continue
        if not resolved_failure.is_relative_to(source_root):
            paths.add(failure_path)
            continue
        failure_text: str | None = None
        try:
            failure_text = resolved_failure.read_text(encoding="utf-8")
            failure = _strict_json_loads(failure_text)
        except (OSError, UnicodeError, json.JSONDecodeError):
            failure = None
        if failure is None:
            # The receipt's protocol cannot be identified; surface it for rejection.
            paths.add(failure_path)
            continue
        registration_path = failure_path.with_name("registration_receipt.json")
        try:
            resolved_registration = registration_path.resolve(strict=True)
        except FileNotFoundError:
            resolved_registration = None
        except (OSError, RuntimeError):
            paths.add(failure_path)
            continue
        if (resolved_registration is not None
                and not resolved_registration.is_relative_to(source_root)):
            paths.add(failure_path)
            continue
        registration_text: str | None = None
        try:
            if resolved_registration is None:
                registration = None
            else:
                registration_text = resolved_registration.read_text(encoding="utf-8")
                registration = _strict_json_loads(registration_text)
        except (OSError, UnicodeError, json.JSONDecodeError):
            paths.add(failure_path)
            continue
        r4_marker_in_damaged_receipt = any(
            text is not None and _contains_r4_receipt_schema(text)
            for text in (failure_text, registration_text)
        )
        if (r4_marker_in_damaged_receipt
                or (isinstance(failure, dict) and failure.get("schema") == R4_FAILURE_SCHEMA)
                or (isinstance(registration, dict)
                    and registration.get("schema") == "routeb.registered-r4-e2-e3-receipt.v1")):
            paths.add(failure_path)
    return sorted(paths)


def _unresolved_r4_accesses(registry_path: str | Path) -> list[dict[str, Any]]:
    """Report consumed R4 scopes without a result receipt; never mutates the ledger."""
    path = Path(registry_path).resolve()
    if not path.is_file():
        return []
    try:
        connection = sqlite3.connect(_read_only_sqlite_uri(path), uri=True)
        try:
            rows = connection.execute(
                "SELECT access_key,start_ns,end_ns,state,identity FROM access "
                "WHERE purpose='development' ORDER BY rowid"
            ).fetchall()
            completed = {row[0] for row in connection.execute(
                "SELECT access_key FROM development_result").fetchall()}
            result = []
            for access_key, start_ns, end_ns, state, encoded in rows:
                try:
                    identity = _strict_json_loads(encoded)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "cannot inspect R4 recovery status: "
                        "invalid development access identity"
                    ) from exc
                if not isinstance(identity, dict):
                    raise ValueError(
                        "cannot inspect R4 recovery status: "
                        "invalid development access identity"
                    )
                label = identity.get("label")
                fold_sha = identity.get("fold_sha256")
                if (access_key in completed or not isinstance(label, str)
                        or "-r4-" not in label.lower()):
                    continue
                protocol_id = f"{R4_PROTOCOL_PREFIX}{label}"
                attempts = connection.execute(
                    "SELECT a.attempt_key,a.candidate_id,a.recipe_json,e.status,e.event_key "
                    "FROM trial_attempt AS a LEFT JOIN trial_attempt_event AS e "
                    "ON e.attempt_key=a.attempt_key AND e.rowid=(SELECT MAX(last.rowid) "
                    "FROM trial_attempt_event AS last WHERE last.attempt_key=a.attempt_key) "
                    "WHERE a.protocol_id=? ORDER BY a.candidate_id", (protocol_id,),
                ).fetchall()
                matching_attempts = []
                ambiguous_attempts = []
                for attempt_key, candidate_id, recipe_json, status, event_key in attempts:
                    try:
                        recipe = _strict_json_loads(recipe_json)
                    except (TypeError, json.JSONDecodeError):
                        recipe = None
                    if not isinstance(recipe, dict) or not isinstance(
                            recipe.get("fold_sha256"), str):
                        ambiguous_attempts.append({
                            "attempt_key": attempt_key, "candidate_id": candidate_id,
                            "last_status": status, "last_event_key": event_key,
                        })
                    elif recipe["fold_sha256"] == fold_sha:
                        matching_attempts.append({
                            "attempt_key": attempt_key, "candidate_id": candidate_id,
                            "last_status": status, "last_event_key": event_key,
                        })
                if ambiguous_attempts:
                    state_name = "attempt_identity_ambiguous"
                elif not matching_attempts:
                    state_name = "reservation_without_attempts"
                elif any(attempt["last_status"] in {"registered", "started", "resumed"}
                         for attempt in matching_attempts):
                    state_name = "attempts_unresolved"
                else:
                    state_name = "attempts_terminal_without_result_receipt"
                result.append({
                    "access_key": access_key, "label": label,
                    "start_ns": start_ns, "end_ns": end_ns,
                    "access_state": state, "state": state_name,
                    "attempts": [*matching_attempts, *ambiguous_attempts],
                    "action": "manual_evidence_review_required; consumed scope is not reusable",
                })
            return result
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ValueError("cannot inspect R4 recovery status in the project ledger") from exc


def reconcile(source: str | Path, registry_path: str | Path, *, apply: bool) -> dict[str, Any]:
    source_path = Path(source).absolute()
    paths = _report_paths(source_path)
    source_root = (source_path.parent if source_path.is_file() else source_path).resolve()
    plans: list[ReconciliationPlan] = []
    rejected: list[dict[str, str]] = []
    seen_access: dict[str, str] = {}
    for path in paths:
        try:
            resolved_path = path.resolve(strict=True)
            if not resolved_path.is_relative_to(source_root):
                raise ValueError("report path escapes the selected source directory")
            plan = build_plan(resolved_path, registry_path)
            previous = seen_access.get(plan.access_key)
            if previous is not None:
                if previous != plan.report_sha256:
                    raise ValueError("multiple reports claim the same development access")
                raise ValueError("duplicate report path claims the same development access")
            seen_access[plan.access_key] = plan.report_sha256
            plans.append(plan)
        except (TypeError, ValueError) as exc:
            rejected.append({"report": str(path), "error": str(exc)})
    if apply and rejected:
        raise ValueError("refusing partial reconciliation because at least one report was rejected")
    if apply:
        for plan in plans:
            apply_plan(plan, registry_path)
    return {
        "schema": "routeb.research-ledger-reconciliation.v1",
        "mode": "apply" if apply else "dry-run",
        "reports_discovered": len(paths),
        "reports_accepted": len(plans),
        "reports_rejected": len(rejected),
        "read_only_reports": sum(plan.read_only for plan in plans),
        "candidate_attempts": sum(len(plan.candidates) for plan in plans),
        "access_keys": [plan.access_key for plan in plans],
        "unresolved_r4_accesses": _unresolved_r4_accesses(registry_path),
        "rejected": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=".artifacts/world_model")
    parser.add_argument("--registry", default=".artifacts/research-access.sqlite")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = reconcile(args.source, args.registry, apply=args.apply)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"status": "COMPLETED", **result}, indent=2))
    return 0 if result["reports_rejected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
