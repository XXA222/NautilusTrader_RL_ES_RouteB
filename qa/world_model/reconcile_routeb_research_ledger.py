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
from pathlib import Path
import sqlite3
from typing import Any

from rles_data.core import digest, require_hash
from rles_research.access import ProjectResearchRegistry


REPORT_SCHEMA = "routeb.multi-seed-fold-experiment.v1"
PROTOCOL_PREFIX = "routeb-r2:"


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


def _load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read canonical JSON report: {path}") from exc
    report = _require_dict(value, "report")
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "COMPLETED":
        raise ValueError("report is not a completed Route B multi-seed fold experiment")
    if report.get("test_payload_decoded") is not False:
        raise ValueError("report does not prove that the test payload stayed sealed")
    if report.get("economic_certification") is not False:
        raise ValueError("historical predictive evidence must not claim economic certification")
    recorded = _require_sha256(report.get("report_sha256"), "report_sha256")
    if recorded != _canonical_report_sha256(report):
        raise ValueError("aggregate report SHA256 does not match its canonical payload")
    return report


def _read_development_accesses(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.is_file():
        raise ValueError(f"research registry does not exist: {registry_path}")
    try:
        with sqlite3.connect(f"file:{registry_path.resolve().as_posix()}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT access_key,start_ns,end_ns,purpose,identity,state "
                "FROM access WHERE purpose='development'"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError("cannot read the project research registry") from exc
    result = []
    for key, start_ns, end_ns, purpose, encoded, state in rows:
        try:
            identity = json.loads(encoded)
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
        content = configuration_path.read_bytes()
        configuration = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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


def build_plan(report_path: str | Path, registry_path: str | Path) -> ReconciliationPlan:
    path = Path(report_path).resolve()
    report = _load_report(path)
    access = _matching_access(report, _read_development_accesses(Path(registry_path).resolve()))
    return ReconciliationPlan(
        report_path=path,
        access_key=access["access_key"],
        access_identity=access["identity"],
        report_sha256=report["report_sha256"],
        report_schema=report["schema"],
        candidates=_candidate_receipts(report, access["access_key"], path),
    )


def apply_plan(plan: ReconciliationPlan, registry_path: str | Path) -> None:
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
    registry.record_development_result(
        key=plan.access_key, identity=plan.access_identity, status="completed",
        report_sha=plan.report_sha256,
        detail={
            "report_schema": plan.report_schema,
            "test_payload_decoded": False,
            "economic_certification": False,
        },
    )


def _report_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ValueError(f"report source does not exist: {source}")
    return sorted(source.rglob("multi_seed_fold_experiment_report.json"))


def reconcile(source: str | Path, registry_path: str | Path, *, apply: bool) -> dict[str, Any]:
    paths = _report_paths(Path(source).resolve())
    plans: list[ReconciliationPlan] = []
    rejected: list[dict[str, str]] = []
    seen_access: dict[str, str] = {}
    for path in paths:
        try:
            plan = build_plan(path, registry_path)
            previous = seen_access.get(plan.access_key)
            if previous is not None and previous != plan.report_sha256:
                raise ValueError("multiple reports claim the same development access")
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
        "candidate_attempts": sum(len(plan.candidates) for plan in plans),
        "access_keys": [plan.access_key for plan in plans],
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
