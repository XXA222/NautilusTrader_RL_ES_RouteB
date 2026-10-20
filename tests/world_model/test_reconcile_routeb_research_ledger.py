from __future__ import annotations

from hashlib import sha256
import json

from qa.world_model.reconcile_routeb_research_ledger import (
    apply_plan,
    build_plan,
)
from rles_data.core import digest
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
