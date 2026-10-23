import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from rles_data.core import DAY_NS
from rles_data.core import FoldManifest, Window, MINUTE_NS
from rles_world_model.baselines import BaselineScore
from rles_orchestrator.application import main
import rles_world_model.cli as world_model_cli
from rles_world_model.contracts import MarketFrame, MarketReplay


pytestmark = pytest.mark.unit


def test_code_identity_falls_back_to_explicit_snapshot_commit(monkeypatch):
    monkeypatch.setenv("RLES_WORKSPACE_ROOT", str(Path(__file__).resolve().parents[2]))

    def no_git_commit(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, "git rev-parse HEAD")

    monkeypatch.setattr(world_model_cli.subprocess, "run", no_git_commit)
    identity = world_model_cli._routeb_code_identity()
    assert identity["repository_commit"] == "c24d1bc4a47584f91a4c4db93bf77117d4e8341c"
    assert len(identity["source_code_sha256"]) == 64


def test_code_identity_binds_nested_routeb_sources(tmp_path, monkeypatch):
    def no_git_commit(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, "git rev-parse HEAD")

    monkeypatch.setattr(world_model_cli.subprocess, "run", no_git_commit)
    required = (
        "pyproject.toml", "uv.lock",
        "packages/data/pyproject.toml", "packages/research/pyproject.toml",
        "packages/orchestrator/pyproject.toml", "packages/world_model/pyproject.toml",
        "packages/data/src/rles_data/core.py", "packages/data/src/rles_data/io.py",
        "packages/research/src/rles_research/access.py",
        "packages/orchestrator/src/rles_orchestrator/research_access.py",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    marker = tmp_path / "docs/world_model/REPO_BASELINE.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({
        "schema_version": "rles.routeb.repository_base.v0",
        "repository_commit": "d" * 40,
    }), encoding="utf-8")
    nested = tmp_path / "packages/world_model/src/rles_world_model/routeb/worker.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("version = 1\n", encoding="utf-8")

    first = world_model_cli._routeb_code_identity(tmp_path)
    nested.write_text("version = 2\n", encoding="utf-8")
    second = world_model_cli._routeb_code_identity(tmp_path)
    assert first["repository_commit"] == second["repository_commit"] == "d" * 40
    assert first["source_code_sha256"] != second["source_code_sha256"]


def test_world_model_cli_dispatches_through_rles(tmp_path, capsys):
    minute = 60_000_000_000
    frames = []
    for index in range(1, 4):
        price = 100.0 + index
        frames.append({
            "ts_close_ns": minute * index,
            "available_at_ns": minute * index,
            "instrument": "ETHUSDT-PERP.SIM",
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 1.0,
        })
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({
        "schema_version": "rles.market_replay.v0",
        "evidence_scope": "synthetic_test_only",
        "dataset_hash": "a" * 64,
        "instrument": "ETHUSDT-PERP.SIM",
        "frames": frames,
    }), encoding="utf-8")

    assert main(["world-model", "audit-data", "--synthetic-replay-json", str(replay)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["bars"] == 3
    assert report["market_model_eligible"] is True
    assert report["perpetual_economics_eligible"] is False
    assert report["evidence_scope"] == "synthetic_test_only"


def test_registered_train_cli_routes_through_scoped_reader_before_audit(tmp_path, monkeypatch, capsys):
    minute = 60_000_000_000
    replay = MarketReplay((
        MarketFrame(minute, minute, "ETHUSDT-PERP.SIM", 100, 101, 99, 100, 1),
        MarketFrame(2 * minute, 2 * minute, "ETHUSDT-PERP.SIM", 100, 101, 99, 100, 1),
    ), "b" * 64)
    captured = {}

    def fake_reader(config, window, **kwargs):
        captured["config"] = config
        captured["window"] = window
        captured.update(kwargs)
        return replay

    monkeypatch.setattr(world_model_cli, "project_registered_train_replay", fake_reader)
    config = {
        "schema_version": "rles.world_model.registered_train_audit.v0",
        "config": {"output": "routeb-b0", "environment": {
            "market_path": "market.parquet", "instrument": "ETHUSDT-PERP.SIM",
            "start_ns": minute, "end_ns": 3 * minute,
        }},
        "window": {"start_ns": minute, "end_ns": 3 * minute},
        "marker_sha256": "c" * 64,
        "label": "train-routeb-b0-smoke",
        "instrument": "ETHUSDT-PERP.SIM",
        "allow_legacy_isolated_snapshot": False,
    }
    path = tmp_path / "registered-train.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert main(["world-model", "audit-data", "--registered-train-config", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["evidence_scope"] == "registered_train"
    assert captured["label"] == "train-routeb-b0-smoke"
    assert captured["marker_sha256"] == "c" * 64


def test_registered_train_cli_rejects_evaluation_config_before_reader(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(world_model_cli, "project_registered_train_replay",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reader called")))
    config = {
        "schema_version": "rles.world_model.registered_train_audit.v0",
        "config": {"output": "routeb-b0", "environment": {"market_path": "market.parquet"},
                   "evaluation": {"start_ns": 100, "end_ns": 200}},
        "window": {"start_ns": 10, "end_ns": 20},
        "marker_sha256": "c" * 64,
        "label": "train-routeb-b0-smoke",
        "instrument": "ETHUSDT-PERP.SIM",
        "allow_legacy_isolated_snapshot": False,
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert main(["world-model", "audit-data", "--registered-train-config", str(path)]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "BLOCKED"


def test_registered_fold_baseline_cli_uses_frozen_reader_before_scoring(
    tmp_path, monkeypatch, capsys,
):
    minute = 60_000_000_000
    frames = tuple(
        MarketFrame(index * minute, index * minute, "ETHUSDT-PERP.SIM",
                    100 + index / 10, 101 + index / 10, 99 + index / 10,
                    100 + index / 10, 1.0)
        for index in range(1, 150)
    )
    replay = MarketReplay(frames, "a" * 64)
    captured = {}

    def frozen_reader(config, fold, **kwargs):
        captured["config"] = config
        captured["fold"] = fold
        captured.update(kwargs)
        return replay

    monkeypatch.setattr(world_model_cli, "project_registered_fold_replay", frozen_reader)
    monkeypatch.setattr(world_model_cli, "_routeb_code_identity", lambda: {
        "repository_commit": "d" * 40, "source_code_sha256": "e" * 64,
    })
    fold = {
        "schema_version": "0.4.0", "dataset_hash": "a" * 64, "feature_hash": "b" * 64,
        "train": {"warmup_start_ns": 0, "score_start_ns": DAY_NS,
                  "score_end_ns": 61 * DAY_NS},
        "validation": {"warmup_start_ns": 60 * DAY_NS, "score_start_ns": 61 * DAY_NS,
                       "score_end_ns": 76 * DAY_NS},
        "test": {"warmup_start_ns": 75 * DAY_NS, "score_start_ns": 76 * DAY_NS,
                 "score_end_ns": 91 * DAY_NS},
        "purge_ns": 0, "embargo_ns": 0, "scaler_hash": None,
    }
    monkeypatch.setattr(world_model_cli, "score_baselines",
                        lambda *_args, **_kwargs: (BaselineScore("S0", 1, 0.1),))
    config = {
        "schema_version": "rles.world_model.registered_fold_baseline.v0",
        "config": {"output": "routeb-b0", "environment": {
            "market_path": "market.parquet", "instrument": "ETHUSDT-PERP.SIM",
        }},
        "fold": fold, "marker_sha256": "c" * 64,
        "label": "routeb-fold-fresh-0001", "instrument": "ETHUSDT-PERP.SIM",
        "allow_legacy_isolated_snapshot": False,
    }
    path = tmp_path / "registered-fold.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    report_dir = tmp_path / "b0-reports"
    assert main(["world-model", "baselines", "--registered-fold-config", str(path),
                 "--report-dir", str(report_dir), "--seed", "19",
                 "--ensemble-size", "32"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["evidence_scope"] == "registered_fresh_fold_train_validation"
    assert report["fold_hash"] == captured["fold"].hash
    assert captured["label"] == "routeb-fold-fresh-0001"
    assert len(captured["repository_commit"]) == 40
    assert len(captured["source_code_sha256"]) == 64
    assert report["source_code_sha256"] == captured["source_code_sha256"]
    assert report["marker_sha256"] == "c" * 64
    assert report["read_scope"]["includes_test"] is False
    assert report["data_audit"]["test_time_coverage_preflight"]["test_payload_decoded"] is False
    assert report["availability_policy"] == "closed_1m_bar;available_at_ns<=decision_ts_ns"
    audit_report = json.loads((report_dir / "B0_data_audit.json").read_text())
    baseline_report = json.loads((report_dir / "baselines.json").read_text())
    assert audit_report["fold_hash"] == report["fold_hash"]
    assert audit_report["fold_manifest"]["dataset_hash"] == "a" * 64
    assert baseline_report["fold_manifest"] == audit_report["fold_manifest"]
    assert baseline_report["scores"][0]["name"] == "S0"


def test_registered_fold_cli_rejects_formal_scope_before_reader(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(world_model_cli, "project_registered_fold_replay",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("reader must not be called")))
    config = {
        "schema_version": "rles.world_model.registered_fold_baseline.v0",
        "config": {"environment": {"market_path": "market.parquet"},
                   "evaluation": {"start_ns": 10, "end_ns": 20}},
        "fold": {}, "marker_sha256": "c" * 64,
        "label": "routeb-fold-formal", "instrument": "ETHUSDT-PERP.SIM",
        "allow_legacy_isolated_snapshot": False,
    }
    path = tmp_path / "formal-fold.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert main(["world-model", "baselines", "--registered-fold-config", str(path),
                 "--report-dir", str(tmp_path / "reports")]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "BLOCKED"


def test_registered_fold_cli_rejects_unapproved_split_durations_before_reader(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(world_model_cli, "project_registered_fold_replay",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("reader must not be called")))
    monkeypatch.setattr(world_model_cli, "_routeb_code_identity", lambda: {
        "repository_commit": "d" * 40, "source_code_sha256": "e" * 64,
    })
    fold = {
        "schema_version": "0.4.0", "dataset_hash": "a" * 64, "feature_hash": "b" * 64,
        "train": {"warmup_start_ns": 0, "score_start_ns": DAY_NS,
                  "score_end_ns": 2 * DAY_NS},
        "validation": {"warmup_start_ns": 2 * DAY_NS, "score_start_ns": 2 * DAY_NS,
                       "score_end_ns": 3 * DAY_NS},
        "test": {"warmup_start_ns": 3 * DAY_NS, "score_start_ns": 3 * DAY_NS,
                 "score_end_ns": 4 * DAY_NS},
        "purge_ns": 0, "embargo_ns": 0, "scaler_hash": None,
    }
    config = {
        "schema_version": "rles.world_model.registered_fold_baseline.v0",
        "config": {"environment": {"market_path": "market.parquet",
                                    "instrument": "ETHUSDT-PERP.SIM"}},
        "fold": fold, "marker_sha256": "c" * 64,
        "label": "routeb-fold-short", "instrument": "ETHUSDT-PERP.SIM",
        "allow_legacy_isolated_snapshot": False,
    }
    path = tmp_path / "short-fold.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert main(["world-model", "baselines", "--registered-fold-config", str(path),
                 "--report-dir", str(tmp_path / "reports")]) == 2
    assert "approved 60/15/15 or 360/15/15" in json.loads(capsys.readouterr().err)["detail"]


def test_registered_fold_cli_rejects_existing_report_dir_before_data_reader(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(world_model_cli, "project_registered_fold_replay",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("data reader must not run")))
    report_dir = tmp_path / "existing-reports"
    report_dir.mkdir()
    config_path = tmp_path / "not-opened.json"
    assert main(["world-model", "baselines", "--registered-fold-config", str(config_path),
                 "--report-dir", str(report_dir)]) == 2
    assert "already exists" in json.loads(capsys.readouterr().err)["detail"]


def test_prepare_fold_train_cli_uses_registered_reader_and_writes_bound_artifact(
    tmp_path, monkeypatch, capsys,
):
    fold = FoldManifest(
        "0.4.0", "a" * 64, "b" * 64,
        Window(0, DAY_NS, 61 * DAY_NS),
        Window(60 * DAY_NS, 61 * DAY_NS, 76 * DAY_NS),
        Window(75 * DAY_NS, 76 * DAY_NS, 91 * DAY_NS),
        0, 0, None,
    )
    replay = MarketReplay((MarketFrame(
        MINUTE_NS, MINUTE_NS, "ETHUSDT-PERP.SIM", 100, 101, 99, 100, 1,
    ),), fold.dataset_hash)
    payload = {"label": "routeb-fold-fresh-0002", "marker_sha256": "c" * 64}
    identity = {"repository_commit": "d" * 40, "source_code_sha256": "e" * 64}
    observed = {}
    monkeypatch.setattr(world_model_cli, "_routeb_code_identity", lambda: identity)
    monkeypatch.setattr(world_model_cli, "_registered_fold_baseline",
                        lambda path, code: (payload, fold, replay))

    class FrozenScaler:
        fingerprint = "f" * 64

    def fake_prepare(**kwargs):
        observed.update(kwargs)
        target = kwargs["target_npz"]
        target.write_bytes(b"train-only synthetic fixture")
        target.with_suffix(".json").write_text(json.dumps({
            "artifact_sha256": "1" * 64,
        }), encoding="utf-8")
        return target, FrozenScaler(), {"train_windows": 12}

    monkeypatch.setattr(world_model_cli, "prepare_fold_train_sequences", fake_prepare)
    config = tmp_path / "registered-fold.json"
    config.write_text("{}", encoding="utf-8")
    target = tmp_path / "worker-train.npz"
    assert main([
        "world-model", "prepare-fold-train",
        "--registered-fold-config", str(config),
        "--output-npz", str(target), "--length", "64",
        "--burn-in", "16", "--stride", "32",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PREPARED"
    assert report["evidence_scope"] == "registered_fresh_fold_train_only"
    assert report["fold_hash"] == fold.hash
    assert report["reservation_label"] == payload["label"]
    assert report["artifact_sha256"] == "1" * 64
    assert report["train_windows"] == 12
    assert observed["train_scope_id"] == payload["label"]
    assert observed["fold_replay"] is replay
    assert observed["length"] == 64 and observed["burn_in"] == 16
    assert observed["stride"] == 32


def test_prepare_fold_train_cli_rejects_existing_artifact_before_reader(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(world_model_cli, "_routeb_code_identity",
                        lambda: (_ for _ in ()).throw(AssertionError("identity called")))
    target = tmp_path / "existing.npz"
    target.write_bytes(b"existing")
    config = tmp_path / "not-opened.json"
    assert main([
        "world-model", "prepare-fold-train",
        "--registered-fold-config", str(config), "--output-npz", str(target),
    ]) == 2
    assert "already exists" in json.loads(capsys.readouterr().err)["detail"]


def test_train_prepared_cli_checks_fold_and_dispatches_e1_worker(
    tmp_path, monkeypatch, capsys,
):
    sequences = tmp_path / "train.npz"
    sequences.write_bytes(b"prepared train payload")
    sequences.with_suffix(".json").write_text("{}", encoding="utf-8")
    configuration = tmp_path / "e1.json"
    configuration.write_text(json.dumps({
        "model": {"hidden": 32},
        "training": {"seed": 3, "steps": 7, "batch_size": 2},
    }), encoding="utf-8")
    worker_python = tmp_path / "python.exe"
    worker_python.write_bytes(b"fake executable")
    upstream = tmp_path / "dreamerv3"
    upstream.mkdir()
    checkpoint = tmp_path / "model.ckpt"
    report_path = tmp_path / "run.json"
    fold_hash = "a" * 64
    monkeypatch.setattr(world_model_cli, "load_worker_batch", lambda *args, **kwargs: (
        type("Batch", (), {"features": np.zeros((12, 64, 5))})(),
        {"dataset_id": "b" * 64, "artifact_sha256": "c" * 64,
               "fold_sha256": fold_hash},
    ))
    calls = {}

    class FakeClient:
        def __init__(self, config):
            calls["worker_config"] = config

        def train_e1(self, **kwargs):
            calls["train"] = kwargs
            kwargs["checkpoint"].write_bytes(b"checkpoint bytes")
            kwargs["checkpoint"].with_suffix(".json").write_text("{}", encoding="utf-8")
            return {"status": "COMPLETED", "kind": "E1_TRAIN_DIAGNOSTIC_ONLY",
                    "reports": [{"step": 7, "loss": 0.5, "parameter_count": 101,
                                 "train_examples": 12, "optimizer_updates": 7,
                                 "training_seconds": 1.25, "device": "cpu"}]}

    monkeypatch.setattr(world_model_cli, "WorkerClient", FakeClient)
    assert main([
        "world-model", "train-prepared", "--candidate", "e1",
        "--sequences", str(sequences), "--scope", "routeb-fold-123",
        "--fold-sha256", fold_hash, "--configuration", str(configuration),
        "--checkpoint", str(checkpoint), "--worker-python", str(worker_python),
        "--upstream-checkout", str(upstream), "--report-json", str(report_path),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "routeb.train-run.v1"
    assert report["candidate"] == "e1"
    assert report["fold_hash"] == fold_hash
    assert report["train_examples"] == 12 and report["window_length"] == 64
    assert report["training_config"]["steps"] == 7
    assert report["optimizer_updates"] == 7
    assert report["parameter_count"] == 101
    assert report["device"] == "cpu" and report["training_seconds"] == 1.25
    assert report["worker_result"]["reports"][0]["step"] == 7
    assert json.loads(report_path.read_text(encoding="utf-8"))["checkpoint_sha256"] == report[
        "checkpoint_sha256"
    ]
    assert calls["train"]["scope"] == "routeb-fold-123"
    assert calls["worker_config"].timeout_seconds == 1800


def test_train_prepared_cli_rejects_wrong_fold_before_worker(
    tmp_path, monkeypatch, capsys,
):
    sequences = tmp_path / "train.npz"
    sequences.write_bytes(b"prepared")
    sequences.with_suffix(".json").write_text("{}", encoding="utf-8")
    configuration = tmp_path / "e2.json"
    configuration.write_text(json.dumps({"model": {}, "training": {}}), encoding="utf-8")
    worker_python = tmp_path / "python.exe"
    worker_python.write_bytes(b"fake")
    upstream = tmp_path / "dreamerv3"
    upstream.mkdir()
    monkeypatch.setattr(world_model_cli, "load_worker_batch", lambda *args, **kwargs: (
        None, {"dataset_id": "b" * 64, "artifact_sha256": "c" * 64,
               "fold_sha256": "d" * 64},
    ))
    monkeypatch.setattr(world_model_cli, "WorkerClient",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("worker must not start")))
    assert main([
        "world-model", "train-prepared", "--candidate", "e2",
        "--sequences", str(sequences), "--scope", "routeb-fold-123",
        "--fold-sha256", "a" * 64, "--configuration", str(configuration),
        "--checkpoint", str(tmp_path / "model.ckpt"), "--worker-python", str(worker_python),
        "--upstream-checkout", str(upstream), "--report-json", str(tmp_path / "run.json"),
    ]) == 2
    assert "does not match" in json.loads(capsys.readouterr().err)["detail"]


def test_run_registered_fold_reserves_once_then_hands_same_replay_to_pipeline(
        tmp_path, monkeypatch, capsys):
    from dataclasses import asdict
    from rles_world_model.routeb.jax_gru import GRUConfig
    from rles_world_model.routeb.jax_model import RSSMConfig
    from rles_world_model.routeb.jax_train import TrainingConfig

    fold = FoldManifest(
        "0.4.0", "a" * 64, "b" * 64,
        Window(MINUTE_NS, 2 * MINUTE_NS, 20 * MINUTE_NS),
        Window(20 * MINUTE_NS, 21 * MINUTE_NS, 40 * MINUTE_NS),
        Window(40 * MINUTE_NS, 41 * MINUTE_NS, 60 * MINUTE_NS),
        0, 0, None,
    )
    replay = MarketReplay((MarketFrame(
        MINUTE_NS, MINUTE_NS, "ETH-PERP", 100, 101, 99, 100, 1),), "a" * 64)
    fold_config = tmp_path / "fold.json"
    fold_config.write_text("{}", encoding="utf-8")
    training = asdict(TrainingConfig(steps=2, batch_size=1))
    e1_config = tmp_path / "e1.json"
    e2_config = tmp_path / "e2.json"
    e1_config.write_text(json.dumps({"model": asdict(GRUConfig()), "training": training}))
    e2_config.write_text(json.dumps({"model": asdict(RSSMConfig()), "training": training}))
    worker_python = tmp_path / "python.exe"
    worker_python.write_bytes(b"fake")
    checkout = tmp_path / "dreamerv3"
    checkout.mkdir()
    reads = []
    handoffs = []
    monkeypatch.setattr(world_model_cli, "_routeb_code_identity", lambda: {
        "repository_commit": "c" * 40, "source_code_sha256": "d" * 64})
    monkeypatch.setattr(world_model_cli, "_registered_fold_baseline",
        lambda *_args: (reads.append(True) or {
            "label": "routeb-fold-once", "marker_sha256": "e" * 64}, fold, replay))
    monkeypatch.setattr(world_model_cli, "WorkerClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(world_model_cli, "run_fold_experiment", lambda **kwargs: (
        handoffs.append(kwargs) or {"status": "COMPLETED", "report_sha256": "f" * 64}))
    output_dir = tmp_path / "result"
    assert main([
        "world-model", "run-registered-fold",
        "--registered-fold-config", str(fold_config),
        "--e1-configuration", str(e1_config), "--e2-configuration", str(e2_config),
        "--worker-python", str(worker_python), "--upstream-checkout", str(checkout),
        "--output-dir", str(output_dir),
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "COMPLETED"
    assert len(reads) == 1 and len(handoffs) == 1
    assert handoffs[0]["replay"] is replay and handoffs[0]["fold"] is fold
    assert handoffs[0]["train_scope_id"] == "routeb-fold-once"
    receipt = json.loads((output_dir / "registration_receipt.json").read_text())
    assert receipt["fold_hash"] == fold.hash
    assert receipt["read_scope"]["includes_test"] is False
