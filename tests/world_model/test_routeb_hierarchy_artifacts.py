from __future__ import annotations

import json
import numpy as np
import pytest
import sys
from pathlib import Path

from rles_data.core import FoldManifest, Window
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_world_model.routeb.codec import TrainOnlyStandardizer
from rles_world_model.routeb.clock import MINUTE_NS
from rles_world_model.routeb.fold_experiment import (
    train_fold_b3_diagnostic, verify_fold_experiment_report_digest,
)
from rles_world_model.routeb.hierarchy_artifacts import (
    export_hierarchical_train_artifact,
    hierarchy_scaler_sha256,
    load_hierarchical_train_artifact,
)
from rles_world_model.routeb.hierarchy_dataset import HierarchyScalers, HierarchyTensors
from rles_world_model.routeb import worker_cli
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig


def _fixture() -> tuple[HierarchyTensors, HierarchyScalers]:
    dataset_id = "synthetic-train-artifact"
    scaler = TrainOnlyStandardizer(
        np.arange(5, dtype=np.float64), np.ones(5, dtype=np.float64),
        dataset_id, "train")
    tensors = HierarchyTensors(
        minute_x=np.arange(20, dtype=np.float32).reshape(1, 4, 5),
        hour_x=np.arange(10, dtype=np.float32).reshape(1, 2, 5),
        day_x=np.arange(10, dtype=np.float32).reshape(1, 2, 5),
        hour_index=np.asarray([[0, 0, 1, 1]], dtype=np.int32),
        day_index=np.asarray([[0, 0, 0, 1]], dtype=np.int32),
        minute_mask=np.asarray([[0, 1, 1, 1]], dtype=np.float32),
        hour_mask=np.ones((1, 2), dtype=np.float32),
        day_mask=np.ones((1, 2), dtype=np.float32),
        minute_available_at=np.asarray([1, 2, 3, 4], dtype=np.int64),
        hour_available_at=np.asarray([1, 3], dtype=np.int64),
        day_available_at=np.asarray([1, 4], dtype=np.int64),
        hour_close_at=np.asarray([1, 3], dtype=np.int64),
        day_close_at=np.asarray([1, 4], dtype=np.int64),
    )
    return tensors, HierarchyScalers(scaler, scaler, scaler)


def test_hierarchy_train_artifact_round_trips_with_frozen_identities(tmp_path):
    tensors, scalers = _fixture()
    target = tmp_path / "b3_train.npz"
    artifact, metadata = export_hierarchical_train_artifact(
        target, tensors=tensors, scalers=scalers,
        train_scope_id="train-fold-001", fold_sha256="a" * 64)

    restored, restored_scalers, restored_metadata = load_hierarchical_train_artifact(
        artifact, expected_scope="train-fold-001", expected_fold_sha256="a" * 64)

    assert restored_metadata == metadata
    assert hierarchy_scaler_sha256(restored_scalers) == metadata["scaler_sha256"]
    for name in (
        "minute_x", "hour_x", "day_x", "hour_index", "day_index",
        "minute_mask", "hour_mask", "day_mask", "minute_available_at",
        "hour_available_at", "day_available_at", "hour_close_at", "day_close_at",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(tensors, name))
    assert metadata["validation_payload_in_worker"] is False
    assert metadata["test_payload_in_worker"] is False


def test_hierarchy_train_artifact_rejects_wrong_fold_or_modified_payload(tmp_path):
    tensors, scalers = _fixture()
    target, _ = export_hierarchical_train_artifact(
        tmp_path / "b3_train.npz", tensors=tensors, scalers=scalers,
        train_scope_id="train-fold-001", fold_sha256="a" * 64)

    with pytest.raises(ValueError, match="provenance"):
        load_hierarchical_train_artifact(
            target, expected_scope="train-fold-001", expected_fold_sha256="b" * 64)

    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_hierarchical_train_artifact(
            target, expected_scope="train-fold-001", expected_fold_sha256="a" * 64)


def test_hierarchy_train_artifact_rejects_missing_dataset_identity(tmp_path):
    tensors, scalers = _fixture()
    target, _ = export_hierarchical_train_artifact(
        tmp_path / "b3_train.npz", tensors=tensors, scalers=scalers,
        train_scope_id="train-fold-001", fold_sha256="a" * 64)
    metadata_path = target.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_id"] = None
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset identity"):
        load_hierarchical_train_artifact(
            target, expected_scope="train-fold-001", expected_fold_sha256="a" * 64)


def test_registered_fold_b3_training_stage_is_train_only_and_checksummed(tmp_path):
    dataset_hash = "c" * 64
    fold = FoldManifest(
        "0.4.0", dataset_hash, "d" * 64,
        Window(MINUTE_NS, 64 * MINUTE_NS, 4_370 * MINUTE_NS),
        Window(4_370 * MINUTE_NS, 4_380 * MINUTE_NS, 4_390 * MINUTE_NS),
        Window(4_390 * MINUTE_NS, 4_400 * MINUTE_NS, 4_410 * MINUTE_NS),
        0, 0, None,
    )
    frames = []
    for index in range(1, 4_390):
        open_price = 100.0 + index * 0.001
        close = open_price + 0.01
        frames.append(MarketFrame(
            index * MINUTE_NS, index * MINUTE_NS, "ETH-PERP",
            open_price, close + 0.1, open_price - 0.1, close, 3.0))
    replay = MarketReplay(tuple(frames), dataset_hash)

    class WorkerStub:
        def train_b3_hierarchical(self, *, sequences, scope, fold_sha256,
                checkpoint, configuration, resume):
            assert scope == "train-fold-b3-001"
            assert fold_sha256 == fold.hash
            assert resume is False
            assert sequences.with_suffix(".json").is_file()
            train_tensors, _, train_metadata = load_hierarchical_train_artifact(
                sequences, expected_scope=scope, expected_fold_sha256=fold_sha256)
            assert train_tensors.minute_available_at[-1] < fold.train.score_end_ns
            assert train_metadata["validation_payload_in_worker"] is False
            assert train_metadata["test_payload_in_worker"] is False
            checkpoint.write_bytes(b"synthetic-checkpoint")
            checkpoint.with_suffix(".json").write_text("{}", encoding="utf-8")
            return {
                "status": "COMPLETED", "kind": "B3_TRAIN_DIAGNOSTIC_ONLY",
                "reports": [{
                    "optimizer_updates": 1, "parameter_count": 123,
                    "train_windows": 20, "training_seconds": 0.5,
                    "device": "CPU", "grad_minute_dyn_norm": 0.2,
                    "grad_hour_dyn_norm": 0.3, "grad_day_dyn_norm": 0.4,
                }],
                "validation_scored": False, "test_scored": False,
                "economic_certification": False,
            }

    configuration = {
        "model": {scale: {} for scale in ("minute", "hour", "day")},
        "training": {"steps": 1},
    }
    report = train_fold_b3_diagnostic(
        replay=replay, fold=fold, train_scope_id="train-fold-b3-001",
        configuration=configuration, worker=WorkerStub(), output_dir=tmp_path,
        repository_commit="e" * 40, source_code_sha256="f" * 64,
        reservation_label="routeb-fold-synthetic-001")

    assert verify_fold_experiment_report_digest(report)
    assert report["schema"] == "routeb.b3-train-run.v1"
    assert report["validation_scored"] is False
    assert report["test_scored"] is False
    assert report["economic_certification"] is False
    assert report["scale_gradient_norms"] == {
        "minute": 0.2, "hour": 0.3, "day": 0.4}


def test_b3_worker_command_loads_bound_artifact_and_reports_train_only(
        tmp_path, monkeypatch, capsys):
    tensors, scalers = _fixture()
    artifact, metadata = export_hierarchical_train_artifact(
        tmp_path / "b3_train.npz", tensors=tensors, scalers=scalers,
        train_scope_id="train-worker-001", fold_sha256="a" * 64)
    configuration = tmp_path / "b3_config.json"
    configuration.write_text(json.dumps({
        "model": {scale: {} for scale in ("minute", "hour", "day")},
        "training": {"steps": 1, "batch_size": 1, "window_length": 4,
                     "burn_in": 1, "stride": 1, "report_interval": 1},
    }), encoding="utf-8")

    def fake_train(**kwargs):
        assert kwargs["train_scope_id"] == "train-worker-001"
        assert kwargs["dataset_id"] == "synthetic-train-artifact"
        assert kwargs["scaler_sha256"] == metadata["scaler_sha256"]
        return [{"step": 1}]

    monkeypatch.setattr(
        "rles_world_model.routeb.jax_hierarchy_train.train_hierarchical", fake_train)
    exit_code = worker_cli.main([
        "b3-train", "--sequences", str(artifact), "--scope", "train-worker-001",
        "--fold-sha256", "a" * 64, "--checkout", str(tmp_path),
        "--checkpoint", str(tmp_path / "checkpoint.npz"),
        "--configuration", str(configuration),
    ])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["kind"] == "B3_TRAIN_DIAGNOSTIC_ONLY"
    assert result["validation_scored"] is False
    assert result["test_scored"] is False
    assert result["economic_certification"] is False


def test_b3_worker_client_requires_matching_train_only_checkpoint(tmp_path, monkeypatch):
    worker = WorkerClient(WorkerConfig(Path(sys.executable), tmp_path))
    artifact = tmp_path / "train.npz"
    configuration = tmp_path / "b3_config.json"
    checkpoint = tmp_path / "checkpoint.npz"

    def fake_run(args):
        assert args[0] == "b3-train"
        assert args[args.index("--fold-sha256") + 1] == "b" * 64
        checkpoint.write_bytes(b"checkpoint")
        checkpoint.with_suffix(".json").write_text("{}", encoding="utf-8")
        return {
            "status": "COMPLETED", "kind": "B3_TRAIN_DIAGNOSTIC_ONLY",
            "checkpoint": str(checkpoint), "fold_sha256": "b" * 64,
            "validation_scored": False, "test_scored": False,
            "economic_certification": False, "reports": [{"step": 1}],
        }

    monkeypatch.setattr(worker, "_run", fake_run)
    result = worker.train_b3_hierarchical(
        sequences=artifact, scope="train-worker-001", fold_sha256="b" * 64,
        checkpoint=checkpoint, configuration=configuration)
    assert result["kind"] == "B3_TRAIN_DIAGNOSTIC_ONLY"
