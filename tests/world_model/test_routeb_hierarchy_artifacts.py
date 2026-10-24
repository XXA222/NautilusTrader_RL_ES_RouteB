from __future__ import annotations

import json
import hashlib
import numpy as np
import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

from rles_data.core import FoldManifest, Window
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_world_model.routeb.codec import TrainOnlyStandardizer
from rles_world_model.routeb.clock import MINUTE_NS
from rles_world_model.routeb.fold_experiment import (
    train_fold_b3_diagnostic, verify_fold_experiment_report_digest,
)
from rles_world_model.routeb.fold_b3_executor import run_fold_b3_experiment
from rles_world_model.routeb.hierarchy_artifacts import (
    export_hierarchical_inference_artifact,
    export_hierarchical_train_artifact,
    hierarchy_scaler_sha256,
    load_hierarchical_train_artifact,
)
from rles_world_model.routeb.hierarchy_dataset import HierarchyScalers, HierarchyTensors
from rles_world_model.routeb.market import Candle
from rles_world_model.routeb import worker_cli
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig, WorkerFailure


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
    import hashlib

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
    receipt = {
        "schema": "rles.routeb-fold-reader-receipt.v1",
        "reservation_key": "1" * 64,
        "reservation_token_sha256": "2" * 64,
        "reservation_identity_sha256": "3" * 64,
        "reservation_label": "routeb-fold-synthetic-001",
        "dataset_hash": dataset_hash, "marker_sha256": "a" * 64,
        "fold_sha256": fold.hash, "repository_commit": "e" * 40,
        "source_code_sha256": "f" * 64,
        "read_scope": {"start_ns": fold.train.warmup_start_ns,
                       "end_ns": fold.validation.score_end_ns, "includes_test": False},
        "reservation_committed_before_decode": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    (tmp_path / "registration_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8")
    report = train_fold_b3_diagnostic(
        replay=replay, fold=fold, train_scope_id="train-fold-b3-001",
        configuration=configuration, worker=WorkerStub(), output_dir=tmp_path,
        repository_commit="e" * 40, source_code_sha256="f" * 64,
        reservation_label="routeb-fold-synthetic-001",
        registration_receipt=receipt, marker_sha256="a" * 64)

    assert verify_fold_experiment_report_digest(report)
    assert report["schema"] == "routeb.b3-train-run.v1"
    assert report["validation_scored"] is False
    assert report["test_scored"] is False
    assert report["economic_certification"] is False
    assert report["registration_receipt_sha256"] == receipt["receipt_sha256"]
    assert report["scale_gradient_norms"] == {
        "minute": 0.2, "hour": 0.3, "day": 0.4}


def test_registered_b3_executor_runs_matched_synthetic_e3_e4_batches(tmp_path):
    import hashlib

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
    receipt = {
        "schema": "rles.routeb-fold-reader-receipt.v1",
        "reservation_key": "1" * 64,
        "reservation_token_sha256": "2" * 64,
        "reservation_identity_sha256": "3" * 64,
        "reservation_label": "routeb-fold-synthetic-executor",
        "dataset_hash": dataset_hash, "marker_sha256": "a" * 64,
        "fold_sha256": fold.hash, "repository_commit": "e" * 40,
        "source_code_sha256": "f" * 64,
        "read_scope": {"start_ns": fold.train.warmup_start_ns,
                       "end_ns": fold.validation.score_end_ns, "includes_test": False},
        "reservation_committed_before_decode": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    (tmp_path / "registration_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8")

    class WorkerStub:
        def train_b3_hierarchical(self, *, sequences, scope, fold_sha256,
                checkpoint, configuration, resume):
            assert scope == "routeb-fold-synthetic-executor"
            assert fold_sha256 == fold.hash and resume is False
            checkpoint.write_bytes(b"synthetic-b3-checkpoint")
            checkpoint.with_suffix(".json").write_text("{}", encoding="utf-8")
            return {"status": "COMPLETED", "kind": "B3_TRAIN_DIAGNOSTIC_ONLY",
                "reports": [{
                    "optimizer_updates": 1, "parameter_count": 123,
                    "train_windows": 20, "training_seconds": 0.5,
                    "device": "CPU", "grad_minute_dyn_norm": 0.2,
                    "grad_hour_dyn_norm": 0.3, "grad_day_dyn_norm": 0.4,
                }], "validation_scored": False, "test_scored": False,
                "economic_certification": False}

        def rollout_b3_batch(self, *, request, checkpoint, output, configuration):
            request_data = json.loads(request.read_text(encoding="utf-8"))
            anchors = np.asarray([item["anchor_ts_ns"]
                                  for item in request_data["context_artifacts"]], dtype=np.int64)
            shape = (len(anchors), request_data["n_paths"], request_data["horizon"], 5)
            e3 = np.empty(shape, dtype=np.float64)
            e4 = np.empty(shape, dtype=np.float64)
            e3[...] = [100.0, 101.0, 99.0, 100.0, 2.0]
            e4[...] = [100.001, 101.001, 99.001, 100.001, 2.0]
            np.savez_compressed(output, bars_e3=e3, bars_e4=e4, anchor_ts_ns=anchors)
            attempted = int(np.prod(shape[:3]))
            return {"sampling": {
                name: {"attempted": attempted, "rejected": 0, "rejection_rate": 0.0}
                for name in ("E3", "E4")}}

    configuration = {
        "model": {scale: {} for scale in ("minute", "hour", "day")},
        "training": {"steps": 1},
    }
    report = run_fold_b3_experiment(
        replay=replay, fold=fold, train_scope_id=receipt["reservation_label"],
        configuration=configuration, worker=WorkerStub(), output_dir=tmp_path,
        repository_commit="e" * 40, source_code_sha256="f" * 64,
        reservation_label=receipt["reservation_label"], marker_sha256="a" * 64,
        registration_receipt=receipt, forecast_seed=17, forecast_paths=2,
        horizons=(2,), anchor_interval_minutes=2, batch_size=2,
        max_attempts_per_bar=2, max_total_bar_attempts=1_000,
        volatility_window=2, block_length=1, bootstrap_replicates=100)

    assert report["hierarchical_rollout"]["matched_checkpoint"] is True
    assert report["hierarchical_rollout"]["registration_receipt_sha256"] == receipt["receipt_sha256"]
    assert report["overall"]["E3"][-1]["samples"] > 0
    assert report["overall"]["E4"][-1]["samples"] == report["overall"]["E3"][-1]["samples"]
    assert (tmp_path / "b3_e3_paths.npz").is_file()
    assert (tmp_path / "b3_e4_paths.npz").is_file()
    assert (tmp_path / "b3_e3_e4_validation_report.json").is_file()


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


def test_b3_rollout_batch_worker_validates_contexts_and_returns_paired_paths(
        tmp_path, monkeypatch, capsys):
    tensors, scalers = _fixture()
    fold_sha = "a" * 64
    anchors = (4320 * MINUTE_NS, 4330 * MINUTE_NS)
    items = []
    tensor_hashes = []
    for index, anchor in enumerate(anchors):
        context_path = tmp_path / f"context-{index}.npz"
        observed = tuple(Candle(
            ts_close_ns=anchor - (4319 - offset) * MINUTE_NS,
            available_at_ns=anchor - (4319 - offset) * MINUTE_NS,
            instrument="SYNTH", open=100.0, high=101.0, low=99.0,
            close=100.0, volume=2.0) for offset in range(4320))
        _, metadata = export_hierarchical_inference_artifact(
            context_path, tensors=tensors, scalers=scalers,
            observed_prefix=observed,
            train_scope_id="train-worker-001", fold_sha256=fold_sha,
            anchor_ts_ns=anchor)
        items.append({
            "path": str(context_path),
            "artifact_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
            "anchor_ts_ns": anchor,
        })
        tensor_hashes.append(metadata["hierarchy_tensor_sha256"])

    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"checkpoint")
    configuration = tmp_path / "b3_config.json"
    configuration.write_text(json.dumps({
        "model": {scale: {} for scale in ("minute", "hour", "day")},
        "training": {"steps": 1},
    }), encoding="utf-8")
    request = {
        "schema": "routeb.b3-rollout-batch-request.v1",
        "train_scope_id": "train-worker-001", "dataset_id": scalers.minute.dataset_id,
        "fold_sha256": fold_sha, "scaler_sha256": hierarchy_scaler_sha256(scalers),
        "train_tensor_sha256": "b" * 64,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256(configuration.read_bytes()).hexdigest(),
        "inference_context_sha256": hashlib.sha256(
            b"".join(bytes.fromhex(value) for value in tensor_hashes)).hexdigest(),
        "context_artifacts": items, "horizon": 2, "n_paths": 2,
        "forecast_seed": 13, "hour_availability_delay_ns": 0,
        "day_availability_delay_ns": 0, "max_attempts_per_bar": 2,
        "max_total_bar_attempts": 100,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    class Predictor:
        def reset_context(self, **kwargs):
            self.context = kwargs

    monkeypatch.setattr(
        "rles_world_model.routeb.jax_hierarchy_train.create_stateful_hierarchical_f2_predictor",
        lambda **kwargs: Predictor())

    def fake_generate(**kwargs):
        return SimpleNamespace(
            ensemble=SimpleNamespace(minute=np.full((2, 2, 5),
                100.0 if not kwargs["advance_slow_priors"] else 101.0)),
            attempted_bar_samples=4, rejected_bar_samples=0)

    monkeypatch.setattr(
        "rles_world_model.routeb.jax_hierarchy_train.generate_recursive_hierarchical_f2_paths",
        fake_generate)
    output = tmp_path / "batch.npz"
    exit_code = worker_cli.main([
        "b3-rollout-batch", "--request", str(request_path), "--checkout", str(tmp_path),
        "--checkpoint", str(checkpoint), "--output", str(output),
        "--configuration", str(configuration),
    ])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["kind"] == "B3_ROLLOUT_BATCH_PATHS"
    assert result["sampling"]["E3"]["attempted"] == 8
    with np.load(output, allow_pickle=False) as archive:
        assert archive["bars_e3"].shape == (2, 2, 2, 5)
        assert archive["bars_e4"].shape == (2, 2, 2, 5)
        np.testing.assert_array_equal(archive["anchor_ts_ns"], anchors)
        assert np.all(archive["bars_e3"] == 100.0)
        assert np.all(archive["bars_e4"] == 101.0)


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


def test_b3_worker_client_validates_recursive_rollout_artifact(tmp_path, monkeypatch):
    import hashlib

    worker = WorkerClient(WorkerConfig(Path(sys.executable), tmp_path))
    context = tmp_path / "context.npz"
    checkpoint = tmp_path / "checkpoint.npz"
    configuration = tmp_path / "configuration.json"
    request_path = tmp_path / "request.json"
    output = tmp_path / "paths.npz"
    for path in (context, checkpoint, configuration):
        path.write_bytes(b"bound-by-host-test")
    request = {
        "schema": "routeb.b3-rollout-request.v1", "rollout_mode": "E4",
        "anchor_ts_ns": 100 * MINUTE_NS, "n_paths": 2, "horizon": 3,
        "max_attempts_per_bar": 20,
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")

    def fake_run(args):
        assert args[0] == "b3-rollout"
        bars = np.asarray([[[100, 101, 99, 100, 2]] * 3] * 2, dtype=np.float64)
        np.savez_compressed(output, bars=bars,
                            anchor_ts_ns=np.asarray(request["anchor_ts_ns"], dtype=np.int64))
        return {
            "status": "COMPLETED", "kind": "B3_ROLLOUT_PATHS",
            "candidate": "E4", "anchor_ts_ns": request["anchor_ts_ns"],
            "paths": 2, "horizon": 3, "attempted_bar_samples": 6,
            "rejected_bar_samples": 0, "rejection_rate": 0.0,
            "output": str(output.resolve()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(worker, "_run", fake_run)
    result = worker.rollout_b3_hierarchical(
        request=request_path, context=context, checkpoint=checkpoint,
        output=output, configuration=configuration)
    assert result["kind"] == "B3_ROLLOUT_PATHS"

    request["rollout_mode"] = "E3"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(WorkerFailure, match="identity"):
        worker.rollout_b3_hierarchical(
            request=request_path, context=context, checkpoint=checkpoint,
            output=output, configuration=configuration)
