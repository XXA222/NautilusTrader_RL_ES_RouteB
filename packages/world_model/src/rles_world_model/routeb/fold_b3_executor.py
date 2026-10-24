"""Registered, matched B3 E3/E4 Validation execution on one authorized Fold read."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from rles_data.core import FoldManifest

from ..contracts import MarketReplay
from .evaluation import _numeric_sha256
from .fold_evaluation import build_fold_b3_predictive_report
from .fold_experiment import train_fold_b3_diagnostic, _write_json_fresh
from .fold_protocol import validation_anchor_timestamps, validate_fold_experiment_protocol
from .hierarchy_artifacts import (
    export_hierarchical_inference_artifact,
    hierarchy_scaler_sha256,
    load_hierarchical_train_artifact,
)
from .hierarchy_dataset import build_inference_hierarchy
from .jax_model import PINNED_UPSTREAM
from .jax_train import SNAPSHOT_SHA
from .market import Candle, MarketSequence, MINUTE_NS
from .worker_client import WorkerClient


def _write_paths(path: Path, *, anchors: np.ndarray, paths: np.ndarray) -> str:
    if path.exists() or not path.parent.is_dir():
        raise ValueError(f"refusing to overwrite B3 path artifact: {path.name}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, anchor_ts_ns=anchors, bars=paths)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fold_b3_experiment(*, replay: MarketReplay, fold: FoldManifest,
        train_scope_id: str, configuration: dict, worker: WorkerClient,
        output_dir: Path, repository_commit: str, source_code_sha256: str,
        reservation_label: str, marker_sha256: str,
        registration_receipt: Mapping[str, object], forecast_seed: int,
        forecast_paths: int = 32, horizons: tuple[int, ...] = (1, 5, 15),
        anchor_interval_minutes: int = 15, batch_size: int = 32,
        hour_availability_delay_ns: int = 0, day_availability_delay_ns: int = 0,
        max_attempts_per_bar: int = 20, max_total_bar_attempts: int = 50_000_000,
        burn_in: int = 8, volatility_window: int = 15, block_length: int = 8,
        bootstrap_replicates: int = 2000, seed: int = 0) -> dict[str, object]:
    """Train B3, run paired E3/E4 rollouts, and score common Validation anchors.

    The caller must obtain ``replay`` and ``registration_receipt`` from one
    successful frozen-reader call. All contexts sent to the JAX worker are
    per-anchor, 4,320-bar and clipped to that anchor; no future labels or Test
    rows enter the worker. Batches bound worker memory and startup overhead.
    """
    if not isinstance(replay, MarketReplay) or not isinstance(fold, FoldManifest):
        raise ValueError("an authorized registered Fold replay is required")
    if (type(forecast_paths) is not int or not 2 <= forecast_paths <= 1024
            or type(batch_size) is not int or not 1 <= batch_size <= 64
            or type(forecast_seed) is not int or forecast_seed < 0
            or type(hour_availability_delay_ns) is not int
            or hour_availability_delay_ns < 0
            or type(day_availability_delay_ns) is not int
            or day_availability_delay_ns < 0
            or type(max_attempts_per_bar) is not int
            or not 1 <= max_attempts_per_bar <= 100
            or type(max_total_bar_attempts) is not int
            or not 1 <= max_total_bar_attempts <= 100_000_000):
        raise ValueError("invalid B3 rollout resources, delays, path count or seed")
    validate_fold_experiment_protocol(
        horizons=horizons, anchor_interval_minutes=anchor_interval_minutes,
        context_lookback=3 * 1440, volatility_window=volatility_window,
        block_length=block_length, bootstrap_replicates=bootstrap_replicates,
        seed=seed)
    anchors = validation_anchor_timestamps(
        fold=fold, max_horizon=horizons[-1],
        anchor_interval_minutes=anchor_interval_minutes)
    if not len(anchors):
        raise ValueError("Validation has no complete B3 forecast anchors")
    worst_case = len(anchors) * forecast_paths * horizons[-1] * max_attempts_per_bar
    per_anchor_worst_case = forecast_paths * horizons[-1] * max_attempts_per_bar
    if (per_anchor_worst_case > 1_000_000
            or worst_case > max_total_bar_attempts):
        raise ValueError("registered B3 worst-case bar attempts exceed the explicit Fold budget")
    if (replay.dataset_hash != fold.dataset_hash or not replay.is_contiguous
            or replay.frames[0].ts_close_ns != fold.train.warmup_start_ns
            or replay.frames[-1].ts_close_ns != fold.validation.score_end_ns - MINUTE_NS
            or any(frame.ts_close_ns >= fold.validation.score_end_ns
                   or frame.available_at_ns > fold.validation.score_end_ns
                   for frame in replay.frames)):
        raise ValueError("registered B3 replay must stop at Validation end and exclude Test")
    root = Path(output_dir).resolve()
    training_report = train_fold_b3_diagnostic(
        replay=replay, fold=fold, train_scope_id=train_scope_id,
        configuration=configuration, worker=worker, output_dir=root,
        repository_commit=repository_commit, source_code_sha256=source_code_sha256,
        reservation_label=reservation_label, registration_receipt=registration_receipt,
        marker_sha256=marker_sha256, burn_in=burn_in)
    train_artifact = root / "b3_train_hierarchy.npz"
    train_tensors, scalers, train_metadata = load_hierarchical_train_artifact(
        train_artifact, expected_scope=train_scope_id, expected_fold_sha256=fold.hash)
    if (train_metadata["hierarchy_tensor_sha256"] != training_report["hierarchy_tensor_sha256"]
            or hierarchy_scaler_sha256(scalers) != training_report["scaler_sha256"]):
        raise ValueError("B3 Train report and frozen artifact identity differ")
    checkpoint = root / "b3_checkpoint.npz"
    configuration_path = root / "b3_configuration.json"
    if (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != training_report["checkpoint_sha256"]
            or hashlib.sha256(configuration_path.read_bytes()).hexdigest()
            != training_report["configuration_sha256"]):
        raise ValueError("B3 Train checkpoint/configuration changed before Validation rollout")
    times = np.asarray([frame.ts_close_ns for frame in replay.frames], dtype=np.int64)
    by_time = {int(timestamp): index for index, timestamp in enumerate(times)}
    all_paths = {name: np.empty(
        (len(anchors), forecast_paths, horizons[-1], 5), dtype=np.float64)
        for name in ("E3", "E4")}
    total_sampling = {name: {"attempted": 0, "rejected": 0} for name in ("E3", "E4")}
    context_hashes: list[str] = []
    batch_count = (len(anchors) + batch_size - 1) // batch_size
    for batch_index in range(batch_count):
        start = batch_index * batch_size
        batch_anchors = anchors[start:start + batch_size]
        with tempfile.TemporaryDirectory(prefix="b3-context-batch-", dir=root) as scratch:
            scratch_dir = Path(scratch)
            context_items = []
            for local_index, anchor_value in enumerate(batch_anchors):
                anchor = int(anchor_value)
                anchor_index = by_time.get(anchor)
                if anchor_index is None:
                    raise ValueError("registered replay lacks a complete Validation anchor")
                visible_frames = replay.frames[:anchor_index + 1]
                if any(frame.available_at_ns > anchor for frame in visible_frames[-3 * 1440:]):
                    raise ValueError("B3 inference prefix contains an unavailable Validation candle")
                visible_candles = tuple(Candle(
                    frame.ts_close_ns, frame.available_at_ns, frame.instrument,
                    frame.open, frame.high, frame.low, frame.close, frame.volume,
                ) for frame in visible_frames)
                visible = MarketSequence(
                    visible_candles, dataset_id=fold.dataset_hash,
                    scope="inference_visible")
                tensors, observed = build_inference_hierarchy(
                    visible, scalers=scalers, decision_ts_ns=anchor)
                artifact_path = scratch_dir / f"context-{local_index:03d}.npz"
                _, metadata = export_hierarchical_inference_artifact(
                    artifact_path, tensors=tensors, scalers=scalers,
                    observed_prefix=observed, train_scope_id=train_scope_id,
                    fold_sha256=fold.hash, anchor_ts_ns=anchor)
                artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                context_hashes.append(metadata["hierarchy_tensor_sha256"])
                context_items.append({
                    "path": str(artifact_path), "artifact_sha256": artifact_sha,
                    "anchor_ts_ns": anchor,
                })
            inference_context_sha = hashlib.sha256(b"".join(
                bytes.fromhex(value) for value in context_hashes[-len(batch_anchors):]
            )).hexdigest()
            request = {
                "schema": "routeb.b3-rollout-batch-request.v1",
                "train_scope_id": train_scope_id, "dataset_id": fold.dataset_hash,
                "fold_sha256": fold.hash, "scaler_sha256": training_report["scaler_sha256"],
                "train_tensor_sha256": training_report["hierarchy_tensor_sha256"],
                "checkpoint_sha256": training_report["checkpoint_sha256"],
                "configuration_sha256": training_report["configuration_sha256"],
                "inference_context_sha256": inference_context_sha,
                "context_artifacts": context_items,
                "horizon": horizons[-1], "n_paths": forecast_paths,
                "forecast_seed": forecast_seed,
                "hour_availability_delay_ns": hour_availability_delay_ns,
                "day_availability_delay_ns": day_availability_delay_ns,
                "max_attempts_per_bar": max_attempts_per_bar,
                "max_total_bar_attempts": max_total_bar_attempts,
            }
            request_path = scratch_dir / "request.json"
            request_path.write_text(json.dumps(
                request, sort_keys=True, separators=(",", ":"), allow_nan=False),
                encoding="utf-8")
            output_path = scratch_dir / "paths.npz"
            result = worker.rollout_b3_batch(
                request=request_path, checkpoint=checkpoint,
                output=output_path, configuration=configuration_path)
            with np.load(output_path, allow_pickle=False) as archive:
                if not np.array_equal(archive["anchor_ts_ns"], batch_anchors):
                    raise ValueError("B3 worker changed the ordered Validation anchors")
                for name, member in (("E3", "bars_e3"), ("E4", "bars_e4")):
                    all_paths[name][start:start + len(batch_anchors)] = archive[member]
            for name in ("E3", "E4"):
                total_sampling[name]["attempted"] += result["sampling"][name]["attempted"]
                total_sampling[name]["rejected"] += result["sampling"][name]["rejected"]
    if len(context_hashes) != len(anchors):
        raise RuntimeError("B3 context construction did not cover every common anchor")
    aggregate_context_sha = hashlib.sha256(b"".join(
        bytes.fromhex(value) for value in context_hashes)).hexdigest()
    anchor_sha = _numeric_sha256(anchors, dtype="<i8")
    train_digest = training_report["report_sha256"]
    rollout_reports: dict[str, dict[str, object]] = {}
    for name, mode in (("E3", "FROZEN_OBSERVED_SLOW_POSTERIOR"),
                       ("E4", "ADVANCE_PATH_CONDITIONED_SLOW_POSTERIOR")):
        path_file = root / f"b3_{name.lower()}_paths.npz"
        path_file_sha = _write_paths(path_file, anchors=anchors, paths=all_paths[name])
        attempted = total_sampling[name]["attempted"]
        rejected = total_sampling[name]["rejected"]
        report: dict[str, object] = {
            "schema": "routeb.b3-rollout-run.v1", "status": "COMPLETED",
            "candidate": name, "rollout_mode": mode,
            "b3_train_report_sha256": train_digest,
            "dataset_hash": fold.dataset_hash, "fold_hash": fold.hash,
            "train_scope_id": train_scope_id,
            "registration_receipt_sha256": training_report["registration_receipt_sha256"],
            "reservation_key": training_report["reservation_key"],
            "reservation_token_sha256": training_report["reservation_token_sha256"],
            "train_artifact_sha256": training_report["train_artifact_sha256"],
            "checkpoint_sha256": training_report["checkpoint_sha256"],
            "configuration_sha256": training_report["configuration_sha256"],
            "training_config": training_report["training_config"],
            "optimizer_updates": training_report["optimizer_updates"],
            "train_windows": training_report["train_windows"],
            "parameter_count": training_report["parameter_count"],
            "training_seconds": training_report["training_seconds"],
            "device": training_report["device"],
            "snapshot_sha256": SNAPSHOT_SHA, "upstream_sha256": PINNED_UPSTREAM,
            "forecast_seed": forecast_seed, "anchors": len(anchors),
            "paths": forecast_paths, "horizon": horizons[-1],
            "inference_context_sha256": aggregate_context_sha,
            "anchor_timestamps_sha256": anchor_sha,
            "forecast_ohlcv_sha256": _numeric_sha256(all_paths[name], dtype="<f8"),
            "forecast_artifact": path_file.name,
            "forecast_artifact_sha256": path_file_sha,
            "attempted_bar_samples": attempted, "rejected_bar_samples": rejected,
            "max_attempts_per_bar": max_attempts_per_bar,
            "rejection_rate": rejected / attempted,
            "test_payload_decoded": False, "economic_certification": False,
        }
        report_sha = hashlib.sha256(json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        report["report_sha256"] = report_sha
        _write_json_fresh(root / f"b3_{name.lower()}_rollout.json", report)
        rollout_reports[name] = report
    final_report = build_fold_b3_predictive_report(
        replay, fold, all_paths, training_report, rollout_reports,
        {"E3": anchors, "E4": anchors}, train_scope_id=train_scope_id,
        snapshot_sha256=SNAPSHOT_SHA, forecast_seed=forecast_seed,
        inference_context_sha256=aggregate_context_sha,
        horizons=horizons, anchor_interval_minutes=anchor_interval_minutes,
        context_lookback=3 * 1440, volatility_window=volatility_window,
        block_length=block_length, bootstrap_replicates=bootstrap_replicates,
        seed=seed)
    _write_json_fresh(root / "b3_e3_e4_validation_report.json", final_report)
    return final_report
