"""One-reservation E0/E1/E2 experiment orchestration for a fresh Route B Fold."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping
import hmac

import numpy as np

from rles_data.core import FoldManifest

from ..baselines import score_baselines
from ..contracts import MarketReplay
from ..data_audit import audit_replay
from .fold_evaluation import build_fold_predictive_report
from .fold_protocol import validate_fold_experiment_protocol
from .integration import (
    prepare_fold_forecast_batch, prepare_fold_train_hierarchy,
    prepare_fold_train_sequences,
)
from .jax_model import PINNED_UPSTREAM
from .jax_train import SNAPSHOT_SHA
from .sequences import load_worker_batch
from .worker_client import WorkerClient


def verify_fold_experiment_report_digest(report: Mapping[str, object]) -> bool:
    """Verify the canonical top-level checksum of a registered Fold report."""
    supplied = report.get("report_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        return False
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return False
    expected = sha256(canonical.encode("utf-8")).hexdigest()
    return hmac.compare_digest(supplied, expected)


def _write_json_fresh(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite experiment artifact: {path.name}")
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2,
                                        ensure_ascii=False, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _candidate_run(*, name: str, configuration: dict, configuration_path: Path,
                   checkpoint: Path, worker_result: dict, train_batch,
                   train_metadata: dict, fold: FoldManifest,
                   train_scope_id: str) -> dict[str, object]:
    diagnostics = worker_result.get("reports")
    if not isinstance(diagnostics, list) or not diagnostics or not isinstance(diagnostics[-1], dict):
        raise ValueError(f"{name} worker omitted final training diagnostics")
    final = diagnostics[-1]
    elapsed = final.get("training_seconds")
    if (type(final.get("parameter_count")) is not int or final["parameter_count"] < 1
            or type(final.get("train_examples")) is not int
            or final["train_examples"] != train_batch.features.shape[0]
            or type(final.get("optimizer_updates")) is not int
            or final["optimizer_updates"] != configuration["training"].get("steps")
            or not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
            or not np.isfinite(elapsed) or elapsed < 0
            or not isinstance(final.get("device"), str) or not final["device"].strip()
            or not checkpoint.is_file() or not checkpoint.with_suffix(".json").is_file()):
        raise ValueError(f"{name} worker diagnostics or checkpoint pair are incomplete")
    return {
        "schema": "routeb.train-run.v1", "status": "COMPLETED",
        "candidate": name, "dataset_hash": train_metadata["dataset_id"],
        "fold_hash": fold.hash, "train_scope_id": train_scope_id,
        "train_npz_sha256": train_metadata["artifact_sha256"],
        "train_examples": int(train_batch.features.shape[0]),
        "window_length": int(train_batch.features.shape[1]),
        "training_config": configuration["training"],
        "optimizer_updates": final["optimizer_updates"],
        "parameter_count": final["parameter_count"], "training_seconds": elapsed,
        "device": final["device"],
        "configuration_sha256": sha256(configuration_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "snapshot_sha256": SNAPSHOT_SHA, "upstream_sha256": PINNED_UPSTREAM,
        "resume": False, "worker_result": worker_result,
    }


def train_fold_b3_diagnostic(*, replay: MarketReplay, fold: FoldManifest,
        train_scope_id: str, configuration: dict, worker: WorkerClient,
        output_dir: Path, repository_commit: str, source_code_sha256: str,
        reservation_label: str, burn_in: int = 8) -> dict[str, object]:
    """Train one registered Fold's B3 hierarchy without Validation/Test scoring.

    This is the Train-only B3 stage. It deliberately does not produce E3/E4
    predictive scores: the current B3 path forecast still uses a separate
    sampled-path feedback replay rather than a single-pass path generator.
    """
    if not isinstance(replay, MarketReplay) or not isinstance(fold, FoldManifest):
        raise ValueError("an authorized registered Fold replay is required")
    if (not isinstance(configuration, dict)
            or set(configuration) != {"model", "training"}
            or not isinstance(configuration["model"], dict)
            or set(configuration["model"]) != {"minute", "hour", "day"}
            or not isinstance(configuration["training"], dict)):
        raise ValueError("B3 requires minute/hour/day models and one training configuration")
    if (not isinstance(repository_commit, str) or len(repository_commit) != 40
            or not isinstance(source_code_sha256, str) or len(source_code_sha256) != 64
            or not isinstance(reservation_label, str)
            or not reservation_label.startswith("routeb-fold-")
            or not isinstance(train_scope_id, str) or not train_scope_id.strip()):
        raise ValueError("invalid B3 source, reservation or Train-scope identity")
    report_sha_inputs = (repository_commit, source_code_sha256)
    if any(any(char not in "0123456789abcdef" for char in value)
           for value in report_sha_inputs):
        raise ValueError("B3 source identities must be lowercase hexadecimal digests")

    root = Path(output_dir).resolve()
    existing = {path.name for path in root.iterdir()} if root.is_dir() else set()
    if not root.is_dir() or existing not in (set(), {"registration_receipt.json"}):
        raise ValueError("B3 output directory must be fresh except for its registration receipt")
    audit = audit_replay(replay)
    if not audit.market_model_eligible:
        raise ValueError("registered Fold failed market model eligibility audit")

    artifact_path, _, preparation = prepare_fold_train_hierarchy(
        fold_replay=replay, fold=fold, train_scope_id=train_scope_id,
        target_npz=root / "b3_train_hierarchy.npz", burn_in=burn_in)
    configuration_path = root / "b3_configuration.json"
    _write_json_fresh(configuration_path, configuration)
    checkpoint = root / "b3_checkpoint.npz"
    worker_result = worker.train_b3_hierarchical(
        sequences=artifact_path, scope=train_scope_id, fold_sha256=fold.hash,
        checkpoint=checkpoint, configuration=configuration_path, resume=False)
    diagnostics = worker_result.get("reports")
    if not isinstance(diagnostics, list) or not diagnostics or not isinstance(diagnostics[-1], dict):
        raise ValueError("B3 worker omitted final training diagnostics")
    final = diagnostics[-1]
    steps = configuration["training"].get("steps")
    elapsed = final.get("training_seconds")
    gradient_norms: dict[str, float] = {}
    for scale in ("minute", "hour", "day"):
        value = final.get(f"grad_{scale}_dyn_norm")
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"B3 {scale} scale has no finite nonzero gradient diagnostic")
        gradient_norms[scale] = float(value)
    if (type(steps) is not int or type(final.get("optimizer_updates")) is not int
            or final["optimizer_updates"] != steps
            or type(final.get("parameter_count")) is not int
            or final["parameter_count"] < 1
            or type(final.get("train_windows")) is not int
            or final["train_windows"] < 1
            or not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
            or not np.isfinite(elapsed) or elapsed < 0
            or not isinstance(final.get("device"), str) or not final["device"].strip()
            or not checkpoint.is_file() or not checkpoint.with_suffix(".json").is_file()):
        raise ValueError("B3 worker diagnostics or checkpoint pair are incomplete")
    report: dict[str, object] = {
        "schema": "routeb.b3-train-run.v1", "status": "COMPLETED",
        "candidate": "B3", "dataset_hash": fold.dataset_hash,
        "fold_hash": fold.hash, "train_scope_id": train_scope_id,
        "reservation_label": reservation_label,
        "repository_commit": repository_commit,
        "source_code_sha256": source_code_sha256,
        "train_artifact": artifact_path.name,
        "train_artifact_sha256": preparation["train_npz_sha256"],
        "hierarchy_tensor_sha256": preparation["hierarchy_tensor_sha256"],
        "scaler_sha256": preparation["scaler_sha256"],
        "training_config": configuration["training"],
        "optimizer_updates": final["optimizer_updates"],
        "parameter_count": final["parameter_count"],
        "train_windows": final["train_windows"],
        "training_seconds": elapsed, "device": final["device"],
        "scale_gradient_norms": gradient_norms,
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "configuration_sha256": sha256(configuration_path.read_bytes()).hexdigest(),
        "snapshot_sha256": SNAPSHOT_SHA, "upstream_sha256": PINNED_UPSTREAM,
        "validation_scored": False, "test_scored": False,
        "economic_certification": False,
        "worker_result": worker_result,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    _write_json_fresh(root / "b3_train_run.json", report)
    return report


def run_fold_experiment(*, replay: MarketReplay, fold: FoldManifest,
        train_scope_id: str, candidate_configurations: Mapping[str, dict],
        worker: WorkerClient, output_dir: Path,
        repository_commit: str, source_code_sha256: str,
        marker_sha256: str, reservation_label: str,
        forecast_seed: int, forecast_paths: int = 128,
        horizons: tuple[int, ...] = (1, 5, 15),
        anchor_interval_minutes: int = 15, context_lookback: int = 128,
        volatility_window: int = 15, block_length: int = 8,
        bootstrap_replicates: int = 2000, seed: int = 0) -> dict[str, object]:
    """Train matched E1/E2 candidates and score E0/E1/E2 within one fresh read.

    This function accepts only the in-memory replay returned by the registered
    reader. It has no market-file access and never loads Test rows. The caller
    must reserve/read a fresh Fold before calling it, then retain the receipt.
    """
    if not isinstance(replay, MarketReplay) or not isinstance(fold, FoldManifest):
        raise ValueError("an authorized registered Fold replay is required")
    if (not isinstance(candidate_configurations, Mapping)
            or set(candidate_configurations) != {"E1", "E2"}
            or any(not isinstance(candidate_configurations[name], dict)
                   or set(candidate_configurations[name]) != {"model", "training"}
                   for name in ("E1", "E2"))):
        raise ValueError("matched E1 and E2 model/training configurations are required")
    if candidate_configurations["E1"]["training"] != candidate_configurations["E2"]["training"]:
        raise ValueError("E1 and E2 must use identical Train optimizer budgets and seeds")
    if (len(repository_commit) != 40 or len(source_code_sha256) != 64
            or len(marker_sha256) != 64 or not reservation_label.startswith("routeb-fold-")
            or type(forecast_seed) is not int or forecast_seed < 0
            or type(forecast_paths) is not int or forecast_paths < 2):
        raise ValueError("invalid source, reservation, or forecast identity")
    validate_fold_experiment_protocol(
        horizons=horizons, anchor_interval_minutes=anchor_interval_minutes,
        context_lookback=context_lookback, volatility_window=volatility_window,
        block_length=block_length, bootstrap_replicates=bootstrap_replicates,
        seed=seed)
    root = Path(output_dir).resolve()
    existing = {path.name for path in root.iterdir()} if root.is_dir() else set()
    if not root.is_dir() or existing not in (set(), {"registration_receipt.json"}):
        raise ValueError("experiment output directory must be empty except for its registration receipt")
    audit = audit_replay(replay)
    if not audit.market_model_eligible:
        raise ValueError("registered Fold failed market model eligibility audit")

    sequence_path = root / "train_sequences.npz"
    sequence_path, scaler, preparation = prepare_fold_train_sequences(
        fold_replay=replay, fold=fold, train_scope_id=train_scope_id,
        target_npz=sequence_path)
    train_batch, train_metadata = load_worker_batch(
        sequence_path, expected_scope=train_scope_id,
        snapshot_sha=SNAPSHOT_SHA, upstream_sha=PINNED_UPSTREAM)
    if train_metadata.get("fold_sha256") != fold.hash:
        raise ValueError("Train worker artifact is not bound to this frozen Fold")

    runs: dict[str, dict[str, object]] = {}
    checkpoints: dict[str, Path] = {}
    configurations: dict[str, Path] = {}
    for name in ("E1", "E2"):
        config = candidate_configurations[name]
        config_path = root / f"{name.lower()}_configuration.json"
        _write_json_fresh(config_path, config)
        checkpoint = root / f"{name.lower()}_checkpoint.npz"
        checkpoints[name] = checkpoint
        configurations[name] = config_path
        train_worker = worker.train_e1 if name == "E1" else worker.train
        worker_result = train_worker(
            sequences=sequence_path, scope=train_scope_id, checkpoint=checkpoint,
            configuration=config_path, resume=False)
        runs[name] = _candidate_run(
            name=name, configuration=config, configuration_path=config_path,
            checkpoint=checkpoint, worker_result=worker_result,
            train_batch=train_batch, train_metadata=train_metadata,
            fold=fold, train_scope_id=train_scope_id)
        _write_json_fresh(root / f"{name.lower()}_train_run.json", runs[name])

    context_path = root / "validation_contexts.npz"
    request_path = root / "validation_forecast_request.json"
    request = prepare_fold_forecast_batch(
        fold_replay=replay, fold=fold, frozen_scaler=scaler,
        train_scope_id=train_scope_id,
        train_npz_sha256=str(train_metadata["artifact_sha256"]),
        output_context_npz=context_path, output_request_json=request_path,
        horizon=horizons[-1], n_paths=forecast_paths, seed=forecast_seed,
        anchor_interval_minutes=anchor_interval_minutes,
        context_lookback=context_lookback)
    distribution_request_path = root / "validation_next_distribution_request.json"
    distribution_request = {**request, "horizon": 1}
    _write_json_fresh(distribution_request_path, distribution_request)
    anchors = np.asarray(request["anchor_timestamps_ns"], dtype=np.int64)
    candidate_paths: dict[str, np.ndarray] = {}
    candidate_distributions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    forecast_artifacts: dict[str, dict[str, object]] = {}
    for name in ("E1", "E2"):
        output = root / f"{name.lower()}_validation_forecasts.npz"
        worker_result = worker.forecast_batch(
            request=request_path, checkpoint=checkpoints[name], output=output,
            configuration=configurations[name], candidate=name)
        artifact_sha256 = sha256(output.read_bytes()).hexdigest()
        expected_kind = "E1_BATCH_PRIOR_PATHS" if name == "E1" else "F1_BATCH_PRIOR_PATHS"
        if (worker_result.get("status") != "COMPLETED"
                or worker_result.get("kind") != expected_kind
                or worker_result.get("sha256") != artifact_sha256
                or worker_result.get("anchors") != len(anchors)
                or worker_result.get("paths") != forecast_paths
                or worker_result.get("horizon") != horizons[-1]):
            raise ValueError(f"{name} forecast worker metadata does not match its artifact")
        with np.load(output, allow_pickle=False) as artifact:
            if set(artifact.files) != {"bars"}:
                raise ValueError(f"{name} forecast artifact has unexpected members")
            bars = np.asarray(artifact["bars"], dtype=np.float64)
            if bars.shape != (len(anchors), forecast_paths, horizons[-1], 5):
                raise ValueError(f"{name} forecast artifact dimensions differ from the frozen request")
            candidate_paths[name] = bars.copy()
        forecast_artifacts[name] = {
            "artifact": output.name, "artifact_sha256": artifact_sha256,
            "worker_result": {
                "kind": worker_result["kind"], "status": worker_result["status"],
                "sha256": worker_result["sha256"], "anchors": worker_result["anchors"],
                "paths": worker_result["paths"], "horizon": worker_result["horizon"],
                "economic_certification": worker_result.get("economic_certification", False),
            },
        }

        distribution_output = root / f"{name.lower()}_validation_next_distribution.npz"
        distribution_result = worker.next_distribution_batch(
            request=distribution_request_path, checkpoint=checkpoints[name],
            output=distribution_output, configuration=configurations[name],
            candidate=name)
        distribution_sha256 = sha256(distribution_output.read_bytes()).hexdigest()
        expected_distribution_kind = f"{name}_BATCH_NEXT_DISTRIBUTION"
        if (distribution_result.get("status") != "COMPLETED"
                or distribution_result.get("kind") != expected_distribution_kind
                or distribution_result.get("sha256") != distribution_sha256
                or distribution_result.get("anchors") != len(anchors)
                or distribution_result.get("horizon") != 1):
            raise ValueError(f"{name} next-distribution metadata does not match its artifact")
        with np.load(distribution_output, allow_pickle=False) as artifact:
            if set(artifact.files) != {"standardized_mean", "standardized_scale"}:
                raise ValueError(f"{name} next-distribution artifact has unexpected members")
            mean = np.asarray(artifact["standardized_mean"], dtype=np.float64)
            scale = np.asarray(artifact["standardized_scale"], dtype=np.float64)
            if (mean.shape != (len(anchors), 5) or scale.shape != mean.shape
                    or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                    or np.any(scale <= 0)):
                raise ValueError(f"{name} next-distribution artifact is malformed")
            candidate_distributions[name] = (mean.copy(), scale.copy())
        forecast_artifacts[name]["next_distribution"] = {
            "artifact": distribution_output.name,
            "artifact_sha256": distribution_sha256,
            "kind": expected_distribution_kind,
        }

    predictive = build_fold_predictive_report(
        replay, fold, candidate_paths, runs,
        {"E1": anchors, "E2": anchors}, train_scope_id=train_scope_id,
        snapshot_sha256=SNAPSHOT_SHA, forecast_seed=forecast_seed,
        horizons=horizons, anchor_interval_minutes=anchor_interval_minutes,
        context_lookback=context_lookback, volatility_window=volatility_window,
        block_length=block_length, bootstrap_replicates=bootstrap_replicates, seed=seed,
        candidate_next_distributions=candidate_distributions, frozen_scaler=scaler)
    baselines = [
        {**score.__dict__} for score in score_baselines(
            replay, fold, seed=seed, ensemble_size=forecast_paths,
            volatility_lookback=volatility_window, bootstrap_block=block_length)
    ]
    result: dict[str, object] = {
        "schema": "routeb.fold-experiment.v1", "status": "COMPLETED",
        "evidence_scope": "registered_fresh_fold_train_validation",
        "dataset_hash": fold.dataset_hash, "fold_hash": fold.hash,
        "fold_manifest": {
            "schema_version": fold.schema_version, "dataset_hash": fold.dataset_hash,
            "feature_hash": fold.feature_hash, "train": fold.train.__dict__,
            "validation": fold.validation.__dict__, "test": fold.test.__dict__,
            "purge_ns": fold.purge_ns, "embargo_ns": fold.embargo_ns,
            "scaler_hash": fold.scaler_hash,
        },
        "repository_commit": repository_commit,
        "source_code_sha256": source_code_sha256,
        "marker_sha256": marker_sha256, "reservation_label": reservation_label,
        "read_scope": {
            "start_ns": fold.train.warmup_start_ns,
            "end_ns": fold.validation.score_end_ns, "includes_test": False,
        },
        "data_audit": {**audit.__dict__}, "train_preparation": preparation,
        "train_artifact_sha256": train_metadata["artifact_sha256"],
        "candidate_training_runs": runs,
        "forecast_context": {
            "artifact_sha256": request["context_artifact_sha256"],
            "request_sha256": sha256(request_path.read_bytes()).hexdigest(),
            "next_distribution_request_sha256": sha256(
                distribution_request_path.read_bytes()).hexdigest(),
            "anchors": len(anchors), "paths": forecast_paths,
            "horizon": horizons[-1], "test_payload_decoded": False,
        },
        "forecast_artifacts": forecast_artifacts,
        "baselines": baselines, "predictive_comparison": predictive,
        "test_payload_decoded": False, "economic_certification": False,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["report_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    _write_json_fresh(root / "fold_experiment_report.json", result)
    return result
