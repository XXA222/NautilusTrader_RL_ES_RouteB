"""Registered cross-market comparison of the E2 RSSM and E3 hierarchy."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from typing import Any

import numpy as np

from rles_data.core import DAY_NS, MINUTE_NS, FoldManifest, Window
from rles_data.io import curated_event_time_bounds, curated_source_identity
from rles_data.json_utils import loads_unique_json
from rles_orchestrator.research_access import (
    complete_routeb_fresh_fold,
    fail_routeb_fresh_fold,
    read_registered_fold_window,
    record_routeb_candidate_event,
    register_routeb_r4_baseline_comparison_attempt,
    routeb_fresh_fold_access_identity_sha256,
    routeb_fresh_fold_access_key,
)
from rles_world_model.contracts import MarketReplay
from rles_world_model.routeb.codec import OHLCVCodec
from rles_world_model.routeb.hierarchy_dataset import (
    HierarchyScalers,
    build_train_hierarchy,
    build_validation_hierarchy,
)
from rles_world_model.routeb.integration import (
    prepare_fold_forecast_batch,
    prepare_fold_train_sequences,
)
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchyTrainingConfig,
    forecast_hierarchical_validation_static_context_batch_standardized,
    train_hierarchical,
)
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import SNAPSHOT_SHA, TrainingConfig
from rles_world_model.routeb.market import MarketSequence
from rles_world_model.routeb.sequences import (
    RISK_BARRIER_VOLATILITY_QUANTILE,
    load_worker_batch,
)
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig


CANDIDATES = ("R4_E2_SINGLE_SCALE", "R4_E3_HIERARCHY")
SEED_METRICS: dict[tuple[str, int], dict[str, np.ndarray]] = {}
ATTEMPTS: dict[tuple[str, int], str] = {}
ATTEMPT_STATES: dict[tuple[str, int], str] = {}


def _load_json_file(path: Path) -> Any:
    """Read frozen experiment-control JSON without duplicate-key ambiguity."""
    try:
        return loads_unique_json(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid R4 JSON input {path}: {exc}") from exc


def _matched_e3_training_config(*, e2_model: RSSMConfig, e3_protocol: dict[str, Any],
                               protocol: dict[str, Any], seed: int
                               ) -> HierarchyTrainingConfig:
    """Freeze E3's auxiliary risk objective to the corresponding E2 config."""
    return HierarchyTrainingConfig(
        seed=seed, steps=protocol["e3_training_steps"],
        hierarchy_variant=e3_protocol["variant"],
        window_length=protocol["e3_window_length"], burn_in=protocol["burn_in"],
        hour_context_steps=e3_protocol["hour_context_steps"],
        day_context_steps=e3_protocol["day_context_steps"],
        risk_auxiliary_scale=e2_model.risk_prediction_scale,
        risk_min_scale=e2_model.min_scale,
        risk_horizons=e2_model.risk_horizons,
        risk_horizon_weights=e2_model.risk_horizon_weights,
        risk_barrier_quantile=RISK_BARRIER_VOLATILITY_QUANTILE,
        learning_rate=protocol["learning_rate"], grad_clip=protocol["grad_clip"],
        report_interval=protocol["report_interval"])


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite registered artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _require_preregistered_digest(preregistration: str, label: str,
                                 aliases: tuple[str, ...], actual: str) -> None:
    matches: list[str] = []
    for alias in aliases:
        matches.extend(re.findall(
            rf"^- {re.escape(alias)} SHA-256: `([0-9a-f]{{64}})`$",
            preregistration, flags=re.MULTILINE))
    if len(matches) != 1 or matches[0] != actual:
        raise ValueError(f"R4 {label} digest differs from or is missing in preregistration")


def _verify_preregistered_controls(preregistration: str, *, protocol_path: Path,
                                   fold_config_path: Path, runner_path: Path) -> None:
    controls = (
        ("Protocol JSON", ("Protocol JSON", "Protocol"), protocol_path),
        ("Fold config", ("Registered Fold config", "Fold config"), fold_config_path),
        ("Runner", ("Registered runner", "Runner"), runner_path),
    )
    for label, aliases, path in controls:
        actual = sha256(path.read_bytes()).hexdigest()
        _require_preregistered_digest(preregistration, label, aliases, actual)


def _fold(value: dict) -> FoldManifest:
    expected = {"schema_version", "dataset_hash", "feature_hash", "train", "validation",
                "test", "purge_ns", "embargo_ns", "scaler_hash", "terminal_policy",
                "reset_positions", "reset_orders", "reset_replay", "freeze_validation_scaler"}
    if set(value) != expected:
        raise ValueError("registered FoldManifest schema mismatch")
    return FoldManifest(
        schema_version=value["schema_version"], dataset_hash=value["dataset_hash"],
        feature_hash=value["feature_hash"], train=Window(**value["train"]),
        validation=Window(**value["validation"]), test=Window(**value["test"]),
        purge_ns=value["purge_ns"], embargo_ns=value["embargo_ns"],
        scaler_hash=value["scaler_hash"], terminal_policy=value["terminal_policy"],
        reset_positions=value["reset_positions"], reset_orders=value["reset_orders"],
        reset_replay=value["reset_replay"],
        freeze_validation_scaler=value["freeze_validation_scaler"])


def _validate_fold(fold: FoldManifest, protocol: dict) -> None:
    day = DAY_NS
    if ((fold.train.score_end_ns - fold.train.score_start_ns) != 360 * day
            or fold.validation.score_end_ns - fold.validation.score_start_ns != 15 * day
            or fold.test.score_end_ns - fold.test.score_start_ns != 15 * day
            or fold.validation.score_start_ns - fold.train.score_end_ns != day
            or fold.test.score_start_ns - fold.validation.score_end_ns != day
            or fold.train.warmup_start_ns != fold.train.score_start_ns - 2 * day
            or fold.validation.warmup_start_ns != fold.validation.score_start_ns - 2 * day
            or fold.purge_ns != 60 * MINUTE_NS or fold.embargo_ns != 23 * 60 * MINUTE_NS):
        raise ValueError("Fold differs from its frozen 360/15/15 and 1h/23h protocol")
    expected = protocol["validation"]
    if (expected["anchor_count"] != 60 or expected["anchor_interval_minutes"] != 360
            or expected["horizon_minutes"] != 15
            or expected["candidate_paths_per_anchor"] != 128
            or expected["accepted_paths_per_anchor"] != 32):
        raise ValueError("unsupported frozen E2/E3 validation protocol")


def _preflight_ledger(root: Path, start_ns: int, end_ns: int, market: str) -> dict:
    path = root / ".artifacts/research-access.sqlite"
    if not path.is_file():
        raise ValueError("RouteB research ledger is missing")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        args = (market, "perpetual-market-v1", end_ns, start_ns)
        access = conn.execute(
            "SELECT access_key,start_ns,end_ns FROM access WHERE market=? AND family=? "
            "AND start_ns<? AND end_ns>?", args).fetchall()
        intents = conn.execute(
            "SELECT protocol_id,intent_id,start_ns,end_ns FROM query_intent "
            "WHERE market=? AND family=? AND start_ns<? AND end_ns>?", args).fetchall()
    finally:
        conn.close()
    if access or intents:
        raise ValueError(f"{market} R4 read interval is already consumed: access={access}, intents={intents}")
    return {"market": market, "start_ns": start_ns, "end_ns": end_ns,
            "overlapping_access_count": 0, "overlapping_query_intent_count": 0}


def _source_code_identity(root: Path) -> dict[str, str]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    fixed = [root / "pyproject.toml", root / "uv.lock",
             root / "packages/data/src/rles_data/core.py",
             root / "packages/data/src/rles_data/io.py",
             root / "packages/research/src/rles_research/access.py",
             root / "packages/orchestrator/src/rles_orchestrator/research_access.py"]
    model_dir = root / "packages/world_model/src/rles_world_model"
    files = sorted([*fixed, *model_dir.rglob("*.py")], key=lambda p: p.as_posix())
    digest = sha256(commit.encode("ascii"))
    for path in files:
        if not path.is_file():
            raise ValueError(f"RouteB code identity file missing: {path}")
        relative, contents = path.relative_to(root).as_posix().encode(), path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big")); digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big")); digest.update(contents)
    return {"repository_commit": commit, "source_code_sha256": digest.hexdigest()}


def _register_comparison_attempts(configurations: dict, seeds: tuple[int, ...],
                                  label: str, fold: FoldManifest) -> None:
    """Register the frozen attempt set, terminalizing a partial registration failure."""
    try:
        for candidate in CANDIDATES:
            for seed in seeds:
                _, config = configurations[(candidate, seed)]
                attempt_key = register_routeb_r4_baseline_comparison_attempt(
                    label=label, candidate_id=candidate, configuration=config,
                    fold_sha256=fold.hash, feature_sha256=fold.feature_hash, seed=seed)
                ATTEMPTS[(candidate, seed)] = attempt_key
                ATTEMPT_STATES[(candidate, seed)] = "registered"
    except Exception as exc:
        for key, status in tuple(ATTEMPT_STATES.items()):
            if status == "registered":
                record_routeb_candidate_event(ATTEMPTS[key], event_key="registration-failed",
                    status="failed", detail={"error": type(exc).__name__,
                    "test_payload_decoded": False})
                ATTEMPT_STATES[key] = "failed"
        raise


def _labels(replay: MarketReplay, fold: FoldManifest, anchors: np.ndarray,
            horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_time = {frame.ts_close_ns: frame for frame in replay.frames}
    closes = np.empty((len(anchors), horizon + 1), dtype=np.float64)
    decisions = np.empty(len(anchors), dtype=np.int64)
    for row, timestamp in enumerate(anchors):
        anchor = int(timestamp)
        frame = by_time.get(anchor)
        if frame is None or not fold.validation.score_start_ns <= anchor < fold.validation.score_end_ns:
            raise ValueError("forecast anchor is outside registered Validation")
        closes[row, 0] = frame.close
        decisions[row] = frame.available_at_ns
        for step in range(1, horizon + 1):
            target = by_time.get(anchor + step * MINUTE_NS)
            if (target is None or not fold.validation.score_start_ns <= target.ts_close_ns
                    < fold.validation.score_end_ns or target.available_at_ns <= frame.available_at_ns):
                raise ValueError("forecast label is missing, unavailable, or crosses Validation")
            closes[row, step] = target.close
    return closes, decisions, np.diff(np.log(closes), axis=1)


def _metrics_from_path_samples(samples: np.ndarray, anchors: np.ndarray,
                               replay: MarketReplay, fold: FoldManifest,
                               scalers: HierarchyScalers, required: int) -> dict[str, np.ndarray]:
    from rles_world_model.routeb.evaluation import _crps_per_anchor

    paths = np.asarray(samples, dtype=np.float64)
    if paths.ndim != 4 or paths.shape[0] != len(anchors) or paths.shape[2:] != (15, 5):
        raise ValueError("path forecast must be [anchors,candidates,15,5]")
    closes, _, _ = _labels(replay, fold, anchors, 15)
    by_time = {frame.ts_close_ns: frame for frame in replay.frames}
    returns = np.empty((len(anchors), required), dtype=np.float64)
    log_rv = np.empty_like(returns)
    accepted_count = np.zeros(len(anchors), dtype=np.int32)
    rejected_count = np.zeros(len(anchors), dtype=np.int32)
    for row, timestamp in enumerate(anchors):
        anchor_frame = by_time[int(timestamp)]
        for candidate in paths[row]:
            coords = scalers.minute.inverse(candidate)
            try:
                bars = OHLCVCodec.decode(
                    coords, previous_close=anchor_frame.close,
                    start_close_ns=int(timestamp) + MINUTE_NS,
                    instrument=anchor_frame.instrument)
            except ValueError:
                rejected_count[row] += 1
                continue
            if accepted_count[row] < required:
                generated_closes = np.asarray([bar.close for bar in bars], dtype=np.float64)
                returns[row, accepted_count[row]] = np.log(generated_closes[-1] / anchor_frame.close)
                path_closes = np.concatenate((np.asarray([anchor_frame.close]), generated_closes))
                increments = np.diff(np.log(path_closes))
                log_rv[row, accepted_count[row]] = np.log(max(float(np.sqrt(
                    np.mean(increments ** 2))), 1e-12))
                accepted_count[row] += 1
        if accepted_count[row] < required:
            raise ValueError(f"E3 safe path count {accepted_count[row]}<{required} at {timestamp}")
    observed_return = np.log(closes[:, -1] / closes[:, 0])
    observed_log_rv = np.log(np.maximum(np.sqrt(np.mean(
        np.diff(np.log(closes), axis=1) ** 2, axis=1)), 1e-12))
    return {"return_crps": _crps_per_anchor(returns, observed_return),
            "path_log_rv_crps": _crps_per_anchor(log_rv, observed_log_rv),
            "path_count": accepted_count, "rejected_count": rejected_count}


def _metrics_from_real_bars(bars: np.ndarray, anchors: np.ndarray,
                            replay: MarketReplay, fold: FoldManifest,
                            required: int) -> dict[str, np.ndarray]:
    from rles_world_model.routeb.evaluation import _crps_per_anchor

    paths = np.asarray(bars, dtype=np.float64)
    if paths.shape != (len(anchors), required, 15, 5) or not np.isfinite(paths).all():
        raise ValueError("E2 worker safe path batch differs from the frozen shape")
    closes, _, _ = _labels(replay, fold, anchors, 15)
    anchor_close = closes[:, :1]
    generated_close = paths[:, :, :, 3]
    generated_returns = np.log(generated_close[:, :, -1] / anchor_close)
    increments = np.diff(np.log(np.concatenate(
        (np.broadcast_to(anchor_close[:, :, None], (len(anchors), required, 1)),
         generated_close), axis=2)), axis=2)
    generated_log_rv = np.log(np.maximum(np.sqrt(np.mean(increments ** 2, axis=2)), 1e-12))
    observed_return = np.log(closes[:, -1] / closes[:, 0])
    observed_log_rv = np.log(np.maximum(np.sqrt(np.mean(
        np.diff(np.log(closes), axis=1) ** 2, axis=1)), 1e-12))
    return {"return_crps": _crps_per_anchor(generated_returns, observed_return),
            "path_log_rv_crps": _crps_per_anchor(generated_log_rv, observed_log_rv)}


def _bootstrap(candidate: np.ndarray, baseline: np.ndarray, *, block: int,
               replicates: int, seed: int) -> dict:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.ndim != 1 or len(delta) < block or not np.isfinite(delta).all():
        raise ValueError("paired bootstrap requires finite per-anchor deltas")
    rng = np.random.default_rng(seed)
    starts = np.arange(len(delta), dtype=np.int64)
    n_blocks = int(np.ceil(len(delta) / block))
    offsets = np.arange(block)
    values = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        selected = rng.choice(starts, size=n_blocks, replace=True)
        indices = np.concatenate([(start + offsets) % len(delta) for start in selected])[:len(delta)]
        values[i] = delta[indices].mean()
    lower, upper = np.quantile(values, [0.025, 0.975])
    p = (1 + np.count_nonzero(np.abs(values - values.mean()) >= abs(delta.mean()))) / (replicates + 1)
    return {"mean_difference": float(delta.mean()), "ci_95_low": float(lower),
            "ci_95_high": float(upper), "p_value": float(p),
            "block_length_anchors": block, "replicates": replicates}


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.ones(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-fold-config", required=True, type=Path)
    parser.add_argument("--protocol-json", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--worker-python", required=True, type=Path)
    parser.add_argument("--upstream-checkout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or not args.upstream_checkout.is_dir():
        raise ValueError("output must be new and pinned DreamerV3 checkout must exist")
    registered = _load_json_file(args.registered_fold_config)
    protocol = _load_json_file(args.protocol_json)
    prereg = args.preregistration.read_text(encoding="utf-8")
    frozen_status = (
        "Status: frozen before downloading the BNBUSDT source archives."
        if protocol.get("source_symbol") == "BNBUSDT"
        else "Status: protocol and date scope frozen before downloading or decoding these market rows."
    )
    if (frozen_status not in prereg
            or protocol.get("schema") != "routeb.r4-e2-e3-baseline-comparison.v1"
            or registered.get("instrument") != protocol.get("instrument")
            or registered.get("allow_legacy_isolated_snapshot") is not False):
        raise ValueError("registered Fold or preregistration differs from frozen E2/E3 protocol")
    _verify_preregistered_controls(
        prereg, protocol_path=args.protocol_json,
        fold_config_path=args.registered_fold_config,
        runner_path=Path(__file__).resolve())
    fold = _fold(registered["fold"])
    _validate_fold(fold, protocol)
    environment = registered["config"].get("environment", {})
    market_path = environment.get("market_path")
    if (not market_path or environment.get("instrument") != registered["instrument"]
            or any(registered["config"].get(name) is not None
                   for name in ("evaluation", "test", "holdout"))):
        raise ValueError("only a sealed raw market source with no formal scopes is permitted")
    source = curated_source_identity(market_path)
    prep_path = Path(market_path).with_name("source_preparation.json")
    prep = _load_json_file(prep_path)
    archive_receipt_path = Path(market_path).parents[1] / "official_archive_receipt.json"
    archive_receipt = _load_json_file(archive_receipt_path)
    if (source["schema_version"] != "0.4.1"
            or source["marker_sha256"] != registered["marker_sha256"]
            or source["snapshot_hash"] != fold.dataset_hash
            or prep.get("test_ohlcv_values_excluded") is not True
            or prep.get("market_payload_cutoff_ns") != fold.validation.score_end_ns
            or prep.get("last_market_event_ns") != fold.test.score_end_ns - MINUTE_NS
            or archive_receipt.get("schema") != protocol["archive_receipt_schema"]
            or archive_receipt.get("market_rows_decoded") is not False
            or len(archive_receipt.get("archives", ())) != 14):
        raise ValueError("curated source or archive receipt does not prove frozen Test sealing")
    footer = curated_event_time_bounds(market_path, expected_marker_sha256=registered["marker_sha256"])
    if footer["max_ts_event_ns"] < fold.test.score_end_ns - MINUTE_NS:
        raise ValueError("sealed source footer does not cover the Test timestamp markers")
    read_start = min(fold.train.warmup_start_ns, fold.validation.warmup_start_ns)
    read_end = fold.validation.score_end_ns
    preflight = _preflight_ledger(root, read_start, read_end, protocol["source_symbol"])
    upstream = subprocess.run(["git", "-C", str(args.upstream_checkout.resolve()),
                                "rev-parse", "HEAD"], capture_output=True,
                               text=True, check=True).stdout.strip()
    if upstream != PINNED_UPSTREAM:
        raise ValueError("E2 worker/E3 trainer upstream differs from pinned DreamerV3")
    worker = WorkerClient(WorkerConfig(args.worker_python.absolute(),
                                       args.upstream_checkout.resolve(), timeout_seconds=7200))
    runtime = worker.preflight()
    if not any("cuda" in device.lower() or "gpu" in device.lower()
               for device in runtime["devices"]):
        raise ValueError("RouteB CUDA JAX device is required for the frozen comparison budget")
    code_identity = _source_code_identity(root)
    output.mkdir(parents=True, exist_ok=False)

    seeds = tuple(protocol["training_seeds"])
    if len(seeds) != 3 or tuple(sorted(set(seeds))) != seeds:
        raise ValueError("frozen comparison requires three ordered unique training seeds")
    e2_model = RSSMConfig(**protocol["e2_model"])
    e3 = protocol["e3_hierarchy"]
    e3_minute, e3_hour, e3_day = (RSSMConfig(**e3[key])
                                  for key in ("minute_model", "hour_model", "day_model"))
    validation_protocol = protocol["validation"]
    forecast_seeds = tuple(range(validation_protocol["forecast_seed_start"],
                                 validation_protocol["forecast_seed_start"] +
                                 validation_protocol["candidate_paths_per_anchor"]))
    configurations: dict[
        tuple[str, int],
        tuple[TrainingConfig | HierarchyTrainingConfig, dict[str, Any]],
    ] = {}
    for candidate in CANDIDATES:
        for seed in seeds:
            training: TrainingConfig | HierarchyTrainingConfig
            if candidate == "R4_E2_SINGLE_SCALE":
                training = TrainingConfig(seed=seed, steps=protocol["e2_training_steps"],
                    batch_size=protocol["batch_size"], lr=protocol["learning_rate"],
                    grad_clip=protocol["grad_clip"], report_interval=protocol["report_interval"])
                config = {"model": asdict(e2_model), "training": asdict(training)}
            else:
                training = _matched_e3_training_config(
                    e2_model=e2_model, e3_protocol=e3, protocol=protocol, seed=seed)
                config = {"minute_model": asdict(e3_minute), "hour_model": asdict(e3_hour),
                          "day_model": asdict(e3_day), "training": asdict(training)}
            configurations[(candidate, seed)] = (training, config)

    window = {"start_ns": read_start, "end_ns": read_end}
    scope_config = {**registered["config"], "environment": {
        **environment, "instrument": registered["instrument"]}}
    access_key = routeb_fresh_fold_access_key(
        scope_config, window, label=registered["label"], fold_sha256=fold.hash,
        repository_commit=code_identity["repository_commit"],
        source_code_sha256=code_identity["source_code_sha256"], source_identity=source)
    _write_new(output / "registration_receipt.json", {
        "schema": "routeb.registered-r4-e2-e3-receipt.v1", "label": registered["label"],
        "access_key": access_key, "fold_hash": fold.hash, "dataset_hash": fold.dataset_hash,
        "instrument": registered["instrument"],
        "access_identity_sha256": routeb_fresh_fold_access_identity_sha256(
            scope_config, window, label=registered["label"], fold_sha256=fold.hash,
            repository_commit=code_identity["repository_commit"],
            source_code_sha256=code_identity["source_code_sha256"],
            source_identity=source),
        "marker_sha256": registered["marker_sha256"], **code_identity,
        "read_scope": {**window, "includes_test_payload": False},
        "ledger_preflight": preflight, "worker_runtime": runtime,
        "protocol_sha256": sha256(args.protocol_json.read_bytes()).hexdigest(),
        "preregistration_sha256": sha256(args.preregistration.read_bytes()).hexdigest(),
        "fold_config_sha256": sha256(args.registered_fold_config.read_bytes()).hexdigest(),
        "archive_receipt_sha256": sha256(archive_receipt_path.read_bytes()).hexdigest(),
        "source_preparation_sha256": sha256(prep_path.read_bytes()).hexdigest(),
    })

    failure_report = None
    reservation_state = {"reserved": False}

    def on_access_reserved() -> None:
        reservation_state["reserved"] = True
        _register_comparison_attempts(configurations, seeds, registered["label"], fold)

    try:
        store = read_registered_fold_window(
            scope_config, window, marker_sha256=registered["marker_sha256"],
            dataset_hash=fold.dataset_hash, required_start_ns=read_start,
            required_end_ns=fold.test.score_end_ns, label=registered["label"],
            fold_sha256=fold.hash, repository_commit=code_identity["repository_commit"],
            source_code_sha256=code_identity["source_code_sha256"],
            on_reserved=on_access_reserved,
            allow_legacy_isolated_snapshot=False)
        replay = MarketReplay.from_events(store.events, dataset_hash=fold.dataset_hash,
                                          instrument=registered["instrument"])
        if (not replay.is_contiguous or replay.frames[0].ts_close_ns != read_start
                or replay.frames[-1].ts_close_ns != read_end - MINUTE_NS
                or any(frame.ts_close_ns >= read_end for frame in replay.frames)):
            raise ValueError("registered replay must be continuous through Validation end and exclude Test")
        train_scope_id = registered["label"]
        train_npz = output / "e2_train_sequences.npz"
        train_npz, e2_scaler, train_summary = prepare_fold_train_sequences(
            fold_replay=replay, fold=fold, train_scope_id=train_scope_id,
            target_npz=train_npz, length=protocol["e2_window_length"],
            burn_in=protocol["burn_in"], stride=protocol["e2_sequence_stride"])
        train_batch, train_metadata = load_worker_batch(
            train_npz, expected_scope=train_scope_id,
            snapshot_sha=SNAPSHOT_SHA, upstream_sha=PINNED_UPSTREAM)
        if train_metadata.get("fold_sha256") != fold.hash:
            raise ValueError("E2 Train artifact is not bound to the registered Fold")

        train_frames = tuple(frame for frame in replay.frames
            if fold.train.warmup_start_ns <= frame.ts_close_ns < fold.train.score_end_ns)
        validation_frames = tuple(frame for frame in replay.frames
            if fold.validation.warmup_start_ns <= frame.ts_close_ns < fold.validation.score_end_ns)
        train_sequence = MarketSequence.from_frames(
            train_frames, dataset_id=fold.dataset_hash, scope="train")
        validation_sequence = MarketSequence.from_frames(
            validation_frames, dataset_id=fold.dataset_hash, scope="validation")
        first_cfg = configurations[("R4_E3_HIERARCHY", seeds[0])][0]
        if not isinstance(first_cfg, HierarchyTrainingConfig):
            raise RuntimeError("frozen E3 configuration is not hierarchical")
        train_tensors, e3_scalers = build_train_hierarchy(
            train_sequence, burn_in=first_cfg.burn_in,
            score_start_ns=fold.train.score_start_ns,
            score_end_ns=fold.train.score_end_ns,
            risk_horizons=first_cfg.risk_horizons,
            risk_barrier_quantile=first_cfg.risk_barrier_quantile)
        if (not np.allclose(e2_scaler.mean, e3_scalers.minute.mean, rtol=0, atol=1e-7)
                or not np.allclose(e2_scaler.std, e3_scalers.minute.std, rtol=0, atol=1e-7)):
            raise ValueError("E2 and E3 must share identical Train-only minute scaler moments")
        validation_tensors = build_validation_hierarchy(
            validation_sequence, scalers=e3_scalers,
            score_start_ns=fold.validation.score_start_ns,
            score_end_ns=fold.validation.score_end_ns)
        expected_anchors = fold.validation.score_start_ns + (
            np.arange(validation_protocol["anchor_count"], dtype=np.int64)
            * validation_protocol["anchor_interval_minutes"] * MINUTE_NS)
        minute_times = (validation_tensors.minute_available_at // MINUTE_NS) * MINUTE_NS
        starts = []
        for anchor in expected_anchors:
            rows = np.flatnonzero((minute_times == anchor)
                                  & (validation_tensors.minute_mask[0] > 0))
            if len(rows) != 1:
                raise ValueError("frozen Validation anchor missing or duplicated in E3 tensors")
            start = int(rows[0]) - first_cfg.window_length + 1
            if start < 0:
                raise ValueError("Validation warmup cannot form the frozen E3 context")
            starts.append(start)
        anchors = np.asarray(expected_anchors, dtype=np.int64)
        _labels(replay, fold, anchors, validation_protocol["horizon_minutes"])
        request_path = output / "validation_forecast_request.json"
        context_path = output / "validation_contexts.npz"
        request = prepare_fold_forecast_batch(
            fold_replay=replay, fold=fold, frozen_scaler=e2_scaler,
            train_scope_id=train_scope_id,
            train_npz_sha256=str(train_metadata["artifact_sha256"]),
            output_context_npz=context_path, output_request_json=request_path,
            horizon=validation_protocol["horizon_minutes"],
            n_paths=validation_protocol["accepted_paths_per_anchor"],
            seed=forecast_seeds[0],
            anchor_interval_minutes=validation_protocol["anchor_interval_minutes"],
            context_lookback=protocol["forecast_context_length"],
            anchor_start_ns=fold.validation.score_start_ns)
        if request["anchor_timestamps_ns"] != anchors.tolist():
            raise ValueError("E2 request and E3 evaluation anchors are not identical")

        for candidate in CANDIDATES:
            for seed in seeds:
                key = (candidate, seed)
                record_routeb_candidate_event(ATTEMPTS[key], event_key="started", status="started",
                    detail={"seed": seed, "fold_hash": fold.hash,
                            "test_payload_decoded": False, "access_key": access_key})
                ATTEMPT_STATES[key] = "started"
                training, config = configurations[key]
                run_dir = output / candidate.lower() / f"seed-{seed}"
                run_dir.mkdir(parents=True, exist_ok=False)
                started = time.perf_counter()
                if candidate == "R4_E2_SINGLE_SCALE":
                    config_path = run_dir / "configuration.json"
                    _write_new(config_path, config)
                    checkpoint = run_dir / "checkpoint.npz"
                    result = worker.train(sequences=train_npz, scope=train_scope_id,
                        checkpoint=checkpoint, configuration=config_path)
                    diagnostics = result.get("reports", [])
                    if not diagnostics or diagnostics[-1].get("optimizer_updates") != protocol["e2_training_steps"]:
                        raise ValueError("E2 worker did not complete the frozen update budget")
                    forecast_path = run_dir / "validation-paths.npz"
                    worker_result = worker.forecast_batch(
                        request=request_path, checkpoint=checkpoint, output=forecast_path,
                        configuration=config_path, candidate="E2")
                    if worker_result["sampled_paths_per_anchor"] != validation_protocol[
                            "candidate_paths_per_anchor"]:
                        raise ValueError("E2 worker candidate budget differs from preregistration")
                    with np.load(forecast_path, allow_pickle=False) as stored:
                        if set(stored.files) != {"bars"}:
                            raise ValueError("E2 forecast artifact schema mismatch")
                        metrics = _metrics_from_real_bars(
                            stored["bars"], anchors, replay, fold,
                            validation_protocol["accepted_paths_per_anchor"])
                    forecast_detail = {
                        "forecast_sha256": sha256(forecast_path.read_bytes()).hexdigest(),
                        "forecast_candidates_per_anchor": worker_result["sampled_paths_per_anchor"],
                        "forecast_seed_rule": worker_result["chunk_seed_rule"],
                        "rejected_paths": worker_result["rejected_paths"],
                    }
                    run_summary = {"training": diagnostics[-1], "forecast": forecast_detail,
                                   "train_config": config}
                else:
                    if not isinstance(training, HierarchyTrainingConfig):
                        raise RuntimeError("frozen E3 run requires a hierarchical configuration")
                    config_path = run_dir / "configuration.json"
                    _write_new(config_path, config)
                    checkpoint = run_dir / "checkpoint.npz"
                    reports = train_hierarchical(
                        tensors=train_tensors, train_scope_id=train_scope_id,
                        dataset_id=fold.dataset_hash, checkout=args.upstream_checkout.resolve(),
                        output_checkpoint=checkpoint, minute_config=e3_minute,
                        hour_config=e3_hour, day_config=e3_day,
                        training_config=training)
                    final = reports[-1]
                    if final.get("optimizer_updates") != protocol["e3_training_steps"]:
                        raise ValueError("E3 optimizer did not complete the frozen update budget")
                    forecasts = forecast_hierarchical_validation_static_context_batch_standardized(
                        train_tensors=train_tensors, validation_tensors=validation_tensors,
                        starts=tuple(starts), steps=validation_protocol["horizon_minutes"],
                        forecast_seeds=forecast_seeds,
                        train_scope_id=train_scope_id, dataset_id=fold.dataset_hash,
                        checkout=args.upstream_checkout.resolve(), checkpoint=checkpoint,
                        minute_config=e3_minute, hour_config=e3_hour, day_config=e3_day,
                        training_config=training,
                        forecast_batch_size=validation_protocol["forecast_chunk_size"])
                    metrics = _metrics_from_path_samples(
                        forecasts["samples"], anchors, replay, fold, e3_scalers,
                        validation_protocol["accepted_paths_per_anchor"])
                    forecast_path = run_dir / "validation-paths.npz"
                    with forecast_path.open("xb") as handle:
                        np.savez_compressed(handle, anchors=anchors,
                            samples=forecasts["samples"],
                            return_crps=metrics["return_crps"],
                            path_log_rv_crps=metrics["path_log_rv_crps"])
                    forecast_detail = {
                        "forecast_sha256": sha256(forecast_path.read_bytes()).hexdigest(),
                        "forecast_candidates_per_anchor": len(forecast_seeds),
                        "rejected_paths": int(metrics["rejected_count"].sum()),
                    }
                    run_summary = {"training": final, "forecast": forecast_detail,
                                   "train_config": config}
                run_summary.update({
                    "status": "COMPLETED", "candidate": candidate, "seed": seed,
                    "fold_hash": fold.hash, "dataset_hash": fold.dataset_hash,
                    "train_npz_sha256": train_metadata["artifact_sha256"],
                    "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                    "validation_anchors_sha256": sha256(anchors.astype("<i8").tobytes()).hexdigest(),
                    "training_wall_seconds": time.perf_counter() - started,
                    "metrics": {name: values.tolist() for name, values in metrics.items()
                                if name in {"return_crps", "path_log_rv_crps"}},
                    "test_payload_decoded": False,
                })
                _write_new(run_dir / "run.json", run_summary)
                SEED_METRICS[key] = metrics
                record_routeb_candidate_event(ATTEMPTS[key], event_key="completed",
                    status="completed", detail={"checkpoint_sha256": run_summary["checkpoint_sha256"],
                        "optimizer_updates": (protocol["e2_training_steps"] if candidate ==
                                              "R4_E2_SINGLE_SCALE" else protocol["e3_training_steps"]),
                        "forecast_sha256": forecast_detail["forecast_sha256"],
                        "test_payload_decoded": False})
                ATTEMPT_STATES[key] = "completed"

        comparisons = {}
        primary_p = []
        for seed in seeds:
            pair = {}
            for metric in ("path_log_rv_crps", "return_crps"):
                summary = _bootstrap(
                    SEED_METRICS[("R4_E3_HIERARCHY", seed)][metric],
                    SEED_METRICS[("R4_E2_SINGLE_SCALE", seed)][metric],
                    block=validation_protocol["bootstrap_block_anchors"],
                    replicates=validation_protocol["bootstrap_replicates"],
                    seed=7700 + seed + (0 if metric == "path_log_rv_crps" else 1))
                pair[metric] = summary
                if metric == "path_log_rv_crps":
                    primary_p.append(summary["p_value"])
            comparisons[str(seed)] = pair
        adjusted = _holm(primary_p)
        for seed, p_adjusted in zip(seeds, adjusted, strict=True):
            comparisons[str(seed)]["path_log_rv_crps"]["holm_p_value"] = p_adjusted
        primary_gate = all(
            comparisons[str(seed)]["path_log_rv_crps"]["ci_95_high"] < 0
            and comparisons[str(seed)]["path_log_rv_crps"]["holm_p_value"] < 0.05
            for seed in seeds)
        return_no_harm = all(
            comparisons[str(seed)]["return_crps"]["ci_95_low"] <= 0 for seed in seeds)
        report = {
            "schema": "routeb.registered-r4-e2-e3-comparison.v1",
            "status": "COMPLETED", "evidence_scope": "registered_train_validation_only",
            "comparison_type": "cross_instrument_system_comparison",
            "e2_risk_head_supervision_included": True,
            "protocol": protocol, "fold_hash": fold.hash,
            "reservation_label": registered["label"],
            "instrument": registered["instrument"],
            "dataset_hash": fold.dataset_hash, "feature_hash": fold.feature_hash,
            "source_identity": source, "source_footer": footer,
            "source_preparation_sha256": sha256(prep_path.read_bytes()).hexdigest(),
            "archive_receipt_sha256": sha256(archive_receipt_path.read_bytes()).hexdigest(),
            "fold_config_sha256": sha256(args.registered_fold_config.read_bytes()).hexdigest(),
            "protocol_sha256": sha256(args.protocol_json.read_bytes()).hexdigest(),
            "preregistration_sha256": sha256(args.preregistration.read_bytes()).hexdigest(),
            "repository_commit": code_identity["repository_commit"],
            "source_code_sha256": code_identity["source_code_sha256"],
            "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "access_key": access_key, "access_window": window,
            "access_identity_sha256": routeb_fresh_fold_access_identity_sha256(
                scope_config, window, label=registered["label"], fold_sha256=fold.hash,
                repository_commit=code_identity["repository_commit"],
                source_code_sha256=code_identity["source_code_sha256"],
                source_identity=source),
            "ledger_preflight": preflight, "worker_runtime": runtime,
            "anchor_timestamps_ns": anchors.tolist(),
            "anchor_sha256": sha256(anchors.astype("<i8").tobytes()).hexdigest(),
            "train_preparation": train_summary,
            "training_runs": {f"{candidate}:{seed}": {
                "run_sha256": sha256((output / candidate.lower() / f"seed-{seed}" / "run.json").read_bytes()).hexdigest(),
                "metrics": {name: SEED_METRICS[(candidate, seed)][name].tolist()
                            for name in ("return_crps", "path_log_rv_crps")}}
                for candidate in CANDIDATES for seed in seeds},
            "comparisons_e3_minus_e2": comparisons,
            "decision": {"strict_all_seed_primary_gate_passed": primary_gate,
                         "return_crps_no_harm_passed": return_no_harm,
                         "e3_promoted": bool(primary_gate and return_no_harm),
                         "economic_certification": False,
                         "test_payload_decoded": False},
            "test_payload_decoded": False,
        }
        canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        report["report_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
        report_path = output / "r4_e2_e3_comparison_report.json"
        _write_new(report_path, report)
        completion = complete_routeb_fresh_fold(
            scope_config, window, label=registered["label"], fold_sha256=fold.hash,
            repository_commit=code_identity["repository_commit"],
            source_code_sha256=code_identity["source_code_sha256"],
            source_identity=source, report_sha256=report["report_sha256"],
            detail={"test_payload_decoded": False, "e3_promoted": report["decision"]["e3_promoted"]})
        _write_new(output / "research_completion_receipt.json", completion)
        print(json.dumps({"status": report["status"], "report": str(report_path),
                          "report_sha256": report["report_sha256"],
                          "decision": report["decision"]}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        if (output / "r4_e2_e3_comparison_report.json").is_file():
            # The experiment report is already complete and independently
            # recoverable. A later receipt/printing failure must not relabel it.
            raise
        failure_report = {"schema": "routeb.registered-r4-e2-e3-failure.v2",
                          "status": "FAILED", "error": type(exc).__name__,
                          "detail": str(exc), "fold_hash": fold.hash,
                          "reservation_label": registered["label"],
                          "instrument": registered["instrument"],
                          "access_key": access_key,
                          "access_identity_sha256": routeb_fresh_fold_access_identity_sha256(
                              scope_config, window, label=registered["label"],
                              fold_sha256=fold.hash,
                              repository_commit=code_identity["repository_commit"],
                              source_code_sha256=code_identity["source_code_sha256"],
                              source_identity=source),
                          "test_payload_decoded": False}
        canonical = json.dumps(failure_report, sort_keys=True, separators=(",", ":"))
        failure_report["report_sha256"] = sha256(canonical.encode()).hexdigest()
        failure_path = output / "failure_receipt.json"
        _write_new(failure_path, failure_report)
        for key, status in ATTEMPT_STATES.items():
            if status in {"registered", "started"}:
                record_routeb_candidate_event(ATTEMPTS[key], event_key="failed", status="failed",
                    detail={"error": type(exc).__name__, "test_payload_decoded": False})
                ATTEMPT_STATES[key] = "failed"
        if reservation_state["reserved"]:
            report_sha256 = str(failure_report["report_sha256"])
            fail_routeb_fresh_fold(
                scope_config, window, label=registered["label"], fold_sha256=fold.hash,
                repository_commit=code_identity["repository_commit"],
                source_code_sha256=code_identity["source_code_sha256"],
                source_identity=source, report_sha256=report_sha256,
                detail={"error": type(exc).__name__, "test_payload_decoded": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
