"""Registered E3/E3b/E4 hierarchy comparison on one fresh RouteB Fold.

Reads only Train warmup/score and Validation warmup/score after one-shot
registration. Test payloads remain timestamp-only in the curated source.
"""
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
    register_routeb_r4_hierarchy_attempt,
)
from rles_world_model.contracts import MarketReplay
from rles_world_model.routeb.codec import OHLCVCodec
from rles_world_model.routeb.evaluation import _crps_per_anchor
from rles_world_model.routeb.hierarchy_dataset import (
    HierarchyScalers,
    build_train_hierarchy,
    build_validation_hierarchy,
)
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchyTrainingConfig,
    forecast_hierarchical_validation_static_context_batch_standardized,
    train_hierarchical,
)
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.market import MarketSequence


VARIANTS = {
    "R4_E3_FEATURES": "e3_features",
    "R4_E3B_FUSION": "e3b_fusion",
    "R4_E4_CONDITIONAL": "e4_conditional",
}
ATTEMPTS: dict[tuple[str, int], str] = {}
ATTEMPT_STATES: dict[tuple[str, int], str] = {}


def _load_json_file(path: Path) -> Any:
    """Read frozen experiment-control JSON without duplicate-key ambiguity."""
    try:
        return loads_unique_json(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid R4 JSON input {path}: {exc}") from exc


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite frozen R4 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _parse_fold(value: dict) -> FoldManifest:
    expected = {"schema_version", "dataset_hash", "feature_hash", "train", "validation",
                "test", "purge_ns", "embargo_ns", "scaler_hash", "terminal_policy",
                "reset_positions", "reset_orders", "reset_replay", "freeze_validation_scaler"}
    if set(value) != expected:
        raise ValueError("R4 FoldManifest schema mismatch")
    return FoldManifest(
        schema_version=value["schema_version"], dataset_hash=value["dataset_hash"],
        feature_hash=value["feature_hash"], train=Window(**value["train"]),
        validation=Window(**value["validation"]), test=Window(**value["test"]),
        purge_ns=value["purge_ns"], embargo_ns=value["embargo_ns"],
        scaler_hash=value["scaler_hash"], terminal_policy=value["terminal_policy"],
        reset_positions=value["reset_positions"], reset_orders=value["reset_orders"],
        reset_replay=value["reset_replay"],
        freeze_validation_scaler=value["freeze_validation_scaler"])


def _validate_fold(fold: FoldManifest) -> None:
    if ((fold.train.score_end_ns - fold.train.score_start_ns) != 360 * DAY_NS
            or fold.validation.score_end_ns - fold.validation.score_start_ns != 15 * DAY_NS
            or fold.test.score_end_ns - fold.test.score_start_ns != 15 * DAY_NS
            or fold.validation.score_start_ns - fold.train.score_end_ns != DAY_NS
            or fold.test.score_start_ns - fold.validation.score_end_ns != DAY_NS
            or fold.purge_ns != MINUTE_NS * 60 or fold.embargo_ns != 23 * 60 * MINUTE_NS
            or fold.train.warmup_start_ns > fold.train.score_start_ns - 2 * DAY_NS
            or fold.validation.warmup_start_ns > fold.validation.score_start_ns - 2 * DAY_NS):
        raise ValueError("R4 requires 360/15/15, 1h purge + 23h embargo, and 2d warmup")


def _require_preregistered_digest(preregistration: str, label: str, actual: str) -> None:
    matches = re.findall(
        rf"^- {re.escape(label)} SHA-256: `([0-9a-f]{{64}})`$",
        preregistration, flags=re.MULTILINE,
    )
    if len(matches) != 1 or matches[0] != actual:
        raise ValueError(f"R4 {label} digest differs from the frozen preregistration")


def _verify_preregistered_controls(preregistration: str, *, protocol_path: Path,
                                   fold_config_path: Path, runner_path: Path) -> None:
    for label, path in (("Protocol JSON", protocol_path),
                         ("Fold config", fold_config_path),
                         ("Runner", runner_path)):
        actual = sha256(path.read_bytes()).hexdigest()
        _require_preregistered_digest(preregistration, label, actual)


def _validate_protocol_controls(protocol: dict[str, Any]) -> None:
    train_seeds = protocol.get("training_seeds")
    forecast_seeds = protocol.get("forecast_seeds")
    paths_per_seed = protocol.get("paths_per_seed")
    block_length = protocol.get("bootstrap_block_length")
    replicates = protocol.get("bootstrap_replicates")
    integer_controls = (protocol.get("anchor_interval_minutes"), protocol.get("horizon"),
                        protocol.get("warmup_days"), paths_per_seed, block_length, replicates)
    if any(type(value) is not int for value in integer_controls):
        raise ValueError("R4 protocol integer controls must be integers, not Booleans or floats")
    if (not isinstance(train_seeds, list) or len(train_seeds) != 3
            or any(type(seed) is not int or seed < 0 for seed in train_seeds)
            or train_seeds != sorted(set(train_seeds))):
        raise ValueError("R4 requires three sorted unique nonnegative training seeds")
    if (not isinstance(forecast_seeds, list) or not forecast_seeds
            or any(type(seed) is not int or seed < 0 for seed in forecast_seeds)
            or forecast_seeds != sorted(set(forecast_seeds))):
        raise ValueError("R4 forecast seeds must be a sorted unique nonnegative list")
    anchor_count = 15 * 24 * 60 // 360
    if (protocol.get("anchor_interval_minutes") != 360 or protocol.get("horizon") != 15
            or protocol.get("warmup_days") != 2 or type(paths_per_seed) is not int
            or not 1 <= paths_per_seed <= len(forecast_seeds)
            or type(block_length) is not int or not 1 <= block_length <= anchor_count
            or type(replicates) is not int or replicates < 1):
        raise ValueError("R4 protocol has invalid path, anchor, bootstrap, horizon, or warmup controls")
    if any(not isinstance(protocol.get(key), dict)
           for key in ("training", "minute_model", "hour_model", "day_model")):
        raise ValueError("R4 model and training controls must be JSON objects")


def _source_code_identity(root: Path) -> dict[str, str]:
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    fixed = [root / "pyproject.toml", root / "uv.lock",
             root / "packages/data/src/rles_data/core.py",
             root / "packages/data/src/rles_data/io.py",
             root / "packages/research/src/rles_research/access.py",
             root / "packages/orchestrator/src/rles_orchestrator/research_access.py"]
    model_dir = root / "packages/world_model/src/rles_world_model"
    files = sorted([*fixed, *model_dir.rglob("*.py")], key=lambda p: p.as_posix())
    h = sha256(commit.encode("ascii"))
    for path in files:
        if not path.is_file():
            raise ValueError(f"RouteB code identity file missing: {path}")
        rel, data = path.relative_to(root).as_posix().encode(), path.read_bytes()
        h.update(len(rel).to_bytes(4, "big")); h.update(rel)
        h.update(len(data).to_bytes(8, "big")); h.update(data)
    return {"repository_commit": commit, "source_code_sha256": h.hexdigest()}


def _preflight_ledger(root: Path, start_ns: int, end_ns: int) -> dict:
    path = root / ".artifacts/research-access.sqlite"
    if not path.is_file():
        raise ValueError("RouteB research ledger is missing")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        args = ("AVAXUSDT", "perpetual-market-v1", end_ns, start_ns)
        access = conn.execute(
            "SELECT access_key,start_ns,end_ns FROM access WHERE market=? AND family=? "
            "AND start_ns<? AND end_ns>?", args).fetchall()
        intents = conn.execute(
            "SELECT protocol_id,intent_id,start_ns,end_ns FROM query_intent "
            "WHERE market=? AND family=? AND start_ns<? AND end_ns>?", args).fetchall()
    finally:
        conn.close()
    if access or intents:
        raise ValueError(f"R4 candidate interval has ledger overlap: access={access}, intent={intents}")
    return {"start_ns": start_ns, "end_ns": end_ns,
            "overlapping_access_count": 0, "overlapping_query_intent_count": 0}


def _crps(samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return _crps_per_anchor(samples, observed)


def _paired_bootstrap(candidate: np.ndarray, baseline: np.ndarray, *, block: int,
                      replicates: int, seed: int) -> dict:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[0] < 1 or delta.shape[1] < block:
        raise ValueError("R4 paired bootstrap requires [seeds,anchors] and enough blocks")
    rng = np.random.default_rng(seed)
    n_seeds, n_anchors = delta.shape
    n_blocks = int(np.ceil(n_anchors / block))
    offsets = np.arange(block)
    boot = np.empty(replicates, dtype=np.float64)
    for rep in range(replicates):
        starts = rng.integers(0, n_anchors, size=n_blocks)
        indices = np.concatenate([(start + offsets) % n_anchors for start in starts])[:n_anchors]
        boot[rep] = delta[:, indices].mean()
    low, high = np.quantile(boot, [0.025, 0.975])
    observed = abs(float(delta.mean()))
    centered = boot - boot.mean()
    p = (1 + np.count_nonzero(np.abs(centered) >= observed)) / (replicates + 1)
    return {"mean_difference": float(delta.mean()), "ci_95_low": float(low),
            "ci_95_high": float(high), "p_value": float(p),
            "block_length_anchors": block, "replicates": replicates}


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.ones(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _validation_labels(replay: MarketReplay, fold: FoldManifest, anchors: np.ndarray,
                       horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_time = {frame.ts_close_ns: frame for frame in replay.frames}
    prices = np.empty((len(anchors), horizon + 1), dtype=np.float64)
    bars = np.empty((len(anchors), horizon, 5), dtype=np.float64)
    decisions = np.empty(len(anchors), dtype=np.int64)
    for row, anchor in enumerate(anchors):
        anchor = int(anchor); source = by_time.get(anchor)
        if source is None or not fold.validation.score_start_ns <= anchor < fold.validation.score_end_ns:
            raise ValueError("R4 anchor is outside registered Validation")
        prices[row, 0] = source.close; decisions[row] = source.available_at_ns
        for step in range(1, horizon + 1):
            frame = by_time.get(anchor + step * MINUTE_NS)
            if (frame is None or not fold.validation.score_start_ns <= frame.ts_close_ns
                    < fold.validation.score_end_ns or frame.available_at_ns <= source.available_at_ns):
                raise ValueError("R4 forecast label crosses Validation or violates availability")
            prices[row, step] = frame.close
            bars[row, step - 1] = (frame.open, frame.high, frame.low, frame.close, frame.volume)
    return prices, bars, decisions


def _evaluate_paths(samples: np.ndarray, anchors: np.ndarray, replay: MarketReplay,
                    fold: FoldManifest, scalers: HierarchyScalers,
                    required_paths: int) -> dict[str, np.ndarray]:
    prices, _labels, _decisions = _validation_labels(replay, fold, anchors, 15)
    # [anchor,path,step,coordinate], in the Train-standardized OHLCV space.
    paths = np.asarray(samples, dtype=np.float64)
    if (paths.ndim != 4 or paths.shape[0] != len(anchors)
            or paths.shape[1] < required_paths or paths.shape[2:] != (15, 5)):
        raise ValueError("R4 forecast path tensor has an unexpected shape")
    returns = np.full((len(anchors), required_paths), np.nan, dtype=np.float64)
    log_rv = np.full_like(returns, np.nan)
    accepted_count = np.zeros(len(anchors), dtype=np.int32)
    rejected_count = np.zeros(len(anchors), dtype=np.int32)
    for row, anchor in enumerate(anchors):
        start_close = int(anchor) + MINUTE_NS
        anchor_frame = next(frame for frame in replay.frames if frame.ts_close_ns == anchor)
        for path_index in range(paths.shape[1]):
            real_coords = scalers.minute.inverse(paths[row, path_index])
            try:
                bars = OHLCVCodec.decode(
                    real_coords, previous_close=anchor_frame.close,
                    start_close_ns=start_close, instrument=anchor_frame.instrument)
            except ValueError:
                rejected_count[row] += 1
                continue
            close = np.asarray([bar.close for bar in bars], dtype=np.float64)
            accepted = int(accepted_count[row])
            if accepted < required_paths:
                returns[row, accepted] = np.log(close[-1] / anchor_frame.close)
                increments = np.diff(np.log(np.concatenate(([anchor_frame.close], close))))
                log_rv[row, accepted] = np.log(max(float(np.sqrt(np.mean(increments ** 2))), 1e-12))
                accepted_count[row] += 1
        if accepted_count[row] < required_paths:
            raise ValueError(
                f"insufficient safe forecast paths: anchor={int(anchor)}, "
                f"accepted={int(accepted_count[row])}, rejected={int(rejected_count[row])}, "
                f"candidates={paths.shape[1]}, required={required_paths}")
    realized_returns = np.log(prices[:, -1] / prices[:, 0])
    realized_increments = np.diff(np.log(prices), axis=1)
    realized_log_rv = np.log(np.maximum(np.sqrt(np.mean(realized_increments ** 2, axis=1)), 1e-12))
    return {"return_crps": _crps(returns, realized_returns),
            "path_log_rv_crps": _crps(log_rv, realized_log_rv),
            "path_count": accepted_count, "candidate_count": np.full(
                len(anchors), paths.shape[1], dtype=np.int32),
            "rejected_count": rejected_count}


def main() -> int:
    parser = argparse.ArgumentParser(description="Registered RouteB R4 E3/E3b/E4 hierarchy comparison")
    parser.add_argument("--registered-fold-config", required=True, type=Path)
    parser.add_argument("--protocol-json", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--upstream-checkout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("R4 output directory must be new and its parent must exist")
    if not args.upstream_checkout.is_dir():
        raise ValueError("pinned DreamerV3 checkout is required")
    upstream = subprocess.run(["git", "-C", str(args.upstream_checkout.resolve()),
                               "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    if upstream != PINNED_UPSTREAM:
        raise ValueError("R4 upstream checkout differs from pinned DreamerV3")

    registered = _load_json_file(args.registered_fold_config)
    protocol = _load_json_file(args.protocol_json)
    preregistration = args.preregistration.read_text(encoding="utf-8")
    if "Status: frozen before downloading or decoding the run-03 market rows." not in preregistration:
        raise ValueError("R4 run-03 preregistration is missing or not frozen")
    _verify_preregistered_controls(
        preregistration, protocol_path=args.protocol_json,
        fold_config_path=args.registered_fold_config, runner_path=Path(__file__).resolve(),
    )
    if (registered.get("schema_version") != "rles.world_model.registered_fold_baseline.v0"
            or registered.get("allow_legacy_isolated_snapshot") is not False
            or registered.get("instrument") != "AVAXUSDT-PERP.BINANCE"):
        raise ValueError("R4 requires a fresh row-group-scoped AVAXUSDT Fold")
    expected_protocol_keys = {"schema", "training", "minute_model", "hour_model", "day_model",
                              "training_seeds", "anchor_interval_minutes", "paths_per_seed",
                              "horizon", "bootstrap_block_length", "bootstrap_replicates",
                              "forecast_seeds", "warmup_days"}
    if (set(protocol) != expected_protocol_keys
            or protocol["schema"] != "routeb.r4-hierarchy-protocol.v2"
            or protocol["horizon"] != 15 or protocol["anchor_interval_minutes"] != 360
            or protocol["warmup_days"] != 2):
        raise ValueError("R4 protocol JSON differs from its frozen schema or scope")
    _validate_protocol_controls(protocol)
    seeds = tuple(protocol["training_seeds"])
    forecast_seeds = tuple(protocol["forecast_seeds"])
    if (len(seeds) != 3 or tuple(sorted(set(seeds))) != seeds
            or len(forecast_seeds) < protocol["paths_per_seed"]
            or len(set(forecast_seeds)) != len(forecast_seeds)):
        raise ValueError("R4 requires three ordered train seeds and a unique frozen path-seed set")
    fold = _parse_fold(registered["fold"]); _validate_fold(fold)
    environment = registered["config"].get("environment", {})
    market_path = environment.get("market_path")
    if (registered["config"].get("evaluation") is not None
            or registered["config"].get("test") is not None
            or registered["config"].get("holdout") is not None
            or market_path is None or environment.get("instrument") != registered["instrument"]):
        raise ValueError("R4 accepts only market-source config without formal evaluation scopes")
    source = curated_source_identity(market_path)
    if (source["schema_version"] != "0.4.1"
            or source["marker_sha256"] != registered["marker_sha256"]
            or source["snapshot_hash"] != fold.dataset_hash):
        raise ValueError("R4 curated source identity differs from frozen Fold config")
    prep_path = Path(market_path).with_name("source_preparation.json")
    prep = _load_json_file(prep_path)
    archive_receipt_path = Path(market_path).parents[1] / "official_archive_receipt.json"
    archive_receipt = _load_json_file(archive_receipt_path)
    if (archive_receipt.get("schema") != "routeb.r4-run03.official-archive-receipt.v1"
            or archive_receipt.get("market_rows_decoded") is not False
            or len(archive_receipt.get("archives", ())) != 14):
        raise ValueError("R4 run-02 official archive receipt is incomplete")
    if (prep.get("schema") != "routeb.binance-fold-source-preparation.v1"
            or prep.get("dataset_hash") != fold.dataset_hash
            or prep.get("marker_sha256") != registered["marker_sha256"]
            or prep.get("market_payload_cutoff_ns") != fold.validation.score_end_ns
            or prep.get("test_ohlcv_values_excluded") is not True):
        raise ValueError("R4 source does not prove Test payload sealing")
    footer = curated_event_time_bounds(market_path,
        expected_marker_sha256=registered["marker_sha256"])
    if footer["max_ts_event_ns"] < fold.test.score_end_ns - MINUTE_NS:
        raise ValueError("R4 source footer does not cover sealed Test timestamp markers")
    read_start = min(fold.train.warmup_start_ns, fold.validation.warmup_start_ns)
    read_end = fold.validation.score_end_ns
    preflight = _preflight_ledger(root, read_start, read_end)
    code_identity = _source_code_identity(root)
    output.mkdir(parents=True)

    configs: dict[tuple[str, int], tuple[RSSMConfig, RSSMConfig, RSSMConfig, HierarchyTrainingConfig]] = {}
    for candidate, variant in VARIANTS.items():
        for seed in seeds:
            training_dict = dict(protocol["training"], seed=seed, hierarchy_variant=variant)
            cfg = HierarchyTrainingConfig(**training_dict)
            minute = RSSMConfig(**protocol["minute_model"])
            hour = RSSMConfig(**protocol["hour_model"])
            day = RSSMConfig(**protocol["day_model"])
            configs[(candidate, seed)] = (minute, hour, day, cfg)
            configuration = {"minute_model": asdict(minute), "hour_model": asdict(hour),
                              "day_model": asdict(day), "training": asdict(cfg),
                              "variant": variant}
            ATTEMPTS[(candidate, seed)] = register_routeb_r4_hierarchy_attempt(
                label=registered["label"], candidate_id=candidate,
                configuration=configuration, fold_sha256=fold.hash,
                feature_sha256=fold.feature_hash, seed=seed)
            ATTEMPT_STATES[(candidate, seed)] = "registered"

    access_window = {"start_ns": read_start, "end_ns": read_end}
    scope_config = {**registered["config"], "environment": {
        **environment, "instrument": registered["instrument"]}}
    store = read_registered_fold_window(
        scope_config, access_window, marker_sha256=registered["marker_sha256"],
        dataset_hash=fold.dataset_hash, required_start_ns=read_start,
        required_end_ns=fold.test.score_end_ns, label=registered["label"],
        fold_sha256=fold.hash, repository_commit=code_identity["repository_commit"],
        source_code_sha256=code_identity["source_code_sha256"],
        allow_legacy_isolated_snapshot=False)
    receipt = {"schema": "routeb.registered-r4-receipt.v1", "label": registered["label"],
               "fold_hash": fold.hash, "dataset_hash": fold.dataset_hash,
               "marker_sha256": registered["marker_sha256"], **code_identity,
               "registered_fold_config_sha256": sha256(
                   args.registered_fold_config.read_bytes()).hexdigest(),
               "protocol_sha256": sha256(args.protocol_json.read_bytes()).hexdigest(),
               "preregistration_sha256": sha256(args.preregistration.read_bytes()).hexdigest(),
               "official_archive_receipt_sha256": sha256(
                   archive_receipt_path.read_bytes()).hexdigest(),
               "source_preparation_sha256": sha256(prep_path.read_bytes()).hexdigest(),
               "read_scope": {**access_window, "includes_test_payload": False},
               "ledger_preflight": preflight}
    _write_new(output / "registration_receipt.json", receipt)
    replay = MarketReplay.from_events(store.events, dataset_hash=fold.dataset_hash,
                                      instrument=registered["instrument"])
    if not replay.is_contiguous:
        raise ValueError("R4 registered Train/Validation replay has a minute gap")
    train_frames = tuple(f for f in replay.frames
                         if fold.train.warmup_start_ns <= f.ts_close_ns < fold.train.score_end_ns)
    validation_frames = tuple(f for f in replay.frames
                              if fold.validation.warmup_start_ns <= f.ts_close_ns
                              < fold.validation.score_end_ns)
    train_sequence = MarketSequence.from_frames(train_frames, dataset_id=fold.dataset_hash, scope="train")
    validation_sequence = MarketSequence.from_frames(
        validation_frames, dataset_id=fold.dataset_hash, scope="validation")
    first_cfg = configs[(next(iter(VARIANTS)), seeds[0])][3]
    train_tensors, scalers = build_train_hierarchy(
        train_sequence, burn_in=first_cfg.burn_in,
        score_start_ns=fold.train.score_start_ns, score_end_ns=fold.train.score_end_ns)
    validation_tensors = build_validation_hierarchy(
        validation_sequence, scalers=scalers,
        score_start_ns=fold.validation.score_start_ns,
        score_end_ns=fold.validation.score_end_ns)
    timestamps = (validation_tensors.minute_available_at // MINUTE_NS) * MINUTE_NS
    anchor_times = fold.validation.score_start_ns + np.arange(
        0, 15 * 24 * 60, protocol["anchor_interval_minutes"], dtype=np.int64) * MINUTE_NS
    anchors, starts = [], []
    for anchor in anchor_times:
        rows = np.flatnonzero((timestamps == anchor) & (validation_tensors.minute_mask[0] > 0))
        if len(rows) != 1:
            raise ValueError("R4 fixed Validation anchor missing or duplicated")
        index = int(rows[0]); start = index - first_cfg.window_length + 1
        if start < 0:
            raise ValueError("R4 Validation warmup cannot form the frozen model context")
        anchors.append(int(anchor)); starts.append(start)
    anchor_array = np.asarray(anchors, dtype=np.int64)
    _validation_labels(replay, fold, anchor_array, 15)

    predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {name: {} for name in VARIANTS}
    training_runs: dict[str, dict[str, dict]] = {}
    try:
        for candidate, variant in VARIANTS.items():
            for model_seed in seeds:
                key = (candidate, model_seed); attempt = ATTEMPTS[key]
                record_routeb_candidate_event(attempt, event_key="started", status="started",
                    detail={"candidate": candidate, "seed": model_seed,
                            "access_scope": access_window, "test_payload_decoded": False})
                ATTEMPT_STATES[key] = "started"
                minute, hour, day, cfg = configs[key]
                run_dir = output / candidate.lower() / f"seed-{model_seed}"
                run_dir.mkdir(parents=True, exist_ok=False)
                checkpoint = run_dir / "checkpoint.npz"
                started = time.perf_counter()
                reports = train_hierarchical(
                    tensors=train_tensors, train_scope_id=registered["label"],
                    dataset_id=fold.dataset_hash, checkout=args.upstream_checkout.resolve(),
                    output_checkpoint=checkpoint, minute_config=minute,
                    hour_config=hour, day_config=day, training_config=cfg)
                run = reports[-1]
                if (run["optimizer_updates"] != cfg.steps or run["grad_norm"] <= 0
                        or not np.isfinite(run["loss"])):
                    raise ValueError("R4 optimizer did not complete the frozen finite-update budget")
                forecasts = forecast_hierarchical_validation_static_context_batch_standardized(
                    train_tensors=train_tensors, validation_tensors=validation_tensors,
                    starts=tuple(starts), steps=15, forecast_seeds=forecast_seeds,
                    train_scope_id=registered["label"], dataset_id=fold.dataset_hash,
                    checkout=args.upstream_checkout.resolve(), checkpoint=checkpoint,
                    minute_config=minute, hour_config=hour, day_config=day,
                    training_config=cfg, forecast_batch_size=8)
                metrics = _evaluate_paths(forecasts["samples"], anchor_array, replay, fold,
                                          scalers, protocol["paths_per_seed"])
                if np.any(metrics["path_count"] < protocol["paths_per_seed"]):
                    raise ValueError("R4 invalid generated paths were omitted from the registered ensemble")
                with (run_dir / "validation-paths.npz").open("xb") as stream:
                    np.savez_compressed(stream, anchors=anchor_array,
                                        samples=forecasts["samples"],
                                        return_crps=metrics["return_crps"],
                                        path_log_rv_crps=metrics["path_log_rv_crps"])
                run_info = {**run, "training_wall_seconds": time.perf_counter() - started,
                            "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                            "forecast_sha256": sha256((run_dir / "validation-paths.npz").read_bytes()).hexdigest(),
                            "unique_anchors": len(anchor_array),
                            "forecast_paths_per_anchor": protocol["paths_per_seed"],
                            "forecast_candidates_per_anchor": len(forecast_seeds),
                            "forecast_batch_size": 8,
                            "rejected_forecast_candidates": int(metrics["rejected_count"].sum()),
                            "minimum_valid_paths_per_anchor": int(metrics["path_count"].min()),
                            "parameter_count": run["parameters"],
                            "variant": variant}
                _write_new(run_dir / "run.json", run_info)
                predictions[candidate][model_seed] = metrics
                training_runs.setdefault(candidate, {})[str(model_seed)] = run_info
                record_routeb_candidate_event(attempt, event_key="completed", status="completed",
                    detail={"checkpoint_sha256": run_info["checkpoint_sha256"],
                            "forecast_sha256": run_info["forecast_sha256"],
                            "optimizer_updates": cfg.steps, "device": run["device"],
                            "test_payload_decoded": False})
                ATTEMPT_STATES[key] = "completed"

        comparisons: dict[str, Any] = {}
        primary_adjustments = []
        for metric in ("path_log_rv_crps", "return_crps"):
            for comparator in ("R4_E3_FEATURES", "R4_E3B_FUSION"):
                candidate_matrix = np.stack([predictions["R4_E4_CONDITIONAL"][s][metric]
                                              for s in seeds])
                baseline_matrix = np.stack([predictions[comparator][s][metric] for s in seeds])
                pooled = _paired_bootstrap(candidate_matrix, baseline_matrix,
                    block=protocol["bootstrap_block_length"],
                    replicates=protocol["bootstrap_replicates"], seed=7301 + len(comparisons))
                per_seed = {}
                for index, seed in enumerate(seeds):
                    per_seed[str(seed)] = _paired_bootstrap(
                        candidate_matrix[index:index+1], baseline_matrix[index:index+1],
                        block=protocol["bootstrap_block_length"],
                        replicates=protocol["bootstrap_replicates"], seed=7400 + index)
                    if metric == "path_log_rv_crps":
                        primary_adjustments.append(per_seed[str(seed)]["p_value"])
                comparisons[f"{metric}:E4-minus-{comparator}"] = {
                    "pooled": pooled, "per_seed": per_seed}
        adjusted = _holm(primary_adjustments)
        primary_index = 0
        for metric in ("path_log_rv_crps",):
            for comparator in ("R4_E3_FEATURES", "R4_E3B_FUSION"):
                block = comparisons[f"{metric}:E4-minus-{comparator}"]["per_seed"]
                for seed in seeds:
                    block[str(seed)]["holm_p_value"] = adjusted[primary_index]
                    primary_index += 1
        all_seed_gate = all(
            comparisons[f"path_log_rv_crps:E4-minus-{comparator}"]["per_seed"][str(seed)]["ci_95_high"] < 0
            and comparisons[f"path_log_rv_crps:E4-minus-{comparator}"]["per_seed"][str(seed)]["holm_p_value"] < 0.05
            for comparator in ("R4_E3_FEATURES", "R4_E3B_FUSION") for seed in seeds)
        no_return_harm = all(
            comparisons[f"return_crps:E4-minus-{comparator}"]["per_seed"][str(seed)]["ci_95_low"] <= 0
            for comparator in ("R4_E3_FEATURES", "R4_E3B_FUSION") for seed in seeds)
        report = {
            "schema": "routeb.registered-r4-hierarchy.v1", "status": "COMPLETED",
            "evidence_scope": "registered_train_validation_only",
            "protocol": protocol, "fold_hash": fold.hash, "dataset_hash": fold.dataset_hash,
            "feature_hash": fold.feature_hash, "source_identity": source,
            "source_preparation_sha256": sha256(prep_path.read_bytes()).hexdigest(),
            "official_archive_receipt_sha256": sha256(
                archive_receipt_path.read_bytes()).hexdigest(),
            "registered_fold_config_sha256": sha256(
                args.registered_fold_config.read_bytes()).hexdigest(),
            "protocol_sha256": sha256(args.protocol_json.read_bytes()).hexdigest(),
            "preregistration_sha256": sha256(args.preregistration.read_bytes()).hexdigest(),
            "source_footer": footer, "repository_commit": code_identity["repository_commit"],
            "source_code_sha256": code_identity["source_code_sha256"],
            "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "access_window": access_window,
            "anchors_sha256": sha256(anchor_array.astype("<i8").tobytes()).hexdigest(),
            "anchors": anchor_array.tolist(), "training_runs": training_runs,
            "comparisons": comparisons,
            "decision": {"strict_all_seed_E4_gate_passed": all_seed_gate,
                         "return_crps_no_significant_harm": no_return_harm,
                         "research_path": "retain E4 for follow-up" if all_seed_gate and no_return_harm
                         else "no hierarchy promotion; prefer E3 pending further evidence"},
            "test_payload_decoded": False, "economic_certification": False,
            "automatic_promotion": False,
        }
        report["report_sha256"] = sha256(json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        _write_new(output / "r4_hierarchy_report.json", report)
        receipt = complete_routeb_fresh_fold(
            scope_config, access_window, label=registered["label"], fold_sha256=fold.hash,
            repository_commit=code_identity["repository_commit"],
            source_code_sha256=code_identity["source_code_sha256"], source_identity=source,
            report_sha256=report["report_sha256"],
            detail={"report_schema": report["schema"], "test_payload_decoded": False,
                    "economic_certification": False})
        _write_new(output / "research_completion_receipt.json", receipt)
        print(json.dumps({"status": "COMPLETED", "report": str(output / "r4_hierarchy_report.json"),
                          "report_sha256": report["report_sha256"], "anchors": len(anchor_array),
                          "variants": list(VARIANTS), "training_seeds": list(seeds),
                          "test_payload_decoded": False}, ensure_ascii=False))
        return 0
    except Exception as exc:
        for key, status in ATTEMPT_STATES.items():
            if status in {"registered", "started"}:
                if status == "registered":
                    record_routeb_candidate_event(ATTEMPTS[key], event_key="started",
                        status="started", detail={"optimizer_started": False,
                        "reason": "run stopped before this candidate was scheduled",
                        "test_payload_decoded": False})
                record_routeb_candidate_event(ATTEMPTS[key], event_key="failed", status="failed",
                    detail={"error": type(exc).__name__, "detail": str(exc),
                            "optimizer_started": status == "started",
                            "test_payload_decoded": False})
                ATTEMPT_STATES[key] = "failed"
        if output.is_dir():
            failure = {"schema": "routeb.r4-hierarchy-failure.v1",
                       "error": type(exc).__name__, "detail": str(exc),
                       "test_payload_decoded": False}
            if source and access_window:
                failure_sha256 = sha256(json.dumps(
                    failure, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                failure["failure_report_sha256"] = failure_sha256
                failed_receipt = fail_routeb_fresh_fold(
                    scope_config, access_window, label=registered["label"],
                    fold_sha256=fold.hash, repository_commit=code_identity["repository_commit"],
                    source_code_sha256=code_identity["source_code_sha256"], source_identity=source,
                    report_sha256=failure_sha256, detail=failure)
                _write_new(output / "failure_report.json", failure)
                _write_new(output / "research_failure_receipt.json", failed_receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
