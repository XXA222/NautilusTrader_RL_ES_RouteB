from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from rles_data.core import DAY_NS, MINUTE_NS
from rles_data.io import curated_event_time_bounds, curated_source_identity
from rles_orchestrator.research_access import (
    complete_routeb_fresh_fold,
    fail_routeb_fresh_fold,
    record_routeb_candidate_event,
    register_routeb_r3_candidate_attempt,
    routeb_fresh_fold_access_key,
)
from rles_world_model.cli import (
    _fold, _read_json, _registered_fold_baseline, _routeb_code_identity,
    _validate_registered_b0_fold,
)
from rles_world_model.routeb.jax_gru import GRUConfig
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import TrainingConfig
from rles_world_model.routeb.r2_ablation import R3_LAMBDA_ARMS, score_r2_ablation_ensembles
from rles_world_model.routeb.fold_experiment import (
    run_fold_experiment, verify_fold_experiment_report_digest,
)
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig

LAMBDA_CANDIDATES = {0.0: "R3_LAMBDA_000", 0.01: "R3_LAMBDA_010", 0.05: "R3_LAMBDA_050"}
RUN_STATE: dict = {}
RUN_ATTEMPTS: dict = {}
RUN_FOLD_CONTEXT: dict = {}


def _fresh_json(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def _seal(report: dict) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    report_sha = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()
    report["report_sha256"] = report_sha
    verify = dict(report)
    verify.pop("report_sha256")
    if sha256(json.dumps(verify, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()).hexdigest() != report_sha:
        raise ValueError("R3 aggregate digest did not verify after sealing")
    return report_sha


def _read_e2(directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    request = _read_json(str(directory / "validation_forecast_request.json"))
    report = _read_json(str(directory / "fold_experiment_report.json"))
    path = directory / "e2_validation_forecasts.npz"
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"bars"}:
            raise ValueError("R3 forecast archive has unexpected members")
        bars = np.asarray(archive["bars"], dtype=np.float64)
    return np.asarray(request["anchor_timestamps_ns"], dtype=np.int64), bars, report


def _validation_labels(replay, fold, anchors: np.ndarray, horizon: int):
    by_time = {frame.ts_close_ns: frame for frame in replay.frames}
    realized = np.empty((len(anchors), horizon, 5), dtype=np.float64)
    previous = np.empty(len(anchors), dtype=np.float64)
    for row, anchor_value in enumerate(anchors):
        anchor = int(anchor_value)
        anchor_frame = by_time.get(anchor)
        if anchor_frame is None:
            raise ValueError("R3 Validation anchor is absent from the registered replay")
        previous[row] = anchor_frame.close
        decision_time = anchor_frame.available_at_ns
        for step in range(1, horizon + 1):
            timestamp = anchor + step * MINUTE_NS
            frame = by_time.get(timestamp)
            if (frame is None or not fold.validation.score_start_ns <= timestamp < fold.validation.score_end_ns
                    or frame.available_at_ns <= decision_time):
                raise ValueError("R3 label crosses Validation or violates causal availability")
            realized[row, step - 1] = (frame.open, frame.high, frame.low, frame.close, frame.volume)
    return realized, previous


def main() -> int:
    global RUN_STATE, RUN_ATTEMPTS, RUN_FOLD_CONTEXT
    parser = argparse.ArgumentParser(description="Registered R3 lambda ablation on one Train/Validation Fold.")
    parser.add_argument("--registered-fold-config", required=True, type=Path)
    parser.add_argument("--e1-configuration", required=True, type=Path)
    parser.add_argument("--e2-configuration", required=True, type=Path)
    parser.add_argument("--worker-python", required=True, type=Path)
    parser.add_argument("--upstream-checkout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-seeds", default="31,102,20261018")
    parser.add_argument("--paths-per-seed", type=int, default=32)
    parser.add_argument("--forecast-seed", type=int, default=20261018)
    parser.add_argument("--anchor-interval-minutes", type=int, default=120)
    parser.add_argument("--context-lookback", type=int, default=128)
    parser.add_argument("--volatility-window", type=int, default=15)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    seeds = tuple(int(value.strip()) for value in args.model_seeds.split(","))
    if (len(seeds) < 3 or tuple(sorted(set(seeds))) != seeds or any(x < 0 for x in seeds)
            or args.paths_per_seed < 2 or args.anchor_interval_minutes < 1
            or args.block_length < 1 or args.bootstrap_replicates < 100 or args.seed < 0
            or args.forecast_seed < 0 or args.timeout_seconds < 1):
        raise ValueError("invalid frozen R3 seeds, path count, or paired-bootstrap protocol")
    output = args.output_dir.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("R3 output must be new and its parent must already exist")
    if not args.worker_python.is_file() or not args.upstream_checkout.is_dir():
        raise ValueError("pinned worker Python and upstream checkout are required")
    upstream_commit = subprocess.run(
        ["git", "-C", str(args.upstream_checkout.resolve()), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    if upstream_commit != PINNED_UPSTREAM:
        raise ValueError("upstream checkout differs from pinned DreamerV3")

    registered = _read_json(str(args.registered_fold_config))
    e1_raw, e2_raw = _read_json(str(args.e1_configuration)), _read_json(str(args.e2_configuration))
    expected = {"schema_version", "config", "fold", "marker_sha256", "label", "instrument",
                "allow_legacy_isolated_snapshot"}
    if set(registered) != expected or registered["schema_version"] != "rles.world_model.registered_fold_baseline.v0":
        raise ValueError("registered Fold config schema mismatch")
    if set(e1_raw) != {"model", "training"} or set(e2_raw) != {"model", "training"}:
        raise ValueError("candidate configs must contain exactly model and training")
    if registered["allow_legacy_isolated_snapshot"] is not False:
        raise ValueError("R3 requires row-group scoped source access")
    e1_model = asdict(GRUConfig(**e1_raw["model"]))
    e2_model_base = asdict(RSSMConfig(**e2_raw["model"]))
    e1_training = TrainingConfig(**e1_raw["training"])
    e2_training = TrainingConfig(**e2_raw["training"])
    if asdict(e1_training) != asdict(e2_training):
        raise ValueError("R3 E1/E2 controls must share the same optimizer budget")
    fold = _fold(registered["fold"])
    _validate_registered_b0_fold(fold)
    train_days = (fold.train.score_end_ns - fold.train.score_start_ns) // DAY_NS
    if fold.test.score_end_ns - fold.train.score_start_ns != (train_days + 32) * DAY_NS:
        raise ValueError("R3 requires the frozen 60/15/15-style Fold with both one-day gaps")
    environment = registered["config"].get("environment", {})
    market_path = environment.get("market_path")
    instrument = registered["instrument"]
    if (registered["config"].get("evaluation") is not None
            or registered["config"].get("test") is not None
            or registered["config"].get("holdout") is not None
            or registered["allow_legacy_isolated_snapshot"]
            or instrument != environment.get("instrument")
            or not isinstance(instrument, str) or not instrument.endswith("-PERP.BINANCE")
            or "USDT-PERP.BINANCE" not in instrument or not isinstance(market_path, str)):
        raise ValueError("R3 requires a market-only USD-M perpetual Fold without formal data scopes")
    source_identity = curated_source_identity(market_path)
    if (source_identity["schema_version"] != "0.4.1"
            or source_identity["marker_sha256"] != registered["marker_sha256"]
            or source_identity["snapshot_hash"] != fold.dataset_hash):
        raise ValueError("R3 source identity differs from the frozen Fold")
    source_prep_path = Path(market_path).with_name("source_preparation.json")
    source_prep = _read_json(str(source_prep_path))
    if (source_prep.get("schema") != "routeb.binance-fold-source-preparation.v1"
            or source_prep.get("instrument") != instrument
            or source_prep.get("dataset_hash") != fold.dataset_hash
            or source_prep.get("marker_sha256") != registered["marker_sha256"]
            or source_prep.get("source_file_sha256") != source_identity["file_sha256"]
            or source_prep.get("test_ohlcv_values_excluded") is not True
            or source_prep.get("market_payload_cutoff_ns") != fold.validation.score_end_ns
            or source_prep.get("last_market_event_ns") != fold.test.score_end_ns - MINUTE_NS
            or type(source_prep.get("sealed_market_rows")) is not int
            or source_prep["sealed_market_rows"] <= 0):
        raise ValueError("R3 curated source did not seal Test/embargo OHLCV values")
    footer = curated_event_time_bounds(market_path, expected_marker_sha256=registered["marker_sha256"])
    if footer["min_ts_event_ns"] > fold.train.warmup_start_ns or footer["max_ts_event_ns"] < fold.test.score_end_ns - MINUTE_NS:
        raise ValueError("R3 source footer does not cover the complete frozen Fold")

    code_identity = _routeb_code_identity()
    output.mkdir(parents=True, exist_ok=False)
    worker = WorkerClient(WorkerConfig(args.worker_python.absolute(), args.upstream_checkout.resolve(),
                                       timeout_seconds=args.timeout_seconds))
    worker_environment = worker.preflight()
    configurations: dict[tuple[float, int], dict] = {}
    attempt_keys: dict[tuple[float, int], str] = {}
    states: dict[tuple[float, int], str] = {}
    RUN_STATE, RUN_ATTEMPTS = states, attempt_keys
    for arm in R3_LAMBDA_ARMS:
        model = dict(e2_model_base)
        model["r2_regularizer_scale"] = arm
        checked_model = asdict(RSSMConfig(**model))
        for model_seed in seeds:
            training = asdict(e1_training)
            training["seed"] = model_seed
            config = {"model": checked_model, "training": training}
            key = (arm, model_seed)
            configurations[key] = config
            attempt_keys[key] = register_routeb_r3_candidate_attempt(
                label=registered["label"], candidate_id=LAMBDA_CANDIDATES[arm],
                configuration=config, fold_sha256=fold.hash,
                feature_sha256=fold.feature_hash, seed=model_seed)
            states[key] = "registered"

    seed_forecasts: dict[float, dict[int, np.ndarray]] = {arm: {} for arm in R3_LAMBDA_ARMS}
    seed_runs: dict[str, dict[str, dict]] = {}
    common_anchors = None
    access_window = {"start_ns": fold.train.warmup_start_ns,
                     "end_ns": fold.validation.score_end_ns}
    RUN_FOLD_CONTEXT = {"config": registered["config"], "window": access_window,
                        "label": registered["label"], "fold_sha256": fold.hash,
                        "repository_commit": code_identity["repository_commit"],
                        "source_code_sha256": code_identity["source_code_sha256"],
                        "source_identity": source_identity}
    receipt, fold, replay = _registered_fold_baseline(str(args.registered_fold_config), code_identity)
    access_key = routeb_fresh_fold_access_key(
        registered["config"], access_window, label=receipt["label"], fold_sha256=fold.hash,
        repository_commit=code_identity["repository_commit"],
        source_code_sha256=code_identity["source_code_sha256"], source_identity=source_identity)
    receipt_payload = {
        "schema": "routeb.registered-r3-receipt.v1", "reservation_label": receipt["label"],
        "marker_sha256": receipt["marker_sha256"], "dataset_hash": fold.dataset_hash,
        "fold_hash": fold.hash, "access_key": access_key, **code_identity,
        "read_scope": {"start_ns": fold.train.warmup_start_ns,
                       "end_ns": fold.validation.score_end_ns, "includes_test": False},
    }
    _fresh_json(output / "registration_receipt.json", receipt_payload)
    for arm in R3_LAMBDA_ARMS:
        for model_seed in seeds:
            key = (arm, model_seed)
            record_routeb_candidate_event(
                attempt_keys[key], event_key="started", status="started",
                detail={"access_key": access_key, "candidate": LAMBDA_CANDIDATES[arm],
                        "lambda": arm, "seed": model_seed, "test_payload_decoded": False})
            states[key] = "started"
            run_dir = output / f"lambda-{arm:g}" / f"seed-{model_seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            _fresh_json(run_dir / "registration_receipt.json", receipt_payload)
            try:
                model_seed_training = dict(configurations[key]["training"])
                model_seed_config = dict(configurations[key])
                model_seed_config["training"] = model_seed_training
                run_report = run_fold_experiment(
                    replay=replay, fold=fold, train_scope_id=receipt["label"],
                    candidate_configurations={
                        "E1": {"model": e1_model, "training": model_seed_training},
                        "E2": model_seed_config,
                    }, worker=worker, output_dir=run_dir,
                    repository_commit=code_identity["repository_commit"],
                    source_code_sha256=code_identity["source_code_sha256"],
                    marker_sha256=receipt["marker_sha256"], reservation_label=receipt["label"],
                    forecast_seed=args.forecast_seed + model_seed,
                    forecast_paths=args.paths_per_seed, horizons=(1, 5, 15),
                    anchor_interval_minutes=args.anchor_interval_minutes,
                    context_lookback=args.context_lookback,
                    volatility_window=args.volatility_window, block_length=args.block_length,
                    bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
                if not verify_fold_experiment_report_digest(run_report):
                    raise ValueError("R3 seed experiment report digest failed")
                anchors, paths, _ = _read_e2(run_dir)
                if paths.shape[1] != args.paths_per_seed:
                    raise ValueError("R3 forecast path count differs from the frozen protocol")
                if common_anchors is None:
                    common_anchors = anchors
                elif not np.array_equal(common_anchors, anchors):
                    raise ValueError("R3 arms and seeds do not share identical Validation anchors")
                seed_forecasts[arm][model_seed] = paths
                seed_runs.setdefault(str(model_seed), {})[str(arm)] = {
                    "report_sha256": run_report["report_sha256"],
                    "candidate_training_runs": run_report["candidate_training_runs"],
                    "forecast_path": str(run_dir / "e2_validation_forecasts.npz"),
                }
                e2_run = run_report["candidate_training_runs"]["E2"]
                record_routeb_candidate_event(
                    attempt_keys[key], event_key="completed", status="completed",
                    detail={"seed_report_sha256": run_report["report_sha256"],
                            "checkpoint_sha256": e2_run["checkpoint_sha256"],
                            "optimizer_updates": e2_run["optimizer_updates"],
                            "test_payload_decoded": False})
                states[key] = "completed"
            except Exception as exc:
                record_routeb_candidate_event(
                    attempt_keys[key], event_key="failed", status="failed",
                    detail={"error": type(exc).__name__, "detail": str(exc),
                            "test_payload_decoded": False})
                states[key] = "failed"
                raise

    if common_anchors is None:
        raise ValueError("R3 produced no Validation forecasts")
    realized, previous = _validation_labels(replay, fold, common_anchors, 15)
    paired = score_r2_ablation_ensembles(
        seed_forecasts, realized, previous, common_anchors, horizon=15,
        anchor_interval_ns=args.anchor_interval_minutes * MINUTE_NS,
        block_length=args.block_length, bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed)
    model_identity = {str(arm): asdict(RSSMConfig(**{
        **e2_model_base, "r2_regularizer_scale": arm})) for arm in R3_LAMBDA_ARMS}
    artifacts = {}
    for arm, per_seed in seed_forecasts.items():
        pooled = np.concatenate([per_seed[model_seed] for model_seed in seeds], axis=1)
        path = output / f"lambda-{arm:g}-pooled-validation-forecasts.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, bars=pooled)
        artifacts[str(arm)] = {"path": path.name,
                               "sha256": sha256(path.read_bytes()).hexdigest(),
                               "shape": list(pooled.shape)}
    aggregate = {
        "schema": "routeb.registered-r3-ablation.v1", "status": "COMPLETED",
        "evidence_scope": "registered_fresh_fold_train_validation",
        "protocol": {"lambda_arms": list(R3_LAMBDA_ARMS), "training_seeds": list(seeds),
                     "paths_per_seed": args.paths_per_seed, "horizon": 15,
                     "anchor_interval_minutes": args.anchor_interval_minutes,
                     "block_length": args.block_length,
                     "bootstrap_replicates": args.bootstrap_replicates},
        "dataset_hash": fold.dataset_hash, "fold_hash": fold.hash,
        "fold_manifest": {"train": asdict(fold.train), "validation": asdict(fold.validation),
                           "test": asdict(fold.test), "purge_ns": fold.purge_ns,
                           "embargo_ns": fold.embargo_ns},
        "repository_commit": code_identity["repository_commit"],
        "source_code_sha256": code_identity["source_code_sha256"],
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "marker_sha256": receipt["marker_sha256"], "reservation_label": receipt["label"],
        "source_identity": source_identity, "footer_time_bounds": footer,
        "source_preparation_sha256": sha256(source_prep_path.read_bytes()).hexdigest(),
        "sealed_market_rows": source_prep["sealed_market_rows"],
        "access_key": access_key,
        "read_scope": {"start_ns": fold.train.warmup_start_ns,
                       "end_ns": fold.validation.score_end_ns, "includes_test": False},
        "model_configurations": model_identity, "candidate_runs_by_seed": seed_runs,
        "forecast_artifacts": artifacts, "paired_validation_comparison": paired,
        "test_payload_decoded": False, "economic_certification": False,
        "decision_scope": "R3 Validation evidence only; no automatic promotion or R4-R6 authorization.",
        "worker_environment": worker_environment,
    }
    report_sha = _seal(aggregate)
    _fresh_json(output / "r3_ablation_report.json", aggregate)
    completion = complete_routeb_fresh_fold(
        registered["config"], access_window, label=receipt["label"], fold_sha256=fold.hash,
        repository_commit=code_identity["repository_commit"],
        source_code_sha256=code_identity["source_code_sha256"],
        source_identity=source_identity, report_sha256=report_sha,
        detail={"report_schema": aggregate["schema"], "test_payload_decoded": False,
                "economic_certification": False})
    _fresh_json(output / "research_completion_receipt.json", completion)
    print(json.dumps({"status": "COMPLETED", "report": str(output / "r3_ablation_report.json"),
                      "report_sha256": report_sha, "lambda_arms": list(R3_LAMBDA_ARMS),
                      "training_seeds": list(seeds), "anchors": len(common_anchors),
                      "test_payload_decoded": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            output_arg = sys.argv.index("--output-dir") + 1
            failure_dir = Path(sys.argv[output_arg])
            if failure_dir.is_dir():
                for key, state in list(RUN_STATE.items()):
                    if state in {"registered", "started"}:
                        record_routeb_candidate_event(
                            RUN_ATTEMPTS[key], event_key="aborted",
                            status="failed", detail={"error": type(exc).__name__,
                            "detail": "run aborted before this candidate started",
                            "test_payload_decoded": False})
                failure_path = failure_dir / "runner_failure.json"
                if not failure_path.exists():
                    failure_path.write_text(json.dumps({"schema": "routeb.r3-runner-failure.v1",
                        "status": "BLOCKED", "error": type(exc).__name__, "detail": str(exc),
                        "traceback": traceback.format_exc()}, sort_keys=True, indent=2,
                        ensure_ascii=False) + "\n", encoding="utf-8")
                if RUN_FOLD_CONTEXT:
                    failure_receipt = fail_routeb_fresh_fold(
                        RUN_FOLD_CONTEXT["config"], RUN_FOLD_CONTEXT["window"],
                        label=RUN_FOLD_CONTEXT["label"],
                        fold_sha256=RUN_FOLD_CONTEXT["fold_sha256"],
                        repository_commit=RUN_FOLD_CONTEXT["repository_commit"],
                        source_code_sha256=RUN_FOLD_CONTEXT["source_code_sha256"],
                        source_identity=RUN_FOLD_CONTEXT["source_identity"],
                        report_sha256=sha256(failure_path.read_bytes()).hexdigest(),
                        detail={"error": type(exc).__name__, "test_payload_decoded": False})
                    receipt_path = failure_dir / "research_failure_receipt.json"
                    if not receipt_path.exists():
                        receipt_path.write_text(json.dumps(failure_receipt, sort_keys=True,
                            indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__, "detail": str(exc)},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        raise



