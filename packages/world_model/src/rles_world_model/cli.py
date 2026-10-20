"""JSON CLI for B0 audits and baseline scoring; never reads unregistered files."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

from rles_data.core import DAY_NS, MINUTE_NS, DataError, FoldManifest, Window

from .baselines import score_baselines
from .contracts import MarketFrame, MarketReplay
from .data_audit import audit_replay
from .train_dataset import project_registered_fold_replay, project_registered_train_replay
from .routeb.integration import prepare_fold_train_sequences
from .routeb.worker_client import WorkerClient, WorkerConfig
from .routeb.sequences import load_worker_batch
from .routeb.jax_train import SNAPSHOT_SHA
from .routeb.jax_model import PINNED_UPSTREAM
from .routeb.fold_experiment import run_fold_experiment
from .upstream import inspect_upstream


def _read_json(path: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read valid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise DataError("JSON input root must be an object")
    return value


def _replay(path: str) -> tuple[dict, MarketReplay]:
    payload = _read_json(path)
    if set(payload) != {"schema_version", "evidence_scope", "dataset_hash", "instrument", "frames"}:
        raise DataError("synthetic replay must declare schema, evidence scope, dataset hash, instrument, and frames")
    if (payload["schema_version"] != "rles.market_replay.v0"
            or payload["evidence_scope"] != "synthetic_test_only"
            or not isinstance(payload["frames"], list)):
        raise DataError("only synthetic_test_only replay JSON is accepted by this mechanism interface")
    frames = tuple(MarketFrame(**item) for item in payload["frames"])
    replay = MarketReplay(frames, payload["dataset_hash"])
    if any(frame.instrument != payload["instrument"] for frame in frames):
        raise DataError("market replay frames do not match declared instrument")
    return payload, replay


def _registered_train_replay(path: str) -> tuple[dict, MarketReplay]:
    payload = _read_json(path)
    expected = {"schema_version", "config", "window", "marker_sha256", "label", "instrument",
                "allow_legacy_isolated_snapshot"}
    if set(payload) != expected or payload["schema_version"] != "rles.world_model.registered_train_audit.v0":
        raise DataError("registered Train config does not match the supported audit schema")
    config = payload["config"]
    window = payload["window"]
    if (not isinstance(config, dict) or not isinstance(config.get("environment"), dict)
            or config.get("evaluation") is not None or config.get("test") is not None
            or config.get("holdout") is not None):
        raise DataError("B0 registered audit accepts a Train-only application config")
    if (not isinstance(window, dict) or set(window) != {"start_ns", "end_ns"}
            or not isinstance(payload["marker_sha256"], str)
            or not isinstance(payload["label"], str) or not payload["label"].startswith("train-")
            or not isinstance(payload["instrument"], str)
            or type(payload["allow_legacy_isolated_snapshot"]) is not bool):
        raise DataError("invalid registered Train audit scope")
    replay = project_registered_train_replay(
        config, window, marker_sha256=payload["marker_sha256"], label=payload["label"],
        instrument=payload["instrument"],
        allow_legacy_isolated_snapshot=payload["allow_legacy_isolated_snapshot"],
    )
    return payload, replay


def _validate_registered_b0_fold(fold: FoldManifest) -> None:
    train_days = (fold.train.score_end_ns - fold.train.score_start_ns) // DAY_NS
    validation_ns = fold.validation.score_end_ns - fold.validation.score_start_ns
    test_ns = fold.test.score_end_ns - fold.test.score_start_ns
    boundaries = (fold.train.warmup_start_ns, fold.train.score_start_ns,
                  fold.train.score_end_ns, fold.validation.warmup_start_ns,
                  fold.validation.score_start_ns, fold.validation.score_end_ns,
                  fold.test.warmup_start_ns, fold.test.score_start_ns, fold.test.score_end_ns)
    if (train_days not in {60, 360}
            or fold.train.score_end_ns - fold.train.score_start_ns != train_days * DAY_NS
            or validation_ns != 15 * DAY_NS or test_ns != 15 * DAY_NS
            or fold.validation.score_start_ns - fold.train.score_end_ns != DAY_NS
            or fold.test.score_start_ns - fold.validation.score_end_ns != DAY_NS
            or any(timestamp % MINUTE_NS for timestamp in boundaries)):
        raise DataError("registered B0 Fold must follow the approved 60/15/15 or 360/15/15 protocol")


def _routeb_code_identity(root: Path | None = None) -> dict[str, str]:
    root = (Path(__file__).resolve().parents[4] if root is None else Path(root)).resolve()
    snapshot_marker: Path | None = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        # A copied Route B workspace may be intentionally materialized without
        # Git objects. In that case accept only the explicit base-commit
        # manifest, and bind that manifest into the source fingerprint below.
        snapshot_marker = root / "docs" / "world_model" / "REPO_BASELINE.json"
        try:
            marker = json.loads(snapshot_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise DataError("cannot freeze Route B repository commit before market access") from exc
        if (not isinstance(marker, dict)
                or set(marker) != {"schema_version", "repository_commit"}
                or marker.get("schema_version") != "rles.routeb.repository_base.v0"
                or not isinstance(marker.get("repository_commit"), str)):
            raise DataError("invalid Route B repository base-commit manifest")
        commit = marker["repository_commit"]
    if (len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise DataError("invalid Route B repository commit")
    fixed = [
        root / "pyproject.toml", root / "uv.lock",
        root / "packages" / "data" / "pyproject.toml",
        root / "packages" / "research" / "pyproject.toml",
        root / "packages" / "orchestrator" / "pyproject.toml",
        root / "packages" / "world_model" / "pyproject.toml",
        root / "packages" / "data" / "src" / "rles_data" / "core.py",
        root / "packages" / "data" / "src" / "rles_data" / "io.py",
        root / "packages" / "research" / "src" / "rles_research" / "access.py",
        root / "packages" / "orchestrator" / "src" / "rles_orchestrator" / "research_access.py",
    ]
    if snapshot_marker is not None:
        fixed.append(snapshot_marker)
    model_dir = root / "packages" / "world_model" / "src" / "rles_world_model"
    source_files = sorted([*fixed, *model_dir.rglob("*.py")], key=lambda path: path.as_posix())
    fingerprint = sha256()
    fingerprint.update(commit.encode("ascii"))
    for source in source_files:
        if not source.is_file():
            raise DataError(f"Route B source identity file missing: {source}")
        relative = source.relative_to(root).as_posix().encode("utf-8")
        fingerprint.update(len(relative).to_bytes(4, "big"))
        fingerprint.update(relative)
        content = source.read_bytes()
        fingerprint.update(len(content).to_bytes(8, "big"))
        fingerprint.update(content)
    return {"repository_commit": commit, "source_code_sha256": fingerprint.hexdigest()}


def _registered_fold_baseline(path: str, code_identity: dict[str, str]
                               ) -> tuple[dict, FoldManifest, MarketReplay]:
    payload = _read_json(path)
    expected = {"schema_version", "config", "fold", "marker_sha256", "label",
                "instrument", "allow_legacy_isolated_snapshot"}
    if (set(payload) != expected
            or payload["schema_version"] != "rles.world_model.registered_fold_baseline.v0"):
        raise DataError("registered Fold config does not match the supported B0 schema")
    config = payload["config"]
    if (not isinstance(config, dict) or not isinstance(config.get("environment"), dict)
            or any(config.get(name) is not None for name in ("evaluation", "test", "holdout"))):
        raise DataError("registered B0 Fold config must be market-only and contain no formal scope")
    if (not isinstance(payload["marker_sha256"], str)
            or not isinstance(payload["label"], str)
            or not payload["label"].startswith("routeb-fold-")
            or not isinstance(payload["instrument"], str)
            or type(payload["allow_legacy_isolated_snapshot"]) is not bool):
        raise DataError("invalid registered B0 Fold identity")
    fold = _fold(payload["fold"])
    _validate_registered_b0_fold(fold)
    replay = project_registered_fold_replay(
        config, fold, marker_sha256=payload["marker_sha256"],
        repository_commit=code_identity["repository_commit"],
        source_code_sha256=code_identity["source_code_sha256"],
        label=payload["label"], instrument=payload["instrument"],
        allow_legacy_isolated_snapshot=payload["allow_legacy_isolated_snapshot"],
    )
    return payload, fold, replay


def _preflight_report_dir(value: str) -> Path:
    target = Path(value).resolve()
    if target.exists():
        raise DataError("B0 report directory already exists; fresh Fold evidence cannot be overwritten")
    if not target.parent.is_dir():
        raise DataError("B0 report directory parent must already exist")
    return target


def _preflight_train_npz(value: str) -> Path:
    target = Path(value).resolve()
    if target.suffix.lower() != ".npz":
        raise DataError("Train artifact output must end in .npz")
    if target.exists() or target.with_suffix(".json").exists():
        raise DataError("Train artifact or metadata already exists; use a fresh output path")
    if not target.parent.is_dir():
        raise DataError("Train artifact output parent must already exist")
    return target


def _preflight_training_report(value: str) -> Path:
    target = Path(value).resolve()
    if target.suffix.lower() != ".json":
        raise DataError("training report output must end in .json")
    if target.exists():
        raise DataError("training report already exists; use a fresh output path")
    if not target.parent.is_dir():
        raise DataError("training report parent must already exist")
    return target


def _write_b0_reports(report_dir: Path, data_audit: dict, baselines: dict) -> None:
    report_dir.mkdir(exist_ok=False)
    for name, report in (("B0_data_audit.json", data_audit), ("baselines.json", baselines)):
        path = report_dir / name
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            stream.write("\n")


def _fold(value: dict) -> FoldManifest:
    if set(value) != {"schema_version", "dataset_hash", "feature_hash", "train", "validation",
                      "test", "purge_ns", "embargo_ns", "scaler_hash"}:
        raise DataError("fold JSON does not match the supported FoldManifest schema")
    return FoldManifest(value["schema_version"], value["dataset_hash"], value["feature_hash"],
                        Window(**value["train"]), Window(**value["validation"]),
                        Window(**value["test"]), value["purge_ns"], value["embargo_ns"],
                        value["scaler_hash"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rles-world-model")
    commands = parser.add_subparsers(dest="command", required=True)
    audit_command = commands.add_parser("audit-data")
    audit_source = audit_command.add_mutually_exclusive_group(required=True)
    audit_source.add_argument("--registered-train-config",
                              help="config that reserves an exact Train interval before reading curated rows")
    audit_source.add_argument("--synthetic-replay-json",
                              help="synthetic_test_only JSON for mechanism tests; never real research evidence")
    baseline = commands.add_parser("baselines")
    baseline_source = baseline.add_mutually_exclusive_group(required=True)
    baseline_source.add_argument("--synthetic-replay-json",
                                 help="synthetic_test_only JSON for mechanism tests; never real research evidence")
    baseline_source.add_argument("--registered-fold-config",
                                 help="one fresh, frozen Fold; reserve Train/Validation before reading; never reads Test")
    baseline.add_argument("--fold-json", help="FoldManifest JSON required with --synthetic-replay-json")
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--ensemble-size", type=int, default=128)
    baseline.add_argument("--report-dir", help="new directory for paired B0_data_audit.json and baselines.json")
    upstream = commands.add_parser("inspect-upstream")
    upstream.add_argument("--checkout", required=True)
    prepare = commands.add_parser("prepare-fold-train")
    prepare.add_argument("--registered-fold-config", required=True,
                         help="fresh registered Fold config; reserves Train/Validation before decoding")
    prepare.add_argument("--output-npz", required=True,
                         help="new Train-only sequence artifact path")
    prepare.add_argument("--length", type=int, default=128)
    prepare.add_argument("--burn-in", type=int, default=32)
    prepare.add_argument("--stride", type=int, default=64)
    train = commands.add_parser("train-prepared")
    train.add_argument("--candidate", choices=("e1", "e2"), required=True)
    train.add_argument("--sequences", required=True,
                       help="previously prepared Train-only NPZ from an authorized read")
    train.add_argument("--scope", required=True)
    train.add_argument("--fold-sha256", required=True,
                       help="expected frozen Fold hash recorded in the Train NPZ metadata")
    train.add_argument("--configuration", required=True,
                       help="candidate model/training JSON; use matched training budgets")
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--worker-python", required=True,
                       help="Python 3.12 executable with the pinned JAX environment")
    train.add_argument("--upstream-checkout", required=True)
    train.add_argument("--report-json", required=True)
    train.add_argument("--timeout-seconds", type=int, default=1800)
    train.add_argument("--resume", action="store_true")
    experiment = commands.add_parser("run-registered-fold")
    experiment.add_argument("--registered-fold-config", required=True,
                            help="one fresh Fold reservation; this command performs the only Train/Validation read")
    experiment.add_argument("--e1-configuration", required=True)
    experiment.add_argument("--e2-configuration", required=True)
    experiment.add_argument("--worker-python", required=True)
    experiment.add_argument("--upstream-checkout", required=True)
    experiment.add_argument("--output-dir", required=True,
                            help="new directory for frozen artifacts and the final E0/E1/E2 report")
    experiment.add_argument("--timeout-seconds", type=int, default=1800)
    experiment.add_argument("--forecast-seed", type=int, default=0)
    experiment.add_argument("--forecast-paths", type=int, default=128)
    experiment.add_argument("--horizons", default="1,5,15")
    experiment.add_argument("--anchor-interval-minutes", type=int, default=15)
    experiment.add_argument("--context-lookback", type=int, default=128)
    experiment.add_argument("--volatility-window", type=int, default=15)
    experiment.add_argument("--block-length", type=int, default=8)
    experiment.add_argument("--bootstrap-replicates", type=int, default=2000)
    experiment.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit-data":
            if args.registered_train_config:
                _, replay = _registered_train_replay(args.registered_train_config)
                result = asdict(audit_replay(replay))
                result["evidence_scope"] = "registered_train"
            else:
                _, replay = _replay(args.synthetic_replay_json)
                result = asdict(audit_replay(replay))
                result["evidence_scope"] = "synthetic_test_only"
        elif args.command == "baselines":
            if args.registered_fold_config:
                if args.fold_json:
                    raise DataError("--fold-json is only valid with --synthetic-replay-json")
                if not args.report_dir:
                    raise DataError("--report-dir is required with --registered-fold-config")
                report_dir = _preflight_report_dir(args.report_dir)
                code_identity = _routeb_code_identity()
                payload, fold, replay = _registered_fold_baseline(
                    args.registered_fold_config, code_identity,
                )
                audit_summary = asdict(audit_replay(replay))
                data_audit = {
                    "schema_version": "rles.world_model.b0_audit.v0",
                    "status": "PASS" if audit_summary["market_model_eligible"] else "BLOCKED",
                    "scope": "registered_fresh_fold_train_validation",
                    **audit_summary,
                    "fold_hash": fold.hash,
                    "fold_manifest": asdict(fold),
                    **code_identity,
                    "availability_policy": "closed_1m_bar;available_at_ns<=decision_ts_ns",
                    "reservation_label": payload["label"],
                    "marker_sha256": payload["marker_sha256"],
                    "read_scope": {
                        "start_ns": fold.train.warmup_start_ns,
                        "end_ns": fold.validation.score_end_ns,
                        "includes_test": False,
                    },
                    "test_time_coverage_preflight": {
                        "required_end_ns": fold.test.score_end_ns,
                        "source": "0.4.1_row_group_manifest_and_parquet_footer",
                        "test_payload_decoded": False,
                    },
                }
                scores = []
                reason = None
                if audit_summary["market_model_eligible"]:
                    try:
                        scores = [asdict(score) for score in score_baselines(
                            replay, fold, seed=args.seed, ensemble_size=args.ensemble_size,
                        )]
                    except DataError as exc:
                        reason = str(exc)
                else:
                    reason = "market data audit failed; S0/S1/S2 were not scored"
                baseline_status = "COMPLETED" if scores else "BLOCKED"
                baselines = {
                    "schema_version": "rles.world_model.b0_baselines.v0",
                    "status": baseline_status,
                    "scope": "registered_fresh_fold_train_validation",
                    "dataset_hash": replay.dataset_hash,
                    "fold_hash": fold.hash,
                    "fold_manifest": asdict(fold),
                    **code_identity,
                    "reservation_label": payload["label"],
                    "marker_sha256": payload["marker_sha256"],
                    "read_scope": data_audit["read_scope"],
                    "availability_policy": data_audit["availability_policy"],
                    "models": ["S0_ZERO_DRIFT", "S1_CONDITIONAL_VOL", "S2_BLOCK_BOOTSTRAP"],
                    "primary_score": "one_step_log_return_crps",
                    "seed": args.seed,
                    "ensemble_size": args.ensemble_size,
                    "observations": scores[0]["observations"] if scores else None,
                    "scores": scores,
                    "reason": reason,
                    "limitations": [
                        "S2 is evaluated only on the one-step marginal in B0; dependence preservation requires multi-step B3 scoring.",
                        "No baseline ranking, statistical significance, or economic conclusion is available.",
                    ],
                }
                result = {
                    "status": baseline_status,
                    "evidence_scope": "registered_fresh_fold_train_validation",
                    "dataset_hash": replay.dataset_hash,
                    "fold_hash": fold.hash,
                    "fold_manifest": asdict(fold),
                    **code_identity,
                    "availability_policy": data_audit["availability_policy"],
                    "reservation_label": payload["label"],
                    "marker_sha256": payload["marker_sha256"],
                    "read_scope": data_audit["read_scope"],
                    "data_audit": data_audit,
                    "baselines": baselines,
                }
                _write_b0_reports(report_dir, data_audit, baselines)
            else:
                if not args.fold_json:
                    raise DataError("--fold-json is required with --synthetic-replay-json")
                if args.report_dir:
                    raise DataError("--report-dir is reserved for registered fresh-fold evidence")
                _, replay = _replay(args.synthetic_replay_json)
                fold = _fold(_read_json(args.fold_json))
                result = {"status": "COMPLETED", "evidence_scope": "synthetic_test_only",
                          "dataset_hash": replay.dataset_hash, "fold_hash": fold.hash,
                          "scores": [asdict(score) for score in
                                     score_baselines(replay, fold, seed=args.seed,
                                                     ensemble_size=args.ensemble_size)]}
        elif args.command == "prepare-fold-train":
            target = _preflight_train_npz(args.output_npz)
            code_identity = _routeb_code_identity()
            payload, fold, replay = _registered_fold_baseline(
                args.registered_fold_config, code_identity,
            )
            artifact, scaler, summary = prepare_fold_train_sequences(
                fold_replay=replay, fold=fold, train_scope_id=payload["label"],
                target_npz=target, length=args.length, burn_in=args.burn_in,
                stride=args.stride,
            )
            metadata = _read_json(str(artifact.with_suffix(".json")))
            result = {
                "status": "PREPARED",
                "evidence_scope": "registered_fresh_fold_train_only",
                "dataset_hash": fold.dataset_hash,
                "fold_hash": fold.hash,
                "reservation_label": payload["label"],
                "marker_sha256": payload["marker_sha256"],
                "artifact_npz": str(artifact),
                "artifact_sha256": metadata["artifact_sha256"],
                "scaler_sha256": scaler.fingerprint,
                **code_identity,
                **summary,
            }
        elif args.command == "run-registered-fold":
            from dataclasses import asdict as dataclass_asdict
            from .routeb.jax_gru import GRUConfig
            from .routeb.jax_train import TrainingConfig
            e1_configuration = _read_json(args.e1_configuration)
            e2_configuration = _read_json(args.e2_configuration)
            if (set(e1_configuration) != {"model", "training"}
                    or set(e2_configuration) != {"model", "training"}
                    or not isinstance(e1_configuration["model"], dict)
                    or not isinstance(e2_configuration["model"], dict)
                    or not isinstance(e1_configuration["training"], dict)
                    or not isinstance(e2_configuration["training"], dict)):
                raise DataError("E1/E2 configurations must each contain exactly model and training objects")
            try:
                e1_configuration = {
                    "model": dataclass_asdict(GRUConfig(**e1_configuration["model"])),
                    "training": dataclass_asdict(TrainingConfig(**e1_configuration["training"])),
                }
                from .routeb.jax_model import RSSMConfig
                e2_configuration = {
                    "model": dataclass_asdict(RSSMConfig(**e2_configuration["model"])),
                    "training": dataclass_asdict(TrainingConfig(**e2_configuration["training"])),
                }
            except (TypeError, ValueError) as exc:
                raise DataError(f"invalid candidate model or training configuration: {exc}") from exc
            if e1_configuration["training"] != e2_configuration["training"]:
                raise DataError("E1 and E2 must use an identical training configuration")
            try:
                horizons = tuple(int(part.strip()) for part in args.horizons.split(","))
            except ValueError as exc:
                raise DataError("--horizons must be a comma-separated sequence of integers") from exc
            if (not horizons or tuple(sorted(set(horizons))) != horizons
                    or any(value < 1 or value > 60 for value in horizons)
                    or type(args.timeout_seconds) is not int or args.timeout_seconds < 1
                    or type(args.forecast_seed) is not int or args.forecast_seed < 0
                    or type(args.forecast_paths) is not int or not 2 <= args.forecast_paths <= 1024):
                raise DataError("invalid forecast horizon, worker timeout, seed, or ensemble size")
            worker_python = Path(args.worker_python).absolute()
            upstream_checkout = Path(args.upstream_checkout).resolve()
            if not worker_python.is_file() or not upstream_checkout.is_dir():
                raise DataError("pinned Python 3.12 worker and upstream checkout are required")
            output_dir = _preflight_report_dir(args.output_dir)
            code_identity = _routeb_code_identity()
            output_dir.mkdir(exist_ok=False)
            receipt = None
            try:
                receipt, fold, replay = _registered_fold_baseline(
                    args.registered_fold_config, code_identity,
                )
                with (output_dir / "registration_receipt.json").open(
                        "x", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps({
                        "schema": "routeb.registered-fold-receipt.v1",
                        "reservation_label": receipt["label"],
                        "marker_sha256": receipt["marker_sha256"],
                        "dataset_hash": fold.dataset_hash, "fold_hash": fold.hash,
                        **code_identity,
                        "read_scope": {"start_ns": fold.train.warmup_start_ns,
                                       "end_ns": fold.validation.score_end_ns,
                                       "includes_test": False},
                    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                result = run_fold_experiment(
                    replay=replay, fold=fold, train_scope_id=receipt["label"],
                    candidate_configurations={"E1": e1_configuration,
                                               "E2": e2_configuration},
                    worker=WorkerClient(WorkerConfig(
                        worker_python, upstream_checkout,
                        timeout_seconds=args.timeout_seconds)),
                    output_dir=output_dir, repository_commit=code_identity["repository_commit"],
                    source_code_sha256=code_identity["source_code_sha256"],
                    marker_sha256=receipt["marker_sha256"],
                    reservation_label=receipt["label"],
                    forecast_seed=args.forecast_seed, forecast_paths=args.forecast_paths,
                    horizons=horizons, anchor_interval_minutes=args.anchor_interval_minutes,
                    context_lookback=args.context_lookback,
                    volatility_window=args.volatility_window, block_length=args.block_length,
                    bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
                )
            except Exception as exc:
                failure = {"schema": "routeb.fold-experiment-failure.v1",
                           "status": "BLOCKED", "error": type(exc).__name__,
                           "detail": str(exc), "registered_read_started": receipt is not None,
                           **code_identity}
                failure_path = output_dir / "run_failure.json"
                if not failure_path.exists():
                    failure_path.write_text(json.dumps(
                        failure, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
                raise
        elif args.command == "train-prepared":
            sequences = Path(args.sequences).resolve()
            configuration = Path(args.configuration).resolve()
            checkpoint = Path(args.checkpoint).resolve()
            worker_python = Path(args.worker_python).absolute()
            upstream_checkout = Path(args.upstream_checkout).resolve()
            report_path = _preflight_training_report(args.report_json)
            if (not sequences.is_file() or not sequences.with_suffix(".json").is_file()
                    or not configuration.is_file() or not worker_python.is_file()
                    or not upstream_checkout.is_dir()):
                raise DataError("Train NPZ/metadata, configuration, worker Python, and upstream checkout are required")
            if (type(args.timeout_seconds) is not int or args.timeout_seconds < 1
                    or len(args.fold_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in args.fold_sha256)):
                raise DataError("invalid timeout or Fold SHA-256")
            train_batch, train_metadata = load_worker_batch(
                sequences, expected_scope=args.scope, snapshot_sha=SNAPSHOT_SHA,
                upstream_sha=PINNED_UPSTREAM,
            )
            if train_metadata.get("fold_sha256") != args.fold_sha256:
                raise DataError("prepared Train artifact does not match the expected Fold hash")
            config_payload = _read_json(str(configuration))
            if set(config_payload) != {"model", "training"}:
                raise DataError("candidate configuration must contain exactly model and training")
            if args.resume:
                if not checkpoint.is_file() or not checkpoint.with_suffix(".json").is_file():
                    raise DataError("resume requires an existing complete checkpoint pair")
            elif checkpoint.exists() or checkpoint.with_suffix(".json").exists():
                raise DataError("checkpoint already exists; pass --resume only to continue its bound run")
            client = WorkerClient(WorkerConfig(
                worker_python, upstream_checkout, timeout_seconds=args.timeout_seconds,
            ))
            run = (client.train_e1 if args.candidate == "e1" else client.train)(
                sequences=sequences, scope=args.scope, checkpoint=checkpoint,
                configuration=configuration, resume=args.resume,
            )
            training_reports = run.get("reports")
            if not isinstance(training_reports, list) or not training_reports:
                raise DataError("worker omitted final training diagnostics")
            final_diagnostics = training_reports[-1]
            elapsed = final_diagnostics.get("training_seconds")
            if (type(final_diagnostics.get("parameter_count")) is not int
                    or final_diagnostics["parameter_count"] < 1
                    or type(final_diagnostics.get("train_examples")) is not int
                    or final_diagnostics["train_examples"] != train_batch.features.shape[0]
                    or type(final_diagnostics.get("optimizer_updates")) is not int
                    or final_diagnostics.get("optimizer_updates") != config_payload["training"].get("steps")
                    or not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
                    or not math.isfinite(elapsed) or elapsed < 0
                    or not isinstance(final_diagnostics.get("device"), str)
                    or not final_diagnostics["device"].strip()):
                raise DataError("worker training diagnostics are incomplete or inconsistent")
            result = {
                "schema": "routeb.train-run.v1",
                "status": "COMPLETED",
                "candidate": args.candidate,
                "dataset_hash": train_metadata["dataset_id"],
                "fold_hash": args.fold_sha256,
                "train_scope_id": args.scope,
                "train_npz_sha256": train_metadata["artifact_sha256"],
                "train_examples": int(train_batch.features.shape[0]),
                "window_length": int(train_batch.features.shape[1]),
                "training_config": config_payload["training"],
                "optimizer_updates": final_diagnostics["optimizer_updates"],
                "parameter_count": final_diagnostics["parameter_count"],
                "training_seconds": elapsed,
                "device": final_diagnostics["device"],
                "configuration_sha256": sha256(configuration.read_bytes()).hexdigest(),
                "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                "snapshot_sha256": SNAPSHOT_SHA,
                "upstream_sha256": PINNED_UPSTREAM,
                "resume": args.resume,
                "worker_result": run,
            }
            with report_path.open("x", encoding="utf-8", newline="\n") as report_file:
                report_file.write(json.dumps(result, ensure_ascii=False, indent=2,
                                             allow_nan=False))
                report_file.write("\n")
        else:
            result = asdict(inspect_upstream(args.checkout))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result.get("status", "READY") not in {"BLOCKED", "FAILED"} else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__, "detail": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2
