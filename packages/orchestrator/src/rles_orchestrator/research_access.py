"""Resolve the shared project ledger; never derive it from an experiment output."""
from contextvars import ContextVar
from hashlib import sha256
import json
from pathlib import Path
from time import time_ns
from typing import Any, Callable
import uuid

from rles_data.core import (CuratedStore, DataError, FoldManifest, MINUTE_NS, digest,
                            require_hash, require_ns)
from rles_data.io import curated_event_time_bounds, curated_source_identity, read_curated_window
from rles_research.access import ProjectResearchRegistry


ACTIVE_OWNER: ContextVar[str | None] = ContextVar("research_owner", default=None)
DATA_FAMILY = "perpetual-market-v1"


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise json.JSONDecodeError(f"duplicate JSON object key: {key!r}", "", 0)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise json.JSONDecodeError(f"non-standard JSON constant: {value}", "", 0)


def _strict_inventory_json(content: bytes) -> dict[str, Any]:
    try:
        inventory = json.loads(content, object_pairs_hook=_json_object_without_duplicate_keys,
                               parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise DataError(
            f"legacy research inventory JSON is invalid or ambiguous: {exc.msg}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataError("legacy research inventory JSON is not valid UTF-8") from exc
    if (not isinstance(inventory, dict)
            or inventory.get("schema") != "rles.legacy-research-access.v1"
            or not isinstance(inventory.get("entries"), list)
            or not isinstance(inventory.get("unresolved"), list)):
        raise DataError("legacy research inventory has an unsupported schema")
    entry_fields = {
        "key", "market", "family", "start_ns", "end_ns", "purpose", "identity",
    }
    if any(
        not isinstance(entry, dict) or set(entry) != entry_fields
        or any(not isinstance(entry.get(field), str) or not entry[field]
               for field in ("key", "market", "family"))
        or entry["key"] != entry["key"].strip()
        or entry["market"] != entry["market"].strip().upper()
        or entry["family"] != entry["family"].strip().lower()
        or type(entry.get("start_ns")) is not int
        or type(entry.get("end_ns")) is not int
        or not 0 < entry["start_ns"] < entry["end_ns"]
        or not isinstance(entry.get("purpose"), str)
        or entry["purpose"] not in {"development", "test", "holdout"}
        or not isinstance(entry.get("identity"), dict)
        for entry in inventory["entries"]
    ):
        raise DataError("legacy research inventory contains an invalid access entry")
    access_keys = [entry["key"] for entry in inventory["entries"]]
    if len(access_keys) != len(set(access_keys)):
        raise DataError("legacy research inventory contains duplicate access keys")
    return inventory


def project_registry() -> ProjectResearchRegistry:
    # Source checkout identity is stable even if the CLI's cwd/output changes.
    source = Path(__file__).resolve().parents[4]
    candidates = (source, *Path.cwd().resolve().parents, Path.cwd().resolve())
    for candidate in candidates:
        metadata = candidate / "pyproject.toml"
        if metadata.is_file() and "[tool.uv.workspace]" in metadata.read_text(encoding="utf-8"):
            registry = ProjectResearchRegistry(candidate / ".artifacts" / "research-access.sqlite")
            inventory_path = candidate / "docs" / "audit_evidence" / "20261016" / "legacy_research_access_v2.json"
            if not inventory_path.is_file():
                raise DataError("legacy research inventory missing; restore the tracked inventory before research")
            content = inventory_path.read_bytes()
            inventory_sha = sha256(content).hexdigest()
            marker = inventory_path.with_suffix(".json.sha256")
            if not marker.is_file() or marker.read_text(encoding="ascii") != inventory_sha:
                raise DataError("legacy research inventory integrity mismatch")
            inventory = _strict_inventory_json(content)
            if inventory["unresolved"]:
                raise DataError("legacy research inventory contains unresolved data access")
            registry.import_inventory(inventory["entries"], inventory_sha)
            return registry
    raise DataError("project research registry unavailable; run from a workspace checkout")


def market_key(instrument: str) -> str:
    return instrument.split(".")[0].removesuffix("-PERP").upper()


def owner_for(config: dict) -> str:
    return ACTIVE_OWNER.get() or digest([str(Path(config["output"]).resolve()), "standalone"])


def record_development(config: dict, window: dict, *, label: str,
                       source_identity: dict | None = None) -> None:
    owner = owner_for(config)
    identity = {"window": window, "environment": config["environment"], "label": label}
    if source_identity is not None:
        required = {"schema_version", "snapshot_hash", "file_sha256", "marker_sha256", "events"}
        if (not isinstance(source_identity, dict) or set(source_identity) != required
                or not isinstance(source_identity["schema_version"], str)
                or source_identity["schema_version"] not in {"0.4.0", "0.4.1"}
                or type(source_identity["events"]) is not int or source_identity["events"] <= 0):
            raise DataError("invalid curated source identity for development reservation")
        for name in ("snapshot_hash", "file_sha256", "marker_sha256"):
            require_hash(source_identity[name])
        identity["source_identity"] = dict(source_identity)
    registry = project_registry()
    registry.reserve(
        # Keep the reservation key stable when an existing owner retries the
        # same named scope. The immutable identity comparison must see a changed
        # source/config and reject it; hashing the identity into the key would
        # let that retry create a second development reservation.
        key=digest([owner, label, window, config["environment"]]), owner=owner,
        market=market_key(config["environment"]["instrument"]), family=DATA_FAMILY,
        start_ns=window["start_ns"], end_ns=window["end_ns"], purpose="development",
        identity=identity,
    )


def _routeb_fresh_fold_access_record(config: dict, window: dict, *, label: str,
                                     fold_sha256: str, repository_commit: str,
                                     source_code_sha256: str,
                                     source_identity: dict) -> tuple[str, str, dict]:
    environment = config.get("environment")
    if (not isinstance(environment, dict)
            or not isinstance(environment.get("market_path"), str)
            or not isinstance(environment.get("instrument"), str)
            or not isinstance(window, dict) or set(window) != {"start_ns", "end_ns"}
            or type(window.get("start_ns")) is not int
            or type(window.get("end_ns")) is not int
            or window["start_ns"] >= window["end_ns"]
            or not isinstance(label, str) or not label.startswith("routeb-fold-")
            or not isinstance(source_identity, dict)):
        raise DataError("invalid fresh Route B fold scope")
    fold_sha256 = require_hash(fold_sha256)
    source_code_sha256 = require_hash(source_code_sha256)
    if (not isinstance(repository_commit, str) or len(repository_commit) != 40
            or any(character not in "0123456789abcdef" for character in repository_commit)):
        raise DataError("invalid frozen repository commit identity")
    required = {"schema_version", "snapshot_hash", "file_sha256", "marker_sha256", "events"}
    if (set(source_identity) != required
            or source_identity.get("schema_version") != "0.4.1"
            or type(source_identity.get("events")) is not int
            or source_identity["events"] <= 0):
        raise DataError("invalid curated source identity for fresh Route B fold")
    for name in ("snapshot_hash", "file_sha256", "marker_sha256"):
        require_hash(source_identity[name])
    environment_identity = {
        **environment,
        "curated_marker_sha256": source_identity["marker_sha256"],
        "curated_read_mode": "whole_snapshot_isolated_v1" if config.get(
            "allow_legacy_isolated_snapshot", False) else "row_group_scoped_v1",
    }
    identity = {
        "protocol": "routeb-world-model-b0-fresh-fold-v0",
        "availability_policy": "closed_1m_bar;available_at_ns<=decision_ts_ns",
        "fold_sha256": fold_sha256,
        "repository_commit": repository_commit,
        "source_code_sha256": source_code_sha256,
        "window": dict(window),
        "environment": environment_identity,
        "label": label,
        "source_identity": dict(source_identity),
    }
    owner = digest(["routeb-world-model-b0", market_key(environment["instrument"]), fold_sha256])
    key = digest([owner, label, window, source_identity])
    return key, owner, identity


def reserve_routeb_fresh_fold(config: dict, window: dict, *, label: str,
                              fold_sha256: str, repository_commit: str,
                              source_code_sha256: str, source_identity: dict) -> dict:
    """Freeze a fresh Route B fold scope before any curated payload is decoded."""
    key, owner, identity = _routeb_fresh_fold_access_record(
        config, window, label=label, fold_sha256=fold_sha256,
        repository_commit=repository_commit, source_code_sha256=source_code_sha256,
        source_identity=source_identity,
    )
    token = project_registry().reserve_fresh_development(
        key=key, owner=owner,
        market=market_key(identity["environment"]["instrument"]),
        family=DATA_FAMILY, start_ns=window["start_ns"], end_ns=window["end_ns"],
        identity=identity,
    )
    return {"access_key": key, "token": token, "identity": identity}


def routeb_fresh_fold_access_key(config: dict, window: dict, *, label: str,
                                 fold_sha256: str, repository_commit: str,
                                 source_code_sha256: str,
                                 source_identity: dict) -> str:
    key, _, _ = _routeb_fresh_fold_access_record(
        config, window, label=label, fold_sha256=fold_sha256,
        repository_commit=repository_commit, source_code_sha256=source_code_sha256,
        source_identity=source_identity,
    )
    return key


def routeb_fresh_fold_access_identity_sha256(
        config: dict, window: dict, *, label: str, fold_sha256: str,
        repository_commit: str, source_code_sha256: str,
        source_identity: dict) -> str:
    """Return the canonical digest of the complete pre-read ledger identity."""
    _, _, identity = _routeb_fresh_fold_access_record(
        config, window, label=label, fold_sha256=fold_sha256,
        repository_commit=repository_commit, source_code_sha256=source_code_sha256,
        source_identity=source_identity,
    )
    return digest(identity)


def complete_routeb_fresh_fold(config: dict, window: dict, *, label: str,
                               fold_sha256: str, repository_commit: str,
                               source_code_sha256: str, source_identity: dict,
                               report_sha256: str, detail: dict | None = None) -> dict:
    """Bind a completed aggregate report to its exact fresh-fold access."""
    key, _, identity = _routeb_fresh_fold_access_record(
        config, window, label=label, fold_sha256=fold_sha256,
        repository_commit=repository_commit, source_code_sha256=source_code_sha256,
        source_identity=source_identity,
    )
    project_registry().record_development_result(
        key=key, identity=identity, status="completed",
        report_sha=report_sha256, detail=detail,
    )
    return {
        "schema": "routeb.development-result-receipt.v1",
        "access_key": key,
        "status": "completed",
        "report_sha256": report_sha256,
    }


def fail_routeb_fresh_fold(config: dict, window: dict, *, label: str,
                            fold_sha256: str, repository_commit: str,
                            source_code_sha256: str, source_identity: dict,
                            report_sha256: str, detail: dict | None = None) -> dict:
    """Bind a terminal failed-run receipt to its exact consumed fresh Fold."""
    key, _, identity = _routeb_fresh_fold_access_record(
        config, window, label=label, fold_sha256=fold_sha256,
        repository_commit=repository_commit, source_code_sha256=source_code_sha256,
        source_identity=source_identity,
    )
    project_registry().record_development_result(
        key=key, identity=identity, status="failed",
        report_sha=report_sha256, detail=detail,
    )
    return {"schema": "routeb.development-result-receipt.v1",
            "access_key": key, "status": "failed", "report_sha256": report_sha256}


def register_routeb_candidate_attempt(*, label: str, candidate_id: str,
                                      configuration: dict, fold_sha256: str,
                                      feature_sha256: str, seed: int) -> str:
    if (not isinstance(label, str) or not label.startswith("routeb-fold-")
            or candidate_id not in {"E1", "E2"}
            or not isinstance(configuration, dict)):
        raise ValueError("invalid Route B candidate attempt")
    require_hash(fold_sha256)
    require_hash(feature_sha256)
    protocol_id = f"routeb-r2:{label}"
    recipe = {
        "candidate": candidate_id,
        "configuration": configuration,
        "fold_sha256": fold_sha256,
    }
    attempt_key = digest([
        "routeb-world-model-candidate-v1", protocol_id, candidate_id,
        seed, feature_sha256, recipe,
    ])
    project_registry().register_trial_attempt(
        attempt_key=attempt_key, protocol_id=protocol_id,
        candidate_id=candidate_id, recipe=recipe,
        feature_sha=feature_sha256, seed=seed,
    )
    return attempt_key


def register_routeb_r3_candidate_attempt(*, label: str, candidate_id: str,
                                         configuration: dict, fold_sha256: str,
                                         feature_sha256: str, seed: int) -> str:
    """Append one frozen R3 regularizer arm/seed to the RouteB trial ledger."""
    allowed = {"R3_LAMBDA_000", "R3_LAMBDA_010", "R3_LAMBDA_050"}
    if (not isinstance(label, str) or not label.startswith("routeb-fold-")
            or candidate_id not in allowed or not isinstance(configuration, dict)):
        raise ValueError("invalid Route B R3 candidate attempt")
    require_hash(fold_sha256)
    require_hash(feature_sha256)
    protocol_id = f"routeb-r3:{label}"
    recipe = {"candidate": candidate_id, "configuration": configuration,
              "fold_sha256": fold_sha256}
    attempt_key = digest([
        "routeb-world-model-r3-candidate-v1", protocol_id, candidate_id,
        seed, feature_sha256, recipe,
    ])
    project_registry().register_trial_attempt(
        attempt_key=attempt_key, protocol_id=protocol_id,
        candidate_id=candidate_id, recipe=recipe,
        feature_sha=feature_sha256, seed=seed,
    )
    return attempt_key


def register_routeb_r4_hierarchy_attempt(*, label: str, candidate_id: str,
                                          configuration: dict, fold_sha256: str,
                                          feature_sha256: str, seed: int) -> str:
    """Append a frozen E3/E3b/E4 hierarchy variant and seed to the trial ledger."""
    allowed = {"R4_E3_FEATURES", "R4_E3B_FUSION", "R4_E4_CONDITIONAL"}
    if (not isinstance(label, str) or not label.startswith("routeb-fold-")
            or candidate_id not in allowed or not isinstance(configuration, dict)):
        raise ValueError("invalid Route B R4 hierarchy candidate")
    require_hash(fold_sha256)
    require_hash(feature_sha256)
    protocol_id = f"routeb-r4:{label}"
    recipe = {"candidate": candidate_id, "configuration": configuration,
              "fold_sha256": fold_sha256}
    attempt_key = digest([
        "routeb-world-model-r4-hierarchy-v1", protocol_id, candidate_id,
        seed, feature_sha256, recipe,
    ])
    project_registry().register_trial_attempt(
        attempt_key=attempt_key, protocol_id=protocol_id,
        candidate_id=candidate_id, recipe=recipe,
        feature_sha=feature_sha256, seed=seed,
    )
    return attempt_key


def register_routeb_r4_baseline_comparison_attempt(
        *, label: str, candidate_id: str, configuration: dict,
        fold_sha256: str, feature_sha256: str, seed: int) -> str:
    """Append one frozen single-scale/hierarchical comparison run to the ledger."""
    allowed = {"R4_E2_SINGLE_SCALE", "R4_E3_HIERARCHY"}
    if (not isinstance(label, str) or not label.startswith("routeb-fold-")
            or candidate_id not in allowed or not isinstance(configuration, dict)
            or type(seed) is not int or seed < 0):
        raise ValueError("invalid Route B R4 baseline-comparison candidate")
    require_hash(fold_sha256)
    require_hash(feature_sha256)
    protocol_id = f"routeb-r4-baseline:{label}"
    recipe = {"candidate": candidate_id, "configuration": configuration,
              "fold_sha256": fold_sha256}
    attempt_key = digest([
        "routeb-world-model-r4-baseline-comparison-v1", protocol_id,
        candidate_id, seed, feature_sha256, recipe,
    ])
    project_registry().register_trial_attempt(
        attempt_key=attempt_key, protocol_id=protocol_id,
        candidate_id=candidate_id, recipe=recipe,
        feature_sha=feature_sha256, seed=seed,
    )
    return attempt_key


def record_routeb_candidate_event(attempt_key: str, *, event_key: str,
                                  status: str, detail: dict) -> None:
    project_registry().record_trial_attempt_event(
        attempt_key, event_key=event_key, status=status, detail=detail,
    )


def read_registered_fold_window(config: dict, window: dict, *, marker_sha256: str,
                                dataset_hash: str, required_start_ns: int, required_end_ns: int,
                                label: str, fold_sha256: str, repository_commit: str,
                                source_code_sha256: str,
                                on_reserved: Callable[[], None] | None = None,
                                allow_legacy_isolated_snapshot: bool = False) -> CuratedStore:
    """Preflight a frozen Route B source, reserve a fresh Fold, then read its rows."""
    environment = config.get("environment")
    if (not isinstance(environment, dict) or not isinstance(environment.get("market_path"), str)
            or any(config.get(name) is not None for name in ("evaluation", "test", "holdout"))
            or type(allow_legacy_isolated_snapshot) is not bool
            or allow_legacy_isolated_snapshot):
        raise DataError("fresh Route B Fold reads require row-group scoped 0.4.1 data and no formal scopes")
    marker_sha256 = require_hash(marker_sha256)
    dataset_hash = require_hash(dataset_hash)
    required_start_ns, required_end_ns = require_ns(required_start_ns), require_ns(required_end_ns)
    if (not isinstance(window, dict) or set(window) != {"start_ns", "end_ns"}
            or type(window.get("start_ns")) is not int or type(window.get("end_ns")) is not int
            or window["start_ns"] != required_start_ns
            or window["end_ns"] > required_end_ns
            or required_start_ns >= required_end_ns):
        raise DataError("invalid frozen Fold source coverage bounds")
    source_identity = curated_source_identity(environment["market_path"])
    if source_identity["marker_sha256"] != marker_sha256:
        raise DataError("curated COMPLETE marker differs from frozen Route B fold config")
    if source_identity["snapshot_hash"] != dataset_hash:
        raise DataError("curated dataset hash differs from frozen Route B fold config")
    bounds = curated_event_time_bounds(
        environment["market_path"], expected_marker_sha256=marker_sha256,
    )
    if (bounds["min_ts_event_ns"] > required_start_ns
            or bounds["max_ts_event_ns"] < required_end_ns - MINUTE_NS):
        raise DataError("curated source time bounds do not cover the complete frozen Fold")
    reserve_routeb_fresh_fold(
        config, window, label=label, fold_sha256=fold_sha256,
        repository_commit=repository_commit, source_code_sha256=source_code_sha256,
        source_identity=source_identity,
    )
    if on_reserved is not None:
        on_reserved()
    return read_curated_window(
        environment["market_path"], window["start_ns"], window["end_ns"],
        expected_marker_sha256=marker_sha256,
        allow_legacy_isolated_snapshot=allow_legacy_isolated_snapshot,
    )


def read_registered_train_window(config: dict, window: dict, *, marker_sha256: str,
                                  label: str,
                                  allow_legacy_isolated_snapshot: bool = False) -> CuratedStore:
    """Reserve a development interval before decoding a scoped curated snapshot."""
    environment = config.get("environment")
    if (not isinstance(environment, dict) or not isinstance(environment.get("market_path"), str)
            or not isinstance(window, dict) or set(window) != {"start_ns", "end_ns"}
            or type(window.get("start_ns")) is not int or type(window.get("end_ns")) is not int
            or window["start_ns"] >= window["end_ns"]
            or not isinstance(label, str) or not label.startswith("train-")
            or type(allow_legacy_isolated_snapshot) is not bool):
        raise DataError("invalid registered Train-inner market window")
    start_ns, end_ns = require_ns(window["start_ns"]), require_ns(window["end_ns"])
    training_start, training_end = environment.get("start_ns"), environment.get("end_ns")
    if (type(training_start) is not int or type(training_end) is not int
            or start_ns > training_start or end_ns < training_end):
        raise DataError("registered read scope must cover the configured Train-inner interval")
    evaluation = config.get("evaluation")
    if isinstance(evaluation, dict):
        eval_start, eval_end = evaluation.get("start_ns"), evaluation.get("end_ns")
        if (type(eval_start) is int and type(eval_end) is int
                and start_ns < eval_end and end_ns > eval_start):
            raise DataError("registered Train-inner read scope overlaps the evaluation interval")
    marker_sha256 = require_hash(marker_sha256)
    scope = {"start_ns": start_ns, "end_ns": end_ns}
    scope_config = {**config, "environment": {
        **environment, "curated_marker_sha256": marker_sha256,
        "curated_read_mode": ("whole_snapshot_isolated_v1"
                              if allow_legacy_isolated_snapshot else "row_group_scoped_v1"),
    }}
    # The reservation includes the exact marker receipt and occurs before the
    # low-level reader opens any row-group payload.
    record_development(scope_config, scope, label=label)
    return read_curated_window(
        environment["market_path"], scope["start_ns"], scope["end_ns"],
        expected_marker_sha256=marker_sha256,
        allow_legacy_isolated_snapshot=allow_legacy_isolated_snapshot,
    )


def read_registered_validation_window(config: dict, window: dict, *, fold: FoldManifest,
                                      marker_sha256: str, label: str) -> CuratedStore:
    """Reserve exact Train+Validation source scope before any payload is decoded.

    This reader is only for a frozen B0 baseline comparison. Its single scope
    covers the Fold's Train warmup through Validation score end, while the
    baseline scorer selects Train, Validation warmup, and Validation score
    segments separately. Test/Holdout configuration is rejected before read.
    """
    environment = config.get("environment")
    evaluation = config.get("evaluation")
    if (not isinstance(environment, dict)
            or not isinstance(environment.get("market_path"), str)
            or not isinstance(environment.get("instrument"), str)
            or not isinstance(fold, FoldManifest)
            or not isinstance(evaluation, dict)
            or set(evaluation) - {"start_ns", "end_ns", "split", "coverage_policy", "mode"}
            or not {"start_ns", "end_ns", "split"} <= set(evaluation)
            or evaluation.get("split") != "validation"
            or type(evaluation.get("start_ns")) is not int
            or type(evaluation.get("end_ns")) is not int
            or type(environment.get("start_ns")) is not int
            or type(environment.get("end_ns")) is not int
            or any(config.get(name) is not None for name in ("test", "holdout"))
            or not isinstance(window, dict) or set(window) != {"start_ns", "end_ns"}
            or type(window.get("start_ns")) is not int
            or type(window.get("end_ns")) is not int
            or not isinstance(label, str) or not label.startswith("validation-world-model-")
            or window["start_ns"] != fold.train.warmup_start_ns
            or window["end_ns"] != fold.validation.score_end_ns
            or evaluation.get("start_ns") != fold.validation.score_start_ns
            or evaluation.get("end_ns") != fold.validation.score_end_ns
            or environment.get("start_ns") != fold.train.score_start_ns
            or environment.get("end_ns") != fold.train.score_end_ns):
        raise DataError("registered Validation scope does not match the frozen Train/Fold boundaries")
    start_ns, end_ns = require_ns(window["start_ns"]), require_ns(window["end_ns"])
    marker_sha256 = require_hash(marker_sha256)
    scope = {"start_ns": start_ns, "end_ns": end_ns}
    scope_config = {**config, "environment": {
        **environment, "curated_marker_sha256": marker_sha256,
        "curated_read_mode": "row_group_scoped_v1",
    }}
    # Development reservations are shared with the existing Validation path;
    # a different owner cannot reopen an already reserved market interval.
    record_development(scope_config, scope, label=label)
    return read_curated_window(
        environment["market_path"], start_ns, end_ns,
        expected_marker_sha256=marker_sha256,
        allow_legacy_isolated_snapshot=False,
    )


def begin_application_trial(*, run_id: str, recipe: dict, feature_sha: str,
                            seed: int, resumed: bool, detail: dict) -> str | None:
    """Register a standalone learner attempt; WFO owns its candidate lifecycle separately."""
    if ACTIVE_OWNER.get() is not None:
        return None
    if (not isinstance(run_id, str) or not run_id or not isinstance(recipe, dict)
            or type(seed) is not int or type(resumed) is not bool
            or not isinstance(detail, dict)):
        raise ValueError("invalid standalone application trial identity")
    attempt_key = digest(["application-training-attempt-v1", run_id])
    registry = project_registry()
    registry.register_trial_attempt(
        attempt_key=attempt_key, protocol_id="standalone-application-training-v1",
        candidate_id=run_id, recipe=recipe, feature_sha=feature_sha, seed=seed,
    )
    if resumed:
        history = registry.trial_attempt_events(attempt_key)
        previous = history[-1] if history else None
        if previous is not None and previous["status"] in {"started", "resumed"}:
            prior_key = previous["event_key"]
            registry.record_trial_attempt_event(
                attempt_key, event_key=f"inferred-interrupted-after-{prior_key}",
                status="interrupted",
                detail={"resolution": "resume_observed_without_terminal_event",
                        "prior_event_key": prior_key},
            )
    status = "resumed" if resumed else "started"
    registry.record_trial_attempt_event(
        attempt_key, event_key=f"{status}-{time_ns()}-{uuid.uuid4().hex}",
        status=status, detail=detail,
    )
    return attempt_key


def finish_application_trial(attempt_key: str | None, *, status: str,
                             detail: dict) -> None:
    """Append a terminal event for standalone training; no-op for WFO-owned calls."""
    if attempt_key is None:
        return
    if status not in {"completed", "failed", "interrupted"} or not isinstance(detail, dict):
        raise ValueError("invalid standalone application trial result")
    project_registry().record_trial_attempt_event(
        attempt_key, event_key=f"{status}-{time_ns()}-{uuid.uuid4().hex}",
        status=status, detail=detail,
    )
