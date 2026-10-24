"""Resolve the shared project ledger; never derive it from an experiment output."""
from contextvars import ContextVar
from hashlib import sha256
import json
from pathlib import Path
from time import time_ns
import uuid

from rles_data.core import (
    CuratedStore,
    DataError,
    MINUTE_NS,
    digest,
    require_hash,
    require_ns,
    routeb_workspace_root,
)
from rles_data.io import curated_event_time_bounds, curated_source_identity, read_curated_window
from rles_research.access import ProjectResearchRegistry


ACTIVE_OWNER: ContextVar[str | None] = ContextVar("research_owner", default=None)
DATA_FAMILY = "perpetual-market-v1"


def project_registry() -> ProjectResearchRegistry:
    # Source checkout identity is stable even if the CLI's cwd/output changes.
    candidate = routeb_workspace_root(__file__)
    if candidate is None:
        raise DataError("project research registry unavailable; configure RLES_WORKSPACE_ROOT")
    registry = ProjectResearchRegistry(candidate / ".artifacts" / "research-access.sqlite")
    inventory_path = candidate / "docs" / "audit_evidence" / "20261016" / "legacy_research_access_v2.json"
    if not inventory_path.is_file():
        raise DataError("legacy research inventory missing; restore the tracked inventory before research")
    content = inventory_path.read_bytes()
    inventory_sha = sha256(content).hexdigest()
    marker = inventory_path.with_suffix(".json.sha256")
    if not marker.is_file() or marker.read_text(encoding="ascii") != inventory_sha:
        raise DataError("legacy research inventory integrity mismatch")
    try:
        registry.import_inventory(content)
    except ValueError as exc:
        raise DataError("legacy research inventory is invalid or unresolved") from exc
    return registry


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


def reserve_routeb_fresh_fold(config: dict, window: dict, *, label: str,
                              fold_sha256: str, repository_commit: str,
                              source_code_sha256: str, source_identity: dict) -> dict:
    """Freeze a fresh Route B fold scope before any curated payload is decoded."""
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
    reservation_token = project_registry().reserve_fresh_development(
        key=key, owner=owner, market=market_key(environment["instrument"]),
        family=DATA_FAMILY, start_ns=window["start_ns"], end_ns=window["end_ns"],
        identity=identity,
    )
    identity_bytes = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    receipt: dict = {
        "schema": "rles.routeb-fold-reader-receipt.v1",
        "reservation_key": key,
        "reservation_token_sha256": sha256(reservation_token.encode("ascii")).hexdigest(),
        "reservation_identity_sha256": sha256(identity_bytes).hexdigest(),
        "reservation_label": label,
        "dataset_hash": source_identity["snapshot_hash"],
        "marker_sha256": source_identity["marker_sha256"],
        "fold_sha256": fold_sha256,
        "repository_commit": repository_commit,
        "source_code_sha256": source_code_sha256,
        "read_scope": {"start_ns": window["start_ns"],
                       "end_ns": window["end_ns"], "includes_test": False},
        "reservation_committed_before_decode": True,
    }
    receipt["receipt_sha256"] = sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    return receipt


def read_registered_fold_window(config: dict, window: dict, *, marker_sha256: str,
                                dataset_hash: str, required_start_ns: int, required_end_ns: int,
                                label: str, fold_sha256: str, repository_commit: str,
                                source_code_sha256: str,
                                allow_legacy_isolated_snapshot: bool = False,
                                return_receipt: bool = False) -> CuratedStore | tuple[CuratedStore, dict]:
    """Preflight the frozen COMPLETE marker, reserve a fresh Fold scope, then decode rows."""
    environment = config.get("environment")
    if (not isinstance(environment, dict) or not isinstance(environment.get("market_path"), str)
            or any(config.get(name) is not None for name in ("evaluation", "test", "holdout"))
            or type(allow_legacy_isolated_snapshot) is not bool
            or allow_legacy_isolated_snapshot or type(return_receipt) is not bool):
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
    receipt = reserve_routeb_fresh_fold(
        config, window, label=label, fold_sha256=fold_sha256,
        repository_commit=repository_commit, source_code_sha256=source_code_sha256,
        source_identity=source_identity,
    )
    store = read_curated_window(
        environment["market_path"], window["start_ns"], window["end_ns"],
        expected_marker_sha256=marker_sha256,
        allow_legacy_isolated_snapshot=allow_legacy_isolated_snapshot,
    )
    return (store, receipt) if return_receipt else store

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
