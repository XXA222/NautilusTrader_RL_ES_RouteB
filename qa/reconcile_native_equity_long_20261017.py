"""Reconcile a frozen 90-day Train-inner linear ledger without selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rles_data.core import CuratedStore, DataError
from rles_data.io import read_curated
from rles_data.json_utils import loads_unique_json
from rles_gym_bridge.historical import HistoricalEnvironmentFactory, load_rules
from rles_orchestrator.application import _file_hash, _verify_file, _write_verified, rollout
from rles_orchestrator.research_access import record_development
from rles_orchestrator.wfo_application import _scaler

from profile_native_history_growth_20261016 import (
    DAY_NS, _routeb_path, coverage_for_window, make_predictor,
)


def _load_json_object(path: Path) -> dict:
    try:
        value = loads_unique_json(path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise DataError(f"cannot read unambiguous equity-reconciliation JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DataError(f"equity-reconciliation JSON root must be an object: {path}")
    return value


def run(*, plan_path: Path, output: Path) -> dict:
    plan_path = _routeb_path(plan_path, "equity reconciliation plan", must_exist=True)
    output = _routeb_path(output, "equity reconciliation output", must_exist=False)
    if output.exists() or not output.parent.is_dir():
        raise DataError("equity reconciliation output must be new and its parent must exist")
    plan = _load_json_object(plan_path)
    if plan.get("split") != "train_inner":
        raise DataError("requires the frozen Train-inner plan and a new output directory")
    original_plan_path = _routeb_path(plan["source_plan"], "frozen Train plan", must_exist=True)
    if _file_hash(original_plan_path) != plan["source_plan_sha256"]:
        raise DataError("frozen Train-inner source plan checksum changed")
    source_plan = _load_json_object(original_plan_path)
    bundle_root = _routeb_path(source_plan["bundle"], "frozen candidate bundle", must_exist=True)
    if _verify_file(bundle_root / "bundle.json") != source_plan["bundle_sha256"]:
        raise DataError("frozen candidate bundle checksum changed")
    bundle = _load_json_object(bundle_root / "bundle.json")
    source = plan["source"]
    source_paths = {
        key: _routeb_path(source[key + "_path"], f"{key} source", must_exist=True)
        for key in ("market", "rules")
    }
    environment = bundle["environment"]
    for key in ("market", "rules"):
        environment_path = _routeb_path(
            environment[key + "_path"], f"bundle {key} source", must_exist=True,
        )
        if environment_path != source_paths[key]:
            raise DataError(f"bundle {key} source differs from the reconciliation plan")
        if _file_hash(source_paths[key]) != source[key + "_sha256"]:
            raise DataError(f"frozen {key} source checksum changed")
    start_ns = source_plan["windows"][0]["start_ns"]
    window = {"start_ns": start_ns, "end_ns": start_ns + 90 * DAY_NS}
    if window["end_ns"] > bundle["training_end_ns"]:
        raise DataError("90-day reconciliation exceeds frozen Train boundary")
    output.mkdir(parents=True)
    registry_config = {"environment": bundle["environment"], "output": str(output),
                       "seed": source_plan["seed"]}
    record_development(registry_config, window, label="m0-1-native-equity-90d")

    rules = load_rules(source_paths["rules"])
    store = CuratedStore((*read_curated(source_paths["market"]).events, *rules))
    if store.snapshot_hash != bundle["preprocessing"]["identities"]["dataset"]:
        raise DataError("frozen archive does not match candidate bundle data identity")
    coverage = coverage_for_window(environment["coverage_policy"], window)
    scaler = _scaler(bundle_root)
    factory = HistoricalEnvironmentFactory(**{
        **environment, **window, "coverage_policy": coverage,
        "split": "train_inner", "mode": "base", "scaler": scaler,
        "fit_scaler": False, "reward_scale": bundle.get("reward_scale", 1.0),
        "compact_report_snapshots": True, "prepared_store": store,
    })
    report, _ = rollout(
        factory, make_predictor("long", factory.rows), seed=source_plan["seed"],
        soft_parameters=bundle["soft_multipliers"], collect_observations=False,
        include_account_rows=False,
    )
    reconciliation = report["equity_reconciliation"]
    if (not report["terminal_flat"] or not reconciliation["terminal_checked"]
            or reconciliation["unchecked_open_barriers"] != 0):
        raise DataError("90-day linear ledger lacks a flat terminal or complete open-barrier marks")
    tolerance = float(reconciliation["rounding_per_event"])
    evidence = {
        "schema": "rles.audit-m0-1-native-equity-reconciliation.v1",
        "task": "M0.1 / A08",
        "split": "train_inner",
        "purpose": "native linear ledger verification only; no model selection",
        "window": window,
        "seed": source_plan["seed"],
        "market_sha256": source["market_sha256"],
        "rules_sha256": source["rules_sha256"],
        "bundle_sha256": source_plan["bundle_sha256"],
        "source_snapshot_hash": store.snapshot_hash,
        "historical_rule_provenance": "scenario-only: engineering assumption, not historical exchange-rule evidence",
        "metrics": {
            "decision_steps": report["action_diagnostics"]["count"],
            "snapshot_count": len(factory.snapshots),
            "terminal_flat": report["terminal_flat"],
            "net_return_descriptive_only": report["metrics"]["net_return"],
            "final_equity": str(factory.snapshots[-1].equity),
            "fee": report["metrics"]["total_fee"],
            "funding": report["metrics"]["total_funding"],
            "slippage_cost_attribution_only": report["metrics"]["total_slippage_cost"],
        },
        "equity_reconciliation": reconciliation,
        "reconciliation_tolerance": {
            "rounding_per_event": reconciliation["rounding_per_event"],
            "max_abs_residual": reconciliation["max_abs_residual"],
            "verified_within_per_event_bound": float(reconciliation["max_abs_residual"]) <=
                tolerance * (reconciliation["unique_fills"]
                             + reconciliation["unique_funding"] + 1),
            "slippage_double_counted": False,
        },
        "data_access": {"validation_test_holdout_access": False,
                        "market_data_read": True, "research_registry_recorded": True},
        "limits": [
            "Uses previously frozen Train-inner archive and scenario-only execution rules.",
            "Certifies one quote-settled unit-multiplier linear instrument without external cashflows.",
            "Does not certify inverse/multi-asset accounts, deposits/withdrawals, or historical rule accuracy.",
        ],
    }
    _write_verified(output / "evidence.json", evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = run(plan_path=args.plan, output=args.output)
    summary = evidence["equity_reconciliation"]
    print(json.dumps({"decision_steps": evidence["metrics"]["decision_steps"],
                      "open_barriers_checked": summary["checked_open_barriers"],
                      "open_barriers_unchecked": summary["unchecked_open_barriers"],
                      "max_abs_residual": summary["max_abs_residual"],
                      "evidence": str(args.output / "evidence.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
