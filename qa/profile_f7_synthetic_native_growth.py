"""Profile native replay memory and trajectory parity on synthetic minute bars.

This is engineering-only evidence. It does not read curated market files or
touch the project research-access registry.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from rles_data.core import Availability, CuratedStore, Event, MINUTE_NS
from rles_gym_bridge.historical import HistoricalEnvironmentFactory
from rles_research.native_metrics import NativeReportAccumulator

_profile_helper_name = (
    "profile_native_history_growth_20261016" if __package__ in (None, "")
    else "qa.profile_native_history_growth_20261016"
)
_profile_helper = importlib.import_module(_profile_helper_name)
RssSampler = _profile_helper.RssSampler
_routeb_path = _profile_helper._routeb_path
available_memory = _profile_helper.available_memory
process_rss = _profile_helper.process_rss

DAY_MINUTES = 24 * 60
INSTRUMENT = "ETHUSDT-PERP.SIM"
START_NS = 61 * MINUTE_NS
ACTION = np.asarray([0.5], dtype=np.float32)
SOURCE_HASH = "f" * 64


def _canonical(value):
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, float):
        return {"float64": value.hex()}
    if isinstance(value, dict):
        # Nautilus assigns fresh UUIDs to simulator initialization and fills;
        # they are audit identities, not trajectory/economic state. Normalize
        # only these two fields so otherwise complete terminal-fill facts stay
        # in the online-summary-vs-lean trajectory digest.
        return {
            str(key): ("<run-local-native-id>" if key in {"event_id", "init_id"}
                       else _canonical(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    raise TypeError(f"unsupported synthetic trace value: {type(value).__name__}")


def _update_trace(digest, observation, reward, terminated, truncated, info) -> None:
    row = {
        "observation": np.asarray(observation, dtype=np.float32),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
    }
    encoded = json.dumps(_canonical(row), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _synthetic_store(max_minutes: int) -> CuratedStore:
    """Build deterministic OHLCV/mark/index/Funding facts entirely in memory."""
    final_minute = START_NS // MINUTE_NS + max_minutes + 1
    rule = {
        "instrument": INSTRUMENT, "quantity_unit": "ETH", "base_currency": "ETH",
        "quote_currency": "USDT", "contract_multiplier": "1",
        "tick_size": "0.01", "step_size": "0.001",
        "min_qty": "0.001", "min_notional": "1", "maker_fee": "0.001",
        "taker_fee": "0.001", "initial_margin_rate": "1",
        "maintenance_margin_rate": "0.35", "effective_ns": MINUTE_NS,
        "valid_until_ns": (final_minute + 1) * MINUTE_NS,
        "rule_version": "SYNTHETIC-F7-NATIVE-GROWTH",
    }
    events = [Event("synthetic-instrument-rule", "instrument_rule", MINUTE_NS,
                    MINUTE_NS, MINUTE_NS, Availability.NATIVE, SOURCE_HASH, rule,
                    "synthetic:f7-native-growth")]
    # Establish a valid pre-window funding observation. Coverage validation
    # requires the first in-window update to be fresh relative to this fact.
    events.append(Event("synthetic-funding-initial", "funding_update", MINUTE_NS,
                        MINUTE_NS, MINUTE_NS, Availability.NATIVE, SOURCE_HASH,
                        {"instrument": INSTRUMENT, "rate": "0.0001"},
                        "synthetic:f7-funding"))
    previous_close = 1_000.0
    for minute in range(1, final_minute + 1):
        timestamp = minute * MINUTE_NS
        close = 1_000.0 * math.exp(0.00008 * math.sin(minute / 31.0))
        opening = previous_close
        high = max(opening, close) * (1.0002 + 0.00002 * math.sin(minute / 17.0) ** 2)
        low = min(opening, close) * (0.9998 - 0.00002 * math.cos(minute / 23.0) ** 2)
        base_payload = {"instrument": INSTRUMENT}
        payloads = (
            ("ohlcv_1m", {"open": opening, "high": high, "low": low,
                          "close": close, "volume": 100.0 + minute % 19}),
            ("mark", {"price": close}),
            ("index", {"price": close}),
        )
        for kind, values in payloads:
            payload = {**base_payload, **{key: format(value, ".12f")
                                             for key, value in values.items()}}
            events.append(Event(f"synthetic-{kind}-{minute}", kind, timestamp,
                                timestamp, timestamp, Availability.NATIVE,
                                SOURCE_HASH, payload, f"synthetic:f7-{kind}"))
        if minute >= 480 and minute % 480 == 0:
            events.append(Event(f"synthetic-funding-{minute}", "funding_update",
                                timestamp, timestamp, timestamp, Availability.NATIVE,
                                SOURCE_HASH, {"instrument": INSTRUMENT, "rate": "0.0001"},
                                "synthetic:f7-funding"))
        previous_close = close
    return CuratedStore(events)


def _factory_args(store: CuratedStore, days: int) -> dict:
    minutes = days * DAY_MINUTES
    end_ns = START_NS + minutes * MINUTE_NS
    return {
        # prepared_store supplies all event data; these paths are never opened.
        "market_path": Path("synthetic-market-in-memory.parquet"),
        "rules_path": Path("synthetic-rules-in-memory.json"),
        "instrument": INSTRUMENT,
        "start_ns": START_NS,
        "end_ns": end_ns,
        "coverage_policy": {
            "maximum_age_ns": {"mark": MINUTE_NS, "index": MINUTE_NS,
                                "funding_update": 480 * MINUTE_NS},
            "settlement_times_ns": {INSTRUMENT: ()},
            "provenance": "synthetic:f7-native-growth",
        },
        "execution": {
            "schema_version": "0.4.0", "deadband": "0.01",
            "pending_order_policy": "cancel_replace", "order_type": "MARKET",
            "tif": "GTC", "terminal_policy": "force_flatten_with_cost",
            "base_latency": {"model_id": "base", "decision_ns": 1,
                              "submit_ns": 1, "venue_ns": 1},
            "stress_latency": {"model_id": "stress", "decision_ns": 1,
                                "submit_ns": 1, "venue_ns": 2},
            "base_fill": {"model_id": "base", "slippage_bps": "1",
                          "data_granularity": "bar"},
            "stress_fill": {"model_id": "stress", "slippage_bps": "2",
                            "data_granularity": "bar"},
        },
        "hard_risk": {"max_notional": "1000", "max_leverage": "1",
                       "margin_buffer_fraction": "0.1", "min_equity": "1",
                       "incomplete_data_max_leverage": "1", "require_mark": True,
                       "require_funding": True},
        "soft_risk": {"target_scale": "1", "max_turnover_fraction": "1",
                       "drawdown_throttle": "0", "funding_sensitivity": "0"},
        "starting_balance": "10000", "risk_budget_notional": "1000",
        "fit_scaler": True, "split": "train_inner", "prepared_store": store,
        "compact_report_snapshots": True,
    }


def _run_case(store: CuratedStore, days: int, *, online_summary: bool) -> dict:
    import gc

    gc.collect()
    rss_start = process_rss()
    started = time.perf_counter()
    digest = hashlib.sha256()
    factory = None
    report_summary = None
    accumulator = None
    with RssSampler() as sampler:
        factory = HistoricalEnvironmentFactory(**_factory_args(store, days))
        if online_summary:
            accumulator = NativeReportAccumulator(
                sample_interval_ns=MINUTE_NS,
                max_return_count=days * DAY_MINUTES + 1,
                allow_price_improvement=False,
                contract_multiplier=factory.contract_multiplier,
                linear_equity_rounding=Decimal("0.00000001"),
            )
        env = factory.make_env(
            retain_report_snapshots=online_summary,
            report_snapshot_sink=(None if accumulator is None else accumulator.add_snapshot),
        )
        try:
            observation, reset_info = env.reset(seed=31)
            _update_trace(digest, observation, 0.0, False, False, {"reset": reset_info})
            steps = 0
            while True:
                observation, reward, terminated, truncated, info = env.step(ACTION)
                _update_trace(digest, observation, reward, terminated, truncated, info)
                steps += 1
                if terminated or truncated:
                    break
        finally:
            env.close()
        report = None
        snapshot_count = 0 if accumulator is None else accumulator.snapshot_count
        if accumulator is not None:
            report = accumulator.finish()
            metric_bytes = json.dumps(_canonical(report["metrics"]), sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()
            report_summary = {
                "snapshot_count": snapshot_count,
                "fill_count": len(report.get("fills", ())),
                "funding_count": len(report.get("funding", ())),
                "terminal_flat": report.get("terminal_flat"),
                "metrics_sha256": hashlib.sha256(metric_bytes).hexdigest(),
            }
    record = {
        "days": days,
        "mode": "online_summary_sink" if online_summary else "lean_execution",
        "report_snapshot_sink_enabled": online_summary,
        "report_snapshot_sequence_retained": False,
        "decision_steps": steps, "trajectory_sha256": digest.hexdigest(),
        "rss_start_bytes": rss_start, "rss_peak_bytes": sampler.peak,
        "rss_peak_delta_bytes": max(0, sampler.peak - rss_start),
        "rss_end_bytes": process_rss(), "elapsed_seconds": time.perf_counter() - started,
        "native_report": report_summary,
    }
    del factory, report
    gc.collect()
    return record


def _isolated_case(days: int, max_days: int, *, online_summary: bool) -> dict:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--days", str(days), "--max-days", str(max_days),
        "--online-summary", "true" if online_summary else "false",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"isolated {days}-day {'online-summary' if online_summary else 'lean'} profile failed: "
            f"{completed.stderr[-4000:]}"
        )
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("isolated synthetic profile returned invalid JSON") from exc


def _worker_case(days: int, max_days: int, *, online_summary: bool) -> dict:
    if available_memory() < 2 * 1024**3:
        raise MemoryError("less than 2 GiB free before isolated synthetic profile")
    source_started = time.perf_counter()
    with RssSampler() as source_sampler:
        store = _synthetic_store(max_days * DAY_MINUTES)
    source_summary = {
        "generation_seconds": time.perf_counter() - source_started,
        "event_count": len(store.events), "snapshot_sha256": store.snapshot_hash,
        "rss_peak_bytes": source_sampler.peak,
        "rss_end_bytes": process_rss(),
    }
    return {
        "case": _run_case(store, days, online_summary=online_summary),
        "source": source_summary,
    }


def run(days: tuple[int, ...], output: Path) -> dict:
    project = Path(__file__).resolve().parents[1]
    output = _routeb_path(output, "synthetic profile output", must_exist=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("profile output must be a new file under an existing RouteB directory")
    if (not days or tuple(sorted(set(days))) != days
            or any(type(day) is not int or day < 1 or day > 90 for day in days)):
        raise ValueError("days must be sorted unique integers in [1, 90]")
    if available_memory() < 2 * 1024**3:
        raise MemoryError("less than 2 GiB free before synthetic profile")
    cases = []
    source_summary = None
    max_days = max(days)
    for day in days:
        if available_memory() < 2 * 1024**3:
            raise MemoryError("less than 2 GiB free before profile case")
        print(f"starting isolated {day}-day online-summary profile", flush=True)
        summary_result = _isolated_case(day, max_days, online_summary=True)
        print(f"starting isolated {day}-day lean profile", flush=True)
        lean_result = _isolated_case(day, max_days, online_summary=False)
        summary, lean = summary_result["case"], lean_result["case"]
        if source_summary is None:
            source_summary = summary_result["source"]
        if (summary["decision_steps"] != lean["decision_steps"]
                or summary["trajectory_sha256"] != lean["trajectory_sha256"]):
            raise RuntimeError(f"online-summary/lean native trajectories differ for {day} days")
        cases.append({"online_summary": summary, "lean_execution": lean,
                      "trajectory_equivalent": True})
        print(json.dumps({"days": day, "steps": summary["decision_steps"],
                          "trajectory_equivalent": True,
                          "online_summary_rss_peak_delta_bytes": summary["rss_peak_delta_bytes"],
                          "lean_execution_rss_peak_delta_bytes": lean["rss_peak_delta_bytes"]}),
              flush=True)
    source_files = (
        Path(__file__),
        project / "packages/data/src/rles_data/core.py",
        project / "packages/data/src/rles_data/features.py",
        project / "packages/gym_bridge/src/rles_gym_bridge/episode.py",
        project / "packages/gym_bridge/src/rles_gym_bridge/historical.py",
        project / "packages/orchestrator/src/rles_orchestrator/application.py",
        project / "packages/research/src/rles_research/native_metrics.py",
        project / "packages/research/src/rles_research/metrics.py",
    )
    result = {
        "schema": "rles.f7-synthetic-native-growth.v2",
        "status": "PASS_SYNTHETIC_ENGINEERING_ONLY",
        "scope": "synthetic Train-only native-engine replay; no market data or registry access",
        "market_data_used": False, "test_data_used": False,
        "research_registry_accessed": False,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "source_files_sha256": {
            path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "source": source_summary, "cases": cases,
        "measurement_note": (
            "Online-summary-sink and lean-execution cases run in independent child processes. "
            "The online summary callback consumes each compact snapshot without retaining a "
            "per-barrier snapshot sequence. Each child's rss_start is measured after identical "
            "synthetic source generation; online-mode peaks include per-barrier metric reduction "
            "and final report assembly."
        ),
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", nargs="+", type=int, default=[1, 30, 90])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--online-summary", choices=("true", "false"))
    args = parser.parse_args()
    if args.worker:
        if (args.output is not None or len(args.days) != 1 or args.max_days is None
                or args.online_summary is None
                or not 1 <= args.days[0] <= args.max_days <= 90):
            parser.error("invalid isolated worker arguments")
        result = _worker_case(args.days[0], args.max_days,
                              online_summary=args.online_summary == "true")
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    if args.output is None:
        parser.error("--output is required outside worker mode")
    result = run(tuple(sorted(set(args.days))), args.output)
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

