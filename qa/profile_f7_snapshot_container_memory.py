"""Measure retained Python container overhead for compact native snapshots."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import tracemalloc
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rles_gym_bridge.historical import (
    _CompactReportSnapshot,
    _CompactReportSnapshotSeries,
    _EMPTY_COMPACT_REPORT_ACCOUNT,
)


def _snapshot(values: tuple[Any, ...]) -> _CompactReportSnapshot:
    (equity, currency, positions, orders, fills, funding, all_orders,
     liquidation_orders, mark, liquidation_capability, decision_ts_ns,
     submitted, filled) = values
    return _CompactReportSnapshot(
        equity=equity, currency=currency, positions=positions, orders=orders,
        account=_EMPTY_COMPACT_REPORT_ACCOUNT, account_events=(), fills=fills,
        funding=funding, all_orders=all_orders,
        liquidation_orders=liquidation_orders, margins=(), mark=mark,
        liquidation_capability=liquidation_capability,
        decision_ts_ns=decision_ts_ns, cumulative_submitted_qty=submitted,
        cumulative_filled_qty=filled,
    )


def _measure(count: int, *, column_store: bool) -> tuple[int, float, Any]:
    # Keep identical scalar/fact objects alive outside tracing. This isolates
    # retained container structure instead of counting native-engine objects.
    positions = (MappingProxyType({
        "position_id": "P1", "side": "LONG", "quantity": "1",
        "multiplier": "1", "is_inverse": False,
        "settlement_currency": "USDT",
    }),)
    zeros = Decimal(0)
    rows = tuple(
        (Decimal(index + 1000), "USDT", positions, (), (), (), (), (),
         Decimal(index % 1000 + 10000), True,
         1_700_000_000_000_000_000 + index * 60_000_000_000,
         zeros, zeros)
        for index in range(count)
    )
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    if column_store:
        stored: Any = _CompactReportSnapshotSeries()
        for row in rows:
            stored.append(_snapshot(row))
    else:
        stored = [_snapshot(row) for row in rows]
    elapsed = time.perf_counter() - started
    gc.collect()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return retained, elapsed, stored


def _measure_with_dynamic_payloads(count: int, *, column_store: bool) -> tuple[int, float, Any]:
    positions = (MappingProxyType({
        "position_id": "P1", "side": "LONG", "quantity": "1",
        "multiplier": "1", "is_inverse": False,
        "settlement_currency": "USDT",
    }),)
    zeros = Decimal(0)
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    if column_store:
        stored: Any = _CompactReportSnapshotSeries()
        for index in range(count):
            row = (
                Decimal(f"{index + 1000}.{index % 10000:04d}"), "USDT", positions,
                (), (), (), (), (), Decimal(f"{10000 + index % 1000}.{index % 100:02d}"),
                True, 1_700_000_000_000_000_000 + index * 60_000_000_000,
                zeros, zeros,
            )
            stored.append(_snapshot(row))
    else:
        stored = []
        for index in range(count):
            row = (
                Decimal(f"{index + 1000}.{index % 10000:04d}"), "USDT", positions,
                (), (), (), (), (), Decimal(f"{10000 + index % 1000}.{index % 100:02d}"),
                True, 1_700_000_000_000_000_000 + index * 60_000_000_000,
                zeros, zeros,
            )
            stored.append(_snapshot(row))
    elapsed = time.perf_counter() - started
    gc.collect()
    retained, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return retained, elapsed, stored


def run(count: int) -> dict[str, Any]:
    old_bytes, old_seconds, old = _measure(count, column_store=False)
    new_bytes, new_seconds, new = _measure(count, column_store=True)
    equivalent = len(old) == len(new)
    for index, original in enumerate(old):
        if original != new[index]:
            equivalent = False
            break
    if not equivalent:
        raise AssertionError("column-store snapshot facts differ from dataclass snapshots")
    dynamic_old_bytes, dynamic_old_seconds, dynamic_old = _measure_with_dynamic_payloads(
        count, column_store=False,
    )
    dynamic_new_bytes, dynamic_new_seconds, dynamic_new = _measure_with_dynamic_payloads(
        count, column_store=True,
    )
    dynamic_equivalent = len(dynamic_old) == len(dynamic_new)
    for index, original in enumerate(dynamic_old):
        if original != dynamic_new[index]:
            dynamic_equivalent = False
            break
    if not dynamic_equivalent:
        raise AssertionError("dynamic column-store facts differ from dataclass snapshots")
    return {
        "schema": "rles.f7-snapshot-container-memory.v1",
        "count": count,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "common_scalar_payloads_excluded_from_tracing": True,
        "semantic_equivalence": equivalent,
        "dataclass_list": {"retained_container_bytes": old_bytes,
                           "construction_seconds": old_seconds},
        "column_store": {"retained_container_bytes": new_bytes,
                         "construction_seconds": new_seconds},
        "retained_container_bytes_saved": old_bytes - new_bytes,
        "saved_bytes_per_snapshot": (old_bytes - new_bytes) / count,
        "dynamic_payload_profile": {
            "semantic_equivalence": dynamic_equivalent,
            "dataclass_list_retained_bytes": dynamic_old_bytes,
            "column_store_retained_bytes": dynamic_new_bytes,
            "retained_bytes_saved": dynamic_old_bytes - dynamic_new_bytes,
            "saved_bytes_per_snapshot": (dynamic_old_bytes - dynamic_new_bytes) / count,
            "dataclass_list_construction_seconds": dynamic_old_seconds,
            "column_store_construction_seconds": dynamic_new_seconds,
        },
        "note": "Isolated Python allocations only. The common-value profile excludes shared values; dynamic_payload_profile creates and retains changing equity, mark and timestamp values. Neither includes native-engine RSS.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=129600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if type(args.count) is not int or args.count < 1:
        parser.error("--count must be a positive integer")
    result = run(args.count)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"python", "platform", "note"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
