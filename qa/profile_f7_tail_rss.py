"""Compare process RSS for exact streamed tail selection implementations.

This is synthetic engineering evidence only. It does not open market or account
data and must not be used as trading-performance evidence.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

if os.name != "nt":
    raise RuntimeError("this process-RSS profiler currently requires Windows")


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetCurrentProcess.restype = ctypes.c_void_p
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_get_process_memory_info = _psapi.GetProcessMemoryInfo
_get_process_memory_info.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_ProcessMemoryCounters),
    ctypes.c_ulong,
]
_get_process_memory_info.restype = ctypes.c_int


def _rss_bytes() -> int:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not _get_process_memory_info(
        _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.WorkingSetSize)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "research" / "src"))

from rles_research import metrics as metrics_module  # noqa: E402


def _value(index: int) -> float:
    return ((index % 101) - 50) * 1e-9


def _sha256(path: Path) -> str:
    # Git may check out the same Python source with CRLF or LF depending on
    # core.autocrlf. Bind evidence to normalized source text, not checkout EOLs.
    contents = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(contents).hexdigest()


def _worker(count: int, mode: str) -> dict[str, Any]:
    metrics_module._TAIL_HEAP_MAX = count if mode == "heap" else 8_192
    accumulator = metrics_module._StreamingSeriesMetrics(
        max_count=count,
        risk_free_per_period=0.0,
        target_per_period=0.0,
        tail_quantile=0.05,
    )
    baseline = _rss_bytes()
    peak = [baseline]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.005):
            try:
                peak[0] = max(peak[0], _rss_bytes())
            except OSError:
                return

    sampler = threading.Thread(target=sample_rss, name="rss-sampler", daemon=True)
    sampler.start()
    started = time.perf_counter()
    for index in range(count):
        accumulator.add(_value(index))
    result = accumulator.result(periods_per_year=525_600, require_exact_count=True)
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join()
    peak[0] = max(peak[0], _rss_bytes())
    if accumulator.returns is not None:
        accumulator.returns.close()
    return {
        "mode": mode,
        "observations": count,
        "tail_count": max(1, int((count * 0.05) + 0.999999999)),
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": peak[0],
        "peak_rss_delta_bytes": peak[0] - baseline,
        "elapsed_seconds": elapsed,
        "metrics": result,
    }


def run(count: int) -> dict[str, Any]:
    runs = {}
    for mode in ("heap", "radix"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
             "--count", str(count)],
            check=True,
            capture_output=True,
            text=True,
        )
        runs[mode] = json.loads(completed.stdout)
    if runs["heap"]["metrics"] != runs["radix"]["metrics"]:
        raise AssertionError("streamed heap and disk-radix metrics differ")
    return {
        "schema": "rles.f7-tail-process-rss.v1",
        "synthetic_only": True,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "rss_measurement": "Windows process WorkingSetSize sampled every 5ms",
        "metrics_source_sha256": _sha256(
            PROJECT_ROOT / "packages" / "research" / "src" / "rles_research" / "metrics.py",
        ),
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "runs": runs,
        "exact_metrics_match": True,
        "rss_delta_reduction_fraction": (
            1 - runs["radix"]["peak_rss_delta_bytes"]
            / max(1, runs["heap"]["peak_rss_delta_bytes"])
        ),
        "elapsed_ratio_radix_to_heap": (
            runs["radix"]["elapsed_seconds"]
            / max(1e-12, runs["heap"]["elapsed_seconds"])
        ),
        "limitation": (
            "Process RSS sampled every 5ms over deterministic synthetic returns. "
            "This is not a full Native historical Train trace and carries no "
            "economic or strategy-validation meaning."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5_000_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("heap", "radix"))
    args = parser.parse_args()
    if type(args.count) is not int or args.count < 2:
        parser.error("--count must be an integer >= 2")
    result = _worker(args.count, args.worker) if args.worker else run(args.count)
    rendered = json.dumps(result, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
