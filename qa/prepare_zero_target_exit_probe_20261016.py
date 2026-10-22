"""Freeze a deterministic Train-only zero-target semantics probe."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from rles_data.core import DataError
from rles_data.json_utils import loads_unique_json
from rles_orchestrator.application import _file_hash, _verify_file, _write_verified


def _read_json_object(path: Path) -> dict:
    try:
        value = loads_unique_json(path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise DataError(f"cannot read unambiguous zero-target JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DataError(f"zero-target JSON root must be an object: {path}")
    return value


def stamp(value: datetime) -> int:
    return int(value.timestamp()) * 1_000_000_000


def prepare(root: Path) -> Path:
    baseline_path = root / "configs/train_signal_baselines_20261016.json"
    baseline = _read_json_object(baseline_path)
    bundle_root = Path(baseline["bundle"])
    if _verify_file(bundle_root / "bundle.json") != baseline["bundle_sha256"]:
        raise ValueError("frozen bundle checksum changed")
    bundle = _read_json_object(bundle_root / "bundle.json")
    start = datetime(2025, 2, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    source_start = start - timedelta(days=1)
    plan = {
        "schema": "rles.zero-target-exit-probe.v1", "date": "2026-10-16",
        "split": "train_inner", "scope": "deterministic execution semantics only; no strategy claim",
        "bundle": str(bundle_root), "bundle_sha256": baseline["bundle_sha256"],
        "dataset_sha256": bundle["preprocessing"]["identities"]["dataset"],
        "source": baseline["source"], "source_plan_sha256": _file_hash(baseline_path),
        "window": {"source_start_ns": stamp(source_start), "start_ns": stamp(start),
                   "end_ns": stamp(end)},
        "instrument": bundle["environment"]["instrument"],
        "modes": ["deadband", "close"], "seeds": [102, 103, 104],
        "action_path": {"step_0": 0.25, "steps_1_through_21": 0.05,
                        "step_22_onward": 0.0},
        "max_probe_steps_after_zero": 60,
        "expected_semantic_difference": "under 0.10 deadband, a valid small residual position persists for zero target; close policy sends Reduce-Only close",
        "execution": "same frozen base latency/fill/fees/rules; only zero_target_policy changes",
        "selection": "none; compare plans, native exposure and fills, not net return",
        "stop_free_bytes": 2 * 1024**3,
        "historical_economics": "scenario-only; not verified historical execution facts",
    }
    path = root / "configs/zero_target_exit_probe_20261016.json"
    if path.exists():
        raise FileExistsError(path)
    _write_verified(path, plan)
    return path


if __name__ == "__main__":
    print(prepare(Path(__file__).resolve().parents[1]))
