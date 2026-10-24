"""Bounded, synthetic-only CPU profile for cached B3 rollout transitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchicalModelConfig,
    HierarchicalTrainingConfig,
    create_stateful_hierarchical_f2_predictor,
    hierarchy_tensor_sha256,
    train_hierarchical,
)
from rles_world_model.routeb.jax_model import RSSMConfig
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS


def _synthetic_bars(count: int = 4_320) -> tuple[Candle, ...]:
    price = 2_000.0
    rows = []
    for index in range(1, count + 1):
        open_price = price * math.exp(0.00002 * math.sin(index / 5))
        price = open_price * math.exp(0.0006 * math.sin(index / 13))
        high = max(open_price, price) * math.exp(0.0002)
        low = min(open_price, price) * math.exp(-0.0002)
        rows.append(Candle(index * MINUTE_NS, index * MINUTE_NS,
                           "SYNTHETIC.STATEFUL.PROFILE", open_price,
                           high, low, price, 100.0 + index % 13))
    return tuple(rows)


def _current_rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    guard = subprocess.run(
        [sys.executable, "-B", str(root / "qa/assert_canonical_workspace.py")],
        cwd=root, check=False, capture_output=True, text=True,
    )
    if guard.returncode != 0:
        raise RuntimeError(f"canonical workspace guard failed: {guard.stderr.strip()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-steps", type=int, default=256)
    parser.add_argument("--paths", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--max-rss-mib", type=float, default=4096.0)
    args = parser.parse_args()
    if (type(args.profile_steps) is not int or not 2 <= args.profile_steps <= 10_000
            or type(args.paths) is not int or not 1 <= args.paths <= 256
            or args.profile_steps * args.paths > 2_560_000
            or not math.isfinite(args.max_seconds) or not 1 <= args.max_seconds <= 600
            or not math.isfinite(args.max_rss_mib) or not 512 <= args.max_rss_mib <= 16_384):
        raise ValueError("profile request exceeds bounded steps, paths, time or RSS limits")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or not output_dir.parent.is_dir():
        raise ValueError("profile output must be a new directory under an existing parent")
    output_dir.mkdir()

    replay = MarketSequence(_synthetic_bars(),
                            "synthetic-stateful-cpu-profile-NOT-MARKET", "train")
    tensors, scalers = build_train_hierarchy(
        replay, burn_in=8, score_start_ns=replay.candles[8].ts_close_ns,
        score_end_ns=replay.candles[-1].ts_close_ns + MINUTE_NS)
    model = RSSMConfig(deter=8, hidden=8, stoch=1, classes=2, blocks=1)
    model_config = HierarchicalModelConfig(model, model, model)
    training_config = HierarchicalTrainingConfig(
        seed=47, steps=1, batch_size=1, window_length=16, burn_in=3,
        stride=8, report_interval=1)
    scaler_sha256 = hashlib.sha256((
        scalers.minute.fingerprint + scalers.hour.fingerprint
        + scalers.day.fingerprint).encode()).hexdigest()
    tensor_sha256 = hierarchy_tensor_sha256(tensors)
    checkpoint = output_dir / "profile_checkpoint.npz"
    train_hierarchical(
        tensors=tensors, train_scope_id="synthetic-profile-train-only",
        dataset_id=replay.dataset_id, scaler_sha256=scaler_sha256,
        checkout=args.checkout, output_checkpoint=checkpoint,
        model_config=model_config, training_config=training_config)
    predictor = create_stateful_hierarchical_f2_predictor(
        tensors=tensors, train_scope_id="synthetic-profile-train-only",
        dataset_id=replay.dataset_id, scaler_sha256=scaler_sha256,
        checkout=args.checkout, checkpoint=checkpoint,
        model_config=model_config, training_config=training_config,
        training_tensor_sha256=tensor_sha256,
        inference_context_sha256=tensor_sha256, paths=args.paths,
        forecast_seed=811, advance_slow_priors=True)

    rng = np.random.default_rng(811)
    minute = np.zeros((args.paths, 5), dtype=np.float32)
    hour = np.zeros_like(minute)
    day = np.zeros_like(minute)
    durations: list[float] = []
    compiled_event_shapes: set[tuple[bool, bool]] = set()
    start = time.perf_counter()
    first_output: tuple[np.ndarray, np.ndarray] | None = None
    for step in range(args.profile_steps):
        if step:
            minute = rng.normal(0.0, 0.01, size=(args.paths, 5)).astype(np.float32)
            event_close = int(tensors.minute_available_at[-1]) + step * MINUTE_NS
            hour_event = event_close % (60 * MINUTE_NS) == 0
            day_event = event_close % (1_440 * MINUTE_NS) == 0
            hour = (rng.normal(0.0, 0.01, size=(args.paths, 5)).astype(np.float32)
                    if hour_event else np.zeros_like(minute))
            day = (rng.normal(0.0, 0.01, size=(args.paths, 5)).astype(np.float32)
                   if day_event else np.zeros_like(minute))
        else:
            hour_event = day_event = False
        shape = hour_event, day_event
        compiled_event_shapes.add(shape)
        step_start = time.perf_counter()
        mean, scale = predictor.predict_step(
            step=step, minute_feedback=minute, hour_feedback=hour,
            day_feedback=day, hour_event=hour_event, day_event=day_event)
        durations.append(time.perf_counter() - step_start)
        if step == 0:
            first_output = mean, scale
        if (not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0)):
            raise RuntimeError("stateful CPU profile produced an invalid distribution")
        elapsed = time.perf_counter() - start
        current_rss = _current_rss_mib()
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        if elapsed > args.max_seconds:
            raise TimeoutError("stateful CPU profile exceeded its wall-clock budget")
        if (math.isfinite(current_rss) and current_rss > args.max_rss_mib
                or peak_rss > args.max_rss_mib):
            raise MemoryError("stateful CPU profile exceeded its resident-memory budget")
    assert first_output is not None
    elapsed = time.perf_counter() - start
    steady = np.asarray(durations[1:], dtype=np.float64)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    result = {
        "status": "PASS",
        "scope": "synthetic_stateful_rollout_mechanism_only",
        "profile_steps": args.profile_steps,
        "paths": args.paths,
        "parameter_count": 3_208,
        "first_step_seconds_including_jit": durations[0],
        "steady_step_seconds_median": float(np.median(steady)),
        "steady_step_seconds_p95": float(np.quantile(steady, 0.95)),
        "steady_steps_per_second": float(1.0 / np.median(steady)),
        "rollout_seconds": elapsed,
        "peak_process_rss_mib": peak_rss,
        "current_rss_mib": _current_rss_mib(),
        "compiled_event_shapes": [list(shape) for shape in sorted(compiled_event_shapes)],
        "finite_heads": True,
        "budgets": {"max_seconds": args.max_seconds,
                    "max_rss_mib": args.max_rss_mib},
        "data_boundary": "Synthetic standardized feedback only; no market or evaluation data.",
    }
    target = output_dir / "result.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
