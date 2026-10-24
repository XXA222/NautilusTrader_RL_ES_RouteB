"""Bounded synthetic-only B3 CPU profile for audit M4/RB-09.

This is a resource probe, not a training-quality or real-market benchmark. Run
one duration per fresh process so peak RSS is attributable to that duration.
All outputs must stay under the canonical RouteB checkout or system temp.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchicalModelConfig, HierarchicalTrainingConfig, train_hierarchical,
)
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import SNAPSHOT_SHA
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS


def synthetic_bars(days: int) -> tuple[Candle, ...]:
    count = days * 24 * 60
    price = 2_000.0
    bars: list[Candle] = []
    for index in range(1, count + 1):
        opening = price * math.exp(0.00002 * math.sin(index / 5))
        price = opening * math.exp(0.0006 * math.sin(index / 13))
        high = max(opening, price) * math.exp(
            0.0002 + 0.0001 * math.sin(index / 7) ** 2)
        low = min(opening, price) * math.exp(
            -0.0002 - 0.0001 * math.cos(index / 11) ** 2)
        bars.append(Candle(index * MINUTE_NS, index * MINUTE_NS,
                           "SYNTHETIC.PROFILE", opening, high, low, price,
                           float(100 + index % 30)))
    return tuple(bars)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from qa.assert_canonical_workspace import assert_canonical_workspace

    root = assert_canonical_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--days", type=int, choices=(30, 90, 360), required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("profile outputs must remain inside the canonical checkout")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"b3_cpu_profile_{args.days}d.json"
    checkpoint = output_dir / f"b3_cpu_profile_{args.days}d_checkpoint.npz"
    if output.exists() or checkpoint.exists():
        raise FileExistsError("profile output already exists; choose a fresh output directory")

    replay = MarketSequence(synthetic_bars(args.days),
                            f"synthetic-b3-profile-{args.days}d-NOT-REAL-MARKET", "train")
    score_start_ns = replay.candles[8].ts_close_ns
    score_end_ns = replay.candles[-1].ts_close_ns + MINUTE_NS
    prepare_started = time.perf_counter()
    tensors, scalers = build_train_hierarchy(
        replay, burn_in=8, score_start_ns=score_start_ns,
        score_end_ns=score_end_ns)
    prepare_seconds = time.perf_counter() - prepare_started

    small = RSSMConfig(deter=8, hidden=8, stoch=1, classes=2, blocks=1)
    config = HierarchicalModelConfig(small, small, small)
    training = HierarchicalTrainingConfig(
        seed=47, steps=1, batch_size=1, window_length=16,
        burn_in=3, stride=8, report_interval=1)
    scaler_sha = hashlib.sha256((
        scalers.minute.fingerprint + scalers.hour.fingerprint
        + scalers.day.fingerprint).encode("ascii")).hexdigest()
    train_started = time.perf_counter()
    reports = train_hierarchical(
        tensors=tensors, train_scope_id=f"synthetic-b3-profile-{args.days}d",
        dataset_id=replay.dataset_id, scaler_sha256=scaler_sha,
        checkout=args.checkout, output_checkpoint=checkpoint,
        model_config=config, training_config=training)
    total_train_call_seconds = time.perf_counter() - train_started
    final = reports[-1]
    getrusage = getattr(resource, "getrusage")
    peak_rss = getrusage(getattr(resource, "RUSAGE_SELF")).ru_maxrss
    payload = {
        "schema": "routeb.b3-synthetic-cpu-profile.v1",
        "status": "COMPLETED",
        "synthetic_only": True,
        "real_market_data_accessed": False,
        "validation_test_holdout_accessed": False,
        "economic_certification": False,
        "days": args.days,
        "minute_rows": int(tensors.minute_x.shape[1]),
        "hour_rows": int(tensors.hour_x.shape[1]),
        "day_rows": int(tensors.day_x.shape[1]),
        "window_length": training.window_length,
        "batch_size": training.batch_size,
        "optimizer_updates": final["optimizer_updates"],
        "training_seconds_reported": final["training_seconds"],
        "train_call_seconds_including_jit": total_train_call_seconds,
        "prepare_seconds": prepare_seconds,
        "peak_rss_platform_units": peak_rss,
        "peak_rss_units": "KiB on Linux",
        "device": final["device"],
        "model_parameter_count": final["parameter_count"],
        "python_snapshot_sha256": SNAPSHOT_SHA,
        "dreamerv3_upstream_sha": PINNED_UPSTREAM,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "One optimizer update with a tiny model on deterministic synthetic data; "
            "not a convergence, GPU, throughput-at-production, or market-quality claim."
        ),
    }
    output.write_text(json.dumps(payload, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
