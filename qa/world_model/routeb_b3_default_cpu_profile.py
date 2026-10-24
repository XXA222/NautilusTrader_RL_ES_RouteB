"""Bounded multi-update synthetic device profile with the repository B3 defaults."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from qa.assert_canonical_workspace import assert_canonical_workspace
    from qa.world_model.routeb_b3_cpu_profile import synthetic_bars

    root = assert_canonical_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--days", type=int, choices=(30, 90, 360), default=30)
    parser.add_argument("--updates", type=int, choices=(1, 3, 5), default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "cuda" else "cpu"
    from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
    from rles_world_model.routeb.jax_hierarchy_train import (
        HierarchicalModelConfig, HierarchicalTrainingConfig, train_hierarchical,
    )
    from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
    from rles_world_model.routeb.jax_train import SNAPSHOT_SHA
    from rles_world_model.routeb.market import MINUTE_NS, MarketSequence
    import jax

    devices = jax.devices()
    if args.device == "cuda" and not any(
            device.platform in {"cuda", "gpu"} for device in devices):
        raise RuntimeError("CUDA profile requested but JAX exposes no CUDA device")

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("profile outputs must remain inside the canonical checkout")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.updates == 1 else f"_{args.updates}updates"
    output = output_dir / f"b3_default_{args.device}_profile_{args.days}d{suffix}.json"
    checkpoint = output_dir / f"b3_default_{args.device}_profile_{args.days}d{suffix}_checkpoint.npz"
    if output.exists() or checkpoint.exists():
        raise FileExistsError("profile output already exists; choose a fresh output directory")

    replay = MarketSequence(synthetic_bars(args.days),
                            f"synthetic-b3-default-profile-{args.days}d-NOT-REAL-MARKET", "train")
    tensors, scalers = build_train_hierarchy(
        replay, burn_in=8, score_start_ns=replay.candles[8].ts_close_ns,
        score_end_ns=replay.candles[-1].ts_close_ns + MINUTE_NS)
    default = RSSMConfig()
    model = HierarchicalModelConfig(default, default, default)
    training = HierarchicalTrainingConfig(
        seed=47, steps=args.updates, batch_size=1, window_length=16,
        burn_in=3, stride=8, report_interval=1)
    scaler_sha = hashlib.sha256((
        scalers.minute.fingerprint + scalers.hour.fingerprint
        + scalers.day.fingerprint).encode("ascii")).hexdigest()
    started = time.perf_counter()
    reports = train_hierarchical(
        tensors=tensors, train_scope_id="synthetic-b3-default-profile-30d",
        dataset_id=replay.dataset_id, scaler_sha256=scaler_sha,
        checkout=args.checkout, output_checkpoint=checkpoint,
        model_config=model, training_config=training)
    elapsed = time.perf_counter() - started
    final = reports[-1]
    update_seconds = [float(report["update_seconds"]) for report in reports]
    if len(update_seconds) != args.updates or any(value <= 0 for value in update_seconds):
        raise RuntimeError("per-update training timings are missing or invalid")
    steady = np.asarray(update_seconds[1:], dtype=np.float64)
    payload = {
        "schema": "routeb.b3-synthetic-default-device-profile.v3",
        "status": "COMPLETED",
        "synthetic_only": True,
        "real_market_data_accessed": False,
        "validation_test_holdout_accessed": False,
        "economic_certification": False,
        "days": args.days,
        "minute_rows": int(tensors.minute_x.shape[1]),
        "hour_rows": int(tensors.hour_x.shape[1]),
        "day_rows": int(tensors.day_x.shape[1]),
        "model_config": default.__dict__ if hasattr(default, "__dict__") else {
            name: getattr(default, name) for name in default.__dataclass_fields__},
        "model_parameter_count": final["parameter_count"],
        "optimizer_updates": final["optimizer_updates"],
        "updates_requested": args.updates,
        "devices": [{
            "id": int(device.id), "platform": device.platform,
            "kind": device.device_kind,
        } for device in devices],
        "update_seconds": update_seconds,
        "first_update_seconds_including_any_jit": update_seconds[0],
        "steady_update_seconds": ({
            "count": int(steady.size), "mean": float(np.mean(steady)),
            "median": float(np.median(steady)),
            "p95": float(np.quantile(steady, 0.95)),
        } if steady.size else None),
        "train_call_seconds_including_first_update_jit": elapsed,
        "training_seconds_reported": final["training_seconds"],
        "peak_rss_kib": getattr(resource, "getrusage")(
            getattr(resource, "RUSAGE_SELF")).ru_maxrss,
        "device": final["device"],
        "python_snapshot_sha256": SNAPSHOT_SHA,
        "dreamerv3_upstream_sha": PINNED_UPSTREAM,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            f"Bounded deterministic synthetic {final['device']} updates using "
            "repository-default RSSM dimensions; first update may include JIT compile. "
            "Not convergence, production capacity, or market-quality evidence."
        ),
    }
    output.write_text(json.dumps(payload, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
