"""Run the repository-default 100-update B3 config on bounded synthetic data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from qa.assert_canonical_workspace import assert_canonical_workspace

    workspace = assert_canonical_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--days", type=int, choices=(30, 90, 360), default=360)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "cuda" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from qa.world_model.routeb_b3_cpu_profile import synthetic_bars
    from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
    from rles_world_model.routeb.jax_hierarchy_train import (
        HierarchicalModelConfig, HierarchicalTrainingConfig, train_hierarchical,
    )
    from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
    from rles_world_model.routeb.jax_train import SNAPSHOT_SHA
    from rles_world_model.routeb.market import MINUTE_NS, MarketSequence
    import jax

    checkout = Path(args.checkout).resolve()
    upstream = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if upstream != PINNED_UPSTREAM:
        raise ValueError("DreamerV3 checkout does not match the pinned source revision")
    devices = jax.devices()
    if args.device == "cuda" and not any(
            device.platform in {"cuda", "gpu"} for device in devices):
        raise RuntimeError("CUDA profile requested but JAX exposes no CUDA device")

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_relative_to(workspace) or output_dir.exists():
        raise ValueError("configured profile output must be new and inside the canonical workspace")
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "b3_configured_checkpoint.npz"
    replay = MarketSequence(
        synthetic_bars(args.days),
        f"synthetic-b3-default-training-{args.days}d-NOT-REAL-MARKET", "train",
    )
    tensors, scalers = build_train_hierarchy(
        replay, burn_in=8, score_start_ns=replay.candles[8].ts_close_ns,
        score_end_ns=replay.candles[-1].ts_close_ns + MINUTE_NS,
    )
    rssm = RSSMConfig()
    model = HierarchicalModelConfig(rssm, rssm, rssm)
    training = HierarchicalTrainingConfig()
    scaler_sha = hashlib.sha256((
        scalers.minute.fingerprint + scalers.hour.fingerprint + scalers.day.fingerprint
    ).encode("ascii")).hexdigest()
    started = time.perf_counter()
    reports = train_hierarchical(
        tensors=tensors, train_scope_id="synthetic-b3-configured-profile",
        dataset_id=replay.dataset_id, scaler_sha256=scaler_sha,
        checkout=str(checkout), output_checkpoint=checkpoint,
        model_config=model, training_config=training,
    )
    wall_seconds = time.perf_counter() - started
    final = reports[-1]
    report_steps = [int(report["step"]) for report in reports]
    observed_update_seconds = [float(report["update_seconds"]) for report in reports]
    if (final.get("optimizer_updates") != training.steps
            or not checkpoint.exists()
            or checkpoint.stat().st_size == 0):
        raise RuntimeError("configured B3 training did not complete and checkpoint")
    payload = {
        "schema": "routeb.b3-configured-synthetic-device-profile.v1",
        "status": "COMPLETED",
        "synthetic_only": True,
        "real_market_data_accessed": False,
        "validation_test_holdout_accessed": False,
        "economic_certification": False,
        "days": args.days,
        "rows": {
            "minute": int(tensors.minute_x.shape[1]),
            "hour": int(tensors.hour_x.shape[1]),
            "day": int(tensors.day_x.shape[1]),
        },
        "model_config": {
            name: getattr(rssm, name) for name in rssm.__dataclass_fields__
        },
        "model_parameter_count": int(final["parameter_count"]),
        "training_config": {
            name: getattr(training, name)
            for name in training.__dataclass_fields__
        },
        "optimizer_updates": int(final["optimizer_updates"]),
        "checkpoint_report_steps": report_steps,
        "observed_update_seconds_at_report_steps": observed_update_seconds,
        "training_seconds_reported": float(final["training_seconds"]),
        "process_wall_seconds": wall_seconds,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "device": str(devices[0]),
        "devices": [{
            "id": int(device.id), "platform": device.platform,
            "kind": device.device_kind,
        } for device in devices],
        "python_snapshot_sha256": SNAPSHOT_SHA,
        "dreamerv3_upstream_sha": PINNED_UPSTREAM,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "One deterministic synthetic run using the repository-default B3 training "
            "budget and dimensions. This is compute evidence, not convergence or "
            "market-quality evidence."
        ),
    }
    target = output_dir / "result.json"
    target.write_text(json.dumps(payload, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
