"""Synthetic-only B2/E2 CUDA smoke for diagnosing isolated JAX worker crashes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rles_world_model.routeb.codec import OHLCVCodec, TrainOnlyStandardizer
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import SNAPSHOT_SHA, TrainingConfig, train
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS
from rles_world_model.routeb.sequences import export_train_sequences, make_windows


def _bars(count: int = 8_000) -> tuple[Candle, ...]:
    close = 100.0
    bars = []
    for minute in range(1, count + 1):
        opening = close * math.exp(0.00004 * math.sin(minute / 19.0))
        close = opening * math.exp(0.0008 * math.sin(minute / 43.0))
        high = max(opening, close) * (1.0003 + 0.00005 * math.sin(minute / 23.0) ** 2)
        low = min(opening, close) * (0.9997 - 0.00005 * math.cos(minute / 29.0) ** 2)
        timestamp = minute * MINUTE_NS
        bars.append(Candle(timestamp, timestamp, "SYNTHETIC-PERP", opening,
                           high, low, close, 50.0 + minute % 17))
    return tuple(bars)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-platform", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--feature-scale", type=float, default=1.0,
                        help="multiply synthetic standardized features for numeric stress tests")
    args = parser.parse_args()
    if not math.isfinite(args.feature_scale) or args.feature_scale <= 0:
        parser.error("--feature-scale must be a finite positive number")
    import jax

    devices = tuple(str(device) for device in jax.devices())
    if not any(args.expected_platform in device.lower() for device in devices):
        raise RuntimeError(f"expected {args.expected_platform} JAX device; found {devices}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sequence = MarketSequence(_bars(), "synthetic-e2-cuda-smoke-only", "train")
    encoded = OHLCVCodec.encode(sequence.candles[1:],
                                previous_close=sequence.candles[0].close)
    scaler = TrainOnlyStandardizer.fit(
        encoded, dataset_id=sequence.dataset_id, scope="train")
    batch = make_windows(sequence, scaler=scaler, length=args.window_length,
                         burn_in=32, stride=max(64, args.window_length // 2))
    # This intentionally breaks the fitted-scaler relationship and must only
    # be used with synthetic inputs to probe model numeric stability.
    if args.feature_scale != 1.0:
        from dataclasses import replace
        batch = replace(batch, features=batch.features * args.feature_scale)
    scope = "synthetic-e2-cuda-smoke-only"
    npz = output / "train.npz"
    export_train_sequences(npz, batch, scaler, snapshot_sha=SNAPSHOT_SHA,
                           upstream_sha=PINNED_UPSTREAM, train_scope_id=scope,
                           fold_sha256="c" * 64)
    reports = train(
        dataset_npz=npz, train_scope_id=scope, checkout=args.checkout,
        output_checkpoint=output / "checkpoint.npz", model_config=RSSMConfig(),
        training_config=TrainingConfig(seed=args.seed, steps=args.steps, batch_size=1,
                                       lr=3e-4, grad_clip=100.0,
                                       report_interval=50))
    final = reports[-1]
    final_loss = final.get("loss")
    final_grad_norm = final.get("grad_norm")
    if (final.get("optimizer_updates") != args.steps
            or not isinstance(final_loss, (int, float)) or not math.isfinite(final_loss)
            or not isinstance(final_grad_norm, (int, float))
            or not math.isfinite(final_grad_norm) or final_grad_norm <= 0):
        raise RuntimeError("synthetic E2 training did not complete with a finite gradient")
    result = {"status": "PASS_SYNTHETIC_ENGINEERING_ONLY", "devices": devices,
              "steps": final["optimizer_updates"], "seed": args.seed, "loss": final_loss,
              "gradient_norm": final_grad_norm, "device": final["device"],
              "window_length": args.window_length,
              "synthetic_feature_scale": args.feature_scale,
              "market_data_used": False, "test_data_used": False}
    (output / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
