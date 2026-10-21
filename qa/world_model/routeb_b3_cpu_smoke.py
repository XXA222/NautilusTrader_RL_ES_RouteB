"""Synthetic end-to-end gradient/resume smoke for the experimental B3 trainer.

Uses only an in-memory three-day Train sequence. It makes no exchange-data,
Validation/Test/Holdout, account, or credential access and is not skill evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchyTrainingConfig, train_hierarchical,
)
from rles_world_model.routeb.jax_model import RSSMConfig
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS


def synthetic_train_bars(count: int = 6_000) -> tuple[Candle, ...]:
    price = 2_000.0
    result = []
    for minute in range(1, count + 1):
        opening = price * math.exp(0.00003 * math.sin(minute / 13.0))
        price = opening * math.exp(0.0005 * math.sin(minute / 37.0))
        high = max(opening, price) * (1.0002 + 0.00005 * math.sin(minute / 17.0) ** 2)
        low = min(opening, price) * (0.9998 - 0.00005 * math.cos(minute / 19.0) ** 2)
        result.append(Candle(minute * MINUTE_NS, minute * MINUTE_NS,
                             "ETHUSDT-PERP.SIM", opening, high, low, price,
                             100.0 + minute % 23))
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-platform", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    import jax
    platforms = {device.platform for device in jax.devices()}
    required = {"cpu"} if args.expected_platform == "cpu" else {"gpu", "cuda"}
    if not platforms.intersection(required):
        raise RuntimeError(f"required {args.expected_platform} device missing: {sorted(platforms)}")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    replay = MarketSequence(synthetic_train_bars(),
                            "synthetic-b3-smoke-not-market-data", "train")
    tensors, _ = build_train_hierarchy(replay, burn_in=32)
    model = RSSMConfig(deter=8, hidden=8, stoch=1, classes=2, blocks=1)
    cfg_one = HierarchyTrainingConfig(seed=31, steps=1, window_length=256,
                                      hour_context_steps=16, day_context_steps=8,
                                      report_interval=1)
    cfg_two = HierarchyTrainingConfig(seed=31, steps=2, window_length=256,
                                      hour_context_steps=16, day_context_steps=8,
                                      report_interval=1)
    resumed_path = output / "resumed.npz"
    direct_path = output / "direct.npz"
    initial = train_hierarchical(
        tensors=tensors, train_scope_id="train-synthetic-only",
        dataset_id=replay.dataset_id, checkout=args.checkout,
        output_checkpoint=resumed_path, minute_config=model,
        hour_config=model, day_config=model, training_config=cfg_one)
    continuation = train_hierarchical(
        tensors=tensors, train_scope_id="train-synthetic-only",
        dataset_id=replay.dataset_id, checkout=args.checkout,
        output_checkpoint=resumed_path, minute_config=model,
        hour_config=model, day_config=model, training_config=cfg_two, resume=True)
    train_hierarchical(
        tensors=tensors, train_scope_id="train-synthetic-only",
        dataset_id=replay.dataset_id, checkout=args.checkout,
        output_checkpoint=direct_path, minute_config=model,
        hour_config=model, day_config=model, training_config=cfg_two)
    if initial[-1]["step"] != 1 or continuation[-1]["step"] != 2:
        raise AssertionError("hierarchical checkpoint did not resume at the next update")
    if (not np.isfinite(continuation[-1]["loss"])
            or continuation[-1]["grad_norm"] <= 0
            or continuation[-1]["parameters"] <= 0):
        raise AssertionError("hierarchical smoke did not produce a finite nonzero gradient")
    with np.load(resumed_path, allow_pickle=False) as resumed, \
            np.load(direct_path, allow_pickle=False) as uninterrupted:
        if set(resumed.files) != set(uninterrupted.files) or not all(
                np.array_equal(resumed[name], uninterrupted[name]) for name in resumed.files):
            raise AssertionError("resumed and uninterrupted B3 training states differ")
    report = {
        "status": "PASS_SYNTHETIC_ENGINEERING_ONLY",
        "device": str(jax.devices()[0]),
        "minute_steps": int(tensors.minute_x.shape[1]),
        "hour_steps": int(tensors.hour_x.shape[1]),
        "day_steps": int(tensors.day_x.shape[1]),
        "optimizer_updates": 2,
        "window_length": cfg_two.window_length,
        "eligible_window_count": continuation[-1]["eligible_window_count"],
        "window_start_sha256": continuation[-1]["window_start_sha256"],
        "loss": continuation[-1]["loss"],
        "grad_norm": continuation[-1]["grad_norm"],
        "parameters": continuation[-1]["parameters"],
        "train_tensor_sha256": continuation[-1]["train_tensor_sha256"],
        "test_or_holdout_accessed": False,
        "predictive_or_economic_claim": False,
    }
    (output / "smoke_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
