"""Synthetic CPU smoke for the Train-only R2 representation loss wiring."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from rles_world_model.routeb.codec import OHLCVCodec, TrainOnlyStandardizer
from rles_world_model.routeb.jax_model import RSSMConfig, make_model
from rles_world_model.routeb.jax_train import TrainingConfig, train
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS
from rles_world_model.routeb.sequences import export_train_sequences, make_windows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import ninjax as nj

    if not any(device.platform == "cpu" for device in jax.devices()):
        raise RuntimeError("R2 smoke requires the isolated CPU JAX runtime")
    config = RSSMConfig(deter=8, hidden=8, stoch=1, classes=2, blocks=1,
                        r2_regularizer_scale=0.01,
                        r2_regularizer_alpha=0.005,
                        r2_regularizer_interval=20)
    model = make_model(args.checkout, config)
    rng = np.random.default_rng(31)
    features = jnp.asarray(rng.normal(size=(2, 128, 5)).astype(np.float32))
    reset = jnp.zeros((2, 128), dtype=jnp.bool_).at[:, 0].set(True)
    mask = jnp.ones((2, 128), dtype=jnp.float32)
    params = nj.init(model)({}, features, reset, mask, True, seed=31)
    apply = nj.pure(model)
    _, diagnostics = apply(params, features, reset, mask, True, seed=41)
    baseline = make_model(args.checkout, replace(config, r2_regularizer_scale=0.0))
    _, baseline_diagnostics = nj.pure(baseline)(
        params, features, reset, mask, True, seed=41)
    high = make_model(args.checkout, replace(config, r2_regularizer_scale=0.05))
    _, high_diagnostics = nj.pure(high)(params, features, reset, mask, True, seed=41)
    loss_delta = float(diagnostics["loss"] - baseline_diagnostics["loss"])
    high_delta = float(high_diagnostics["loss"] - baseline_diagnostics["loss"])
    expected_delta = config.r2_regularizer_scale * float(
        diagnostics["r2_regularizer_loss"])
    gradients = jax.grad(lambda p: apply(
        p, features, reset, mask, True, seed=41)[1]["r2_regularizer_loss"])(params)
    grad_norm = float(jnp.sqrt(sum(jnp.square(x).sum()
                                   for x in jax.tree_util.tree_leaves(gradients))))
    if (not np.isfinite(loss_delta) or not np.isclose(loss_delta, expected_delta,
                                                      rtol=2e-4, atol=2e-5)
            or float(diagnostics["r2_valid_samples"]) != 12.0
            or float(diagnostics["r2_regularizer_loss"]) <= 0.0
            or not np.isfinite(grad_norm) or grad_norm <= 0.0
            or not np.isfinite(float(diagnostics["r2_condition_number"]))
            or not np.isclose(high_delta, 0.05 * float(
                high_diagnostics["r2_regularizer_loss"]), rtol=2e-4, atol=2e-5)):
        raise AssertionError("R2 model loss, mask, or gradient contract failed")

    bars = []
    close = 100.0
    for minute in range(1, 257):
        opening = close * math.exp(0.0002 * math.sin(minute / 9.0))
        close = opening * math.exp(0.0005 * math.cos(minute / 17.0))
        bars.append(Candle(minute * MINUTE_NS, minute * MINUTE_NS,
                           "R2-SYNTHETIC-PERP.SIM", opening,
                           max(opening, close) * 1.0002,
                           min(opening, close) * 0.9998, close, 10.0 + minute % 5))
    replay = MarketSequence(tuple(bars), "synthetic-r2-not-market-data", "train")
    coordinates = OHLCVCodec.encode(tuple(bars[1:]), previous_close=bars[0].close)
    scaler = TrainOnlyStandardizer.fit(
        coordinates, dataset_id=replay.dataset_id, scope="train")
    batch = make_windows(replay, scaler=scaler, length=32, burn_in=8, stride=16)
    output = Path(".tmp/routeb-r2-trainer-smoke").resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_npz = output / "train_only.npz"
    scope = "synthetic-r3-train-only"
    export_train_sequences(train_npz, batch, scaler,
                           snapshot_sha="80194a587444bf4a11f06f6e273723e5754f2483",
                           upstream_sha="e01491fad6434b2245a3b8ca201dd7faedcc458c",
                           train_scope_id=scope)
    train_config = replace(config, r2_regularizer_scale=0.01,
                           r2_regularizer_interval=4)
    reports = train(dataset_npz=train_npz, train_scope_id=scope,
                    checkout=args.checkout, output_checkpoint=output / "model.npz",
                    model_config=train_config,
                    training_config=TrainingConfig(seed=31, steps=2,
                                                   batch_size=2, report_interval=2))
    if (not reports or reports[-1].get("r2_batch_count") != 2
            or not np.isfinite(reports[-1].get("r2_batch_loss_cv", float("nan")))
            or reports[-1].get("r2_valid_samples", 0) <= 0):
        raise AssertionError("R2 optimizer training/reporting contract failed")
    print(json.dumps({
        "status": "PASS_SYNTHETIC_ENGINEERING_ONLY",
        "device": str(jax.devices()[0]),
        "r2_scale": config.r2_regularizer_scale,
        "r2_alpha": config.r2_regularizer_alpha,
        "r2_interval": config.r2_regularizer_interval,
        "valid_samples": float(diagnostics["r2_valid_samples"]),
        "r2_loss": float(diagnostics["r2_regularizer_loss"]),
        "r2_loss_delta_matches_weight": bool(np.isclose(
            loss_delta, expected_delta, rtol=2e-4, atol=2e-5)),
        "r2_high_weight_delta_matches_weight": bool(np.isclose(
            high_delta, 0.05 * float(high_diagnostics["r2_regularizer_loss"]),
            rtol=2e-4, atol=2e-5)),
        "r2_gradient_norm": grad_norm,
        "r2_condition_number": float(diagnostics["r2_condition_number"]),
        "optimizer_updates": reports[-1]["step"],
        "r2_batch_loss_cv": reports[-1]["r2_batch_loss_cv"],
        "r2_optimizer_training_finite": bool(np.isfinite(reports[-1]["loss"])),
        "market_data_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
