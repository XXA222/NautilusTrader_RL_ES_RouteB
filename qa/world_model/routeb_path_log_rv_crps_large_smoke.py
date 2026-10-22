"""One-update full-size synthetic CUDA smoke for Train-only path log-RV CRPS."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

from rles_world_model.routeb.codec import OHLCVCodec, TrainOnlyStandardizer
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import SNAPSHOT_SHA, TrainingConfig, train
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS
from rles_world_model.routeb.sequences import export_train_sequences, make_windows


def main() -> int:
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    price = 20.0
    bars = []
    for index in range(1, 481):
        previous = price
        opening = previous * math.exp(0.00003 * math.sin(index / 7))
        price = opening * math.exp(0.0008 * math.sin(index / 17))
        high = max(opening, price) * math.exp(0.0002 + 0.0001 * math.sin(index / 9) ** 2)
        low = min(opening, price) * math.exp(-0.0002 - 0.0001 * math.cos(index / 13) ** 2)
        bars.append(Candle(index * MINUTE_NS, index * MINUTE_NS,
                           "SYNTHETIC-PERP.SIM", opening, high, low, price,
                           100.0 + index % 31))
    replay = MarketSequence(tuple(bars), "synthetic-path-crps-large", "train")
    coordinates = OHLCVCodec.encode(replay.candles[1:],
                                    previous_close=replay.candles[0].close)
    scaler = TrainOnlyStandardizer.fit(coordinates, dataset_id=replay.dataset_id,
                                       scope="train")
    batch = make_windows(replay, scaler=scaler, length=128, burn_in=32, stride=32)
    if batch.features.shape[0] < 8:
        raise AssertionError("large smoke did not produce a full training batch")
    scope = "train-SYNTHETIC-PATH-CRPS-LARGE-SMOKE"
    dataset = output / "train_sequences.npz"
    export_train_sequences(dataset, batch, scaler, snapshot_sha=SNAPSHOT_SHA,
                           upstream_sha=PINNED_UPSTREAM, train_scope_id=scope)
    model = RSSMConfig(forecast_objective="path_log_rv_crps",
                       path_log_rv_crps_horizon=15,
                       path_log_rv_crps_horizons=(15,30,60),
                       path_log_rv_crps_paths=4)
    reports = train(dataset_npz=dataset, train_scope_id=scope,
                    checkout=os.environ["ROUTEB_DREAMERV3_CHECKOUT"],
                    output_checkpoint=output / "checkpoint.npz",
                    model_config=model,
                    training_config=TrainingConfig(steps=1, batch_size=8,
                                                   seed=31, report_interval=1))
    final = reports[-1]
    devices = [str(device) for device in jax.devices()]
    if (not math.isfinite(final["path_log_rv_crps"])
            or final["valid_path_log_rv_crps"] <= 0
            or final["device"] != "cuda:0"):
        raise AssertionError(f"full-size CUDA path CRPS smoke failed: {final}")
    report = {
        "status": "PASS",
        "scope": "synthetic_mechanism_only",
        "device": final["device"],
        "devices": devices,
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "model": {"deter": model.deter, "hidden": model.hidden,
                  "stoch": model.stoch, "classes": model.classes,
                  "paths": model.path_log_rv_crps_paths,
                  "horizon": model.path_log_rv_crps_horizon,
                  "horizons": list(model.path_log_rv_crps_horizons)},
        "train_sequences": int(batch.features.shape[0]),
        "path_log_rv_crps": final["path_log_rv_crps"],
        "valid_path_log_rv_crps": final["valid_path_log_rv_crps"],
        "optimizer_updates": final["optimizer_updates"],
        "training_seconds": final["training_seconds"],
        "device_memory_stats": jax.devices()[0].memory_stats(),
        "market_data_read": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, default=lambda value: int(value)) + "\n",
        encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2, default=lambda value: int(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
