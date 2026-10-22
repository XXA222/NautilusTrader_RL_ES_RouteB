"""Synthetic E1 GRU train/resume/forecast smoke in the locked Python 3.12 env."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
import os
import sys
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

from rles_world_model.routeb.codec import OHLCVCodec, TrainOnlyStandardizer
from rles_world_model.routeb.jax_gru import GRUConfig
from rles_world_model.routeb.jax_gru_train import forecast_gru, train_gru
from rles_world_model.routeb.jax_gru_train import forecast_gru_distribution
from rles_world_model.routeb.evaluation import diagonal_gaussian_return_nll
from rles_world_model.routeb.jax_train import TrainingConfig, SNAPSHOT_SHA, PINNED_UPSTREAM
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS
from rles_world_model.routeb.sequences import export_train_sequences, make_windows
from rles_world_model.routeb.integration import prepare_forecast_context
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig


def synthetic_bars(count: int = 480) -> tuple[Candle, ...]:
    price = 2000.0
    result = []
    for i in range(1, count + 1):
        opening = price * math.exp(0.0001 * math.sin(i / 11))
        price = opening * math.exp(0.0008 * math.sin(i / 17))
        high = max(opening, price) * math.exp(0.0003 + 0.0001 * math.sin(i / 7) ** 2)
        low = min(opening, price) * math.exp(-0.0003 - 0.0001 * math.cos(i / 9) ** 2)
        result.append(Candle(i * MINUTE_NS, i * MINUTE_NS, "ETHUSDT-PERP.SIM",
                             opening, high, low, price, float(90 + i % 29)))
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkout", required=True,
                        help="pinned DreamerV3 checkout used by the shared worker contract")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bars = synthetic_bars()
    replay = MarketSequence(bars, "synthetic-E1-NOT-REAL-MARKET", "train")
    coordinates = OHLCVCodec.encode(bars[1:], previous_close=bars[0].close)
    scaler = TrainOnlyStandardizer.fit(
        coordinates, dataset_id=replay.dataset_id, scope="train"
    )
    windows = make_windows(replay, scaler=scaler, length=32, burn_in=8, stride=16)
    scope = "train-SYNTHETIC-E1-MECHANISM-ONLY"
    dataset = root / "train_sequences.npz"
    export_train_sequences(
        dataset, windows, scaler, snapshot_sha=SNAPSHOT_SHA,
        upstream_sha=PINNED_UPSTREAM, train_scope_id=scope,
    )
    model = GRUConfig(hidden=32)
    two_steps = TrainingConfig(steps=2, batch_size=4, seed=41, report_interval=1)
    four_steps = TrainingConfig(steps=4, batch_size=4, seed=41, report_interval=2)
    resumed = root / "e1_resumed.npz"
    uninterrupted = root / "e1_uninterrupted.npz"
    first = train_gru(dataset_npz=dataset, train_scope_id=scope,
                      output_checkpoint=resumed, model_config=model,
                      training_config=two_steps)
    second = train_gru(dataset_npz=dataset, train_scope_id=scope,
                       output_checkpoint=resumed, model_config=model,
                       training_config=four_steps, resume=True)
    full = train_gru(dataset_npz=dataset, train_scope_id=scope,
                     output_checkpoint=uninterrupted, model_config=model,
                     training_config=four_steps)
    with np.load(resumed, allow_pickle=False) as resumed_file, \
            np.load(uninterrupted, allow_pickle=False) as full_file:
        if set(resumed_file.files) != set(full_file.files) or not all(
            np.array_equal(resumed_file[name], full_file[name])
            for name in resumed_file.files
        ):
            raise AssertionError("E1 resume differs from uninterrupted optimizer state")
    history = windows.features[:3]
    forecast = forecast_gru(
        checkpoint=resumed,
        model_config=model,
        training_config=four_steps,
        train_scope_id=scope,
        dataset_id=replay.dataset_id,
        train_npz_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,
        history=history,
        steps=8,
        forecast_seed=901,
    )
    if forecast.shape != (3, 8, 5) or not np.isfinite(forecast).all():
        raise AssertionError("E1 forecast shape or values invalid")
    if np.array_equal(forecast[0], forecast[1]):
        raise AssertionError("independent E1 forecast paths did not diverge")
    mean, scale = forecast_gru_distribution(
        checkpoint=resumed,
        model_config=model,
        training_config=four_steps,
        train_scope_id=scope,
        dataset_id=replay.dataset_id,
        train_npz_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,
        history=history[:, :-1],
    )
    one_step_nll = diagonal_gaussian_return_nll(
        mean, scale, history[:, -1], scaler_mean=scaler.mean, scaler_std=scaler.std)
    if mean.shape != (3, 5) or scale.shape != (3, 5) or not np.isfinite(one_step_nll).all():
        raise AssertionError("E1 one-step distribution or exact return NLL is invalid")
    extended = synthetic_bars(481)
    request_path = root / "worker_request.json"
    prepare_forecast_context(
        visible_history=extended[:-1], frozen_scaler=scaler,
        anchor_ns=extended[-2].ts_close_ns, train_scope_id=scope,
        train_npz_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        output_context_npz=root / "worker_context.npz",
        output_request_json=request_path, horizon=1, n_paths=2, seed=1201,
    )
    configuration_path = root / "worker_configuration.json"
    configuration_path.write_text(json.dumps({
        "model": asdict(model), "training": asdict(four_steps),
    }), encoding="utf-8")
    worker = WorkerClient(WorkerConfig(Path(sys.executable),Path(args.checkout).resolve()))
    worker_checkpoint = root / "e1_worker_checkpoint.npz"
    worker_training = worker.train_e1(
        sequences=dataset, scope=scope, checkpoint=worker_checkpoint,
        configuration=configuration_path,
    )
    worker_forecast_path = root / "e1_worker_forecast.npz"
    worker_forecast_result = worker.forecast_e1(
        request=request_path, checkpoint=worker_checkpoint,
        output=worker_forecast_path, configuration=configuration_path,
    )
    worker_distribution_path = root / "e1_worker_distribution.npz"
    worker_distribution_result = worker.next_distribution_e1(
        request=request_path, checkpoint=worker_checkpoint,
        output=worker_distribution_path, configuration=configuration_path,
    )
    with np.load(worker_forecast_path, allow_pickle=False) as worker_artifact:
        worker_paths = np.asarray(worker_artifact["bars"])
    with np.load(worker_distribution_path, allow_pickle=False) as worker_artifact:
        worker_mean = np.asarray(worker_artifact["standardized_mean"])
        worker_scale = np.asarray(worker_artifact["standardized_scale"])
    future_coordinates = OHLCVCodec.encode(
        (extended[-1],), previous_close=extended[-2].close,
    )
    worker_nll = diagonal_gaussian_return_nll(
        worker_mean, worker_scale, scaler.transform(future_coordinates),
        scaler_mean=scaler.mean, scaler_std=scaler.std,
    )
    if (worker_training.get("kind") != "E1_TRAIN_DIAGNOSTIC_ONLY"
            or worker_forecast_result.get("kind") != "E1_PRIOR_PATHS"
            or worker_distribution_result.get("kind") != "E1_NEXT_DISTRIBUTION"
            or worker_paths.shape != (2, 1, 5)
            or worker_mean.shape != (1, 5) or worker_scale.shape != (1, 5)
            or not np.isfinite(worker_nll).all()):
        raise AssertionError("E1 subprocess training/forecast IPC contract failed")
    result = {
        "status": "PASS",
        "scope": "synthetic_mechanism_only",
        "optimizer_updates": len(first) + len(second),
        "final_step": full[-1]["step"],
        "resume_parity": True,
        "forecast_shape": list(forecast.shape),
        "one_step_distribution_shape": list(mean.shape),
        "one_step_return_nll_finite": True,
        "independent_paths": int(forecast.shape[0]),
        "worker_forecast_shape": list(worker_paths.shape),
        "worker_one_step_return_nll_finite": True,
        "real_market_validation": False,
        "economic_claim": False,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
