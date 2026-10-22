"""Drive the host train CLI through both isolated workers on synthetic data.

Run in the workspace Python 3.14 environment with the locked Python 3.12 JAX
worker and pinned DreamerV3 checkout. This is mechanism evidence only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from rles_world_model.cli import main as world_model_main
from rles_world_model.routeb.codec import OHLCVCodec, TrainOnlyStandardizer
from rles_world_model.routeb.jax_gru import GRUConfig
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import SNAPSHOT_SHA, TrainingConfig
from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS
from rles_world_model.routeb.sequences import export_train_sequences, make_windows
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_world_model.routeb.fold_experiment import run_fold_experiment
from rles_data.core import FoldManifest, Window


def synthetic_bars(count: int = 256) -> tuple[Candle, ...]:
    close = 2000.0
    bars = []
    for index in range(1, count + 1):
        opening = close * math.exp(0.0001 * math.sin(index / 9))
        close = opening * math.exp(0.0005 * math.sin(index / 13))
        high = max(opening, close) * math.exp(0.0003)
        low = min(opening, close) * math.exp(-0.0003)
        bars.append(Candle(index * MINUTE_NS, index * MINUTE_NS,
                           "ETHUSDT-PERP.SIM", opening, high, low, close,
                           float(80 + index % 17)))
    return tuple(bars)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    bars = synthetic_bars()
    market = MarketSequence(bars, "synthetic-routeb-cli-smoke", "train")
    coordinates = OHLCVCodec.encode(bars[1:], previous_close=bars[0].close)
    scaler = TrainOnlyStandardizer.fit(
        coordinates, dataset_id=market.dataset_id, scope="train",
    )
    batch = make_windows(market, scaler=scaler, length=16, burn_in=4, stride=8)
    scope = "routeb-fold-SYNTHETIC-CLI-SMOKE"
    fold_hash = "a" * 64
    sequences = root / "train_sequences.npz"
    export_train_sequences(
        sequences, batch, scaler, snapshot_sha=SNAPSHOT_SHA,
        upstream_sha=PINNED_UPSTREAM, train_scope_id=scope, fold_sha256=fold_hash,
    )

    training = TrainingConfig(steps=2, batch_size=2, seed=73, report_interval=1)
    candidates = {
        "e1": asdict(GRUConfig(hidden=16)),
        "e2": asdict(RSSMConfig(deter=32, hidden=32, stoch=2, classes=4, blocks=4)),
    }
    reports: dict[str, dict] = {}
    for name, model_config in candidates.items():
        configuration = root / f"{name}_configuration.json"
        configuration.write_text(json.dumps({
            "model": model_config, "training": asdict(training),
        }), encoding="utf-8")
        report = root / f"{name}_train_report.json"
        checkpoint = root / f"{name}_checkpoint.npz"
        status = world_model_main([
            "train-prepared", "--candidate", name,
            "--sequences", str(sequences), "--scope", scope,
            "--fold-sha256", fold_hash,
            "--configuration", str(configuration),
            "--checkpoint", str(checkpoint),
            "--worker-python", str(Path(args.worker_python).resolve()),
            "--upstream-checkout", str(Path(args.checkout).resolve()),
            "--report-json", str(report),
        ])
        if status != 0:
            raise RuntimeError(f"train-prepared failed for {name}")
        reports[name] = json.loads(report.read_text(encoding="utf-8"))

    left, right = reports["e1"], reports["e2"]
    shared = ("fold_hash", "train_scope_id", "train_npz_sha256", "train_examples",
              "window_length", "training_config")
    if any(left[key] != right[key] for key in shared):
        raise AssertionError("E1/E2 smoke candidates did not share identical Train budget and artifact")
    if (left["worker_result"]["kind"] != "E1_TRAIN_DIAGNOSTIC_ONLY"
            or right["worker_result"]["kind"] != "TRAIN_DIAGNOSTIC_ONLY"):
        raise AssertionError("worker returned an unexpected candidate type")

    # Exercise the real batched forecast IPC with the newly trained checkpoints.
    # The two anchors and contexts below are synthetic mechanism inputs only.
    contexts=[]; close_times=[]; availability=[]; anchor_closes=[]
    anchor_times=[]
    for end in (len(bars)-2,len(bars)-1):
        history_bars=bars[end-31:end+1]
        encoded=OHLCVCodec.encode(history_bars[1:],previous_close=history_bars[0].close)
        contexts.append(scaler.transform(encoded))
        close_times.append([bar.ts_close_ns for bar in history_bars[1:]])
        availability.append([bar.available_at_ns for bar in history_bars[1:]])
        anchor_times.append(history_bars[-1].ts_close_ns)
        anchor_closes.append(history_bars[-1].close)
    batch_context=root/"synthetic_forecast_batch_context.npz"
    np.savez_compressed(batch_context,histories=np.asarray(contexts,dtype=np.float32),
        scaler_mean=scaler.mean,scaler_std=scaler.std,
        anchor_timestamps_ns=np.asarray(anchor_times,dtype=np.int64),
        close_timestamps=np.asarray(close_times,dtype=np.int64),
        available_timestamps=np.asarray(availability,dtype=np.int64),
        anchor_closes=np.asarray(anchor_closes,dtype=np.float64))
    batch_request={
        "schema":"routeb.forecast-batch-request.v1","train_scope_id":scope,
        "dataset_id":market.dataset_id,"history_npz":str(batch_context.resolve()),
        "context_sha256":hashlib.sha256(batch_context.read_bytes()).hexdigest(),
        "scaler_sha256":scaler.fingerprint,"horizon":3,"n_paths":2,"seed":17,
        "train_npz_sha256":left["train_npz_sha256"],
        "instrument":bars[-1].instrument,"anchor_timestamps_ns":anchor_times,
    }
    batch_request_path=root/"synthetic_forecast_batch_request.json"
    batch_request_path.write_text(json.dumps(batch_request),encoding="utf-8")
    worker=WorkerClient(WorkerConfig(Path(args.worker_python).resolve(),
        Path(args.checkout).resolve()))
    batch_shapes={}
    for name in ("e1","e2"):
        output=root/f"{name}_forecast_batch.npz"
        response=worker.forecast_batch(request=batch_request_path,
            checkpoint=root/f"{name}_checkpoint.npz",output=output,
            configuration=root/f"{name}_configuration.json",candidate=name)
        with np.load(output,allow_pickle=False) as artifact:
            batch_shapes[name]=list(artifact["bars"].shape)
        if response["anchors"]!=2 or batch_shapes[name]!=[2,2,3,5]:
            raise AssertionError(f"{name} batched forecast shape mismatch: {batch_shapes[name]}")

    # Exercise the entire one-replay orchestration without research data.
    # This synthetic run validates wiring, not model selection or market skill.
    experiment_bars=synthetic_bars(620)
    dataset_hash="f"*64
    experiment_fold=FoldManifest(
        "0.4.0",dataset_hash,"e"*64,
        Window(MINUTE_NS,20*MINUTE_NS,320*MINUTE_NS),
        Window(260*MINUTE_NS,320*MINUTE_NS,560*MINUTE_NS),
        Window(540*MINUTE_NS,580*MINUTE_NS,640*MINUTE_NS),0,0,None)
    replay=MarketReplay(tuple(MarketFrame(
        bar.ts_close_ns,bar.available_at_ns,bar.instrument,bar.open,bar.high,
        bar.low,bar.close,bar.volume) for bar in experiment_bars
        if bar.ts_close_ns<experiment_fold.validation.score_end_ns),dataset_hash)
    experiment_dir=root/"synthetic_registered_fold_experiment"
    experiment_dir.mkdir()
    short_training=TrainingConfig(steps=2,batch_size=2,seed=83,report_interval=1)
    experiment_configurations={
        "E1":{"model":asdict(GRUConfig(hidden=8)),"training":asdict(short_training)},
        "E2":{"model":asdict(RSSMConfig(deter=32,hidden=32,stoch=2,classes=4,blocks=4)),
              "training":asdict(short_training)},
    }
    full_experiment=run_fold_experiment(
        replay=replay,fold=experiment_fold,train_scope_id="routeb-fold-SYNTHETIC-ONE-SHOT",
        candidate_configurations=experiment_configurations,worker=worker,
        output_dir=experiment_dir,repository_commit="d"*40,source_code_sha256="c"*64,
        marker_sha256="b"*64,reservation_label="routeb-fold-SYNTHETIC-ONE-SHOT",
        forecast_seed=83,forecast_paths=4,horizons=(1,5,15),
        anchor_interval_minutes=15,context_lookback=32,volatility_window=15,
        block_length=8,bootstrap_replicates=100,seed=83)
    comparison=full_experiment.get("predictive_comparison")
    if (full_experiment["status"]!="COMPLETED"
            or full_experiment["test_payload_decoded"]
            or not isinstance(comparison,dict)
            or comparison.get("authorization_verified") is not False):
        raise AssertionError("synthetic one-shot E0/E1/E2 pipeline evidence is invalid")

    # Separate fixed synthetic-fit sanity: with a longer budget, E2's masked
    # next-bar NLL should improve on this deterministic toy sequence.
    sanity_training = TrainingConfig(steps=100, batch_size=4, seed=73,
                                     report_interval=10)
    sanity_config = root / "e2_nll_sanity_configuration.json"
    sanity_config.write_text(json.dumps({
        "model": candidates["e2"], "training": asdict(sanity_training),
    }), encoding="utf-8")
    sanity_report_path = root / "e2_nll_sanity_report.json"
    sanity_checkpoint = root / "e2_nll_sanity_checkpoint.npz"
    status = world_model_main([
        "train-prepared", "--candidate", "e2",
        "--sequences", str(sequences), "--scope", scope,
        "--fold-sha256", fold_hash,
        "--configuration", str(sanity_config),
        "--checkpoint", str(sanity_checkpoint),
        "--worker-python", str(Path(args.worker_python).resolve()),
        "--upstream-checkout", str(Path(args.checkout).resolve()),
        "--report-json", str(sanity_report_path),
    ])
    if status != 0:
        raise RuntimeError("fixed-sequence E2 NLL sanity training failed")
    sanity = json.loads(sanity_report_path.read_text(encoding="utf-8"))
    nll_reports = sanity["worker_result"]["reports"]
    initial_nll = float(nll_reports[0]["pred_nll"])
    final_nll = float(nll_reports[-1]["pred_nll"])
    if not math.isfinite(initial_nll + final_nll) or final_nll >= initial_nll:
        raise AssertionError("E2 masked next-bar NLL did not decrease on fixed synthetic data")
    result = {
        "status": "PASS",
        "scope": "synthetic_mechanism_only",
        "e1_e2_share_train_artifact": True,
        "e1_e2_share_training_budget": True,
        "e1_checkpoint_sha256": left["checkpoint_sha256"],
        "e2_checkpoint_sha256": right["checkpoint_sha256"],
        "train_examples": left["train_examples"],
        "window_length": left["window_length"],
        "optimizer_steps": training.steps,
        "e2_synthetic_nll_initial": initial_nll,
        "e2_synthetic_nll_final": final_nll,
        "e2_synthetic_nll_decreased": True,
        "synthetic_batched_forecast_shapes": batch_shapes,
        "synthetic_one_shot_fold_pipeline": "COMPLETED",
        "synthetic_one_shot_report_sha256": full_experiment["report_sha256"],
        "real_market_validation": False,
        "economic_claim": False,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
