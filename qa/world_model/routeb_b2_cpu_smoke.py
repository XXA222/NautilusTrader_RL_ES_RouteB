"""On-host full B2 JAX gradient/checkpoint smoke, synthetic only.

Run using the SNAPSHOT'S locked Python 3.12.10 world model environment after
checking out the pinned upstream DreamerV3 in an ignored .tmp directory.
Defaults to CPU. Set ``JAX_PLATFORMS=cuda`` and pass ``--expected-platform cuda``
to exercise the isolated WSL2 GPU worker. No exchange data, account data,
credentials, or Test/Holdout are read.
"""
from __future__ import annotations
import argparse
import json
from importlib.metadata import version as package_version
import math
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax
import numpy as np
from rles_world_model.routeb.market import Candle,MarketSequence,MINUTE_NS
from rles_world_model.routeb.codec import OHLCVCodec,TrainOnlyStandardizer
from rles_world_model.routeb.sequences import (
    RISK_BARRIER_VOLATILITY_QUANTILE,
    make_windows,
    export_train_sequences,
)
import hashlib
from rles_world_model.routeb.jax_model import RSSMConfig,PINNED_UPSTREAM
from rles_world_model.routeb.jax_train import train,TrainingConfig,SNAPSHOT_SHA
from rles_world_model.routeb.integration import prepare_forecast_context
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig


def synthetic_bars(n:int=480):
    price=2000.;records=[]
    for i in range(1,n+1):
        old=price
        op=old*math.exp(.00002*math.sin(i/5))
        price=op*math.exp(.0006*math.sin(i/13))
        hi=max(op,price)*math.exp(.0002+.0001*math.sin(i/7)**2)
        lo=min(op,price)*math.exp(-.0002-.0001*math.cos(i/11)**2)
        records.append(Candle(i*MINUTE_NS,i*MINUTE_NS,"ETHUSDT-PERP.SIM",
                              op,hi,lo,price,100+i%30))
    return tuple(records)


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkout',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--expected-platform',choices=('cpu','cuda'))
    ap.add_argument('--forecast-objective',choices=(
        'nll','crps','open_loop_crps','path_log_rv_crps'),
                    default='nll')
    args=ap.parse_args()
    devices=jax.devices()
    platforms={device.platform for device in devices}
    if args.expected_platform=='cpu' and 'cpu' not in platforms:
        raise RuntimeError(f'CPU was required but JAX exposed {sorted(platforms)}')
    if args.expected_platform=='cuda' and not platforms.intersection({'gpu','cuda'}):
        raise RuntimeError(f'CUDA was required but JAX exposed {sorted(platforms)}')
    uses_cuda=bool(platforms.intersection({'gpu','cuda'}))
    workspace_root=Path(__file__).resolve().parents[2]
    worker_lock=(workspace_root/'packages/world_model/requirements-world-model-wsl-cuda.lock'
                 if uses_cuda else workspace_root/'packages/world_model/environment/uv.lock')
    worker_lock_sha256=hashlib.sha256(worker_lock.read_bytes()).hexdigest()
    root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    bars=synthetic_bars()
    replay=MarketSequence(bars,'synthetic-smoke-NOT-REAL-MARKET','train')
    coord=OHLCVCodec.encode(bars[1:],previous_close=bars[0].close)
    scaler=TrainOnlyStandardizer.fit(coord,dataset_id=replay.dataset_id,scope='train')
    batch=make_windows(replay,scaler=scaler,length=32,burn_in=8,stride=16)
    scope='train-SYNTHETIC-MECHANISM-ONLY'
    input_path=root/'input.npz'
    export_train_sequences(input_path,batch,scaler,snapshot_sha=SNAPSHOT_SHA,
                           upstream_sha=PINNED_UPSTREAM,train_scope_id=scope)
    model=RSSMConfig(deter=32,hidden=32,stoch=2,classes=4,blocks=4,
                     forecast_objective=args.forecast_objective,
                     open_loop_crps_horizon=15,
                     path_log_rv_crps_horizon=15,
                     path_log_rv_crps_paths=4)
    cfg2=TrainingConfig(steps=2,batch_size=4,seed=31,report_interval=1)
    cfg4=TrainingConfig(steps=4,batch_size=4,seed=31,report_interval=2)
    resumed=root/'model_resumed.npz'
    uninterrupted=root/'model_uninterrupted.npz'
    first=train(dataset_npz=input_path,train_scope_id=scope,checkout=args.checkout,
          output_checkpoint=resumed,model_config=model,training_config=cfg2)
    if ("valid_risk_by_horizon" not in first[-1]
            or not any(count > 0 for count in first[-1]["valid_risk_by_horizon"])):
        raise RuntimeError("initial risk training had no valid multi-horizon targets")
    rest=train(dataset_npz=input_path,train_scope_id=scope,checkout=args.checkout,
               output_checkpoint=resumed,model_config=model,training_config=cfg4,resume=True)
    full=train(dataset_npz=input_path,train_scope_id=scope,checkout=args.checkout,
               output_checkpoint=uninterrupted,model_config=model,training_config=cfg4)
    if ("valid_risk_by_horizon" not in full[-1]
            or not any(count > 0 for count in full[-1]["valid_risk_by_horizon"])):
        raise RuntimeError("risk supervision did not receive valid multi-horizon targets")
    if not np.isfinite(full[-1]["prior_risk_nll"]):
        raise RuntimeError("risk training objective is nonfinite")
    if args.forecast_objective=='path_log_rv_crps':
        if (not np.isfinite(full[-1].get('path_log_rv_crps',float('nan')))
                or full[-1].get('valid_path_log_rv_crps',0.0)<=0):
            raise RuntimeError("path log-RV CRPS received no finite valid Train anchors")
    sampler_rng_parity=(
        rest[-1]["sampler_seed"]==full[-1]["sampler_seed"]
        and rest[-1]["sampler_order_sha256"]==full[-1]["sampler_order_sha256"]
        and rest[-1]["model_rng_seed"]==full[-1]["model_rng_seed"])
    if not sampler_rng_parity:
        raise RuntimeError("sampler order or stateless model PRNG seed changed after resume")
    with np.load(resumed,allow_pickle=False) as resumed_npz, np.load(uninterrupted,allow_pickle=False) as full_npz:
        if set(resumed_npz.files)!=set(full_npz.files) or not all(
            np.array_equal(resumed_npz[name],full_npz[name]) for name in resumed_npz.files
        ):
            raise AssertionError('Interrupted/resumed vs uninterrupted optimizer states differ')
    from rles_world_model.routeb.jax_train import (
        alignment_distributions_standardized, forecast_distribution_standardized,
        forecast_risk_distribution_standardized, forecast_standardized,
    )
    from rles_world_model.routeb.prior_forecast import (
        build_prior_alignment_table, write_prior_alignment_table,
    )
    from rles_world_model.routeb.evaluation import diagonal_gaussian_return_nll
    future=forecast_standardized(checkout=args.checkout,checkpoint=resumed,
        model_config=model,training_config=cfg4,train_scope_id=scope,
        train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,
        dataset_id=replay.dataset_id,history=batch.features[:3],steps=3,forecast_seed=701)
    assert future.shape==(3,3,5) and np.isfinite(future).all()
    repeated_future=forecast_standardized(checkout=args.checkout,checkpoint=resumed,
        model_config=model,training_config=cfg4,train_scope_id=scope,
        train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,
        dataset_id=replay.dataset_id,history=batch.features[:3],steps=3,forecast_seed=701)
    assert np.array_equal(future,repeated_future), 'fixed-seed forecast is not reproducible'
    conditioned_model=replace(model,rollout_risk_conditioning=True)
    conditioned_checkpoint=root/'model_risk_conditioned.npz'
    conditioned_reports=train(dataset_npz=input_path,train_scope_id=scope,
        checkout=args.checkout,output_checkpoint=conditioned_checkpoint,
        model_config=conditioned_model,training_config=cfg4)
    conditioned_future=forecast_standardized(
        checkout=args.checkout,checkpoint=conditioned_checkpoint,
        model_config=conditioned_model,training_config=cfg4,train_scope_id=scope,
        train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,dataset_id=replay.dataset_id,
        history=batch.features[:3],steps=3,forecast_seed=701,feature_std=scaler.std)
    conditioned_repeat=forecast_standardized(
        checkout=args.checkout,checkpoint=conditioned_checkpoint,
        model_config=conditioned_model,training_config=cfg4,train_scope_id=scope,
        train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,dataset_id=replay.dataset_id,
        history=batch.features[:3],steps=3,forecast_seed=701,feature_std=scaler.std)
    assert conditioned_future.shape==(3,3,5)
    assert np.isfinite(conditioned_future).all()
    assert np.array_equal(conditioned_future,conditioned_repeat), (
        'risk-conditioned fixed-seed forecast is not reproducible')
    assert np.isfinite(conditioned_reports[-1]['prior_risk_nll'])
    one_step_distribution=forecast_distribution_standardized(
        checkout=args.checkout,checkpoint=resumed,model_config=model,training_config=cfg4,
        train_scope_id=scope,train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,dataset_id=replay.dataset_id,
        history=batch.features[:3,:-1])
    if isinstance(one_step_distribution,dict):
        raise AssertionError("Gaussian B2 smoke unexpectedly returned mixture components")
    mean,scale=one_step_distribution
    one_step_nll=diagonal_gaussian_return_nll(
        mean,scale,batch.features[:3,-1],scaler_mean=scaler.mean,scaler_std=scaler.std)
    assert mean.shape==(3,5) and scale.shape==(3,5) and np.isfinite(one_step_nll).all()
    risk_mean,risk_scale,barrier_logits=forecast_risk_distribution_standardized(
        checkout=args.checkout,checkpoint=resumed,model_config=model,training_config=cfg4,
        train_scope_id=scope,train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,dataset_id=replay.dataset_id,
        history=batch.features[:3,:-1])
    assert risk_mean.shape==(3,len(model.risk_horizons),4)
    assert risk_scale.shape==risk_mean.shape and barrier_logits.shape==(3,len(model.risk_horizons),2)
    assert np.isfinite(risk_mean).all() and np.isfinite(risk_scale).all()
    assert np.all(risk_scale>0) and np.isfinite(barrier_logits).all()
    alignment_distributions=alignment_distributions_standardized(
        checkout=args.checkout,checkpoint=resumed,model_config=model,
        training_config=cfg4,train_scope_id=scope,
        train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,dataset_id=replay.dataset_id,
        history=batch.features[:3])
    alignment_table=build_prior_alignment_table(
        anchor_timestamps_ns=batch.timestamps[:3,:-1],
        target_timestamps_ns=batch.timestamps[:3,1:],
        target_features=batch.features[:3,1:],valid_mask=batch.masks[:3,1:],
        **alignment_distributions)
    alignment_path,alignment_sha=write_prior_alignment_table(
        root/'prior_alignment.npz',alignment_table)
    assert alignment_table['valid_mask'].any() and alignment_path.is_file()
    extended=synthetic_bars(481)
    request_path=root/'next_distribution_request.json'
    context_path=root/'next_distribution_context.npz'
    prepare_forecast_context(
        visible_history=extended[:-1],frozen_scaler=scaler,
        anchor_ns=extended[-2].ts_close_ns,train_scope_id=scope,
        train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        output_context_npz=context_path,output_request_json=request_path,
        horizon=1,n_paths=2,seed=907)
    configuration_path=root/'worker_configuration.json'
    configuration_path.write_text(json.dumps({
        'model':asdict(model),'training':asdict(cfg4)}),encoding='utf-8')
    distribution_path=root/'worker_next_distribution.npz'
    worker=WorkerClient(WorkerConfig(Path(sys.executable),Path(args.checkout).resolve()))
    worker_result=worker.next_distribution(
        request=request_path,checkpoint=resumed,output=distribution_path,
        configuration=configuration_path)
    with np.load(distribution_path,allow_pickle=False) as artifact:
        worker_mean=np.asarray(artifact['standardized_mean'])
        worker_scale=np.asarray(artifact['standardized_scale'])
    future_coordinate=OHLCVCodec.encode(
        (extended[-1],),previous_close=extended[-2].close)
    future_standard=scaler.transform(future_coordinate)
    worker_nll=diagonal_gaussian_return_nll(
        worker_mean,worker_scale,future_standard,
        scaler_mean=scaler.mean,scaler_std=scaler.std)
    assert worker_result['kind']=='E2_NEXT_DISTRIBUTION'
    assert worker_mean.shape==(1,5) and np.isfinite(worker_nll).all()
    risk_distribution_path=root/'worker_risk_distribution.npz'
    risk_worker_result=worker.forecast_risk_distribution(
        request=request_path,checkpoint=resumed,output=risk_distribution_path,
        configuration=configuration_path,risk_horizons=model.risk_horizons)
    with np.load(risk_distribution_path,allow_pickle=False) as artifact:
        worker_risk_mean=np.asarray(artifact['risk_mean'])
        worker_risk_scale=np.asarray(artifact['risk_scale'])
        worker_barrier_logits=np.asarray(artifact['barrier_logits'])
        worker_risk_horizons=np.asarray(artifact['risk_horizons'])
    assert risk_worker_result['kind']=='E2_RISK_DISTRIBUTION'
    assert worker_risk_mean.shape==(1,len(model.risk_horizons),4)
    assert worker_risk_scale.shape==worker_risk_mean.shape
    assert worker_barrier_logits.shape==(1,len(model.risk_horizons),2)
    assert np.array_equal(worker_risk_horizons,np.asarray(model.risk_horizons))
    assert np.isfinite(worker_risk_mean).all() and np.all(worker_risk_scale>0)
    assert np.isfinite(worker_barrier_logits).all()
    forecast_path=root/'worker_f1_forecast.npz'
    forecast_result=worker.forecast(
        request=request_path,checkpoint=resumed,output=forecast_path,
        configuration=configuration_path)
    with np.load(forecast_path,allow_pickle=False) as artifact:
        worker_paths=np.asarray(artifact['bars'])
    assert forecast_result['kind']=='F1_PRIOR_PATHS'
    assert worker_paths.shape==(2,1,5) and np.isfinite(worker_paths).all()
    repeat_forecast_path=root/'worker_f1_forecast_repeat.npz'
    worker.forecast(request=request_path,checkpoint=resumed,output=repeat_forecast_path,
        configuration=configuration_path)
    with np.load(repeat_forecast_path,allow_pickle=False) as artifact:
        repeated_worker_paths=np.asarray(artifact['bars'])
    assert np.array_equal(worker_paths,repeated_worker_paths), (
        'fixed-seed worker forecast is not reproducible')
    batch_histories=[];batch_closes=[];batch_available=[];batch_anchors=[];batch_prices=[]
    for end_index in range(470,476):
        history_bars=bars[end_index-31:end_index+1]
        encoded=OHLCVCodec.encode(history_bars[1:],previous_close=history_bars[0].close)
        batch_histories.append(scaler.transform(encoded))
        batch_closes.append([bar.ts_close_ns for bar in history_bars[1:]])
        batch_available.append([bar.available_at_ns for bar in history_bars[1:]])
        batch_anchors.append(history_bars[-1].ts_close_ns)
        batch_prices.append(history_bars[-1].close)
    batch_context_path=root/'synthetic_six_anchor_context.npz'
    np.savez_compressed(batch_context_path,histories=np.asarray(batch_histories,dtype=np.float32),
        scaler_mean=scaler.mean,scaler_std=scaler.std,
        anchor_timestamps_ns=np.asarray(batch_anchors,dtype=np.int64),
        decision_timestamps_ns=np.asarray(batch_anchors,dtype=np.int64),
        close_timestamps=np.asarray(batch_closes,dtype=np.int64),
        available_timestamps=np.asarray(batch_available,dtype=np.int64),
        anchor_closes=np.asarray(batch_prices,dtype=np.float64))
    batch_request_path=root/'synthetic_six_anchor_request.json'
    batch_request={
        'schema':'routeb.forecast-batch-request.v1','train_scope_id':scope,
        'dataset_id':replay.dataset_id,'history_npz':str(batch_context_path.resolve()),
        'context_sha256':hashlib.sha256(batch_context_path.read_bytes()).hexdigest(),
        'scaler_sha256':scaler.fingerprint,'horizon':3,'n_paths':2,'seed':907,
        'train_npz_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'instrument':bars[-1].instrument,'anchor_timestamps_ns':batch_anchors,
        'decision_timestamps_ns':batch_anchors}
    batch_request_path.write_text(json.dumps(batch_request),encoding='utf-8')
    batch_forecast_path=root/'worker_six_anchor_chunked_forecast.npz'
    batch_forecast_result=worker.forecast_batch(
        request=batch_request_path,checkpoint=resumed,output=batch_forecast_path,
        configuration=configuration_path,candidate='E2')
    with np.load(batch_forecast_path,allow_pickle=False) as artifact:
        worker_chunked_paths=np.asarray(artifact['bars'])
    assert batch_forecast_result['anchors']==6
    assert batch_forecast_result['anchor_chunk_size']==4
    assert batch_forecast_result['chunk_seed_rule']=='request_seed_plus_chunk_start_index'
    assert worker_chunked_paths.shape==(6,2,3,5)
    assert np.isfinite(worker_chunked_paths).all()
    memory_stats=devices[0].memory_stats()
    memory_stats=({key:int(value) for key,value in memory_stats.items()}
                  if memory_stats is not None else None)
    result={"status":"PASS","scope":"synthetic_mechanism_only",
        "forecast_objective":model.forecast_objective,
        "open_loop_crps_horizon":model.open_loop_crps_horizon,
        "path_log_rv_crps_train_loss":full[-1].get('path_log_rv_crps'),
        "path_log_rv_crps_valid_train_anchors":full[-1].get(
            'valid_path_log_rv_crps'),
            "optimizer_updates":full[-1]['optimizer_updates'],"final_step":full[-1]['step'],
            "device":full[-1]['device'],"jax_platforms":sorted(platforms),
            "python":sys.version.split()[0],"jax":jax.__version__,
            "jaxlib":package_version('jaxlib'),"upstream_sha256":PINNED_UPSTREAM,
            "snapshot_sha256":SNAPSHOT_SHA,
            "worker_lock_sha256":worker_lock_sha256,
            "uninterrupted_training_seconds":full[-1]['training_seconds'],
            "device_memory_stats_bytes":memory_stats,
            "resume_parity":True,"resumed_reports":len(rest),"forecast_shape":list(future.shape),
            "one_step_distribution_shape":list(mean.shape),"one_step_return_nll_finite":True,
            "worker_distribution_shape":list(worker_mean.shape),
            "direct_prior_risk_distribution_shape":list(risk_mean.shape),
            "worker_prior_risk_distribution_shape":list(worker_risk_mean.shape),
            "worker_risk_horizons":worker_risk_horizons.tolist(),
            "worker_risk_distribution_finite":True,
            "worker_one_step_return_nll_finite":True,
            "worker_f1_forecast_shape":list(worker_paths.shape),
            "worker_chunked_forecast_shape":list(worker_chunked_paths.shape),
            "worker_chunked_forecast_anchor_chunk_size":batch_forecast_result['anchor_chunk_size'],
            "worker_chunked_forecast_seed_rule":batch_forecast_result['chunk_seed_rule'],
            "risk_training_loss_finite":bool(np.isfinite(full[-1]["prior_risk_nll"])),
            "risk_valid_targets_by_horizon":full[-1]["valid_risk_by_horizon"],
            "risk_barrier_volatility_quantile":RISK_BARRIER_VOLATILITY_QUANTILE,
            "risk_barrier_fractions":batch.risk_barrier_fractions.tolist(),
            "sampler_rng_resume_parity":sampler_rng_parity,
            "prior_alignment_rows":len(alignment_table["valid_mask"]),
            "prior_alignment_valid_rows":int(alignment_table["valid_mask"].sum()),
            "prior_alignment_sha256":alignment_sha,
            "direct_forecast_seed_repeatability":True,
            "risk_conditioned_forecast_shape":list(conditioned_future.shape),
            "risk_conditioned_forecast_finite":True,
            "risk_conditioned_forecast_seed_repeatability":True,
            "worker_forecast_seed_repeatability":True,
            "worker_f1_forecast_ohlcv_valid":True,
            "real_market_validation":False,"economic_claim":False}
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
