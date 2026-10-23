"""On-host full B2 JAX gradient/checkpoint smoke, synthetic only.

Run using the SNAPSHOT'S locked Python 3.12.10 world model environment after
checking out the pinned upstream DreamerV3 in an ignored .tmp directory.
No exchange data, account data, credentials, or Test/Holdout are read.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import numpy as np
from rles_world_model.routeb.market import Candle,MarketSequence,MINUTE_NS
from rles_world_model.routeb.codec import OHLCVCodec,TrainOnlyStandardizer
from rles_world_model.routeb.sequences import make_windows,export_train_sequences
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
    args=ap.parse_args()
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
    model=RSSMConfig(deter=32,hidden=32,stoch=2,classes=4,blocks=4)
    cfg2=TrainingConfig(steps=2,batch_size=4,seed=31,report_interval=1)
    cfg4=TrainingConfig(steps=4,batch_size=4,seed=31,report_interval=2)
    resumed=root/'model_resumed.npz'
    uninterrupted=root/'model_uninterrupted.npz'
    first=train(dataset_npz=input_path,train_scope_id=scope,checkout=args.checkout,
                output_checkpoint=resumed,model_config=model,training_config=cfg2)
    rest=train(dataset_npz=input_path,train_scope_id=scope,checkout=args.checkout,
               output_checkpoint=resumed,model_config=model,training_config=cfg4,resume=True)
    full=train(dataset_npz=input_path,train_scope_id=scope,checkout=args.checkout,
               output_checkpoint=uninterrupted,model_config=model,training_config=cfg4)
    with np.load(resumed,allow_pickle=False) as resumed_npz, np.load(uninterrupted,allow_pickle=False) as full_npz:
        if set(resumed_npz.files)!=set(full_npz.files) or not all(
            np.array_equal(resumed_npz[name],full_npz[name]) for name in resumed_npz.files
        ):
            raise AssertionError('Interrupted/resumed vs uninterrupted optimizer states differ')
    from rles_world_model.routeb.jax_train import (
        forecast_distribution_standardized, forecast_standardized,
    )
    from rles_world_model.routeb.evaluation import diagonal_gaussian_return_nll
    future=forecast_standardized(checkout=args.checkout,checkpoint=resumed,
        model_config=model,training_config=cfg4,train_scope_id=scope,
        train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,
        dataset_id=replay.dataset_id,history=batch.features[:3],steps=3,forecast_seed=701)
    assert future.shape==(3,3,5) and np.isfinite(future).all()
    mean,scale=forecast_distribution_standardized(
        checkout=args.checkout,checkpoint=resumed,model_config=model,training_config=cfg4,
        train_scope_id=scope,train_npz_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        scaler_sha256=scaler.fingerprint,dataset_id=replay.dataset_id,
        history=batch.features[:3,:-1])
    one_step_nll=diagonal_gaussian_return_nll(
        mean,scale,batch.features[:3,-1],scaler_mean=scaler.mean,scaler_std=scaler.std)
    assert mean.shape==(3,5) and scale.shape==(3,5) and np.isfinite(one_step_nll).all()
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
    forecast_path=root/'worker_f1_forecast.npz'
    forecast_result=worker.forecast(
        request=request_path,checkpoint=resumed,output=forecast_path,
        configuration=configuration_path)
    with np.load(forecast_path,allow_pickle=False) as artifact:
        worker_paths=np.asarray(artifact['bars'])
    assert forecast_result['kind']=='F1_PRIOR_PATHS'
    assert worker_paths.shape==(2,1,5) and np.isfinite(worker_paths).all()
    result={"status":"PASS","scope":"synthetic_mechanism_only",
            "optimizer_updates":len(first)+len(rest),"final_step":full[-1]['step'],
            "resume_parity":True,"resumed_reports":len(rest),"forecast_shape":list(future.shape),
            "one_step_distribution_shape":list(mean.shape),"one_step_return_nll_finite":True,
            "worker_distribution_shape":list(worker_mean.shape),
            "worker_one_step_return_nll_finite":True,
            "worker_f1_forecast_shape":list(worker_paths.shape),
            "worker_f1_forecast_ohlcv_valid":True,
            "real_market_validation":False,"economic_claim":False}
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
