"""Standalone JAX worker entry point, invoked by Python 3.14 host via subprocess.

This worker takes ONLY prepared Train NPZ or already-observable context NPZ.
It never imports NautilusTrader or reads curated raw market parquet.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import os
import tempfile
from pathlib import Path
import sys
import numpy as np
from rles_data.io import strict_json_loads
from .jax_model import RSSMConfig
from .jax_gru import GRUConfig
from .jax_train import (train, forecast_distribution_standardized,
                        forecast_standardized, TrainingConfig)
from .jax_gru_train import (train_gru, forecast_gru,
                            forecast_gru_distribution)
from .codec import OHLCVCodec,TrainOnlyStandardizer
from .market import MINUTE_NS


def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(prog="routeb-jax-worker")
    subs=ap.add_subparsers(dest="command",required=True)
    t=subs.add_parser("train")
    t.add_argument("--sequences",required=True)
    t.add_argument("--scope",required=True)
    t.add_argument("--checkout",required=True)
    t.add_argument("--checkpoint",required=True)
    t.add_argument("--configuration",required=True)
    t.add_argument("--resume",action="store_true")
    bt=subs.add_parser("b3-train")
    bt.add_argument("--sequences",required=True)
    bt.add_argument("--scope",required=True)
    bt.add_argument("--fold-sha256",required=True)
    bt.add_argument("--checkout",required=True)
    bt.add_argument("--checkpoint",required=True)
    bt.add_argument("--configuration",required=True)
    bt.add_argument("--resume",action="store_true")
    f=subs.add_parser("forecast")
    f.add_argument("--request",required=True)
    f.add_argument("--checkout",required=True)
    f.add_argument("--checkpoint",required=True)
    f.add_argument("--output",required=True)
    f.add_argument("--configuration",required=True)
    d=subs.add_parser("next-distribution")
    d.add_argument("--request",required=True)
    d.add_argument("--checkout",required=True)
    d.add_argument("--checkpoint",required=True)
    d.add_argument("--output",required=True)
    d.add_argument("--configuration",required=True)
    et=subs.add_parser("e1-train")
    et.add_argument("--sequences",required=True)
    et.add_argument("--scope",required=True)
    et.add_argument("--checkout",required=True)
    et.add_argument("--checkpoint",required=True)
    et.add_argument("--configuration",required=True)
    et.add_argument("--resume",action="store_true")
    ef=subs.add_parser("e1-forecast")
    ef.add_argument("--request",required=True)
    ef.add_argument("--checkout",required=True)
    ef.add_argument("--checkpoint",required=True)
    ef.add_argument("--output",required=True)
    ef.add_argument("--configuration",required=True)
    bf=subs.add_parser("forecast-batch")
    bf.add_argument("--request",required=True)
    bf.add_argument("--checkout",required=True)
    bf.add_argument("--checkpoint",required=True)
    bf.add_argument("--output",required=True)
    bf.add_argument("--configuration",required=True)
    ebf=subs.add_parser("e1-forecast-batch")
    ebf.add_argument("--request",required=True)
    ebf.add_argument("--checkout",required=True)
    ebf.add_argument("--checkpoint",required=True)
    ebf.add_argument("--output",required=True)
    ebf.add_argument("--configuration",required=True)
    bd=subs.add_parser("next-distribution-batch")
    bd.add_argument("--request",required=True)
    bd.add_argument("--checkout",required=True)
    bd.add_argument("--checkpoint",required=True)
    bd.add_argument("--output",required=True)
    bd.add_argument("--configuration",required=True)
    ebd=subs.add_parser("e1-next-distribution-batch")
    ebd.add_argument("--request",required=True)
    ebd.add_argument("--checkout",required=True)
    ebd.add_argument("--checkpoint",required=True)
    ebd.add_argument("--output",required=True)
    ebd.add_argument("--configuration",required=True)
    ed=subs.add_parser("e1-next-distribution")
    ed.add_argument("--request",required=True)
    ed.add_argument("--checkout",required=True)
    ed.add_argument("--checkpoint",required=True)
    ed.add_argument("--output",required=True)
    ed.add_argument("--configuration",required=True)
    args=ap.parse_args(argv)
    try:
        config=strict_json_loads(
            Path(args.configuration).read_bytes(), context='Route B worker configuration',
        )
        if args.command == "b3-train":
            from .hierarchy_artifacts import (
                hierarchy_scaler_sha256, load_hierarchical_train_artifact,
            )
            from .jax_hierarchy_train import (
                HierarchicalModelConfig, HierarchicalTrainingConfig,
                train_hierarchical,
            )

            if (not isinstance(config, dict) or set(config) != {"model", "training"}
                    or not isinstance(config["model"], dict)
                    or set(config["model"]) != {"minute", "hour", "day"}
                    or any(not isinstance(config["model"][scale], dict)
                           for scale in ("minute", "hour", "day"))
                    or not isinstance(config["training"], dict)):
                raise ValueError("B3 config requires minute/hour/day models and training")
            model_config = HierarchicalModelConfig(**{
                scale: RSSMConfig(**config["model"][scale])
                for scale in ("minute", "hour", "day")
            })
            training_config = HierarchicalTrainingConfig(**config["training"])
            tensors, scalers, metadata = load_hierarchical_train_artifact(
                args.sequences, expected_scope=args.scope,
                expected_fold_sha256=args.fold_sha256)
            reports = train_hierarchical(
                tensors=tensors, train_scope_id=args.scope,
                dataset_id=metadata["dataset_id"],
                scaler_sha256=hierarchy_scaler_sha256(scalers),
                checkout=args.checkout, output_checkpoint=args.checkpoint,
                model_config=model_config, training_config=training_config,
                resume=args.resume)
            result = {
                "status": "COMPLETED",
                "kind": "B3_TRAIN_DIAGNOSTIC_ONLY",
                "checkpoint": args.checkpoint,
                "fold_sha256": args.fold_sha256,
                "hierarchy_tensor_sha256": metadata["hierarchy_tensor_sha256"],
                "scaler_sha256": metadata["scaler_sha256"],
                "reports": reports,
                "validation_scored": False,
                "test_scored": False,
                "economic_certification": False,
            }
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        if set(config)!={"model","training"}:
            raise ValueError("unknown model/training configuration keys")
        e1=args.command.startswith('e1-')
        mc=(GRUConfig(**config['model']) if e1 else RSSMConfig(**config['model']))
        tc=TrainingConfig(**config['training'])
        if args.command in {'train','e1-train'}:
            if e1:
                assert isinstance(mc,GRUConfig)
                report=train_gru(dataset_npz=args.sequences,train_scope_id=args.scope,
                                 output_checkpoint=args.checkpoint,model_config=mc,
                                 training_config=tc,resume=args.resume)
            else:
                assert isinstance(mc,RSSMConfig)
                report=train(dataset_npz=args.sequences,train_scope_id=args.scope,
                             checkout=args.checkout,output_checkpoint=args.checkpoint,
                             model_config=mc,training_config=tc,resume=args.resume)
            result={"status":"COMPLETED",
                    "kind":"E1_TRAIN_DIAGNOSTIC_ONLY" if e1 else "TRAIN_DIAGNOSTIC_ONLY",
                    "checkpoint":args.checkpoint,"reports":report}
        else:
            request=strict_json_loads(
                Path(args.request).read_bytes(), context='Route B worker forecast request',
            )
            batch=args.command in {
                'forecast-batch','e1-forecast-batch',
                'next-distribution-batch','e1-next-distribution-batch'}
            expected={"schema","train_scope_id","dataset_id","history_npz","context_sha256","scaler_sha256","horizon",
                      "n_paths","seed","train_npz_sha256","instrument"}
            expected |= ({"anchor_timestamps_ns"} if batch else
                         {"anchor_ts_ns","anchor_close","first_future_close_ns"})
            schema="routeb.forecast-batch-request.v1" if batch else "routeb.forecast-request.v2"
            if set(request)!=expected or request['schema']!=schema:
                raise ValueError("unexpected forecast request schema")
            if batch and (not isinstance(request.get('anchor_timestamps_ns'),list)
                or not request['anchor_timestamps_ns']
                or any(type(value) is not int for value in request['anchor_timestamps_ns'])):
                raise ValueError('batched anchor timestamps must be a nonempty integer list')
            int_fields=('horizon','n_paths','seed') if batch else ('horizon','n_paths','seed','anchor_ts_ns','first_future_close_ns')
            if any(type(request[k]) is not int for k in int_fields):
                raise ValueError('forecast integer fields must not be coerced from strings or bools')
            h=request['horizon'];p=request['n_paths']
            if not batch:
                if int(request["first_future_close_ns"])!=int(request["anchor_ts_ns"])+MINUTE_NS:
                    raise ValueError("forecast time does not start after anchor close")
                if request['seed']<0 or request['anchor_ts_ns']<=0 or request['anchor_ts_ns']%MINUTE_NS:
                    raise ValueError('forecast seed or anchor timestamp invalid')
            elif request['seed']<0:
                raise ValueError('forecast seed invalid')
            if h<1 or h>60 or p<2 or p>1024:
                raise ValueError("forecast size exceeds supported research limits")
            context_path=Path(request['history_npz'])
            if hashlib.sha256(context_path.read_bytes()).hexdigest()!=request['context_sha256']:
                raise ValueError('forecast context checksum mismatch')
            with np.load(context_path,allow_pickle=False) as arr:
                expected_members=({'histories','scaler_mean','scaler_std','anchor_timestamps_ns',
                                   'close_timestamps','available_timestamps','anchor_closes'} if batch else
                                  {'history','scaler_mean','scaler_std','anchor_ns','close_timestamps','available_timestamps'})
                if set(arr.files)!=expected_members:
                    raise ValueError('invalid forecast context members')
                context=np.asarray(arr['histories' if batch else 'history'],dtype=np.float32)
                mean=np.asarray(arr['scaler_mean'],dtype=np.float64)
                std=np.asarray(arr['scaler_std'],dtype=np.float64)
                context_anchors=np.asarray(arr['anchor_timestamps_ns'],dtype=np.int64) if batch else np.asarray([int(arr['anchor_ns'])],dtype=np.int64)
                anchor_closes=np.asarray(arr['anchor_closes'],dtype=np.float64) if batch else np.asarray([request['anchor_close']],dtype=np.float64)
                closes=np.asarray(arr['close_timestamps'],dtype=np.int64)
                available=np.asarray(arr['available_timestamps'],dtype=np.int64)
            if batch:
                if (context.ndim!=3 or context.shape[0]<1 or context.shape[2]!=5 or context.shape[1]<2
                    or context_anchors.shape!=(context.shape[0],) or anchor_closes.shape!=(context.shape[0],)
                    or closes.shape!=context.shape[:2] or available.shape!=closes.shape
                    or not np.array_equal(closes[:,-1],context_anchors)
                    or not np.all(np.diff(closes,axis=1)==MINUTE_NS)
                    or np.any(available>context_anchors[:,None])
                    or np.any(available<closes) or np.any(np.diff(available,axis=1)<0)
                    or not np.isfinite(anchor_closes).all() or np.any(anchor_closes<=0)
                    or not np.array_equal(np.asarray(request['anchor_timestamps_ns'],dtype=np.int64),context_anchors)):
                    raise ValueError('batched forecast context violates causal timestamps or request anchors')
                if (np.any(context_anchors<=0) or np.any(context_anchors%MINUTE_NS)
                    or np.any(np.diff(context_anchors)<=0)):
                    raise ValueError('batched anchor timestamps must be ordered and minute aligned')
            else:
                if context.ndim!=2 or context.shape[1]!=5 or context.shape[0]<2:
                    raise ValueError("expected [T,5] standardized historical context")
                if context_anchors[0]!=int(request["anchor_ts_ns"]):
                    raise ValueError("context anchor differs from forecast request")
                if (closes.shape!=(len(context),) or available.shape!=closes.shape
                    or not np.all(np.diff(closes)==MINUTE_NS)
                    or closes[-1]!=context_anchors[0] or np.any(available>context_anchors[0])
                    or np.any(available<closes) or np.any(np.diff(available)<0)):
                    raise ValueError('forecast context violates causal timestamps')
            if (not np.isfinite(context).all() or mean.shape!=(5,) or std.shape!=(5,)
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any()):
                raise ValueError("invalid frozen scaler or forecast context")
            codec=TrainOnlyStandardizer(mean,std,request['dataset_id'],'train')
            if codec.fingerprint!=request['scaler_sha256']:
                raise ValueError('frozen Train scaler fingerprint mismatch')
            if args.command in {'forecast','e1-forecast','forecast-batch','e1-forecast-batch'}:
                if batch:
                    histories=np.repeat(context,p,axis=0)
                else:
                    histories=np.broadcast_to(context[None,:,:],(p,*context.shape))
                if e1:
                    assert isinstance(mc,GRUConfig)
                    std_paths=forecast_gru(
                        checkpoint=args.checkpoint,model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],history=histories,
                        steps=h,forecast_seed=request['seed'])
                else:
                    assert isinstance(mc,RSSMConfig)
                    std_paths=forecast_standardized(
                        checkout=args.checkout,checkpoint=args.checkpoint,
                        model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],history=histories,
                        steps=h,forecast_seed=request['seed'])
                market: list[np.ndarray]=[]
                if batch:
                    std_paths=std_paths.reshape((len(context),p,h,5))
                    for i,anchor_paths in enumerate(std_paths):
                        market_anchor: list[list[list[float]]]=[]
                        for x in anchor_paths:
                            bars=OHLCVCodec.decode(codec.inverse(x),
                                previous_close=float(anchor_closes[i]),
                                start_close_ns=int(context_anchors[i])+MINUTE_NS,
                                instrument=request['instrument'])
                            market_anchor.append([[b.open,b.high,b.low,b.close,b.volume] for b in bars])
                        market.append(np.asarray(market_anchor,dtype=np.float64))
                    members={"bars":np.asarray(market,dtype=np.float64)}
                else:
                    for x in std_paths:
                        bars=OHLCVCodec.decode(codec.inverse(x),
                                              previous_close=float(request['anchor_close']),
                                              start_close_ns=int(request['first_future_close_ns']),
                                              instrument=request['instrument'])
                        market.append(np.asarray(
                            [[b.open,b.high,b.low,b.close,b.volume] for b in bars],
                            dtype=np.float64))
                    members={"bars":np.asarray(market,dtype=np.float64)}
                kind=("E1_BATCH_PRIOR_PATHS" if batch else "E1_PRIOR_PATHS") if e1 else (
                    "F1_BATCH_PRIOR_PATHS" if batch else "F1_PRIOR_PATHS")
            else:
                if h!=1:
                    raise ValueError("next-distribution request horizon must equal one")
                if e1:
                    assert isinstance(mc,GRUConfig)
                    mean,scale=forecast_gru_distribution(
                        checkpoint=args.checkpoint,model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],
                        history=context if batch else context[None,:,:])
                else:
                    assert isinstance(mc,RSSMConfig)
                    mean,scale=forecast_distribution_standardized(
                        checkout=args.checkout,checkpoint=args.checkpoint,
                        model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],
                        history=context if batch else context[None,:,:])
                # DreamerV3 commonly emits bfloat16; NumPy otherwise serializes
                # that dtype as opaque |V2, which downstream scoring cannot read.
                members={"standardized_mean":np.asarray(mean,dtype=np.float32),
                         "standardized_scale":np.asarray(scale,dtype=np.float32)}
                kind=("E1_BATCH_NEXT_DISTRIBUTION" if batch else "E1_NEXT_DISTRIBUTION") if e1 else (
                    "E2_BATCH_NEXT_DISTRIBUTION" if batch else "E2_NEXT_DISTRIBUTION")
            output=Path(args.output).resolve()
            output.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.npz',delete=False) as tmp_file:
                temp_path=Path(tmp_file.name)
                np.savez_compressed(tmp_file,**members)  # type: ignore[arg-type]
            try:
                os.replace(temp_path,output)
            finally:
                temp_path.unlink(missing_ok=True)
            result={"status":"COMPLETED","kind":kind,
                    "output":str(output),
                    "paths":p if args.command in {'forecast','e1-forecast','forecast-batch','e1-forecast-batch'} else 1,
                    "anchors":len(context_anchors) if batch else 1,
                    "horizon":h,
                    "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
                    "economic_certification":False}
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED","error":type(exc).__name__,
                          "detail":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
