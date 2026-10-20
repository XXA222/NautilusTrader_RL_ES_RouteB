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
from .jax_model import RSSMConfig
from .jax_gru import GRUConfig
from .jax_train import (train, forecast_distribution_standardized,
                        forecast_risk_distribution_standardized,
                        forecast_standardized, TrainingConfig)
from .jax_gru_train import (train_gru, forecast_gru,
                            forecast_gru_distribution)
from .codec import OHLCVCodec,TrainOnlyStandardizer,select_decodable_paths
from .market import MINUTE_NS

BATCH_ANCHOR_CHUNK_SIZE = 4


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
    rd=subs.add_parser("risk-distribution")
    rd.add_argument("--request",required=True)
    rd.add_argument("--checkout",required=True)
    rd.add_argument("--checkpoint",required=True)
    rd.add_argument("--output",required=True)
    rd.add_argument("--configuration",required=True)
    rdb=subs.add_parser("risk-distribution-batch")
    rdb.add_argument("--request",required=True)
    rdb.add_argument("--checkout",required=True)
    rdb.add_argument("--checkpoint",required=True)
    rdb.add_argument("--output",required=True)
    rdb.add_argument("--configuration",required=True)
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
    ed=subs.add_parser("e1-next-distribution")
    ed.add_argument("--request",required=True)
    ed.add_argument("--checkout",required=True)
    ed.add_argument("--checkpoint",required=True)
    ed.add_argument("--output",required=True)
    ed.add_argument("--configuration",required=True)
    args=ap.parse_args(argv)
    try:
        config=json.loads(Path(args.configuration).read_text('utf-8'))
        if set(config)!={"model","training"}:
            raise ValueError("unknown model/training configuration keys")
        e1=args.command.startswith('e1-')
        batch=args.command in {'forecast-batch','e1-forecast-batch','risk-distribution-batch'}
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
            request=json.loads(Path(args.request).read_text('utf-8'))
            batch=args.command in {'forecast-batch','e1-forecast-batch','risk-distribution-batch'}
            expected={"schema","train_scope_id","dataset_id","history_npz","context_sha256","scaler_sha256","horizon",
                      "n_paths","seed","train_npz_sha256","instrument"}
            expected |= ({"anchor_timestamps_ns","decision_timestamps_ns"} if batch else
                         {"anchor_ts_ns","anchor_close","first_future_close_ns"})
            schema="routeb.forecast-batch-request.v1" if batch else "routeb.forecast-request.v2"
            if set(request)!=expected or request['schema']!=schema:
                raise ValueError("unexpected forecast request schema")
            if batch and (not isinstance(request.get('anchor_timestamps_ns'),list)
                or not request['anchor_timestamps_ns']
                or any(type(value) is not int for value in request['anchor_timestamps_ns'])
                or not isinstance(request.get('decision_timestamps_ns'),list)
                or len(request['decision_timestamps_ns'])!=len(request['anchor_timestamps_ns'])
                or any(type(value) is not int for value in request['decision_timestamps_ns'])):
                raise ValueError('batched anchors and decision clocks must be nonempty aligned integer lists')
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
                                   'decision_timestamps_ns','close_timestamps','available_timestamps','anchor_closes'} if batch else
                                  {'history','scaler_mean','scaler_std','anchor_ns','close_timestamps','available_timestamps'})
                if set(arr.files)!=expected_members:
                    raise ValueError('invalid forecast context members')
                context=np.asarray(arr['histories' if batch else 'history'],dtype=np.float32)
                mean=np.asarray(arr['scaler_mean'],dtype=np.float64)
                std=np.asarray(arr['scaler_std'],dtype=np.float64)
                context_anchors=np.asarray(arr['anchor_timestamps_ns'],dtype=np.int64) if batch else np.asarray([int(arr['anchor_ns'])],dtype=np.int64)
                decisions=np.asarray(arr['decision_timestamps_ns'],dtype=np.int64) if batch else context_anchors.copy()
                anchor_closes=np.asarray(arr['anchor_closes'],dtype=np.float64) if batch else np.asarray([request['anchor_close']],dtype=np.float64)
                closes=np.asarray(arr['close_timestamps'],dtype=np.int64)
                available=np.asarray(arr['available_timestamps'],dtype=np.int64)
            if batch:
                if (context.ndim!=3 or context.shape[0]<1 or context.shape[2]!=5 or context.shape[1]<2
                    or context_anchors.shape!=(context.shape[0],) or decisions.shape!=(context.shape[0],)
                    or anchor_closes.shape!=(context.shape[0],)
                    or closes.shape!=context.shape[:2] or available.shape!=closes.shape
                    or not np.array_equal(closes[:,-1],context_anchors)
                    or not np.all(np.diff(closes,axis=1)==MINUTE_NS)
                    or np.any(decisions<context_anchors) or np.any(decisions>=context_anchors+MINUTE_NS)
                    or np.any(available>decisions[:,None])
                    or np.any(available<closes) or np.any(np.diff(available,axis=1)<0)
                    or not np.isfinite(anchor_closes).all() or np.any(anchor_closes<=0)
                    or not np.array_equal(np.asarray(request['anchor_timestamps_ns'],dtype=np.int64),context_anchors)
                    or not np.array_equal(np.asarray(request['decision_timestamps_ns'],dtype=np.int64),decisions)):
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
                batch_candidate_count=p*4 if batch else p
                starts=(range(0,len(context),BATCH_ANCHOR_CHUNK_SIZE) if batch else (0,))
                path_chunks=[]
                for chunk_start in starts:
                    chunk_end=(min(chunk_start+BATCH_ANCHOR_CHUNK_SIZE,len(context))
                               if batch else len(context))
                    chunk_context=context[chunk_start:chunk_end] if batch else context
                    histories=(np.repeat(chunk_context,batch_candidate_count,axis=0) if batch
                               else np.broadcast_to(context[None,:,:],(p,*context.shape)))
                    # Each ordered anchor chunk receives a deterministic, disjoint
                    # seed while E1/E2 share the same chunk seed schedule.
                    forecast_seed=request['seed']+chunk_start if batch else request['seed']
                    if e1:
                        assert isinstance(mc,GRUConfig)
                        chunk_paths=forecast_gru(
                            checkpoint=args.checkpoint,model_config=mc,training_config=tc,
                            train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                            train_npz_sha256=request["train_npz_sha256"],
                            scaler_sha256=request['scaler_sha256'],history=histories,
                            steps=h,forecast_seed=forecast_seed)
                    else:
                        assert isinstance(mc,RSSMConfig)
                        chunk_paths=forecast_standardized(
                            checkout=args.checkout,checkpoint=args.checkpoint,
                            model_config=mc,training_config=tc,
                            train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                            train_npz_sha256=request["train_npz_sha256"],
                            scaler_sha256=request['scaler_sha256'],history=histories,
                            steps=h,forecast_seed=forecast_seed,feature_std=std)
                    path_chunks.append(np.asarray(chunk_paths,dtype=np.float32))
                std_paths=np.concatenate(path_chunks,axis=0)
                market: list[np.ndarray]=[]
                if batch:
                    std_paths=std_paths.reshape((len(context),batch_candidate_count,h,5))
                    rejected_by_anchor=[]
                    for i,anchor_paths in enumerate(std_paths):
                        def decode_path(x):
                            return OHLCVCodec.decode(codec.inverse(x),
                                previous_close=float(anchor_closes[i]),
                                start_close_ns=int(context_anchors[i])+MINUTE_NS,
                                instrument=request['instrument'])
                        decoded,rejected,candidates=select_decodable_paths(
                            anchor_paths,decoder=decode_path,required=p)
                        rejected_by_anchor.append(rejected)
                        market_anchor=[[[b.open,b.high,b.low,b.close,b.volume] for b in bars]
                                       for bars in decoded]
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
            elif args.command in {'risk-distribution','risk-distribution-batch'}:
                if e1:
                    raise ValueError('risk distribution requires the E2 RSSM model')
                assert isinstance(mc,RSSMConfig)
                histories=context if batch else context[None,:,:]
                risk_mean,risk_scale,barrier_logits=forecast_risk_distribution_standardized(
                    checkout=args.checkout,checkpoint=args.checkpoint,
                    model_config=mc,training_config=tc,
                    train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                    train_npz_sha256=request['train_npz_sha256'],
                    scaler_sha256=request['scaler_sha256'],history=histories)
                members={"risk_mean":risk_mean.astype(np.float32),
                         "risk_scale":risk_scale.astype(np.float32),
                         "barrier_logits":barrier_logits.astype(np.float32),
                         "risk_horizons":np.asarray(mc.risk_horizons,dtype=np.int64)}
                kind="E2_RISK_DISTRIBUTION_BATCH" if batch else "E2_RISK_DISTRIBUTION"
            else:
                if h!=1:
                    raise ValueError("next-distribution request horizon must equal one")
                if e1:
                    assert isinstance(mc,GRUConfig)
                    mean,scale=forecast_gru_distribution(
                        checkpoint=args.checkpoint,model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],history=context[None,:,:])
                else:
                    assert isinstance(mc,RSSMConfig)
                    distribution=forecast_distribution_standardized(
                        checkout=args.checkout,checkpoint=args.checkpoint,
                        model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],history=context[None,:,:])
                # DreamerV3 commonly emits bfloat16; NumPy otherwise serializes
                # that dtype as opaque |V2, which downstream scoring cannot read.
                if isinstance(distribution,dict):
                    members={
                        "standardized_mean":np.asarray(distribution["mean"],dtype=np.float32),
                        "standardized_scale":np.asarray(distribution["scale"],dtype=np.float32),
                        "mixture_means":np.asarray(distribution["mixture_means"],dtype=np.float32),
                        "mixture_scales":np.asarray(distribution["mixture_scales"],dtype=np.float32),
                        "mixture_weights":np.asarray(distribution["mixture_weights"],dtype=np.float32),
                    }
                else:
                    mean,scale=distribution
                    members={"standardized_mean":np.asarray(mean,dtype=np.float32),
                             "standardized_scale":np.asarray(scale,dtype=np.float32)}
                kind="E1_NEXT_DISTRIBUTION" if e1 else "E2_NEXT_DISTRIBUTION"
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
        if args.command in {'next-distribution','e1-next-distribution'}:
            if e1:
                distribution_family="gaussian"
                student_t_df=None
                mixture_components=None
            else:
                assert isinstance(mc,RSSMConfig)
                distribution_family=mc.forecast_distribution
                student_t_df=(mc.forecast_student_t_df
                              if mc.forecast_distribution=="student_t" else None)
                mixture_components=(2 if mc.forecast_distribution=="mixture_gaussian"
                                    else None)
            result["distribution_family"]=distribution_family
            result["scale_semantics"]="marginal_standard_deviation"
            result["student_t_df"]=student_t_df
            result["mixture_components"]=mixture_components
        if args.command in {'forecast-batch','e1-forecast-batch'}:
            result["sampled_paths_per_anchor"]=batch_candidate_count
            result["anchor_chunk_size"]=BATCH_ANCHOR_CHUNK_SIZE
            result["chunk_seed_rule"]="request_seed_plus_chunk_start_index"
            result["rejected_paths_by_anchor"]=rejected_by_anchor
            result["rejected_paths"]=int(sum(rejected_by_anchor))
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED","error":type(exc).__name__,
                          "detail":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
