"""Optional integration with *existing* RLES authorization and World Model B0.

No original project source included. This code is placed under the existing
rles_world_model package when overlayed onto the pinned snapshot.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
from .market import MarketSequence,Candle,MINUTE_NS
from .codec import OHLCVCodec,TrainOnlyStandardizer
from .sequences import (RISK_BARRIER_VOLATILITY_QUANTILE,make_windows,
                        export_train_sequences)
from ..contracts import MarketReplay
from .jax_model import PINNED_UPSTREAM
from .jax_train import SNAPSHOT_SHA


def prepare_fold_train_sequences(*,fold_replay:MarketReplay,fold,
                                train_scope_id:str,target_npz:Path,
                                length:int=128,burn_in:int=32,
                                stride:int=64)->tuple[Path,TrainOnlyStandardizer,dict]:
    """Build Train artifacts from a *single already-authorized* Fold replay.

    The replay must have been returned by ``project_registered_fold_replay``;
    this function does not authorize or read data. It accepts the Train warmup
    plus Train/Validation rows from that one-shot reader, uses warmup only as
    causal context, fits the scaler only on Train score rows, and emits no
    Validation/Test labels to the worker NPZ.
    """
    from rles_data.core import FoldManifest
    if not isinstance(fold_replay,MarketReplay) or not isinstance(fold,FoldManifest):
        raise ValueError("an authorized MarketReplay and frozen FoldManifest are required")
    if (fold_replay.dataset_hash!=fold.dataset_hash or not train_scope_id
            or not fold_replay.is_contiguous):
        raise ValueError("Fold data identity or minute continuity is invalid")
    frames=fold_replay.frames
    if (frames[0].ts_close_ns!=fold.train.warmup_start_ns
            or frames[-1].ts_close_ns!=fold.validation.score_end_ns-MINUTE_NS
            or any(frame.ts_close_ns>=fold.validation.score_end_ns
                   or frame.available_at_ns>fold.validation.score_end_ns
                   for frame in frames)):
        raise ValueError("Fold replay must cover Train warmup through Validation end and exclude Test")
    train_frames=tuple(frame for frame in frames
        if fold.train.warmup_start_ns<=frame.ts_close_ns<fold.train.score_end_ns)
    if (not train_frames or train_frames[0].ts_close_ns!=fold.train.warmup_start_ns
            or train_frames[-1].ts_close_ns!=fold.train.score_end_ns-MINUTE_NS):
        raise ValueError("Fold replay does not fully cover the Train window")
    train=MarketSequence(tuple(Candle(
        frame.ts_close_ns,frame.available_at_ns,frame.instrument,
        float(frame.open),float(frame.high),float(frame.low),float(frame.close),
        float(frame.volume),
    ) for frame in train_frames),dataset_id=fold.dataset_hash,scope="train")
    coordinates=OHLCVCodec.encode(train.candles[1:],
                                   previous_close=train.candles[0].close)
    coordinate_times=np.asarray([bar.ts_close_ns for bar in train.candles[1:]],
                                 dtype=np.int64)
    train_score_mask=((coordinate_times>=fold.train.score_start_ns)
                      &(coordinate_times<fold.train.score_end_ns))
    scaler=TrainOnlyStandardizer.fit(
        coordinates[train_score_mask],dataset_id=fold.dataset_hash,scope="train")
    batch=make_windows(
        train,scaler=scaler,length=length,burn_in=burn_in,stride=stride,
        loss_interval=(fold.train.score_start_ns,fold.train.score_end_ns))
    if np.any(batch.timestamps[batch.masks]<fold.train.score_start_ns) or np.any(
            batch.timestamps[batch.masks]>=fold.train.score_end_ns):
        raise RuntimeError("Train sequence artifact contains targets outside Train score")
    export_train_sequences(
        target_npz,batch,scaler,snapshot_sha=SNAPSHOT_SHA,
        upstream_sha=PINNED_UPSTREAM,train_scope_id=train_scope_id,
        fold_sha256=fold.hash)
    summary={
        "train_scope_id":train_scope_id,"dataset_id":fold.dataset_hash,
        "fold_hash":fold.hash,"train_warmup_bars":sum(
            bar.ts_close_ns<fold.train.score_start_ns for bar in train.candles),
        "train_score_bars":int(train_score_mask.sum()),
        "loss_target_count":int(batch.masks.sum()),
        "train_windows":len(batch.features),"window_length":length,
        "risk_horizons":list(batch.risk_horizons),
        "risk_barrier_fractions":batch.risk_barrier_fractions.tolist(),
        "risk_barrier_volatility_quantile":RISK_BARRIER_VOLATILITY_QUANTILE,
        "burn_in":burn_in,"validation_payload_in_worker":False,
        "test_payload_in_worker":False,
    }
    return target_npz,scaler,summary


def prepare_registered_train(*,rles_config:dict,window:dict,
                             marker_sha256:str,label:str,instrument:str,
                             target_npz:Path,length:int=128,burn_in:int=32,
                             stride:int=64)->tuple[Path,TrainOnlyStandardizer,dict]:
    """Use the snapshot's registered Train accessor BEFORE reading market events."""
    # Imports EXISTING files at runtime. Neither is present in this source ZIP.
    from rles_world_model.train_dataset import project_registered_train_replay
    from rles_world_model.data_audit import audit_replay
    if not isinstance(rles_config,dict) or any(
        rles_config.get(k) is not None for k in ("evaluation","test","holdout")):
        raise ValueError("Route B Train preparation rejects evaluation/test/holdout in source config")
    if not isinstance(window,dict) or set(window)!={"start_ns","end_ns"}:
        raise ValueError("explicit registered Train interval required")
    if not label.startswith('train-') or not marker_sha256:
        raise ValueError("registered Train token and marker required")
    replay=project_registered_train_replay(
        rles_config,window,marker_sha256=marker_sha256,
        label=label,instrument=instrument,allow_legacy_isolated_snapshot=False)
    quality=audit_replay(replay)
    if not quality.market_model_eligible:
        raise ValueError("Train source is missing/duplicated/otherwise unsuitable")
    candles=MarketSequence(tuple(Candle(
        frame.ts_close_ns, frame.available_at_ns, frame.instrument,
        float(frame.open), float(frame.high), float(frame.low),
        float(frame.close), float(frame.volume),
    ) for frame in replay.frames), dataset_id=replay.dataset_hash, scope='train')
    train_coords=OHLCVCodec.encode(candles.candles[1:],
                                   previous_close=candles.candles[0].close)
    scaler=TrainOnlyStandardizer.fit(train_coords,dataset_id=candles.dataset_id,scope='train')
    batch=make_windows(candles,scaler=scaler,length=length,
                       burn_in=burn_in,stride=stride)
    export_train_sequences(target_npz,batch,scaler,snapshot_sha=SNAPSHOT_SHA,
                           upstream_sha=PINNED_UPSTREAM,train_scope_id=label)
    summary={"train_scope_id":label,"dataset_id":candles.dataset_id,
             "input_bars":len(candles.candles),"train_windows":len(batch.features),
             "window_length":length,"burn_in":burn_in,
             "perpetual_economics_eligible":quality.perpetual_economics_eligible}
    return target_npz,scaler,summary


def prepare_forecast_context(*,visible_history:tuple[Candle,...],
                             frozen_scaler:TrainOnlyStandardizer,
                             anchor_ns:int,train_scope_id:str,train_npz_sha256:str,
                             output_context_npz:Path,output_request_json:Path,
                             horizon:int,n_paths:int,seed:int)->dict:
    """Prepare zero-future-label F1 request from caller-authorized visible bars.

    For Validation, the caller MUST use project's evaluation authorization before
    obtaining `visible_history`; this method neither opens nor authorizes it.
    """
    if len(visible_history)<3 or visible_history[-1].ts_close_ns!=anchor_ns:
        raise ValueError("forecast anchor must be last visible completed candle")
    if any(b.available_at_ns>anchor_ns or b.ts_close_ns>anchor_ns for b in visible_history):
        raise ValueError("unavailable candle in inference context")
    if type(horizon) is not int or not 1<=horizon<=60 or type(n_paths) is not int or not 2<=n_paths<=1024 or type(seed) is not int or seed<0:
        raise ValueError('forecast size/seed outside supported bounds')
    market=MarketSequence(tuple(visible_history),frozen_scaler.dataset_id,'inference_visible')
    coords=OHLCVCodec.encode(market.candles[1:],previous_close=market.candles[0].close)
    standard=frozen_scaler.transform(coords)
    output_context_npz=Path(output_context_npz)
    output_request_json=Path(output_request_json)
    output_context_npz.parent.mkdir(parents=True,exist_ok=True)
    output_request_json.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_context_npz.parent,suffix='.npz',delete=False) as temp:
        tmp=Path(temp.name)
        np.savez_compressed(temp,history=standard,scaler_mean=frozen_scaler.mean,
                            scaler_std=frozen_scaler.std,anchor_ns=np.int64(anchor_ns),
                            close_timestamps=np.asarray([b.ts_close_ns for b in visible_history[1:]],dtype=np.int64),
                            available_timestamps=np.asarray([b.available_at_ns for b in visible_history[1:]],dtype=np.int64))
    try:
        os.replace(tmp,output_context_npz)
    finally:
        tmp.unlink(missing_ok=True)
    if len(train_npz_sha256)!=64:
        raise ValueError("expected frozen training NPZ SHA256")
    context_sha256=hashlib.sha256(output_context_npz.read_bytes()).hexdigest()
    request={"schema":"routeb.forecast-request.v2", "train_scope_id":train_scope_id,
             "context_sha256":context_sha256,"scaler_sha256":frozen_scaler.fingerprint,
             "train_npz_sha256":train_npz_sha256,
             "dataset_id":frozen_scaler.dataset_id,"history_npz":str(output_context_npz.resolve()),
             "horizon":horizon,"n_paths":n_paths,"seed":seed,
             "anchor_ts_ns":anchor_ns,"anchor_close":visible_history[-1].close,
             "first_future_close_ns":anchor_ns+MINUTE_NS,"instrument":visible_history[-1].instrument}
    tmp_req=output_request_json.with_suffix('.json.partial')
    tmp_req.write_text(json.dumps(request,sort_keys=True),encoding='utf-8')
    os.replace(tmp_req,output_request_json)
    return request


def prepare_fold_forecast_batch(*,fold_replay:MarketReplay,fold,
        frozen_scaler:TrainOnlyStandardizer,train_scope_id:str,train_npz_sha256:str,
        output_context_npz:Path,output_request_json:Path,horizon:int,n_paths:int,
        seed:int,anchor_interval_minutes:int=15,context_lookback:int=128,
        anchor_start_ns:int|None=None)->dict:
    """Materialize one causal, ordered Validation-anchor batch for the JAX worker.

    ``fold_replay`` must come from the registered Fold reader and end at
    Validation score end. The batch contains no future labels; each context is
    clipped to its own anchor. Test is rejected by the replay boundary check.
    """
    from rles_data.core import FoldManifest
    from .jax_model import PINNED_UPSTREAM
    from .jax_train import SNAPSHOT_SHA
    if not isinstance(fold_replay,MarketReplay) or not isinstance(fold,FoldManifest):
        raise ValueError("an authorized Fold replay and frozen FoldManifest are required")
    if (fold_replay.dataset_hash!=fold.dataset_hash or not fold_replay.is_contiguous
            or frozen_scaler.dataset_id!=fold.dataset_hash or frozen_scaler.train_scope!="train"
            or not train_scope_id or len(train_npz_sha256)!=64
            or any(character not in "0123456789abcdef" for character in train_npz_sha256)):
        raise ValueError("Fold, Train artifact, and scaler identities must match")
    if (fold_replay.frames[0].ts_close_ns!=fold.train.warmup_start_ns
            or fold_replay.frames[-1].ts_close_ns!=fold.validation.score_end_ns-MINUTE_NS
            or any(frame.ts_close_ns>=fold.validation.score_end_ns
                   or frame.available_at_ns>fold.validation.score_end_ns
                   for frame in fold_replay.frames)):
        raise ValueError("forecast replay must cover Train/Validation only and exclude Test")
    if (type(horizon) is not int or not 1<=horizon<=60
            or type(n_paths) is not int or not 2<=n_paths<=1024
            or type(seed) is not int or seed<0
            or type(anchor_interval_minutes) is not int or anchor_interval_minutes<1
            or type(context_lookback) is not int or context_lookback<3
            or (anchor_start_ns is not None and
                (type(anchor_start_ns) is not int or anchor_start_ns%MINUTE_NS))):
        raise ValueError("forecast batch size or Validation protocol is outside supported bounds")
    output_context_npz=Path(output_context_npz).resolve()
    output_request_json=Path(output_request_json).resolve()
    if (output_context_npz.exists() or output_request_json.exists()
            or not output_context_npz.parent.is_dir()
            or output_context_npz.parent!=output_request_json.parent):
        raise ValueError("batch outputs must be new files in an existing shared directory")
    frames=fold_replay.frames
    times=np.asarray([frame.ts_close_ns for frame in frames],dtype=np.int64)
    by_time={int(timestamp):index for index,timestamp in enumerate(times)}
    max_horizon=horizon
    first_anchor=(fold.validation.score_start_ns-MINUTE_NS
                  if anchor_start_ns is None else anchor_start_ns)
    last_anchor=fold.validation.score_end_ns-max_horizon*MINUTE_NS
    if anchor_start_ns is not None and first_anchor<fold.validation.score_start_ns:
        raise ValueError("explicit anchor grid cannot precede the Validation scoring interval")
    stride=anchor_interval_minutes*MINUTE_NS
    anchors=np.arange(first_anchor,last_anchor+1,stride,dtype=np.int64)
    if anchors.size<1:
        raise ValueError("Validation interval has no complete forecast anchors")
    contexts=[]; close_times=[]; availability=[]; anchor_closes=[]; decision_times=[]
    for anchor in anchors:
        anchor_index=by_time.get(int(anchor))
        if anchor_index is None or anchor_index+1<context_lookback:
            raise ValueError("authorized replay lacks complete causal lookback at a Validation anchor")
        anchor_frame=frames[anchor_index]
        first_future=frames[anchor_index+1]
        decision_ns=anchor_frame.available_at_ns
        if (anchor_frame.ts_close_ns!=int(anchor) or decision_ns<int(anchor)
                or decision_ns>=first_future.available_at_ns):
            raise ValueError("anchor decision clock must follow availability and precede the first target")
        selected=frames[anchor_index-context_lookback+1:anchor_index+1]
        if any(frame.available_at_ns>decision_ns or frame.ts_close_ns>int(anchor)
               for frame in selected):
            raise ValueError("future or unavailable Validation candle entered a forecast context")
        candles=tuple(Candle(frame.ts_close_ns,frame.available_at_ns,frame.instrument,
            float(frame.open),float(frame.high),float(frame.low),float(frame.close),float(frame.volume))
            for frame in selected)
        encoded=OHLCVCodec.encode(candles[1:],previous_close=candles[0].close)
        contexts.append(frozen_scaler.transform(encoded))
        close_times.append([bar.ts_close_ns for bar in candles[1:]])
        availability.append([bar.available_at_ns for bar in candles[1:]])
        anchor_closes.append(candles[-1].close)
        decision_times.append(decision_ns)
    histories=np.asarray(contexts,dtype=np.float32)
    context_closes=np.asarray(close_times,dtype=np.int64)
    context_availability=np.asarray(availability,dtype=np.int64)
    context_decisions=np.asarray(decision_times,dtype=np.int64)
    with tempfile.NamedTemporaryFile(dir=output_context_npz.parent,suffix=".npz",delete=False) as temp:
        tmp_context=Path(temp.name)
        np.savez_compressed(temp,histories=histories,scaler_mean=frozen_scaler.mean,
            scaler_std=frozen_scaler.std,anchor_timestamps_ns=anchors,
            decision_timestamps_ns=context_decisions,
            close_timestamps=context_closes,available_timestamps=context_availability,
            anchor_closes=np.asarray(anchor_closes,dtype=np.float64))
    try:
        os.replace(tmp_context,output_context_npz)
    finally:
        tmp_context.unlink(missing_ok=True)
    context_sha256=hashlib.sha256(output_context_npz.read_bytes()).hexdigest()
    request={"schema":"routeb.forecast-batch-request.v1","train_scope_id":train_scope_id,
        "dataset_id":fold.dataset_hash,"history_npz":str(output_context_npz),
        "context_sha256":context_sha256,"scaler_sha256":frozen_scaler.fingerprint,
        "horizon":horizon,"n_paths":n_paths,"seed":seed,
        "train_npz_sha256":train_npz_sha256,"instrument":frames[0].instrument,
        "anchor_timestamps_ns":[int(value) for value in anchors],
        "decision_timestamps_ns":[int(value) for value in context_decisions]}
    tmp_request=output_request_json.with_suffix(".json.partial")
    tmp_request.write_text(json.dumps(request,sort_keys=True),encoding="utf-8")
    os.replace(tmp_request,output_request_json)
    return {**request,"status":"PREPARED","evidence_scope":"authorized_fold_validation_context_only",
        "context_artifact_sha256":context_sha256,"test_payload_decoded":False,
        "anchors":len(anchors),"context_lookback":context_lookback,
        "anchor_interval_minutes":anchor_interval_minutes,
        "snapshot_sha256":SNAPSHOT_SHA,"upstream_sha256":PINNED_UPSTREAM}
