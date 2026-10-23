"""Train-only causal sequence sampler and safe NPZ materialization.

The authorized RLES Train reader MUST be called before this module. This module
accepts in-memory frames and does not open an arbitrary raw market dataset.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import os
import tempfile
import numpy as np
from rles_data.io import strict_json_loads
from .market import MarketSequence, MINUTE_NS
from .codec import OHLCVCodec, TrainOnlyStandardizer

SCHEMA="routeb.train-sequences.v1"


@dataclass(frozen=True)
class SequenceBatch:
    features: np.ndarray  # [N,T,5], standardized
    reset: np.ndarray     # [N,T] bool
    masks: np.ndarray     # [N,T] bool: target-time loss mask, burn-in is False
    timestamps: np.ndarray   # [N,T] UTC close ns

    def __post_init__(self):
        if (self.features.ndim!=3 or self.features.shape[-1]!=5 or
            self.reset.shape!=self.features.shape[:2] or
            self.masks.shape!=self.reset.shape or
            self.timestamps.shape!=self.reset.shape or
            not np.isfinite(self.features).all()):
            raise ValueError("invalid causal sequence tensors")
        if not np.all(np.diff(self.timestamps,axis=1)==MINUTE_NS):
            raise ValueError("sample crosses market history gap")
        if (self.features.shape[1] < 3 or self.features.shape[0] < 1
            or not np.issubdtype(self.features.dtype, np.floating)
            or self.reset.dtype != np.dtype('bool')
            or self.masks.dtype != np.dtype('bool')
            or self.timestamps.dtype != np.dtype('int64')):
            raise ValueError('invalid sequence batch dtypes')
        if not np.all(self.reset[:,0]) or np.any(self.reset[:,1:]):
            raise ValueError("each training window must reset exactly at its start")
        if np.any(self.masks[:,0]) or np.any(~self.masks[:,-1]):
            raise ValueError('prediction masks must exclude warmup and include last target')
        if np.any(np.diff(self.masks.astype(np.int8),axis=1)<0):
            raise ValueError('prediction target mask cannot turn off after burn-in')


def make_windows(replay: MarketSequence, *, scaler: TrainOnlyStandardizer,
                 length: int=128, burn_in: int=32, stride: int=64,
                 decision_cutoff_ns: int | None=None,
                 loss_interval: tuple[int,int] | None=None) -> SequenceBatch:
    if replay.scope!="train" or scaler.dataset_id!=replay.dataset_id or scaler.train_scope!="train":
        raise ValueError("not an authorized same-identity Train sequence")
    if length<3 or burn_in<1 or burn_in>=length-1 or stride<=0:
        raise ValueError("bad sequence length, burn_in or stride")
    if (loss_interval is not None
            and (not isinstance(loss_interval,tuple) or len(loss_interval)!=2
                 or any(type(value) is not int for value in loss_interval)
                 or loss_interval[0]>=loss_interval[1])):
        raise ValueError("loss interval must be a valid half-open integer-ns range")
    bars=replay.candles
    if decision_cutoff_ns is not None:
        bars=tuple(x for x in bars if x.available_at_ns<=decision_cutoff_ns)
    if len(bars)<length+1:
        raise ValueError("not enough source bars for one full window and anchor")
    for a,b in zip(bars,bars[1:]):
        if b.ts_close_ns-a.ts_close_ns!=MINUTE_NS:
            raise ValueError("after availability filtering, bars must remain contiguous")
    coords=OHLCVCodec.encode(bars[1:],previous_close=bars[0].close)
    standardized=scaler.transform(coords)
    feature_windows=[]; reset=[]; masks=[]; timestamps=[]
    # Prevent crossing end-of-Train boundary: caller explicitly passes only Train.
    for i in range(0,len(coords)-length+1,stride):
        # The final target IS supervised: at t=T-2 we predict the final bar.
        # Previous code masked m[-1] and wasted one valid prediction per window.
        window_timestamps=np.asarray(
            [x.ts_close_ns for x in bars[i+1:i+1+length]],dtype=np.int64)
        if loss_interval is None:
            m=np.ones(length,dtype=np.bool_)
        else:
            m=((window_timestamps>=loss_interval[0])
               &(window_timestamps<loss_interval[1]))
        m[:burn_in]=False
        # Omit context-only windows and windows whose final target is outside
        # the Train score interval. SequenceBatch requires a supervised tail.
        if not m.any() or not m[-1]:
            continue
        feature_windows.append(standardized[i:i+length])
        r=np.zeros(length,dtype=np.bool_);r[0]=True;reset.append(r)
        masks.append(m)
        timestamps.append(window_timestamps)
    if not feature_windows:
        raise ValueError("no windows produced")
    return SequenceBatch(np.stack(feature_windows),np.stack(reset),
                         np.stack(masks),np.asarray(timestamps,dtype=np.int64))


def export_train_sequences(path: str|Path, batch: SequenceBatch,
                           scaler: TrainOnlyStandardizer, *, snapshot_sha: str,
                           upstream_sha: str, train_scope_id: str,
                           fold_sha256: str | None = None)->tuple[Path,Path]:
    """Atomically materialize a host-authorized Train-only batch.

    Data carries no Validation/Test/Holdout targets; worker is not given a raw path.
    """
    if not train_scope_id or len(snapshot_sha)!=40 or len(upstream_sha)!=40:
        raise ValueError("requires snapshot/upstream SHA and registered Train scope")
    if (fold_sha256 is not None
            and (len(fold_sha256)!=64
                 or any(character not in "0123456789abcdef" for character in fold_sha256))):
        raise ValueError("Fold identity must be a lowercase SHA-256 digest")
    path=Path(path).resolve(); path.parent.mkdir(parents=True,exist_ok=True)
    meta_path=path.with_suffix('.json')
    payload={"schema":SCHEMA,"purpose":"train","train_scope_id":train_scope_id,
             "dataset_id":scaler.dataset_id,"scaler_sha256":scaler.fingerprint,"snapshot_sha":snapshot_sha,
             "upstream_sha":upstream_sha,"features_shape":list(batch.features.shape),
             "burn_in_masked":True}
    if fold_sha256 is not None:
        payload["fold_sha256"] = fold_sha256
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".npz",delete=False) as temp:
        tmp=Path(temp.name)
        np.savez_compressed(temp,features=batch.features,reset=batch.reset,
                            masks=batch.masks,timestamps=batch.timestamps,
                            **scaler.arrays())  # type: ignore[arg-type]
    try:
        os.replace(tmp,path)
        payload["artifact_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        t=meta_path.with_suffix('.json.partial')
        t.write_text(json.dumps(payload,sort_keys=True),encoding='utf-8')
        os.replace(t,meta_path)
    finally:
        tmp.unlink(missing_ok=True)
    return path,meta_path


def load_worker_batch(path: str|Path, *, expected_scope: str,
                      snapshot_sha: str,upstream_sha: str)->tuple[SequenceBatch,dict]:
    """Loads prepared features only; validates metadata but does not grant data access."""
    path=Path(path)
    meta=strict_json_loads(
        path.with_suffix('.json').read_bytes(), context='Route B Train sequence manifest',
    )
    if (meta.get('schema')!=SCHEMA or meta.get('purpose')!='train' or
        meta.get('train_scope_id')!=expected_scope or
        meta.get('snapshot_sha')!=snapshot_sha or meta.get('upstream_sha')!=upstream_sha):
        raise ValueError("Train provenance or pinned code identity mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta.get("artifact_sha256"):
        raise ValueError("prepared Train NPZ checksum differs from registered metadata")
    with np.load(path,allow_pickle=False) as arr:
        expected={'features','reset','masks','timestamps','scaler_mean','scaler_std'}
        if set(arr.files)!=expected:
            raise ValueError('Train NPZ member schema mismatch')
        mean=np.asarray(arr['scaler_mean'],dtype=np.float64)
        std=np.asarray(arr['scaler_std'],dtype=np.float64)
        if (mean.shape!=(5,) or std.shape!=(5,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std<=0).any()):
            raise ValueError('Train NPZ contains an invalid scaler')
        validated_scaler=TrainOnlyStandardizer(mean,std,meta['dataset_id'],'train')
        if validated_scaler.fingerprint!=meta.get('scaler_sha256'):
            raise ValueError('Train scaler data does not match identity')
        batch=SequenceBatch(arr['features'].copy(),arr['reset'].copy(),
                            arr['masks'].copy(),arr['timestamps'].copy())
    if list(batch.features.shape)!=meta['features_shape']:
        raise ValueError("feature artifact shape/metadata mismatch")
    return batch,meta
