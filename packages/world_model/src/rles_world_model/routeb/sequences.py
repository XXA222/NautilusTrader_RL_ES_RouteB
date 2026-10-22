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
from .market import MarketSequence, MINUTE_NS
from .codec import OHLCVCodec, TrainOnlyStandardizer
from ..json_utils import loads_unique_json

SCHEMA="routeb.train-sequences.v3"
RISK_HORIZONS=(1,5,15,30,60)
RISK_BARRIER_VOLATILITY_QUANTILE=0.75


@dataclass(frozen=True)
class SequenceBatch:
    features: np.ndarray  # [N,T,5], standardized
    reset: np.ndarray     # [N,T] bool
    masks: np.ndarray     # [N,T] bool: target-time loss mask, burn-in is False
    timestamps: np.ndarray   # [N,T] UTC close ns
    risk_targets: np.ndarray = None  # type: ignore[assignment]  # [N,T,H,6]
    risk_masks: np.ndarray = None    # type: ignore[assignment]  # [N,T,H] bool
    risk_horizons: tuple[int,...]=RISK_HORIZONS
    risk_barrier_fractions: np.ndarray = None  # type: ignore[assignment]  # [H], Train fitted

    def __post_init__(self):
        if self.risk_targets is None and self.risk_masks is None:
            # Legacy synthetic callers can still validate sequence invariants;
            # production v2 NPZ loading always requires explicit risk tensors.
            object.__setattr__(self,"risk_targets",np.zeros(
                (*self.reset.shape,len(self.risk_horizons),6),dtype=np.float64))
            object.__setattr__(self,"risk_masks",np.zeros(
                (*self.reset.shape,len(self.risk_horizons)),dtype=np.bool_))
        if self.risk_barrier_fractions is None:
            object.__setattr__(self,"risk_barrier_fractions",np.full(
                len(self.risk_horizons),0.01,dtype=np.float64))
        if self.risk_targets is None or self.risk_masks is None:
            raise ValueError('risk targets and masks must be supplied together')
        if (self.features.ndim!=3 or self.features.shape[-1]!=5 or
            self.reset.shape!=self.features.shape[:2] or
            self.masks.shape!=self.reset.shape or
            self.timestamps.shape!=self.reset.shape or
            self.risk_targets.shape[:2]!=self.reset.shape or
            self.risk_targets.shape[2:]!=(len(self.risk_horizons),6) or
            self.risk_masks.shape!=self.risk_targets.shape[:-1] or
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
        if (self.risk_masks.dtype!=np.dtype('bool') or not np.isfinite(self.risk_targets).all()
                or self.risk_barrier_fractions.shape!=(len(self.risk_horizons),)
                or not np.isfinite(self.risk_barrier_fractions).all()
                or (self.risk_barrier_fractions<=0).any()
                or (self.risk_barrier_fractions>=1).any()
                or tuple(sorted(set(self.risk_horizons)))!=self.risk_horizons):
            raise ValueError('invalid multi-horizon risk tensors')


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
    from .risk_targets import build_risk_targets,estimate_train_barrier_fractions
    # Build each anchor's path from Train bars only. Missing tail bars are padded
    # with a geometrically valid candle and an intentionally invalid timestamp.
    raw=np.asarray([[x.open,x.high,x.low,x.close,x.volume] for x in bars],dtype=np.float64)
    bar_ts=np.asarray([x.ts_close_ns for x in bars],dtype=np.int64)
    score_bars=np.ones(len(bars),dtype=bool)
    if loss_interval is not None:
        score_bars=(bar_ts>=loss_interval[0])&(bar_ts<loss_interval[1])
    barriers=estimate_train_barrier_fractions(
        raw[score_bars,3],train_timestamps_ns=bar_ts[score_bars],
        horizons=RISK_HORIZONS,
        volatility_quantile=RISK_BARRIER_VOLATILITY_QUANTILE)
    ncoords=len(coords); width=RISK_HORIZONS[-1]
    anchor_idx=np.arange(1,ncoords+1)
    future=np.empty((ncoords,width,5),dtype=np.float64)
    future_ts=np.zeros((ncoords,width),dtype=np.int64)
    for j,source_idx in enumerate(anchor_idx):
        available=raw[source_idx+1:min(source_idx+1+width,len(raw))]
        take=len(available)
        if take:
            future[j,:take]=available
            future_ts[j,:take]=bar_ts[source_idx+1:source_idx+1+take]
            future[j,take:]=available[-1]
        else:
            future[j]=raw[source_idx]
        # A zero timestamp invalidates all horizons that need padded data.
    anchor_risk=build_risk_targets(future,anchor_close=raw[anchor_idx,3],
        anchor_timestamps_ns=bar_ts[anchor_idx],future_timestamps_ns=future_ts,
        horizons=RISK_HORIZONS,barrier_fraction=barriers)
    risk_values=np.stack([anchor_risk.log_return,
        np.log(anchor_risk.realized_volatility+1e-8),anchor_risk.max_high_return,
        anchor_risk.min_low_return,anchor_risk.upper_barrier_hit.astype(np.float64),
        anchor_risk.lower_barrier_hit.astype(np.float64)],axis=-1)
    risk_valid=anchor_risk.valid.copy()
    if loss_interval is not None:
        for hidx,horizon in enumerate(RISK_HORIZONS):
            end_idx=anchor_idx+horizon
            end_ts=np.zeros(ncoords,dtype=np.int64)
            inside=end_idx<len(bar_ts)
            end_ts[inside]=bar_ts[end_idx[inside]]
            risk_valid[:,hidx]&=(bar_ts[anchor_idx]>=loss_interval[0]) & (end_ts<loss_interval[1])
    feature_windows=[]; reset=[]; masks=[]; timestamps=[]; risk_windows=[]; risk_mask_windows=[]
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
        risk_windows.append(risk_values[i:i+length])
        risk_mask_windows.append(risk_valid[i:i+length])
    if not feature_windows:
        raise ValueError("no windows produced")
    return SequenceBatch(np.stack(feature_windows),np.stack(reset),
                         np.stack(masks),np.asarray(timestamps,dtype=np.int64),
                         np.stack(risk_windows),np.stack(risk_mask_windows),
                         RISK_HORIZONS,barriers)


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
             "risk_horizons":list(batch.risk_horizons),
             "risk_barrier_fractions":batch.risk_barrier_fractions.tolist(),
             "risk_barrier_volatility_quantile":RISK_BARRIER_VOLATILITY_QUANTILE,
             "burn_in_masked":True}
    if fold_sha256 is not None:
        payload["fold_sha256"] = fold_sha256
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".npz",delete=False) as temp:
        tmp=Path(temp.name)
        np.savez_compressed(temp,features=batch.features,reset=batch.reset,
                            masks=batch.masks,timestamps=batch.timestamps,
                            risk_targets=batch.risk_targets,risk_masks=batch.risk_masks,
                            risk_horizons=np.asarray(batch.risk_horizons,dtype=np.int32),
                            risk_barrier_fractions=batch.risk_barrier_fractions,
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
    meta=loads_unique_json(path.with_suffix('.json').read_text('utf-8'))
    if (meta.get('schema')!=SCHEMA or meta.get('purpose')!='train' or
        meta.get('train_scope_id')!=expected_scope or
        meta.get('snapshot_sha')!=snapshot_sha or meta.get('upstream_sha')!=upstream_sha):
        raise ValueError("Train provenance or pinned code identity mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta.get("artifact_sha256"):
        raise ValueError("prepared Train NPZ checksum differs from registered metadata")
    with np.load(path,allow_pickle=False) as arr:
        expected={'features','reset','masks','timestamps','risk_targets','risk_masks',
                  'risk_horizons','risk_barrier_fractions','scaler_mean','scaler_std'}
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
                            arr['masks'].copy(),arr['timestamps'].copy(),
                            arr['risk_targets'].copy(),arr['risk_masks'].copy(),
                            tuple(int(x) for x in arr['risk_horizons']),
                            arr['risk_barrier_fractions'].copy())
    if meta.get('risk_horizons')!=list(batch.risk_horizons):
        raise ValueError('risk horizon metadata mismatch')
    if meta.get('risk_barrier_fractions')!=batch.risk_barrier_fractions.tolist():
        raise ValueError('risk barrier metadata mismatch')
    if meta.get('risk_barrier_volatility_quantile')!=RISK_BARRIER_VOLATILITY_QUANTILE:
        raise ValueError('risk barrier quantile protocol mismatch')
    if list(batch.features.shape)!=meta['features_shape']:
        raise ValueError("feature artifact shape/metadata mismatch")
    return batch,meta
