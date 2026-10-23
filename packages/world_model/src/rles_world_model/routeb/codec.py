"""OHLCV path parameterization with valid price geometry.

Coordinates: log(open/previous_close), log(close/open), log(upper_wick+eps),
log(lower_wick+eps), log1p(volume). Coordinates are real-valued and can be
modeled with normal / mixture distributions, then deterministically inverted.
"""
from __future__ import annotations
import math
import hashlib
from dataclasses import dataclass
import numpy as np
from .market import Candle, MINUTE_NS


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    mean: np.ndarray
    std: np.ndarray
    dataset_id: str
    train_scope: str

    @classmethod
    def fit(cls, features: np.ndarray, *, dataset_id: str, scope: str) -> TrainOnlyStandardizer:
        if scope != "train":
            raise ValueError("fit must use authorized Train data only")
        x=np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 5 or x.shape[0] < 2 or not np.isfinite(x).all():
            raise ValueError("valid [N,5] finite training features required")
        return cls(x.mean(0), np.maximum(x.std(0), 1e-6), dataset_id, scope)

    def __post_init__(self) -> None:
        # Freeze own copies, not caller-owned mutable arrays.
        object.__setattr__(self,'mean',np.array(self.mean,dtype=np.float64,copy=True))
        object.__setattr__(self,'std',np.array(self.std,dtype=np.float64,copy=True))
        if (self.mean.shape!=(5,) or self.std.shape!=(5,)
            or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
            or np.any(self.std<=0) or not self.dataset_id or self.train_scope!='train'):
            raise ValueError('frozen Train scaler identity or moments invalid')
        self.mean.setflags(write=False)
        self.std.setflags(write=False)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x=np.asarray(x,dtype=np.float64)
        if x.shape[-1] != 5 or not np.isfinite(x).all():
            raise ValueError("bad feature shape or value")
        return ((x-self.mean)/self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x=np.asarray(x,dtype=np.float64)
        if x.shape[-1] != 5 or not np.isfinite(x).all():
            raise ValueError("bad feature shape or value")
        return x*self.std+self.mean

    @property
    def fingerprint(self)->str:
        # Bind the frozen fitting moments and their origin to the JAX checkpoint.
        digest=hashlib.sha256()
        digest.update(self.dataset_id.encode('utf-8'))
        digest.update(b'\0train\0')
        digest.update(np.asarray(self.mean,dtype='<f8').tobytes())
        digest.update(np.asarray(self.std,dtype='<f8').tobytes())
        return digest.hexdigest()

    def arrays(self)->dict[str,np.ndarray]:
        return {"scaler_mean":self.mean,"scaler_std":self.std}


class OHLCVCodec:
    EPS = 1e-6

    @staticmethod
    def encode(candles: tuple[Candle,...], *, previous_close: float) -> np.ndarray:
        if previous_close <= 0 or not math.isfinite(previous_close) or not candles:
            raise ValueError("valid previous close and nonempty candles required")
        rows=[]
        last=previous_close
        for f in candles:
            upper=math.log(f.high/max(f.open,f.close))
            lower=math.log(min(f.open,f.close)/f.low)
            rows.append((math.log(f.open/last),math.log(f.close/f.open),
                         math.log(upper+OHLCVCodec.EPS),
                         math.log(lower+OHLCVCodec.EPS), math.log1p(f.volume)))
            last=f.close
        return np.asarray(rows,dtype=np.float32)

    @staticmethod
    def decode(y: np.ndarray, *, previous_close: float, start_close_ns: int,
               instrument: str, availability_lag_ns: int = 0,
               max_abs_log: float = 18.) -> tuple[Candle,...]:
        if type(start_close_ns) is not int or start_close_ns<=0 or start_close_ns%MINUTE_NS:
            raise ValueError("forecast timestamp must be UTC minute close aligned")
        vals=np.asarray(y,dtype=np.float64)
        if vals.ndim != 2 or vals.shape[1]!=5 or not np.isfinite(vals).all():
            raise ValueError("decoded features must be finite [T,5]")
        if previous_close<=0 or not math.isfinite(previous_close):
            raise ValueError("invalid price anchor")
        if availability_lag_ns<0 or type(availability_lag_ns) is not int:
            raise ValueError("invalid availability lag")
        if np.any(np.abs(vals)>max_abs_log):
            raise ValueError("unsafe extrapolated feature; reject path instead of clipping")
        out=[]
        last=previous_close
        for i,(gap,body,upper,lower,logv) in enumerate(vals):
            op=last*math.exp(float(gap))
            cl=op*math.exp(float(body))
            wick_up=max(0.0, math.exp(float(upper))-OHLCVCodec.EPS)
            wick_down=max(0.0, math.exp(float(lower))-OHLCVCodec.EPS)
            if wick_up>2 or wick_down>2:
                raise ValueError("unphysical generated wick; reject path")
            hi=max(op,cl)*math.exp(wick_up)
            lo=min(op,cl)/math.exp(wick_down)
            vol=max(0.,math.expm1(float(logv)))
            ts=start_close_ns+i*MINUTE_NS
            if not all(math.isfinite(v) and v>0 for v in (op,hi,lo,cl)) or not math.isfinite(vol):
                raise ValueError("nonfinite/invalid forecast path")
            out.append(Candle(ts,ts+availability_lag_ns,instrument,op,hi,lo,cl,vol))
            last=cl
        return tuple(out)
