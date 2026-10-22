"""Causal 1m/hour/day event-plan for a future multi-RSSM hierarchy.

This is a COMPLETE causal schedule contract, not a trained 3-RSSM neural model.
The later JAX hierarchy trainer must consume these masks/indexes unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .clock import MultiScaleClock
from .market import Candle

@dataclass(frozen=True)
class HierarchyPlan:
    minute_closes: np.ndarray      # [T]
    hour_last_visible: np.ndarray  # [T], index in hourly buckets or -1
    day_last_visible: np.ndarray   # [T], index in daily buckets or -1
    hour_events: np.ndarray        # [T] boolean (newly available at decision time)
    day_events: np.ndarray         # [T] boolean
    hour_closes: np.ndarray
    day_closes: np.ndarray
    hour_available_at: np.ndarray
    day_available_at: np.ndarray


def build_plan(candles:tuple[Candle,...])->HierarchyPlan:
    clock=MultiScaleClock(candles)
    decisions=np.asarray([c.available_at_ns for c in candles],dtype=np.int64)
    # Decisions can occur between closes when availability is delayed.
    if np.any(np.diff(decisions)<0):
        raise ValueError("nonmonotonic availability")
    hour=clock.buckets(60,as_of_ns=int(decisions[-1]))
    day=clock.buckets(1440,as_of_ns=int(decisions[-1]))
    ha=np.asarray([b.available_at_ns for b in hour],dtype=np.int64)
    da=np.asarray([b.available_at_ns for b in day],dtype=np.int64)
    # searchsorted(right) reveals buckets only after they are actually available.
    hi=np.searchsorted(ha,decisions,side='right')-1
    di=np.searchsorted(da,decisions,side='right')-1
    he=np.r_[False,np.diff(hi)>0]
    de=np.r_[False,np.diff(di)>0]
    if len(hi):
        he[0]=(hi[0]>=0)
        de[0]=(di[0]>=0)
    return HierarchyPlan(np.asarray([c.ts_close_ns for c in candles],dtype=np.int64),
                         hi,di,he,de,np.asarray([b.close_ns for b in hour],dtype=np.int64),
                         np.asarray([b.close_ns for b in day],dtype=np.int64),ha,da)


def causal_state_lookup(states:np.ndarray, indices:np.ndarray)->np.ndarray:
    """Gather last available slow-state vector without using future states.

    Missing state uses all zero vectors and must carry its own availability mask.
    """
    x=np.asarray(states)
    idx=np.asarray(indices)
    if x.ndim!=2 or idx.ndim!=1 or (idx>=len(x)).any() or (idx < -1).any():
        raise ValueError("invalid causal slow state gather")
    if len(x)==0:
        return np.zeros((len(idx),0),dtype=np.float32)
    return np.where((idx>=0)[:,None],x[np.maximum(idx,0)],0)
