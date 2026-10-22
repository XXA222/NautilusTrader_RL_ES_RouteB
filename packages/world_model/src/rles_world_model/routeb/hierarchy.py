"""Causal 1m/hour/day event planning for the RouteB hierarchical RSSM."""
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


def derive_future_scale_events(*, minute_decisions_ns: np.ndarray,
        hour_available_at_ns: np.ndarray, day_available_at_ns: np.ndarray
        ) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Map slow-bar availability timestamps onto future minute decisions.

    Slow timestamp inputs are an exogenous availability schedule, not OHLCV
    observations. A state is emitted at the first decision whose barrier is at
    or after its actual availability time, matching ``build_plan``'s right-
    sided search. More than one new bar of one scale at a decision is rejected
    because F2 advances each slow RSSM one step per event.
    """
    decisions = np.asarray(minute_decisions_ns)
    hours = np.asarray(hour_available_at_ns)
    days = np.asarray(day_available_at_ns)
    if (decisions.ndim != 1 or not len(decisions) or hours.ndim != 1 or days.ndim != 1
            or any(x.dtype.kind not in "iu" for x in (decisions, hours, days))
            or np.any(decisions <= 0) or np.any(hours <= 0) or np.any(days <= 0)
            or np.any(np.diff(decisions) <= 0) or np.any(np.diff(hours) < 0)
            or np.any(np.diff(days) < 0)):
        raise ValueError("future minute decisions and slow availability timestamps must be positive, sorted integers")
    if ((hours.size and hours[-1] > decisions[-1])
            or (days.size and days[-1] > decisions[-1])):
        raise ValueError("future slow availability schedule extends beyond forecast decisions")

    def events(available: np.ndarray, name: str) -> tuple[bool, ...]:
        prior = 0
        result = []
        for decision in decisions:
            visible = int(np.searchsorted(available, decision, side="right"))
            delta = visible - prior
            if delta not in (0, 1):
                raise ValueError(f"multiple future {name} states become visible at one minute decision")
            result.append(delta == 1)
            prior = visible
        return tuple(result)

    hour_events = events(hours, "hour")
    day_events = events(days, "day")
    return hour_events, day_events
