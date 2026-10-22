"""F2 synthetic high-timeframe bar builder, no access to realized future data.

Maintains a contiguous observed history and simulated continuation. High-timeframe
callbacks receive ONLY fully closed bars. Does not itself train/advance RSSMs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal
from .market import Candle,MINUTE_NS
from .clock import MultiScaleClock,ClosedBucket


class SyntheticFeedback:
    def __init__(self, observed_prefix:tuple[Candle,...], *, anchor_ns:int):
        if not observed_prefix or observed_prefix[-1].ts_close_ns!=anchor_ns:
            raise ValueError("observed prefix must end exactly at anchor")
        if any(c.ts_close_ns>anchor_ns or c.available_at_ns>anchor_ns for c in observed_prefix):
            raise ValueError("future/unavailable observed candle at forecast anchor")
        self.anchor_ns=anchor_ns
        self.observed_prefix=tuple(observed_prefix)
        self.simulated:list[Candle]=[]
        self.clock=MultiScaleClock(self.observed_prefix)
        self._hour_close=((anchor_ns//(60*MINUTE_NS))+1)*60*MINUTE_NS
        self._day_close=((anchor_ns//(1440*MINUTE_NS))+1)*1440*MINUTE_NS
        self._hour_rows=self._partial_bucket(60,self._hour_close)
        self._day_rows=self._partial_bucket(1440,self._day_close)

    def _partial_bucket(self,timeframe:int,close_ns:int)->list[Candle]:
        start=close_ns-(timeframe-1)*MINUTE_NS
        return [bar for bar in self.observed_prefix
                if start<=bar.ts_close_ns<=self.anchor_ns]

    @staticmethod
    def _close_bucket(timeframe:int,close_ns:int,rows:list[Candle])->ClosedBucket:
        start=close_ns-(timeframe-1)*MINUTE_NS
        if (len(rows)!=timeframe or rows[0].ts_close_ns!=start
                or rows[-1].ts_close_ns!=close_ns
                or any(b.ts_close_ns-a.ts_close_ns!=MINUTE_NS
                       for a,b in zip(rows,rows[1:]))):
            raise ValueError(
                f'F2 needs sufficient observed prefix for complete {timeframe}m bucket')
        available=max(bar.available_at_ns for bar in rows)
        return ClosedBucket(timeframe,close_ns,available,rows[0].instrument,
            rows[0].open,max(bar.high for bar in rows),min(bar.low for bar in rows),
            rows[-1].close,sum(bar.volume for bar in rows))

    def advance(self,synthetic:Candle)->tuple[ClosedBucket,...]:
        expected=self.anchor_ns+(len(self.simulated)+1)*MINUTE_NS
        if synthetic.ts_close_ns!=expected or synthetic.instrument!=self.observed_prefix[-1].instrument:
            raise ValueError("synthetic continuation must be contiguous and same market")
        if synthetic.available_at_ns>synthetic.ts_close_ns:
            raise ValueError("F2 has no synthetic exchange-latency model; reject delayed synthetic candle")
        result=[]
        self._hour_rows.append(synthetic)
        if synthetic.ts_close_ns==self._hour_close:
            result.append(self._close_bucket(60,self._hour_close,self._hour_rows))
            self._hour_rows=[]
            self._hour_close+=60*MINUTE_NS
        elif synthetic.ts_close_ns>self._hour_close:
            raise ValueError('F2 skipped an hourly close event')
        self._day_rows.append(synthetic)
        if synthetic.ts_close_ns==self._day_close:
            result.append(self._close_bucket(1440,self._day_close,self._day_rows))
            self._day_rows=[]
            self._day_close+=1440*MINUTE_NS
        elif synthetic.ts_close_ns>self._day_close:
            raise ValueError('F2 skipped a daily close event')
        self.simulated.append(synthetic)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class F2Result:
    mode: Literal["F2_SLOW_ONLY", "F2_FULL_FILTER"]
    bars: tuple[Candle, ...]
    minute_states: tuple[object, ...]
    hour_state: object
    day_state: object
    hour_updates: int
    day_updates: int
    score_capability: Literal["SCENARIO_ONLY"] = "SCENARIO_ONLY"
    final_minute_state: object = None
    minute_posterior_updates: int = 0
    minute_posterior_max_abs_logit_delta: float | None = None


def rollout_hierarchical_f2(*, feedback: SyntheticFeedback, steps: int,
        minute_state: object, hour_state: object, day_state: object,
        minute_prior: Callable[[object, object, object, int, int], tuple[object, Candle]],
        hour_posterior: Callable[[object, ClosedBucket, object, int], object],
        day_posterior: Callable[[object, ClosedBucket, int], object],
        mode: Literal["F2_SLOW_ONLY", "F2_FULL_FILTER"],
        minute_posterior: Callable[[object, Candle, object, object], object] | None = None,
        seed: int = 0, record_minute_states: bool = True) -> F2Result:
    """Run an event-driven F2 loop using generated-only slow observations.

    The callbacks are the model boundary: ``minute_prior`` receives the current
    latent states and a stateless seed, never a trade action or realized future.
    At each generated close, day events update before same-time hour events.
    ``F2_SLOW_ONLY`` advances minute latent state through its prior; the separate
    ``F2_FULL_FILTER`` mode also filters that step's synthetic minute bar.
    Results are scenario-only diagnostics, never forecast skill evidence.
    """
    if type(steps) is not int or not 1 <= steps <= 1440:
        raise ValueError("F2 horizon must be an integer in [1,1440]")
    if type(seed) is not int or seed < 0:
        raise ValueError("F2 seed must be a nonnegative integer")
    if type(record_minute_states) is not bool:
        raise ValueError("record_minute_states must be a bool")
    if mode not in {"F2_SLOW_ONLY", "F2_FULL_FILTER"}:
        raise ValueError("unknown F2 update mode")
    if mode == "F2_FULL_FILTER" and minute_posterior is None:
        raise ValueError("F2_FULL_FILTER requires an explicit synthetic-minute posterior callback")
    if mode == "F2_SLOW_ONLY" and minute_posterior is not None:
        raise ValueError("F2_SLOW_ONLY must not receive a minute posterior callback")

    generated: list[Candle] = []
    minute_states: list[object] = []
    hour_updates = 0
    day_updates = 0
    current_minute = minute_state
    current_hour = hour_state
    current_day = day_state
    for index in range(steps):
        prior_minute, synthetic = minute_prior(
            current_minute, current_hour, current_day,
            feedback.anchor_ns + (index + 1) * MINUTE_NS, seed + index)
        events = feedback.advance(synthetic)
        # SyntheticFeedback emits daily then hourly events. Sort explicitly to
        # preserve the same order if another event source is supplied later.
        for event in sorted(events, key=lambda item: -item.timeframe_minutes):
            if event.timeframe_minutes == 1440:
                current_day = day_posterior(current_day, event, seed + index)
                day_updates += 1
            elif event.timeframe_minutes == 60:
                current_hour = hour_posterior(
                    current_hour, event, current_day, seed + index)
                hour_updates += 1
            else:
                raise ValueError("F2 slow callbacks accept only complete hour/day buckets")
        current_minute = prior_minute
        if mode == "F2_FULL_FILTER":
            assert minute_posterior is not None
            current_minute = minute_posterior(
                prior_minute, synthetic, current_hour, current_day)
        generated.append(synthetic)
        if record_minute_states:
            minute_states.append(current_minute)
    return F2Result(mode, tuple(generated), tuple(minute_states), current_hour,
                    current_day, hour_updates, day_updates,
                    final_minute_state=current_minute)
