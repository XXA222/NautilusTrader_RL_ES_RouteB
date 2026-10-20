"""F2 synthetic high-timeframe bar builder, no access to realized future data.

Maintains a contiguous observed history and simulated continuation. High-timeframe
callbacks receive ONLY fully closed bars. Does not itself train/advance RSSMs.
"""
from __future__ import annotations
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
        self._seen_hour={b.close_ns for b in self.clock.buckets(60,as_of_ns=anchor_ns)}
        self._seen_day={b.close_ns for b in self.clock.buckets(1440,as_of_ns=anchor_ns)}

    def advance(self,synthetic:Candle)->tuple[ClosedBucket,...]:
        expected=self.anchor_ns+(len(self.simulated)+1)*MINUTE_NS
        if synthetic.ts_close_ns!=expected or synthetic.instrument!=self.observed_prefix[-1].instrument:
            raise ValueError("synthetic continuation must be contiguous and same market")
        if synthetic.available_at_ns>synthetic.ts_close_ns:
            raise ValueError("F2 has no synthetic exchange-latency model; reject delayed synthetic candle")
        source=self.observed_prefix+tuple(self.simulated)+(synthetic,)
        schedule=MultiScaleClock(source)
        result=[]
        for tf,seen in ((1440,self._seen_day),(60,self._seen_hour)):
            for b in schedule.buckets(tf,as_of_ns=synthetic.ts_close_ns):
                if b.close_ns>self.anchor_ns and b.close_ns not in seen:
                    seen.add(b.close_ns)
                    result.append(b)
        # A boundary was crossed. If the complete candle is absent, the caller
        # provided too short an observed prefix to form a synthetic slow bar.
        # Silently skipping it would corrupt the future RSSM update schedule.
        for tf in (60, 1440):
            if synthetic.ts_close_ns % (tf * MINUTE_NS) == 0:
                if not any(b.timeframe_minutes == tf and b.close_ns == synthetic.ts_close_ns
                           for b in result):
                    raise ValueError(f'F2 needs sufficient observed prefix for complete {tf}m bucket')
        self.simulated.append(synthetic)
        return tuple(result)
