"""Research-only deterministic, net-position simulated execution for OHLC bars.

NOT a replacement for NautilusTrader; no hedge legs, liquidation, orderbook,
funding settlement, mark/index basis, or historical margin rules. Economic
certification remains BLOCKED until native differential testing passes.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from .market import Candle,MINUTE_NS


@dataclass(frozen=True)
class Pending:
    signed_qty:float
    kind:str                     # market | limit
    submitted_at_ns:int
    active_from_ns:int
    limit_price:float|None=None

    def __post_init__(self):
        if (self.kind not in {'market','limit'} or not math.isfinite(self.signed_qty)
            or self.signed_qty==0 or type(self.submitted_at_ns) is not int
            or type(self.active_from_ns) is not int):
            raise ValueError("invalid order")
        if self.active_from_ns<self.submitted_at_ns:
            raise ValueError("order activated before submission")
        if self.kind=='limit' and (self.limit_price is None or not math.isfinite(self.limit_price) or self.limit_price<=0):
            raise ValueError("limit needs price")
        if self.kind=='market' and self.limit_price is not None:
            raise ValueError("market order cannot carry a limit price")


@dataclass(frozen=True)
class Fill:
    ts_close_ns:int
    signed_qty:float
    price:float
    fee:float
    maker:bool


class ImaginedAccount:
    """Market events are passed in, never altered by the action."""
    def __init__(self,*,initial_equity:float,fee_maker:float,fee_taker:float,
                 max_abs_qty:float=1e6):
        if (initial_equity<=0 or max_abs_qty<=0 or
            not all(math.isfinite(v) for v in (initial_equity,fee_maker,fee_taker,max_abs_qty)) or
            not 0<=fee_maker<1 or not 0<=fee_taker<1):
            raise ValueError("invalid scenario account settings")
        self.cash=float(initial_equity)
        self.qty=0.0;self.entry=0.0;self.pending:Pending|None=None
        self.fees_paid=0.0;self.max_abs_qty=max_abs_qty
        self.maker=float(fee_maker);self.taker=float(fee_taker)
        self.last_close:float|None=None
        self.last_bar_ns:int|None=None
        self.last_available_ns:int|None=None
        self.instrument:str|None=None
        self.fills:list[Fill]=[]

    def submit(self,*,target_qty:float,decision_ns:int,kind:str="market",
               limit_price:float|None=None,delay_ns:int=0)->None:
        if self.pending is not None:
            raise ValueError("pending order must be cancelled or completed first")
        if (not math.isfinite(target_qty) or abs(target_qty)>self.max_abs_qty
            or type(decision_ns) is not int or decision_ns<=0
            or decision_ns%MINUTE_NS!=0
            or type(delay_ns) is not int or delay_ns<0):
            raise ValueError("invalid target/risk limit")
        if self.last_available_ns is not None and decision_ns<self.last_available_ns:
            raise ValueError('cannot submit a historical order before the last bar was available')
        delta=target_qty-self.qty
        if abs(delta)<1e-12:
            return
        self.pending=Pending(delta,kind,decision_ns,decision_ns+delay_ns,limit_price)

    def cancel(self)->bool:
        was=self.pending is not None
        self.pending=None
        return was

    def _apply_fill(self,order:Pending,bar:Candle,price:float,*,maker:bool)->Fill:
        old=self.qty
        delta=order.signed_qty
        fee=abs(delta*price)*(self.maker if maker else self.taker)
        # Realized PNL on reduction or reversal, never count both realized and MTM.
        if old and old*delta<0:
            closed=min(abs(old),abs(delta))
            self.cash+=closed*(price-self.entry)*(1. if old>0 else -1.)
        self.cash-=fee
        new=old+delta
        if abs(new)<1e-12:
            self.entry=0.0;new=0.0
        elif abs(old)<1e-12 or old*new<0:
            self.entry=price
        elif old*delta>0:
            self.entry=(abs(old)*self.entry+abs(delta)*price)/abs(new)
        self.qty=new
        self.fees_paid+=fee
        fill=Fill(bar.ts_close_ns,delta,price,fee,maker)
        self.fills.append(fill)
        return fill

    def on_bar(self,bar:Candle)->Fill|None:
        if self.instrument is not None and bar.instrument!=self.instrument:
            raise ValueError('imagined account cannot switch instruments during replay')
        if self.last_bar_ns is not None and bar.ts_close_ns != self.last_bar_ns+MINUTE_NS:
            raise ValueError('imagined bar replay must be strictly increasing and contiguous')
        if self.last_available_ns is not None and bar.available_at_ns<self.last_available_ns:
            raise ValueError('imagined bar availability must be monotonic')
        self.instrument=bar.instrument
        self.last_bar_ns=bar.ts_close_ns
        self.last_available_ns=bar.available_at_ns
        order=self.pending
        result=None
        if order is not None:
            # Without tick data, NEVER fill an order in a bar whose open
            # preceded order activation. This avoids intrabar hindsight fills.
            start=bar.ts_close_ns-MINUTE_NS
            if order.active_from_ns<=start:
                price=None
                maker=False
                limit_price=order.limit_price
                if order.kind=='limit' and limit_price is None:
                    raise ValueError('pending limit order lost its required price')
                if order.kind=='market':
                    price=bar.open
                elif order.signed_qty>0 and limit_price is not None and bar.low<=limit_price:
                    price=min(bar.open,limit_price)  # crossed opening price pays taker fee
                    maker=bar.open>limit_price
                elif order.signed_qty<0 and limit_price is not None and bar.high>=limit_price:
                    price=max(bar.open,limit_price)
                    maker=bar.open<limit_price
                if price is not None:
                    # The maker flag is a conservative OHLC-only heuristic:
                    # a marketable-at-open limit pays taker, an intrabar touch maker.
                    # Native execution parity is not claimed.
                    result=self._apply_fill(order,bar,float(price),maker=maker)
                    self.pending=None
        self.last_close=bar.close
        return result

    @property
    def equity(self)->float:
        return self.cash if self.qty==0 or self.last_close is None else (
            self.cash+self.qty*(self.last_close-self.entry))

    @property
    def economics_eligible(self)->bool:
        # Funding, liquidation, margin tiers, actual historical exchange rules,
        # intrabar sequencing, native pricing, etc. are intentionally not modeled.
        return False
