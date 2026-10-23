"""Research-only deterministic, net-position simulated execution for OHLC bars.

NOT a replacement for NautilusTrader; no hedge legs, liquidation, orderbook,
funding settlement, mark/index basis, or historical margin rules. Economic
certification remains BLOCKED until native differential testing passes.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
import math
from .market import Candle,MINUTE_NS


@dataclass(frozen=True)
class Pending:
    signed_qty:float
    kind:str                     # market | limit
    submitted_at_ns:int
    active_from_ns:int
    limit_price:float|None=None
    initial_signed_qty:float|None=None
    remaining_depth_qty:float|None=None

    def __post_init__(self):
        if (self.kind not in {'market','limit'} or type(self.signed_qty) is bool or not math.isfinite(self.signed_qty)
            or self.signed_qty==0 or type(self.submitted_at_ns) is not int
            or type(self.active_from_ns) is not int):
            raise ValueError("invalid order")
        if self.active_from_ns<self.submitted_at_ns:
            raise ValueError("order activated before submission")
        initial = self.signed_qty if self.initial_signed_qty is None else self.initial_signed_qty
        if (not math.isfinite(initial) or initial * self.signed_qty <= 0
                or abs(initial) + 1e-12 < abs(self.signed_qty)):
            raise ValueError("pending leaves exceed original order quantity")
        object.__setattr__(self, "initial_signed_qty", initial)
        available = abs(self.signed_qty) if self.remaining_depth_qty is None else self.remaining_depth_qty
        if not math.isfinite(available) or not 0 <= available <= abs(initial) + 1e-12:
            raise ValueError("pending simulated depth is outside original order quantity")
        object.__setattr__(self, "remaining_depth_qty", available)
        if self.kind=='limit' and (self.limit_price is None or type(self.limit_price) is bool or not math.isfinite(self.limit_price) or self.limit_price<=0):
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
    """Market events are passed in, never altered by the action.

    The default ``bar_close`` limit policy matches the close-only trigger used
    by the pinned Native bar engine. ``ohlc_touch`` is an explicit scenario
    approximation; OHLC bars cannot establish queue position or fill order.
    """
    def __init__(self,*,initial_equity:float,fee_maker:float,fee_taker:float,
                 max_abs_qty:float=1e6,equity_precision:int=8,
                 limit_fill_policy:str="bar_close",
                 simulated_depth_fraction:float=1.0, market_slippage_bps:float=0.0,
                 price_increment:float=0.01):
        if (any(type(value) is bool for value in (
                initial_equity, fee_maker, fee_taker, max_abs_qty,
                simulated_depth_fraction, market_slippage_bps, price_increment)) or
            initial_equity<=0 or max_abs_qty<=0 or
            not all(math.isfinite(v) for v in (initial_equity,fee_maker,fee_taker,max_abs_qty)) or
            not -1<fee_maker<1 or not -1<fee_taker<1 or
            type(equity_precision) is not int or not 0<=equity_precision<=18 or
            type(limit_fill_policy) is not str or
            limit_fill_policy not in {"bar_close", "ohlc_touch"} or
            not math.isfinite(simulated_depth_fraction) or
            not 0 <= simulated_depth_fraction <= 1 or
            not math.isfinite(market_slippage_bps) or not 0 <= market_slippage_bps < 10_000 or
            not math.isfinite(price_increment) or price_increment <= 0):
            raise ValueError("invalid scenario account settings")
        self.cash=float(initial_equity)
        self.qty=0.0;self.entry=0.0;self.pending:Pending|None=None
        self.fees_paid=0.0;self.max_abs_qty=max_abs_qty;self.equity_precision=equity_precision
        self.maker=float(fee_maker);self.taker=float(fee_taker)
        self.limit_fill_policy=limit_fill_policy
        self.simulated_depth_fraction=float(simulated_depth_fraction)
        self.market_slippage_bps=Decimal(str(market_slippage_bps))
        self.price_increment=Decimal(str(price_increment))
        self.last_close:float|None=None
        self.last_bar_ns:int|None=None
        self.instrument_id: str | None = None
        self.fills:list[Fill]=[]

    def submit(self,*,target_qty:float,decision_ns:int,kind:str="market",
               limit_price:float|None=None,delay_ns:int=0)->None:
        if self.pending is not None:
            raise ValueError("pending order must be cancelled or completed first")
        if (type(target_qty) is bool or not math.isfinite(target_qty) or abs(target_qty)>self.max_abs_qty
            or type(decision_ns) is not int or decision_ns<=0
            or decision_ns%MINUTE_NS!=0
            or type(delay_ns) is not int or delay_ns<0):
            raise ValueError("invalid target/risk limit")
        if self.last_bar_ns is not None and decision_ns<self.last_bar_ns:
            raise ValueError('cannot submit a historical order after consuming later bars')
        delta=target_qty-self.qty
        if abs(delta)<1e-12:
            return
        self.pending=Pending(
            delta, kind, decision_ns, decision_ns+delay_ns, limit_price,
            remaining_depth_qty=abs(delta)*self.simulated_depth_fraction,
        )

    def cancel(self)->bool:
        was=self.pending is not None
        self.pending=None
        return was

    def _market_fill_price(self, bar: Candle, signed_qty: float) -> float:
        reference = Decimal(str(bar.close))
        adverse = self.market_slippage_bps / Decimal("10000")
        if signed_qty > 0:
            raw_price = reference * (Decimal(1) + adverse)
            rounding = ROUND_CEILING
        else:
            raw_price = reference * (Decimal(1) - adverse)
            rounding = ROUND_FLOOR
        with localcontext() as context:
            context.prec = 64
            ticks = (raw_price / self.price_increment).to_integral_value(rounding=rounding)
            price = ticks * self.price_increment
        if price <= 0:
            raise ValueError("adverse market fill price is not positive")
        return float(price)

    def _apply_fill(self,order:Pending,bar:Candle,price:float,*,maker:bool)->Fill:
        old=self.qty
        delta=order.signed_qty
        rate = self.maker if maker else self.taker
        quantum = Decimal(1).scaleb(-self.equity_precision)
        with localcontext() as context:
            context.prec = 64
            fee_decimal = (Decimal(str(abs(delta))) * Decimal(str(price)) * Decimal(str(rate))).quantize(
                quantum,
                rounding=ROUND_HALF_EVEN,
            )
        fee = float(fee_decimal)
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
        if self.last_bar_ns is not None and bar.ts_close_ns != self.last_bar_ns+MINUTE_NS:
            raise ValueError('imagined bar replay must be strictly increasing and contiguous')
        if self.instrument_id is not None and bar.instrument != self.instrument_id:
            raise ValueError('imagined account cannot mix instruments')
        if self.instrument_id is None:
            self.instrument_id = bar.instrument
        self.last_bar_ns=bar.ts_close_ns
        order=self.pending
        result=None
        if order is not None:
            # Without tick data, never fill an order in a bar whose open
            # preceded order activation.
            start=bar.ts_close_ns-MINUTE_NS
            if order.active_from_ns<=start:
                price=None
                maker=False
                limit_price=order.limit_price
                if order.kind=='limit' and limit_price is None:
                    raise ValueError('pending limit order lost its required price')
                if order.kind=='market':
                    price=self._market_fill_price(bar, order.signed_qty)
                elif self.limit_fill_policy == "bar_close":
                    # Native bar matching consumes the bar's close. An OHLC
                    # touch with a close on the other side is insufficient.
                    if order.signed_qty>0 and limit_price is not None and bar.close<=limit_price:
                        price=bar.close
                    elif order.signed_qty<0 and limit_price is not None and bar.close>=limit_price:
                        price=bar.close
                elif order.signed_qty>0 and limit_price is not None and bar.low<=limit_price:
                    price=min(bar.open,limit_price)
                    maker=bar.open>limit_price
                elif order.signed_qty<0 and limit_price is not None and bar.high>=limit_price:
                    price=max(bar.open,limit_price)
                    maker=bar.open<limit_price
                available = order.remaining_depth_qty or 0.0
                fill_quantity = min(abs(order.signed_qty), available)
                if price is not None and fill_quantity > 1e-12:
                    fill_delta = math.copysign(fill_quantity, order.signed_qty)
                    fill_order = Pending(
                        fill_delta, order.kind, order.submitted_at_ns, order.active_from_ns,
                        order.limit_price, order.initial_signed_qty, available,
                    )
                    result=self._apply_fill(fill_order,bar,float(price),maker=maker)
                    remaining = order.signed_qty - fill_delta
                    remaining_depth = max(0.0, available - fill_quantity)
                    self.pending = (None if abs(remaining) <= 1e-12 else Pending(
                        remaining, order.kind, order.submitted_at_ns, order.active_from_ns,
                        order.limit_price, order.initial_signed_qty, remaining_depth,
                    ))
        self.last_close=bar.close
        return result

    @property
    def equity(self)->float:
        value = self.cash if self.qty==0 or self.last_close is None else (
            self.cash+self.qty*(self.last_close-self.entry))
        return round(value, self.equity_precision)
    @property
    def economics_eligible(self)->bool:
        # Funding, liquidation, margin tiers, actual historical exchange rules,
        # intrabar sequencing, native pricing, etc. are intentionally not modeled.
        return False
