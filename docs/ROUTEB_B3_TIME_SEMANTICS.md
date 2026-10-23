# RouteB B3/F2 time semantics

All intervals are half-open and all timestamps below identify candle closes.
The prediction head has one invariant meaning: `head(state[t])` parameterizes
the standardized 1-minute candle at `t+1m`.

| Event | Observable state | Head target | Slow event handling |
|---|---|---|---|
| Training at minute `t` | posterior after observing minute `t`, plus only 1h/1d bars available by `t` | minute `t+1m` | a slow bar available at `t` may condition this head |
| First F2 step at anchor `t` | final history posterior at `t` | minute `t+1m` | no prior advance occurs before this head |
| Later F2 step | prior advanced from the preceding state | next minute after the current state | a slow event available at a target close is applied to the transition and may affect only the following target |

`future_minute_decisions_ns` enumerates target-close barriers. An event at one
of those barriers cannot alter the forecast for the candle that has just closed;
it becomes causal input to the next transition. F2 exports both the anchor head
and the first advanced-prior head so the contract is checked numerically.

The hour/day decoder outputs remain latent-scale diagnostics, not independent
tradable OHLCV forecasts. They must not be interpreted as conserved market
candles until per-path minute sampling and aggregation reconciliation are used.
