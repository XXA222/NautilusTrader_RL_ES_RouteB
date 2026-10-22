# RouteB R4 AVAXUSDT hierarchy comparison — run 01 outcome

Status: failed under a preregistered safety stop condition; the time window is consumed and must not be retried.

- Fold: `406ec709490959d84fc68083728e388ebb5465a403e01c1934352e4d31f32f78`
- Read scope: `[2021-12-26T00:01:00Z, 2023-01-12T00:01:00Z)`
- Source: AVAXUSDT-PERP.BINANCE, Train/Validation payloads only; Test payloads remained sealed.
- Failure: E3, training seed 31 completed its 1,000 updates. During path evaluation, a generated forecast exceeded the OHLCV codec wick limit and raised `ValueError: unphysical generated wick; reject path`.
- Failure report SHA-256: `b1e1e8f4c7a2be345a3ba2a3594875f93dd82f8248bf53d310089bb14d438e4e`
- Failure receipt status: `failed`; Test payload decoded: `false`.

No score was produced from the incomplete ensemble. The codec's physical constraints remain unchanged. Run 02 uses a different, earlier Fold and preregisters 128 candidate forecast seeds per anchor, rejects unsafe candidates without clipping, and requires at least 32 valid paths per anchor. Run 01's timestamps are not reused.
