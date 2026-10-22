# RouteB R4 DOGEUSDT single-scale versus hierarchy comparison

Status: protocol and date scope frozen before downloading or decoding these market rows.

## Purpose and prior evidence

The earlier R4 BNB/XRP comparisons used unequal auxiliary objectives. The registered R4 runner now derives E3's Train-only multi-horizon risk targets, scale floor, barrier quantile, and horizon weights from E2's frozen configuration. This fresh DOGEUSDT comparison evaluates that objective-matched runner on a market and Train/Validation interval with no overlap in the RouteB research ledger. It is a predictive system comparison only; it cannot certify profitability or auto-promote a model.

## Frozen Fold and data boundary

- Instrument: `DOGEUSDT-PERP.BINANCE`; source: official Binance USD-M Futures 1-minute klines.
- Train score: `[2024-08-13T00:01:00Z, 2025-08-08T00:01:00Z)` (360 days).
- Validation score: `[2025-08-09T00:01:00Z, 2025-08-24T00:01:00Z)` (15 days).
- Test score: `[2025-08-25T00:01:00Z, 2025-09-09T00:01:00Z)` (15 days); OHLCV after the Validation cutoff must be sealed as timestamp-only markers.
- Train warmup starts `2024-08-11T00:01:00Z`; Validation warmup starts `2025-08-07T00:01:00Z`. Purge is 60 minutes; embargo is 1,380 minutes.
- Registered read scope is `[2024-08-11T00:01:00Z, 2025-08-24T00:01:00Z)`. A read-only RouteB access and query-intent ledger preflight found zero overlapping intervals. The runner must repeat this check immediately before reservation; any overlap stops the run.
- Source open-time interval is `[2024-08-06T00:00:00Z, 2025-09-09T00:00:00Z)`, giving the frozen warmup and timestamp coverage. Only official monthly Binance archives and their SHA-256 checksum files may be used.
- Train and Validation market values may be read only after the runner reserves this exact scope. Test/Holdout values must not be decoded by the data preparation, model worker, evaluator, or report path.

## Frozen comparison

Controls are defined in [`r4_dogeusdt_e2_e3_baseline_protocol.json`](../../configs/routeb/r4_dogeusdt_e2_e3_baseline_protocol.json). Compare `R4_E2_SINGLE_SCALE` and `R4_E3_HIERARCHY` with ordered training seeds 31, 102, and 20261018.

- E2 uses the established RSSM with 512-minute windows and 2,067 updates (480 targets/update; 992,160 total Train targets).
- E3 uses `e3_features`, 1,024-minute windows and 1,000 updates (992 targets/update; 992,000 total Train targets). Both use batch size 1, burn-in 32, learning rate 0.0003, and gradient clip 100.
- E3's auxiliary multi-horizon risk objective, scale floor, quantile and horizon weights are derived by the frozen runner from E2. No independent retuning is permitted.
- Use 60 Validation anchors six hours apart, a 15-minute horizon, 128 ordered candidate forecasts and 32 accepted paths per anchor. Decode batches of eight. Primary metric: path log-realized-volatility CRPS; return CRPS is a no-harm guardrail. Bootstrap uses 12-anchor blocks and 2,000 replicates, with Holm correction over the three seed-level primary tests.
- Promotion requires every seed's primary 95% CI below zero, every Holm-adjusted p-value below 0.05, and no return-CRPS harm. Otherwise retain E2 as the default. No economic certification, automatic promotion, or production readiness follows.

## One-time attempt policy

Archive download, source preparation, and the registered run must preserve immutable receipts. The date scope and protocol are frozen before downloads. After source preparation seals Test rows, the final Fold config and runner/protocol digests will be added here before registered Train/Validation access. Once the ledger reservation succeeds, any failure consumes the interval permanently; do not retry it or inspect Test/Holdout.

## Frozen control digests before registered access

- Protocol JSON SHA-256: `89de47f04f22d1183f85b8d6371b91c53bfa4f1c930104e80268c8c2edd4d84d`
- Registered Fold config SHA-256: `7ae9b64e167d0e81a60d2b8cc3848afd137642bb5513fa85699be71c1083bc42`
- Registered runner SHA-256: `508be5a97e3a6caef67b9ab1ee70af7ea5669f2d16922353b806952918c353f8`
- Expected Train target budgets: E2 992,160; E3 992,000. Test/Holdout payload decode: `false`.

## Sealed source identity before registered access

All 14 official monthly archives passed published SHA-256 checks. Preparation retained 574,560 continuous minute events. The 23,040 rows after the Validation payload cutoff are timestamp-only markers; their OHLCV values were excluded. These identities were frozen before the registered runner reads Train or Validation.

- Archive receipt SHA-256: `1050aa7a116916c578335df74803977d977fa0179fd3180830b00073e13801af`
- Curated dataset hash: `221082db3217db3bac63a6924952cc461a7c308dc169976401eff98021dff153`
- Curated marker SHA-256: `0d1ffb5a6ff927a96a9697f83febced1b01eb140f4fcd4a6c0358efb955a73f5`
- Fold hash: `6c1cbe344728dd6c3f0047347b7a1d2098bfc54baa9f8afdbe554760fd022a00`
- Source preparation SHA-256: `aeb2089a405cbb277f04417a19f868edd9235b2a837dd2578377f77bed303ea5`
- Source event interval: `[2024-08-06T00:01:00Z, 2025-09-09T00:00:00Z]`; OHLCV payload at and after the Validation cutoff was excluded.
