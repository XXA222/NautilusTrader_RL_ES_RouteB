# R2 pre-registration revision 2: ETHUSDT long-horizon risk weighting

Date frozen: 2026-10-19, before downloading or reading the proposed market payload. Revision 2 preserves the original hypothesis and protocol after revision 1's one-shot scope was consumed by a worker checkpoint-identity failure before Validation forecasts were produced.

## Hypothesis and frozen model

Continue the single fixed hypothesis from [`R2_ETHUSDT_LONG_HORIZON_RISK_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_LONG_HORIZON_RISK_PRE_REGISTRATION_20261019.md): E2 `risk_prediction_scale=2.0` with normalized `risk_horizon_weights=[0.5, 0.5, 1.0, 1.5, 1.5]` in `[1, 5, 15, 30, 60]` minute order. E1 remains the GRU baseline. No additional hyperparameter or model changes are introduced. The weights average 1.0, emphasizing 30/60-minute targets while holding the aggregate risk-loss scale constant.

The revision 1 worker fault was fixed by serializing `risk_horizon_weights` as a JSON list in the optimizer/checkpoint recipe. A four-test isolated JAX run passes, including a synthetic train/checkpoint reload/forecast path using the new weights. Revision 2 tests the frozen hypothesis on a clean, independent market interval.

## Frozen scope and protocol

- Instrument/source: ETHUSDT USD-M perpetual, Binance 1-minute klines, source-open interval `[2023-04-13T00:00:00Z, 2023-07-15T00:00:00Z)`, 93 days / 133,920 expected bars. Use the 93 daily archives from April 13 through July 14 only.
- The RouteB ledger was checked read-only before download. Train/Validation read interval `[2023-04-13T00:01:00Z, 2023-06-29T00:01:00Z)` has no overlapping ETHUSDT access rows or query intents.
- Approved 60/15/15 Fold: Train warmup April 13 00:01; Train score April 14 00:01 through June 13 00:01; Validation warmup June 3 00:01; Validation score June 14 00:01 through June 29 00:01; Test warmup June 19 00:01; Test score June 30 00:01 through July 15 00:01. One-day gaps, one-hour purge, and 23-hour embargo are fixed. Test remains undecoded.
- E1/E2 share seeds `31, 102, 20261018`, 1,000 updates per seed, 32 forecast paths per seed, 180 two-hour Validation anchors, 96 pooled paths per candidate, and 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Primary measure remains 15-minute return CRPS against the strongest statistical baseline selected on Validation. Promotion requires the paired 95% interval to favor E2 and Holm-adjusted p < 0.05. Native 30/60-minute log-volatility CRPS/Brier and free-rollout volatility diagnostics are secondary and cannot override the primary gate.
- No Test/Holdout decoding, economic certification, tuning sweep, or additional candidate is permitted.

## Frozen configuration paths

- Fold: `configs/routeb/fresh_ethusdt_20230413_60_15_15_fold.json`.
- E1: `configs/routeb/fresh_ethusdt_e1_ensemble.json`.
- E2: `configs/routeb/fresh_ethusdt_e2_long_horizon_risk_ensemble.json`.
- Source preparation: 93 official Binance daily archives for April 13 through July 14, with each exchange checksum verified; 133,920 continuous bars. No July 15 archive was downloaded.
- Curated source: `.artifacts/world_model/fresh-ethusdt-20230413-60-15-15-20261019/source/market.parquet`.
- Dataset SHA-256: `f83b783333a1887bab6a994aa4c4dc4fe2b16451a5cef0e52a4e91d8108b76dd`.
- Source marker SHA-256: `7fd778088b561a6f2dc4e512ccea1b6d3f967f710d3612b42294284552299940`.
- Parquet file SHA-256: `909687e761a6d7656ddb3e1e21dca01b9f65d0267f20a3dfaf4646172ac57c47`.
- Fold config SHA-256: `79aedd75cbd202f950a8f0b3607b0232c5c7371851f750ecc47ba1d32065ff54`.
- E2 config SHA-256: `e06ccd0da117b216440f1729d3c3ba9b12702622115c1628f91971361b576599`.
- Frozen Fold manifest hash: `782d929a4386a08d5319795df3af852240c3e1eb1890ac1b8a67c0be5d8544ae`.
- Source preparation receipt: [`B0_ethusdt_202304_source_preparation.json`](B0_ethusdt_202304_source_preparation.json), SHA-256 `1165d41d7eb65ab8770bb9a81cb80ac476a5dbd29c5cbdd6c2aa6f46886ee1b9`.

The exact Train/Validation read range was checked once more in the RouteB ledger before starting. It still has no overlapping access rows or query intents. If the final registered runner preflight rejects the scope, stop before payload access.

Any failed primary gate keeps R3/R4 closed. This one-shot scope cannot be reused for tuning.
