# R2 ETHUSDT 15-step open-loop CRPS pre-registration

Date frozen: 2026-10-19, before downloading or reading this source scope. Previous R2 runs improved some native medium-horizon risk scores but did not pass the primary 15-minute return CRPS gate. This experiment changes only E2's prior observation objective to a 15-step open-loop CRPS: posterior states are transitioned forward using the RSSM prior without feeding future observations, and each future feature distribution is scored for horizons 1 through 15. Default one-step NLL remains unchanged for other configurations.

## Fixed candidate and hypothesis

Use the current two-level RSSM, Gaussian forecast head, and E2 risk scale 0.5. E2 config: `configs/routeb/fresh_ethusdt_e2_openloop_crps_ensemble.json`, SHA-256 `fb53ef0c0100bcc7156ab7a6e91385b58d5ed04d21b6cf35fe4d8b24383cbbfd`. E1 is unchanged at `configs/routeb/fresh_ethusdt_e1_ensemble.json`; E0 is unchanged. The model shape, risk loss, auxiliary losses, KL terms, optimizer, seed set, training budget, forecast sampling and evaluation are unchanged. The sole change from the prior CRPS candidate is scoring 15 prior-only rollout steps per training sequence.

Hypothesis: direct open-loop supervision reduces the free-rollout persistence gap and improves the primary 15-minute return CRPS against the strongest Validation-selected statistical baseline, without degrading native 30/60-minute volatility scores. Promotion still requires the preregistered paired interval and Holm-adjusted p < 0.05 against the strongest statistical baseline. Secondary metrics cannot override the primary gate. No candidate sweep is permitted.

## Frozen scope and protocol

- Source: Binance USD-M ETHUSDT 1-minute klines, open-time interval `[2021-04-05T00:00:00Z, 2021-07-08T00:00:00Z)`, 94 daily archives, 135,360 expected bars.
- Read scope: Train warmup through Validation score end, `[2021-04-05T00:01:00Z, 2021-06-22T00:01:00Z)`. A read-only query of RouteB's `access` and `query_intent` tables against the half-open interval returned zero overlaps for ETHUSDT `perpetual-market-v1`. The one-shot registered reader remains authoritative.
- Fold: approved 60/15/15, with Train warmup April 5 00:01, Train score April 7 through June 6 00:01, Validation warmup May 27 00:01, Validation score June 7 through June 22 00:01, Test warmup June 12 00:01, Test score June 23 through July 8 00:01. Keep 1-hour purge and 23-hour embargo. Test remains undecoded.
- Three seeds `31`, `102`, `20261018`; 1,000 updates per candidate and seed; 32 paths per seed; 180 two-hour Validation anchors; 96 pooled paths per candidate; 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Secondary metrics: native 30/60-minute log-realized-volatility CRPS/Brier, free-rollout absolute-return lag-1 autocorrelation, rolling-RV mean/p95, and path validity. No Test/Holdout access or economic claim.

## Frozen implementation and mechanism evidence

- RouteB repository commit identity: `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`.
- RouteB source-code identity SHA-256: `ed5d97f0a409f21182530e44afd220c734ca97bab374aa09f0a7ac70984cc89a`.
- Prior-head/transition gradient smoke passed with finite nonzero gradients for open-loop CRPS horizons 1/2/3; report `.artifacts/world_model/routeb_open_loop_crps_smoke_20261019.json`, SHA-256 `5f14052a59a97a8f5de85de3895fb64764317fe36d19ffe139d13b24f5003787`.
- Full 15-step synthetic CPU train/checkpoint-resume/worker-forecast smoke passed with resume parity and valid finite worker outputs; report `.artifacts/world_model/routeb_open_loop_crps_train_smoke_20261019/result.json`, SHA-256 `7f69062b9671ca8ac2910c8732be895ce365743938a258742cf5ee2b20445656`. Synthetic evidence only; no real-market or economic claim.
- The candidate, objective, and source interval were frozen before source download. The deterministic UTC day-count correction below was appended after archive checksum verification but before market-row preparation. After preparation, archive, checksum, source, marker, Fold, and refreshed manifest hashes were recorded before the single registered Train/Validation read. If the final ledger preflight rejects the scope, do not decode Train/Validation. R3/R4/R5/R6 remain gated unless the primary R2 rule passes.


## Deterministic calendar correction, before market-row preparation

After daily archive checksums finished, UTC date arithmetic showed that [2021-04-05, 2021-07-08) contains 94 days, not 93. At correction time, no OHLCV row payload had been decoded or prepared. The source interval and objective remain fixed; this dated calendar correction changes the expected count to 94 archives / 135,360 bars and places one additional day in Train warmup so the approved 60/15/15 score durations, one-day gaps, purge and embargo remain exact. The corrected Train/Validation scope [2021-04-05T00:01:00Z, 2021-06-22T00:01:00Z) was rechecked read-only: zero overlapping access and zero overlapping query_intent rows. This calendar-only correction is recorded before source preparation and before the registered reader can decode Train/Validation.





## Frozen source identities recorded before reservation

- Source preparation receipt: `.artifacts/world_model/fresh-ethusdt-20210405-openloop-crps-60-15-15-20261019/source/source_preparation.json`, SHA-256 `00d71bccab61b8bb2e8bac6ae4be0c50561a041a3409e150ed8bba0d9d9f0189`; its 94 daily archive entries were all verified against official Binance `.CHECKSUM` sidecars.
- Prepared source has 135,360 bars. Dataset hash `2933e7010fa51abc05db591e865467964e796c3913697824679200069bed472e`; completion marker SHA-256 `bbd6abb6151f7c9236a6c1f2891dc95208bcfea89a8414ce14ed59bdedf20b87`; Parquet SHA-256 `84336c634f8d456fffdb818bdd487a21c692a6f0d89af6d4746aee918a186f62`.
- Registered Fold: `configs/routeb/fresh_ethusdt_20210405_openloop_crps_60_15_15_fold.json`, SHA-256 `d3c9944fdb60edc300bf64d1c785c9270eea708e3f97491342a288928aa2d740`. The 60/15/15 score windows and one-day gaps validate against the approved protocol; purge is 1 hour and embargo is 23 hours.
- Refreshed RouteB source manifest: 443 files, identity SHA-256 `0ba9a41e11bfa9691ab50b5d5f98ecb3329657492e50e2bcef00ecdd76359f20`; refresh and verification passed before reservation.
- Final read-only access/query-intent preflight for `[2021-04-05T00:01:00Z, 2021-06-22T00:01:00Z)` returned zero overlaps in both tables. The registered one-shot reader remains authoritative and Test/Holdout remain closed.
