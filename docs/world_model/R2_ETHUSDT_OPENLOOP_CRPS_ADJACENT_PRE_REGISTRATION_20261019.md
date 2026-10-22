# R2 ETHUSDT open-loop CRPS adjacent confirmation pre-registration

Date frozen: 2026-10-19, before downloading or reading the new source interval. The prior 15-step open-loop CRPS run had a favorable 15-minute return CRPS point estimate but did not pass the registered Holm significance gate. This single adjacent-period run is a fixed-recipe confirmation. It does not change the objective, architecture, data features, risk loss, optimizer, training budget, or evaluation protocol. No candidate sweep is permitted.

## Fixed candidate and hypothesis

E2 remains the two-level RSSM with a Gaussian prior head, `forecast_objective="open_loop_crps"`, `open_loop_crps_horizon=15`, and risk scale 0.5. The frozen config `configs/routeb/fresh_ethusdt_e2_openloop_crps_ensemble.json` has SHA-256 `fb53ef0c0100bcc7156ab7a6e91385b58d5ed04d21b6cf35fe4d8b24383cbbfd`. E1 is unchanged at `configs/routeb/fresh_ethusdt_e1_ensemble.json`; E0 is unchanged. The prior implementation, synthetic gradient smoke and full 15-step train/checkpoint-resume/forecast smoke are documented in `R2_ETHUSDT_OPENLOOP_CRPS_PRE_REGISTRATION_20261019.md`.

Hypothesis: the open-loop objective's favorable E2-vs-strongest-E0 15m return CRPS difference will replicate on this chronologically adjacent scope and pass the same paired 95% interval plus Holm-adjusted p < 0.05 gate. Secondary native 30/60-minute volatility scores cannot override the primary rule. Test remains undecoded and no economic claim is made.

## Frozen scope and protocol

- Source: Binance USD-M ETHUSDT 1-minute klines, open-time interval `[2021-07-08T00:00:00Z, 2021-10-10T00:00:00Z)`, 94 daily archives, 135,360 expected bars.
- Train/Validation read scope: `[2021-07-08T00:01:00Z, 2021-09-24T00:01:00Z)`. Read-only RouteB access and query-intent table checks found zero overlapping ETHUSDT `perpetual-market-v1` records. The one-shot reader remains authoritative.
- Approved 60/15/15 Fold: Train warmup July 8 00:01; Train score July 10 through September 8 00:01; Validation warmup August 28 00:01; Validation score September 9 through September 24 00:01; Test warmup September 14 00:01; Test score September 25 through October 10 00:01. The extra day in the source is Train warmup. Keep the one-hour purge and 23-hour embargo.
- Three seeds `31`, `102`, `20261018`; 1,000 updates per candidate and seed; 32 paths per seed; 180 two-hour Validation anchors; 96 pooled paths per candidate; 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Primary gate: E2 must beat the Validation-selected strongest E0 baseline on 15-minute return CRPS under the frozen paired bootstrap and Holm correction. Secondary diagnostics: native 30/60-minute log-realized-volatility CRPS/Brier, absolute-return lag-1 autocorrelation, rolling-RV mean/p95, and path validity.

## Frozen implementation identity

- RouteB repository commit identity: `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`.
- RouteB source-code identity SHA-256: `ed5d97f0a409f21182530e44afd220c734ca97bab374aa09f0a7ac70984cc89a`.
- No real-market result from the prior interval is used for fitting this new interval. This experiment is one confirmation, not a threshold or seed sweep. R3/R4/R5/R6 remain gated unless the primary R2 rule passes.
- Before the single registered Train/Validation read, append official archive checksum results, source receipt, dataset/marker/Parquet hashes, Fold hash, refreshed source-manifest identity, and a final read-only ledger preflight. If the authoritative one-shot reservation rejects, do not access Train/Validation rows.

## Frozen source identities recorded before reservation

- All 94 archive entries were verified against the official Binance `.CHECKSUM` sidecars. Source-preparation receipt SHA-256 `c4a5d5f0a628474bb4fd71d9b50abc9e2382a28b6d6cba63b63a7c90eaa12061`.
- Prepared source: 135,360 bars; dataset hash `530de4bce242da2fe31d236270388bdd9d14292e1b8c9d64681774d72d1c9ffd`; marker SHA-256 `338d54a3d663de0aa0ea1b4d2777753e44050394a03ee89f0fd25e141ec0f980`; Parquet SHA-256 `398fa47e2cb46eedb5187d3ffbf402b0d78bbeb00699e3bdcf71dc69d41c9b1c`.
- Registered Fold: `configs/routeb/fresh_ethusdt_20210708_openloop_crps_60_15_15_fold.json`, SHA-256 `df8181ad7e50aad850aabf4dc685a7fa19ac7610762c366be16f8469aa4b2dfe`.
- Refreshed RouteB manifest: 444 files, identity SHA-256 `3134bcaae104b5b1eeca6928cf9e880864886731f1f87927546e42adcd7b61b`; verification passed.
- Final read-only preflight for `[2021-07-08T00:01:00Z, 2021-09-24T00:01:00Z)` returned zero overlapping RouteB access and query-intent rows. The one-shot reservation remains the final authority.
