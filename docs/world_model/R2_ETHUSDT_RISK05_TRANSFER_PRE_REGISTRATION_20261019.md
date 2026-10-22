# R2 ETHUSDT risk-scale-0.5 transfer confirmation pre-registration

Date frozen: 2026-10-19, before downloading or reading this new source scope. A risk-scale-0.5 experiment on February-May 2022 preserved significant 30/60-minute native log-volatility improvements but did not pass primary 15-minute CRPS. A risk-scale-0.25 experiment on May-August 2022 nearly tied its strongest baseline but lost the significant native-risk improvements. This fixed transfer run checks whether the 0.5 tradeoff repeats in a distinct 2023 market period.

## Hypothesis and fixed candidate

Use the already frozen Gaussian E2 with `risk_prediction_scale=0.5`; do not alter any model, target, horizon, feature, optimizer, or forecast setting. E1 and E0 baselines are unchanged. This is one confirmation run of the fixed 0.5 recipe, with no alternative scales or additional candidate. The primary 15-minute return CRPS promotion gate remains unchanged: paired 95% CI must favor E2 and Holm-adjusted p must be below 0.05 against the strongest Validation statistical baseline. The native 30/60-minute log-realized-volatility comparison remains secondary and cannot override that gate.

## Frozen scope and protocol

- Source: Binance USD-M ETHUSDT 1-minute klines, source-open interval `[2023-07-15T00:00:00Z, 2023-10-16T00:00:00Z)`, 93 days / 133,920 expected bars. Download daily archives July 15 through October 15 inclusive.
- RouteB's access registry was queried read-only before freezing. Train/Validation interval `[2023-07-15T00:01:00Z, 2023-09-30T00:01:00Z)` had zero overlapping ETHUSDT perpetual-market accesses or query intents. The one-shot registered reader remains authoritative; a rejection means do not read Train/Validation.
- 60/15/15 Fold: Train warmup July 15 00:01; Train score July 16 through September 14 00:01; Validation warmup September 4 00:01; Validation score September 15 through September 30 00:01; Test warmup September 20 00:01; Test score October 1 through October 16 00:01. Preserve the one-hour purge and 23-hour embargo; do not decode Test.
- Three seeds `31, 102, 20261018`; 1,000 optimizer updates per seed; 32 forecast paths per seed; 180 two-hour Validation anchors; 96 pooled paths per candidate; 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Secondary reports: 30/60-minute native log-realized-volatility CRPS/Brier, free-rollout absolute-return lag-1 autocorrelation, rolling-RV mean/p95 and path validity. No Test/Holdout access or economic claim.

## Frozen implementation/provenance

- E1 config: `configs/routeb/fresh_ethusdt_e1_ensemble.json` (unchanged).
- E2 config: `configs/routeb/fresh_ethusdt_e2_risk05_ensemble.json` (Gaussian, `risk_prediction_scale=0.5`). The config-only candidate has no source-code delta.
- RouteB HEAD `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`; source identity SHA-256 `e1e08a244a8e95e3c3dfd26c1c1e15765a0ec0d6114cc322a257afd63df656db`.
- E1 config SHA-256 `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`; E2 config SHA-256 `e64938514b7ed10ae375b2dc42e100100772a9f406deb8fbd50b299606ca60e9`.
- Source archive sidecar hashes, curated marker/hash, Parquet hash and Fold hash will be appended before the registered reader accesses the data. The scope is consumed after the single reservation. R3/R4/R5/R6 remain gated unless the frozen primary rule passes.

## Frozen data and Fold identity (before registered reader access)

- Downloaded 93 official Binance daily archives (July 15 through October 15, 2023), total 5,480,288 bytes; all published `.CHECKSUM` sidecars were independently rechecked. Canonical archive receipt manifest SHA-256: `a2cf6c581575881597d51426b504fa438004a309a0badf87ea09a641262358d1`.
- Curated source `.artifacts/world_model/fresh-ethusdt-20230715-risk05-transfer-60-15-15-20261019/source/market.parquet`: 133,920 continuous bars, source-open interval `[2023-07-15T00:00:00Z, 2023-10-16T00:00:00Z)`.
- Dataset SHA-256 `0f3e46703e879bcb6d1b7530181106754dbbaed21f5931612e04e2ca9812ee73`; marker SHA-256 `0a8562b563acafd212599fceee2da2bc8335ed568f2d3a77833e308469edcc46`; Parquet SHA-256 `befad181767cfe1f135ea4c725478b5ebe8e15cf962d557a56e50b21f7eed7e8`.
- Source preparation receipt SHA-256 `9589b6122399bd3e54ac679feebc9af0c05275565239f19028234c2ee3c7cc56`.
- Fold config `configs/routeb/fresh_ethusdt_20230715_risk05_transfer_60_15_15_fold.json`, config SHA-256 `9a08ff6f17c939af91c3e1b8e2456f4bf9f2f47746d3e0f25ff5f796769f347d`; Fold hash `7ca42d67f92b14b5306cd80ce0e5f81f1980ec99a3a897a3a7c7aeb89d71479a`.
- FoldManifest 0.4.0 and all temporal boundaries, purge/embargo, source footer and sidecar checks pass; no Train/Validation reader has run and Test remains excluded.
