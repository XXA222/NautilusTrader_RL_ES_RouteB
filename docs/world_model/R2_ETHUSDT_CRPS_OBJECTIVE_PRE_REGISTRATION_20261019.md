# R2 ETHUSDT Gaussian CRPS-objective pre-registration

Date frozen: 2026-10-19, before downloading or reading this source scope. Across multiple R2 runs, the generative model's primary 15-minute return CRPS remains inconclusive, while the implementation is optimized with Gaussian negative log-likelihood. This experiment changes only the fixed observation-prior training score to the closed-form Gaussian CRPS, matching the primary evaluation score. It tests that objective on a separate, earlier ETHUSDT regime. No candidate or scale sweep is permitted.

## Fixed candidate and hypothesis

Use the current two-level RSSM architecture and frozen E2 risk scale `0.5`. E2 uses the existing Gaussian prior forecast head and the new `forecast_objective="crps"`; E1 and E0 are unchanged. The auxiliary posterior and reconstruction losses, risk loss, KL terms, risk weights, optimizer and inference protocol stay fixed. Hypothesis: directly optimizing proper Gaussian CRPS improves 15-minute return CRPS against the strongest Validation-selected statistical baseline without damaging native 30/60-minute volatility scores. The primary gate remains a paired 95% interval favoring E2 with Holm-adjusted p < 0.05 against the strongest statistical baseline. Secondary results cannot override that gate.

## Frozen scope and protocol

- Source: Binance USD-M ETHUSDT 1-minute klines, open times `[2021-01-02T00:00:00Z, 2021-04-05T00:00:00Z)`, 93 days / 133,920 expected bars. Download daily archives January 2 through April 4 inclusive.
- RouteB's access and query-intent ledger was checked read-only before freezing. Proposed Train/Validation read scope `[2021-01-02T00:01:00Z, 2021-03-20T00:01:00Z)` had zero overlapping records. The registered one-shot reader remains authoritative.
- 60/15/15 Fold: Train warmup January 2 00:01; Train score January 3 through March 4 00:01; Validation warmup February 22 00:01; Validation score March 5 through March 20 00:01; Test warmup March 10 00:01; Test score March 21 through April 5 00:01. Preserve 1-hour purge and 23-hour embargo; Test remains undecoded.
- Three seeds `31, 102, 20261018`; 1,000 optimizer updates per seed; 32 paths per seed; 180 two-hour Validation anchors; 96 pooled paths per candidate; 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Secondary: native 30/60-minute log-realized-volatility CRPS/Brier, free-rollout absolute-return lag-1 autocorrelation, rolling-RV mean/p95, path validity. No Test/Holdout access or economic claim.

## Frozen implementation/provenance

- E1: `configs/routeb/fresh_ethusdt_e1_ensemble.json`.
- E2: `configs/routeb/fresh_ethusdt_e2_crps_ensemble.json`, SHA-256 `72f8fb1e4829d98ea3264c880463ac1883e752eb18ee3adc24c4f3c5da94f658`.
- RouteB repository base `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`; source-code identity SHA-256 `85875768e403155d74b0ef718c85fc0435cf83b0234f7c1b2c0222a832c321fa`.
- The isolated JAX CPU synthetic smoke for CRPS prior-head/transition gradients passed before freezing. Its report is `.artifacts/world_model/routeb_crps_objective_smoke_20261019.json`; it is mechanism evidence only.
- Archive checksums, prepared source identities and Fold hash will be recorded here before the registered reader accesses Train/Validation. The scope is consumed after its one reservation. R3/R4/R5/R6 remain gated unless the primary rule passes.
- RouteB source manifest after freezing the model config: 440 files, identity SHA-256 `40dccef96e789d8aee76974373f5e4bbac1d2e126d4514e3ddccc85afd3a5685`; verification passed before download. Synthetic smoke receipt SHA-256 `424ca96a7d66f70825cc4aa7fa18f57734dd28a334463608b4f89183409b2c11`.

## Frozen source identities recorded before reservation

- All 93 archive SHA-256 values matched the official checksum sidecars. Source preparation receipt SHA-256 `ba2349efe9e91da7398ba6d7a2337b84015a5dc20c88d62a5654a4a51ba53ed6`.
- Prepared dataset hash `ffcd7f82770c8f29f0a440756a9596e9abe076141c456f35e626b742d6dd0a16`, marker SHA-256 `94340f12ed48c82f5dc208cc130bf93096169203ac08698dc47f9eb7f275947e`, Parquet/source file SHA-256 `52a1405622c076999d16cc3d07f08f4bc354bb5f060ae91333ae1f97fdd79595`; `133920` bars.
- Fold config `configs/routeb/fresh_ethusdt_20210102_crps_objective_60_15_15_fold.json`, SHA-256 `3010a2c1cb00637d29d1ea0421c2c014750ab1b3d3d29ea891b60a99ead8419f`.
- Read-only access and query-intent preflight immediately before reservation found zero overlaps for the frozen Train/Validation scope. The registered one-shot reader remains authoritative.
- RouteB source manifest after adding the frozen Fold config: 441 files, identity SHA-256 `732fd9fb6f4af9ba48b9a016de5f58271e698c02be205f753e9cc7d5e99fd363`; verification passed before reservation.
