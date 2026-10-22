# R2 pre-registration: ETHUSDT long-horizon risk weighting

Date frozen: 2026-10-19, before downloading or reading the proposed market payload. This is the next R2 hypothesis after the risk-scale 2.0 result failed its primary gate.

## Hypothesis

The previous ETHUSDT run showed that E2 risk loss scale 2.0 moved free-rollout volatility mean and p95 closer to realized values than E1, but E2 had weaker absolute-return lag-1 autocorrelation and remained under-dispersed. The risk objective currently averages all five horizons equally. A single change will emphasize the 30- and 60-minute targets while preserving the mean total risk-loss weight: `risk_horizon_weights=[0.5, 0.5, 1.0, 1.5, 1.5]` in horizon order `[1, 5, 15, 30, 60]`, with `risk_prediction_scale=2.0` unchanged. The weights average to 1.0. The prediction is improved conditional long-horizon risk representation and free-rollout volatility persistence. This is one fixed comparison, not a weight sweep.

## Frozen validation protocol

- Instrument/source: ETHUSDT USD-M perpetual, Binance 1-minute klines, UTC source-open interval `[2024-04-13T00:00:00Z, 2024-07-15T00:00:00Z)`, 93 days / 133,920 expected bars. Daily archives run from 2024-04-13 through 2024-07-14; no 2024-07-15 archive is used.
- The RouteB research ledger was checked read-only before source download. Candidate Train/Validation read range `[2024-04-13T00:01:00Z, 2024-06-29T00:01:00Z)` has no overlapping ETHUSDT access records or query intents. It begins one day after the prior ETH Test score interval ends at 2024-04-12 00:01 and ends before the next recorded ETH access at 2024-07-16.
- Approved 60/15/15 Fold: Train warmup 2024-04-13 00:01; Train score 2024-04-14 00:01 through 2024-06-13 00:01; Validation warmup 2024-06-03 00:01; Validation score 2024-06-14 00:01 through 2024-06-29 00:01; Test warmup 2024-06-19 00:01; Test score 2024-06-30 00:01 through 2024-07-15 00:01. One-day gaps, one-hour purge, and 23-hour embargo are fixed. Test payload is not decoded.
- E1/E2 share three seeds `31, 102, 20261018`, 1,000 optimizer updates per seed, 32 forecast paths per seed, 180 two-hour Validation anchors, 96 pooled paths per candidate, 12-anchor block bootstrap and 2,000 replicates.
- Primary gate remains 15-minute return CRPS against the strongest statistical baseline selected on Validation. Promotion requires the paired 95% block-bootstrap interval to favor E2 and Holm-adjusted p < 0.05. Secondary measures are native 30/60-minute log-realized-volatility CRPS/Brier and free-rollout absolute-return autocorrelation, rolling-volatility mean and p95. Secondary measures cannot override a failed primary gate.
- No Test/Holdout decoding, economic certification, threshold search, or additional candidate is permitted in this experiment.

## Frozen implementation delta

- E1 remains the existing GRU baseline.
- E2 is `configs/routeb/fresh_ethusdt_e2_long_horizon_risk_ensemble.json`. It uses the previous `risk_prediction_scale=2.0`; the only new setting is the normalized horizon-weight vector above. The earlier unweighted config remains unchanged as historical evidence.
- All RSSM dimensions, features, optimizer settings, training budget, seed list and forecast protocol remain fixed.
- Once the source is prepared, its marker, file hash and dataset hash must be recorded in the Fold config before the one-shot runner registers the scope. The experiment must stop before payload access if the ledger no longer shows this scope as unused.

A failed primary gate leaves R3/R4 closed and requires a new hypothesis and a never-consumed scope for further R2 work.

## Frozen inputs before Train/Validation access

The 93 official Binance daily archives from 2024-04-13 through 2024-07-14 and their checksum sidecars were downloaded and individually SHA-256 verified. The curated source contains exactly 133,920 one-minute bars. No 2024-07-15 archive was downloaded. Source preparation completed before the one-shot model runner was invoked.

- Source: `.artifacts/world_model/fresh-ethusdt-20240413-60-15-15-20261019/source/market.parquet`.
- Source dataset hash: `8d3473b9adca35ce51e9c641f50504dd6eca087a69fbda22e7ba72f017e3c764`.
- Source completion marker: `969359cba6d3d910f9c16119329ec276e15ea3801d31b1b8692ed4accb228e64`.
- Source Parquet file SHA-256: `274ff942f965ed97a38295e68b573651210cbca006343bf3533f603c5f48a0f6`.
- Fold config: `configs/routeb/fresh_ethusdt_20240413_60_15_15_fold.json`, SHA-256 `39608c79ce29da7a8227acf5126a988acf7475b11c2831b0be16b52061a23302`.
- Frozen Fold manifest hash: `a1b7488197de359e2af8dd1b9015d8d57e9fc075a2c6d07c18c02fe264f1fb43`.
- E2 config: `configs/routeb/fresh_ethusdt_e2_long_horizon_risk_ensemble.json`, SHA-256 `e06ccd0da117b216440f1729d3c3ba9b12702622115c1628f91971361b576599`.
- Source preparation receipt: [`B0_ethusdt_202404_source_preparation.json`](B0_ethusdt_202404_source_preparation.json), SHA-256 `187e7ed6646d808081650506feebd3ffc8d5de2891a125184927f5804fa97741`.

Before reservation, the RouteB access ledger was checked again for the exact Train/Validation read range; it still had no overlapping access record or query intent. The experiment must not read payload if the final reservation preflight rejects the scope.

## Revision 1 execution record

The one-shot runner registered `routeb-fold-ethusdt-binance-601515-20240413-v1` and read only Train/Validation. E1 and E2 both completed 1,000 optimizer updates for seed 31, but E2's checkpoint could not be reloaded for Validation forecasting because the new horizon-weight tuple was serialized to JSON as a list while the reload recipe retained a tuple. The runner stopped with `checkpoint schema/train/config identity mismatch`; `Test` was not decoded and no market forecast report was produced. This scope is consumed and must not be retried.

The code now normalizes horizon weights in the checkpoint recipe. The optimizer recipe round-trip test passes, as do four isolated JAX regression tests and the end-to-end synthetic worker smoke with horizon-weighted E2. Revision 2 uses the independent 2023 ETHUSDT scope specified in [`R2_ETHUSDT_LONG_HORIZON_RISK_PRE_REGISTRATION_20261019_REV2.md`](R2_ETHUSDT_LONG_HORIZON_RISK_PRE_REGISTRATION_20261019_REV2.md).
