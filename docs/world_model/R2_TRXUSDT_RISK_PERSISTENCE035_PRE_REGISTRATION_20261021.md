# R2 TRXUSDT risk-noise persistence 0.35 — fresh-window pre-registration

Frozen on 2026-10-21 after the preceding registered fresh-window experiment completed and before downloading or decoding the new source interval.

## Hypothesis and one-factor intervention

- The preceding fresh Fold improved pooled 15-minute path log-RV CRPS and brought generated rolling-RV mean close to realized, but generated absolute-return lag-1 autocorrelation overshot (0.2107 vs 0.0913). Its strict all-seed gate did not pass.
- Test whether lowering stationary standardized log-RV AR(1) persistence from `0.65` to `0.35` reduces excess return persistence while retaining risk-to-return conditioning strength `0.25`.
- E2 remains Gaussian RSSM with Train-only equal-weight sampled free-prior path log-RV CRPS at horizons `[15,30,60]`, coefficient `3.0`, 16 Train paths, native risk loss scale `1.0`, risk-conditioned sampling, and the same optimizer recipe.
- E1 remains the unchanged fixed GRU comparator. Both candidates use 1,000 updates, batch size 8, learning rate `0.0003`, report interval 50, and seeds `[31,102,20261018]`.
- Validation remains 180 anchors at 120-minute spacing, 32 paths per seed, 96 pooled paths, 2,000 paired moving-block bootstrap replicates, block length 12.
- The all-seed decision gate is unchanged: for every seed, E2-minus-selected-E0 15-minute path log-RV CRPS 95% CI must lie strictly below zero and Holm `p<0.05) in the fixed 11-test family; no seed may show significant Holm-corrected E2 harm to 15-minute return CRPS; generated OHLCV paths must be valid. A pooled improvement alone does not pass the gate.

## Data scope and isolation

- Binance USD-M TRXUSDT 1m source-open interval `[2026-05-19T00:00:00Z,2026-08-21T00:00:00Z)`: 94 UTC days and 135,360 expected closed bars.
- Approved 60/15/15 Fold with one-hour purge and 23-hour embargo. The exact Fold config, source marker and semantic dataset hash will be frozen after monthly archives are verified and curated, but before the registered reader is allowed to decode Train/Validation.
- Train+Validation reader scope is `[2026-05-19T00:01:00Z,2026-08-05T00:01:00Z)` (`[1779152460000000000,1785891660000000000)`). Test score is `[2026-08-06T00:01:00Z,2026-08-21T00:01:00Z)`.
- This window begins after the preceding Test interval ended on 2026-05-16T00:00Z. No Test/Holdout payload will be read.
- Output root: `.artifacts/world_model/fresh-trxusdt-20260519-risk-persistence035-20261021/`.
- Repeat the read-only project-ledger overlap check immediately before registered access; both access and query-intent overlaps must be zero.

## Frozen candidates

- E1 config: `configs/routeb/fresh_trxusdt_20260519_e1_risk_persistence035_strength025_ensemble.json`, copied unchanged from the fixed GRU comparator.
- E2 config: `configs/routeb/fresh_trxusdt_20260519_e2_risk_persistence035_strength025_ensemble.json`; persistence `0.35`, conditioning strength `0.25`.
- Compared with the immediately preceding candidate, persistence is the sole model-configuration change. No tuning based on the new Validation window is allowed.
- New market archives will be downloaded only after this freeze. The preregistration and configurations are version-controlled before the download; the final commit-bound code fingerprint, config/source hashes, footer-only row-group audit, isolated CUDA worker identity and final read-only ledger state will be recorded in `pre_read_identity.json` before any registered payload access.

