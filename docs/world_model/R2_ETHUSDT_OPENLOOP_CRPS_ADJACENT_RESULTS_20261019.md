# R2 ETHUSDT open-loop CRPS adjacent confirmation results, 2026-10-19

Status: completed; adjacent confirmation failed the primary R2 promotion gate. R3/R4/R5/R6 remain gated.

## Scope and protocol

This was the fixed-recipe adjacent confirmation of the 15-step open-loop CRPS objective after its initial 2021 Validation point estimate favored E2 without clearing the preregistered Holm gate. E2, E1/E0, training budget, seeds and metrics were unchanged. See [`R2_ETHUSDT_OPENLOOP_CRPS_ADJACENT_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_OPENLOOP_CRPS_ADJACENT_PRE_REGISTRATION_20261019.md).

- Source: 94 Binance USD-M ETHUSDT daily 1-minute archives for `[2021-07-08T00:00:00Z, 2021-10-10T00:00:00Z)`. Every official `.CHECKSUM` sidecar passed; source has 135,360 bars.
- The registered reader reserved Train/Validation `[2021-07-08T00:01:00Z, 2021-09-24T00:01:00Z)` and read 112,320 bars. The data audit found no duplicate/missing minutes or availability violations. Test/Holdout were not decoded.
- Approved 60/15/15 score windows, one-day gaps, one-hour purge and 23-hour embargo. Three seeds (`31`, `102`, `20261018`), 1,000 updates per candidate/seed, 32 paths per seed, 180 two-hour Validation anchors and 96 pooled paths per candidate; 2,000 block-bootstrap replicates with 12-anchor blocks.

## Primary result

Lower 15-minute return CRPS is better. E0 S1 remained the strongest statistical baseline.

| Candidate | 15m return CRPS |
|---|---:|
| E0 S0 fixed volatility | 0.0031219078 |
| E0 S1 EWMA volatility | 0.0030902426 |
| E0 S2 stationary block bootstrap | 0.0031263890 |
| E1 GRU | 0.0030684815 |
| E2 15-step open-loop CRPS | 0.0031071156 |

E2 was worse than E0 S1 by `+0.0000168730`; its paired 95% CI was `[-0.0000569696, +0.0001198835]`, `p=0.7006497`, Holm-adjusted `p=1.0`. E1 also had a lower point estimate than E2, with an inconclusive paired comparison. The fixed open-loop objective did not replicate its earlier favorable point estimate and does not pass R2 promotion.

## Risk and rollout diagnostics

E2 native log-realized-volatility scores were mixed: two seeds showed corrected 30/60-minute improvements, while seed 102 did not. These secondary diagnostics do not change the primary outcome.

Generated paths had 100% valid bars/paths but severely understated volatility clustering and dispersion. E2 absolute-return lag-1 autocorrelation was `-0.00879` against realized `0.06870`; rolling-RV mean was `0.0008073` against `0.0012126`, and p95 was `0.0012299` against `0.0025193`.

## Integrity and provenance

- Aggregate report file SHA-256 `f55d6995de95e7605267e9a10aa0b546353ca377b093a17e37ada41d50de8a88`; aggregate canonical digest `0f57c1debb8400de28a4f805e9ca74c7c28ab522d086dda56c7b07bc7604aa5a` and predictive digest `54a8132fd75d5bbec3c44156f8d5f1ffc4c613d1f6132c6ee0d9a074e18341a9` verify in RouteB's Python 3.14.7 coordinator.
- All three seed digests verify. Pooled E1 SHA-256 `ad9d8e7ac714810137330d4f65ef2f6fb19602702a6f5c901c8ac02bf9da9d62`; pooled E2 `25a7043c0b3343c16b996c78b6ee41c27f20a5a5371de4b1bb2327c4edebf0fb`; both shapes `[180, 96, 60, 5]` match the aggregate report.
- Dataset hash `530de4bce242da2fe31d236270388bdd9d14292e1b8c9d64681774d72d1c9ffd`; marker SHA-256 `338d54a3d663de0aa0ea1b4d2777753e44050394a03ee89f0fd25e141ec0f980`; Parquet SHA-256 `398fa47e2cb46eedb5187d3ffbf402b0d78bbeb00699e3bdcf71dc69d41c9b1c`.
- Source-preparation receipt SHA-256 `c4a5d5f0a628474bb4fd71d9b50abc9e2382a28b6d6cba63b63a7c90eaa12061`; Fold SHA-256 `df8181ad7e50aad850aabf4dc685a7fa19ac7610762c366be16f8469aa4b2dfe`.
- RouteB manifest has 444 files, identity SHA-256 `3134bcaae104b5b1eeca6928cf9e880864886731f1f87927546e42adcd7b61b`; verification passes.
- Report marks `test_payload_decoded=false` and `economic_certification=false`. No Test-based selection or economic claim.

Report: `.artifacts/world_model/fresh_ethusdt_20210708_openloop_crps_adjacent_ensemble_20261019/multi_seed_fold_experiment_report.json`.
