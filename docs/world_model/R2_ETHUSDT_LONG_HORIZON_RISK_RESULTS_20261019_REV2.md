# R2 ETHUSDT long-horizon risk weighting, revision 2

Date: 2026-10-19
Status: completed; preregistered primary promotion gate not met

## Frozen experiment and data boundary

This one-shot run tested the preregistered E2 risk objective with `risk_prediction_scale=2.0` and normalized horizon weights `[0.5, 0.5, 1.0, 1.5, 1.5]` for horizons `[1, 5, 15, 30, 60]` minutes. E1, model dimensions, optimizer budget, forecast protocol and seeds were unchanged.

- Instrument/source: ETHUSDT USD-M perpetual, Binance 1-minute klines; 2023-04-13 00:00 UTC through 2023-07-15 00:00 UTC source-open interval; 133,920 bars.
- Registered Train/Validation read scope: `[2023-04-13T00:01:00Z, 2023-06-29T00:01:00Z)`; source and Fold were verified against RouteB's registry before the read.
- Approved 60/15/15 Fold, with the preregistered gaps, purge and embargo. The Test payload was not decoded.
- Seeds: 31, 102 and 20261018; 1,000 updates per candidate and seed; 32 forecast paths per seed; 180 Validation anchors; 96 pooled paths per candidate.
- Paired block bootstrap: 2,000 replicates, 12-anchor blocks; Holm correction over the preregistered model/horizon comparison family.
- No economic certification or model selection was performed.

Preregistration: [`R2_ETHUSDT_LONG_HORIZON_RISK_PRE_REGISTRATION_20261019_REV2.md`](R2_ETHUSDT_LONG_HORIZON_RISK_PRE_REGISTRATION_20261019_REV2.md).

## Primary result

The strongest statistical baseline on the 15-minute Validation score was E0 S1 (EWMA volatility with Train innovations), with CRPS `0.0013427077`. E2 CRPS was `0.0013115379`, an E2-minus-S1 point difference of `-0.0000311698`. Its paired 95% block-bootstrap interval was `[-0.0000579592, -0.0000112561]` and raw p was `0.0089955`, but Holm-adjusted p was `0.3418291`. The corrected comparison therefore does not pass the preregistered promotion rule. E2 also did not significantly improve over E1 at 15 minutes (difference `+0.0000037718`, 95% CI `[-0.0000187698, +0.0000213190]`, Holm p `1.0`). R2 is not promoted; R3/R4/R5/R6 remain gated.

## Risk and free-rollout diagnostics

| Diagnostic | Realized | E1 | E2 |
|---|---:|---:|---:|
| Absolute-return lag-1 autocorrelation | 0.12382 | 0.05028 | 0.01766 |
| Rolling realized-volatility mean | 0.00051083 | 0.00049171 | 0.00042487 |
| Rolling realized-volatility p95 | 0.00112116 | 0.00090871 | 0.00076878 |

The 30/60-minute loss emphasis did not resolve volatility clustering or under-dispersion; E2 was farther from the realized clustering and tail diagnostics than E1. Continue R2 only with a new, preregistered distribution hypothesis and a never-consumed scope. This scope is consumed and cannot be reused for tuning.

## Integrity and artifacts

- Reservation: `routeb-fold-ethusdt-binance-601515-20230413-v1`.
- Dataset SHA-256: `f83b783333a1887bab6a994aa4c4dc4fe2b16451a5cef0e52a4e91d8108b76dd`.
- Source marker SHA-256: `7fd778088b561a6f2dc4e512ccea1b6d3f967f710d3612b42294284552299940`.
- Parquet SHA-256: `909687e761a6d7656ddb3e1e21dca01b9f65d0267f20a3dfaf4646172ac57c47`.
- Fold SHA-256: `782d929a4386a08d5319795df3af852240c3e1eb1890ac1b8a67c0be5d8544ae`.
- Aggregate report: `.artifacts/world_model/fresh_ethusdt_20230413_longrisk_ensemble_20261019/multi_seed_fold_experiment_report.json`.
- Embedded canonical report digest: `a9b3791dc612e386373fb8166dbc57e5632b35df7b5dc96e54166fbbd13509f1`; raw file SHA-256: `a8d3ffe9fe2d6a202b59cc8e927e56a9ae8b982555db93f12057281fdb176424`.
- The aggregate and all three per-seed canonical digests verify using RouteB's Python 3.14 host runtime and UTF-8 decoding. Both pooled forecast files match their recorded hashes.
- `test_payload_decoded=false`; `includes_test=false`; registry receipt records Train/Validation only.

## Decision

Keep the result as research evidence. The result does not authorize R3/R4/R5/R6 and does not establish trading performance. Continue R2 with the v3 design's controlled prior output-distribution work; freeze its distribution family, scoring family and new scope before downloading/reading the next market payload.
