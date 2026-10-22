# R2 TRXUSDT path log-RV CRPS with 16 training paths — results

Date: 2026-10-20. Preregistered evaluation, completed without reading Test/Holdout.

## Decision

The single registered change (path_log_rv_crps_paths=16, with E1 unchanged) did **not** pass the all-seed gate. Keep R2 unpromoted and R3–R6 gated. No trading-economics claim is made.

## Scope and integrity

- Source interval: Binance USD-M TRXUSDT 1m [2023-10-10T00:00Z, 2024-01-12T00:00Z); 94 daily archives and 135,360 expected closed bars.
- Train+Validation reader scope: [2023-10-10T00:01Z, 2023-12-27T00:01Z); 112,320 bars, zero missing/duplicate minutes and zero availability violations. includes_test=false, test_payload_decoded=false, and economic_certification=false.
- The market data passed the model-data audit. Perpetual economics remain blocked because the required instrument rule is absent.
- Aggregate and all three seed report canonical digests verified. All checkpoint, config, forecast, native-risk and pooled forecast artifact hashes matched their reports. Both pooled arrays have shape (180, 96, 60, 5) and contain only finite values. Generated bar and path validity are each 100%.
- Aggregate file SHA-256: 581b69eb6c0faee4f7939da186e8e664026a22583f2e43c786e2aaaf8d105e99.
- Aggregate canonical report digest: bb1e7c8bc81b0956f8458980bd249336e84a01f072fe856dfa47855800507053.
- Validation prediction artifacts: .artifacts/world_model/fresh-trxusdt-20231010-path-crps-paths16-20261020/ensemble/.

## Primary metric

The selected Validation baseline was E0 stationary block bootstrap. Pooled mean 15-bar path log-RV CRPS:

| Candidate | Mean CRPS |
|---|---:|
| E0 stationary block bootstrap | 0.236564 |
| E1 | 0.238989 |
| E2 paths-16 | 0.228686 |

E2 minus E0 was -0.007879 (favorable direction), with paired 95% block-bootstrap CI [-0.022809, +0.013310], p=0.388306. The interval crosses zero.

Per-seed E2-minus-E0 differences and registered Holm p-values:

| Seed | Difference | 95% CI | Holm p |
|---|---:|---:|---:|
| 31 | -0.007906 | [-0.023158, +0.012143] | 0.367316 |
| 102 | -0.007226 | [-0.023275, +0.018973] | 0.911544 |
| 20261018 | +0.009722 | [-0.007735, +0.034469] | 1.000000 |

None of the three seeds passed the preregistered requirement that the CI lie strictly below zero and Holm p be below 0.05. Seed 20261018 reversed direction. Increasing Train path count from 4 to 16 therefore did not establish stable improvement.

## Guardrails and diagnosis

For E2 versus the selected E0 baseline at 15-bar return CRPS, Holm-adjusted p was 1.0 for all seeds; there was no significant corrected return-score harm. Generated OHLCV and path validity both remained 100%.

E2's generated rolling realized-volatility mean was 0.000347 and p95 0.000544, compared with realized 0.000364 and 0.000684. This remains evidence of under-dispersion; the favorable pooled point estimate is inconclusive and does not justify promotion.

## Frozen run identity

- Dataset: e273b372543285281ef57b10c5ec9fd19d4dba3bb5e33bf4a8d46ca7da3dc9bf
- Semantic Fold: 60e5a532f75b0cabe3517bdc47617c4885305f933510714fe5b17cc9a46c8ef6
- Source fingerprint: 593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c
- Aggregate report: .artifacts/world_model/fresh-trxusdt-20231010-path-crps-paths16-20261020/ensemble/multi_seed_fold_experiment_report.json

The next research change must address volatility under-dispersion with one frozen hypothesis and a fresh non-overlapping scope; do not tune on this consumed Validation interval.
