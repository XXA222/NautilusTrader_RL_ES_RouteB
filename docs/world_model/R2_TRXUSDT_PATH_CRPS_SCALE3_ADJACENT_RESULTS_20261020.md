# R2 TRXUSDT path-CRPS scale-3 adjacent transfer — results

## Scope and integrity

- Fixed coefficient-3 transfer on fresh Binance USD-M TRXUSDT 1m source-open interval [2024-10-20, 2025-01-22), using the registered 60/15/15 Fold. Train+Validation reader scope was `[1729382460000000000, 1736121660000000000)` ns.
- Three seeds [31, 102, 20261018], 1,000 updates per candidate, 180 Validation anchors, 32 forecast paths per seed (96 pooled), horizons [1,5,15,30,60]. Test/Holdout were not decoded; no economic certification.
- Data audit: 112,320 Train+Validation bars; zero missing or duplicate minutes and zero availability violations. The market data is model eligible; perpetual economics eligibility is blocked because instrument rules were not supplied.
- Aggregate digest verified using the RouteB Linux Python 3.14 runtime: `563beb189eb6b170b5df12a04a9bc1eb2f1c0ea0876710253bcc6ba871c3f14b`. All per-seed report digests matched the aggregate map. Both pooled forecasts matched their recorded hashes and shape `[180,96,60,5]`; all native risk forecast hashes matched.
- Dataset hash `e5e90419c00cc2cb9ecb697c60b0a66cb000c7f138c9c29f3125f66e6a2e30e0`; Fold hash `e612cd789c39cbe3121185245aa5d9227b6861116042593b527a8e489354542d`.

## Validation results

The Validation-selected E0 baseline was stationary block bootstrap (S2). E2 pooled 15-bar path log-RV CRPS was `0.233027` versus S2 `0.283873`, difference `-0.050846`, 95% block-bootstrap CI `[-0.099325,-0.017006]`, p `0.016992` (2,000 replicates). E1 scored better than E2 at `0.201562`.

| Seed | E2 minus selected E0, 15-bar path log-RV CRPS | 95% CI | Holm p | Path gate |
|---|---:|---:|---:|---|
| 31 | -0.063380 | [-0.112643, -0.027905] | 0.010995 | Pass |
| 102 | -0.007204 | [-0.061248, +0.032775] | 1.000000 | Fail |
| 20261018 | -0.078072 | [-0.123201, -0.040238] | 0.005497 | Pass |

The registered all-seed path gate failed because seed 102's confidence interval crosses zero and its Holm p-value is 1.0. The 15-bar return-CRPS guardrail passed: no seed showed Holm-corrected significant E2 harm against its Validation-selected S2 baseline.

E2 generated rolling-RV mean/p95 were `0.000435815 / 0.000970755`, versus realized `0.000498057 / 0.000865579`; the mean stayed low while the upper tail ran high. Generated absolute-return lag-1 autocorrelation was `0.020382` versus realized `0.034521`. Generated path and bar validity were both 100%.

## Decision

Do not promote R2 or open R3-R6. Coefficient 3.0 avoided the significant 15-bar return-CRPS harm seen in the scale-4 transfer, but did not meet the per-seed path gate, and E1 outscored E2 on average. Keep Test/Holdout sealed. A follow-up should improve the stability of the Train-only path objective and retain the return guardrail under a newly frozen adjacent scope. This run supplies Validation predictive evidence only and makes no trading-economics claim.

Aggregate report: `.artifacts/world_model/fresh-trxusdt-20241020-path-crps-scale3-20261020/ensemble/attempt-2/multi_seed_fold_experiment_report.json`.
