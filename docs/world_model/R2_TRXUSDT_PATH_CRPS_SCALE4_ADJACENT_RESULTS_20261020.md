# R2 TRXUSDT path-CRPS scale-4 adjacent transfer — results

## Scope and integrity

- Fixed scale-4 recipe transferred to fresh Binance USD-M TRXUSDT 1m source-open interval [2024-07-18, 2024-10-20), using the registered 60/15/15 fold and Train+Validation reader scope only.
- Three seeds [31, 102, 20261018], 1,000 updates per candidate, 180 validation anchors, 32 paths per seed (96 pooled), horizons [1,5,15,30,60]. Test/Holdout were not decoded; `read_scope.includes_test=false`; no economic certification.
- Data audit: 112,320 Train+Validation bars; zero missing/duplicate minutes and zero availability violations. The instrument economics rule is absent, so perpetual economics eligibility remains false.
- Aggregate report canonical digest verified: `2e003164589aa71d346ecbbe3cb952ced21db997d3d6c16fa5c6ad6e2faf5d0a`. All three seed report digests matched the aggregate map. Both pooled forecast archives matched their recorded SHA-256 and shape `[180,96,60,5]`; all three native risk forecast hashes also matched.
- Dataset hash `f3a75f7f54a085938dcdc5313c5f48234385494f2e2e06b30e4c29e895e96ade`; Fold hash `036b09a8a048eadd0f905951f1abd1c4eedca153ba5b7a0d145d4fe9cac27dc1`.

## Validation results

Validation selected E0 stationary block bootstrap (S2) as the strongest frozen E0 baseline by 15-bar path log-RV CRPS. E2 mean CRPS was `0.252385` versus S2 `0.306793`; paired E2-minus-S2 difference was `-0.054408`, 95% block-bootstrap CI `[-0.093431,-0.023968]`, p `0.001499` (2,000 replicates).

| Seed | E2 minus selected E0, 15-bar path log-RV CRPS | 95% CI | Holm p | Path gate |
|---|---:|---:|---:|---|
| 31 | -0.072317 | [-0.120347, -0.036452] | 0.005497 | Pass |
| 102 | -0.055032 | [-0.090691, -0.027941] | 0.005497 | Pass |
| 20261018 | -0.054174 | [-0.092782, -0.023985] | 0.005997 | Pass |

The registered 15-bar return-CRPS guardrail failed. Seed 102 had significant E2 harm versus selected S2: difference `+0.000126349`, 95% CI `[+0.000055336,+0.000202628]`, Holm p `0.031984`. The other seeds did not show corrected significant harm at 15 bars. The guardrail requires no such corrected harm, therefore the run does not pass the full promotion gate even though all three path-RV seed gates pass.

Generated E2 rolling-RV mean/p95 were `0.000274630 / 0.000425253`, versus realized `0.000256907 / 0.000461407`. Generated absolute-return lag-1 autocorrelation was `0.003779` versus realized `0.051493`, so volatility clustering remains weak. Generated path and bar validity were both 100%.

## Decision

Do not promote R2 or open R3-R6. The adjacent transfer confirms the path-RV improvement is reproducible across all three seeds, while surfacing a material 15-minute return-score regression in seed 102. Next work should address the return-CRPS guardrail and volatility persistence under a newly frozen, adjacent Train/Validation scope; keep Test/Holdout sealed. This run is Validation predictive evidence only and makes no trading-economics claim.

Aggregate report: `.artifacts/world_model/fresh-trxusdt-20240718-path-crps-scale4-adjacent-20261020/ensemble/multi_seed_fold_experiment_report.json`.
