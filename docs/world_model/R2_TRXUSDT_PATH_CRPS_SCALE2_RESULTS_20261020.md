# R2 TRXUSDT Train path log-RV CRPS scale 2.0 — results

Date: 2026-10-20. Preregistered three-seed Validation evaluation completed.

## Decision

The registered coefficient increase from 1.0 to 2.0 did not pass the all-seed primary gate. Keep R2 unpromoted; R3–R6 remain gated. No trading-economics claim is made.

## Scope and integrity

- Binance USD-M TRXUSDT 1m source-open scope: [2024-01-12T00:00Z, 2024-04-15T00:00Z), 94 official daily archives and 135,360 expected closed bars.
- Registered reader scope: [2024-01-12T00:01Z, 2024-03-30T00:01Z), Train and Validation only; 112,320 bars, zero missing or duplicate minutes, zero availability violations.
- includes_test=false, test_payload_decoded=false, economic_certification=false. Test/Holdout were not read. The data are model eligible; perpetual economics remain blocked because the required instrument rule is missing.
- Aggregate and each seed canonical digests match. Checkpoint, config, forecast, native-risk and pooled file hashes match the reports. Both pooled arrays are finite with shape (180, 96, 60, 5). Generated bar and path validity are 100%.
- Aggregate file SHA-256: f83d0b6fd0011fe8e4f2fa93d764b04e83d1f1de473764dcd96f7ebb776ebf69.
- Aggregate canonical report digest: 63887d655a8f581dbf0ac47f56adebf8ab563139cee4a5e11cd984f91fcbb6c9.
- Artifact directory: .artifacts/world_model/fresh-trxusdt-20240112-path-crps-scale2-20261020/ensemble/.

## Primary metric

The strongest Validation baseline was E0 Train-innovation EWMA. Mean 15-bar path log-RV CRPS:

| Candidate | Mean CRPS |
|---|---:|
| E0 Train-innovation EWMA | 0.317092 |
| E1 | 0.278231 |
| E2 path CRPS scale 2.0 | 0.297176 |

Pooled E2-minus-E0 difference was -0.019916, 95% paired block-bootstrap CI [-0.092988, +0.081086], p=0.651174. The interval crosses zero. E1 also scored better than E2.

| Seed | E2 minus selected E0 | 95% CI | Holm p |
|---|---:|---:|---:|
| 31 | -0.030409 | [-0.101736, +0.060965] | 0.458771 |
| 102 | -0.021311 | [-0.098601, +0.081048] | 0.638181 |
| 20261018 | -0.013575 | [-0.089499, +0.091727] | 0.764118 |

No seed met the registered requirement that the full interval be below zero and Holm p be under 0.05. The point estimates are favorable, but uncertainty is too large to establish a reliable benefit.

## Guardrails and diagnosis

At the registered 15-bar return CRPS comparison, E2 had no Holm-corrected significant harm in any seed (all adjusted p=1.0). Generated OHLCV/path structure was valid.

Volatility remains under-dispersed: E2 generated rolling-RV mean 0.000347 and p95 0.000649; realized values were 0.000515 and 0.001385. Raising the Train path-loss coefficient to 2.0 did not resolve the tail gap in this transfer.

## Frozen identities

- Dataset: e1bc66ea3f2043f9590b0990a8765f1fa792b145e21db928e2428e7df5ba32c2
- Semantic Fold: 34360fb85e58fbb6cee43851a6d055e1b3441b426bc73d87e6304b6f1ca5616b
- Source fingerprint: 593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c
- Aggregate report: .artifacts/world_model/fresh-trxusdt-20240112-path-crps-scale2-20261020/ensemble/multi_seed_fold_experiment_report.json

Do not tune on this consumed Validation interval. The next hypothesis should address the weak return/volatility persistence diagnosis with one frozen change on an unused scope.
