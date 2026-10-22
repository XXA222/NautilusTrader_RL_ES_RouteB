# R2 ETHUSDT risk-weight experiment results

Date: 2026-10-19
Status: completed, primary promotion gate not met

## Frozen experiment

This run tested the preregistered increase of E2 `risk_prediction_scale` from 1.0 to 2.0. E1, model dimensions, training budget, forecast protocol, and seeds were unchanged. The registered fresh Fold used ETHUSDT USD-M 1-minute bars from the ledger-verified 2024-01-10 to 2024-04-12 source interval with the approved 60/15/15 score windows, one-day gaps, one-hour purge, and 23-hour embargo.

- Seeds: 31, 102, 20261018; 1,000 optimizer updates per model and seed.
- Validation: 180 two-hour anchors; 32 paths per seed and 96 pooled paths per candidate.
- Paired block bootstrap: 2,000 replicates, 12-anchor blocks; Holm correction across model-pair and horizon comparisons.
- Test and Holdout were not decoded. This evidence does not certify economic or trading performance.

## Primary result

The strongest statistical baseline at 15 minutes was E0 S1, the EWMA-volatility model with Train innovations. Its mean CRPS was 0.00243469684. E2's mean CRPS was 0.00243324591, a point improvement of 0.00000145092. The paired 95% interval was `[-0.0000701531, +0.0000526914]` and Holm-adjusted p was `1.0`. The interval crosses zero, so the preregistered promotion gate fails. E2 is not promoted; R3/R4 remain gated.

Against S0, E2's point improvement was larger (`-0.0000368223`), but its 95% interval also crossed zero (`[-0.0001177822, +0.0000209358]`, Holm p `1.0`). Baseline selection is based on Validation CRPS as preregistered; E0 S1 remains the comparison baseline for the primary gate.

## Risk rollout diagnostics

| Diagnostic | Realized | E1 | E2 |
|---|---:|---:|---:|
| Absolute-return lag-1 autocorrelation | 0.06434 | 0.03265 | 0.01348 |
| Rolling realized-volatility mean | 0.00106318 | 0.00082488 | 0.00086441 |
| Rolling realized-volatility p95 | 0.00200070 | 0.00161510 | 0.00173498 |

Raising the risk-loss weight moved the rollout volatility mean and p95 closer to realized values, but both remained low. Its absolute-return autocorrelation was farther from realized than E1's, so the risk-weight change did not solve volatility clustering.

## Integrity and outputs

- Registered scope: `routeb-fold-ethusdt-binance-601515-20240110-v2`; Fold SHA-256 `8b58ed540e75d14acf08305a08aa05d66ca30f70a8232a001559bbbbe4246554`.
- Dataset SHA-256: `7aafd059f086d9cc1651c171282de59a48a0c71c7f6b068209cf4b9b76c783cb`.
- Aggregate report: `.artifacts/world_model/fresh_ethusdt_2024_risk2_ensemble_20261019/multi_seed_fold_experiment_report.json`; embedded canonical digest `a5e6bc924780b8dc958ddc4f80b44c1ed308bb6699079177b69d05fca50f8041`; raw file SHA-256 `c953a390e00d724810e5e8f16f4d9e51486f2c5e5e8d4c6b7db5a38bb659129a`.
- All three per-seed report digests independently verify. Both pooled forecast files match their recorded SHA-256 values. The aggregate digest independently verifies after canonical serialization. The runner now seals and verifies this digest immediately before writing; regression tests cover digest creation and resealing after report changes.
- The run receipt records `includes_test: false`; the aggregate report records `test_payload_decoded: false`.

## Engineering verification

- `pytest tests/world_model -q`: **135 passed, 4 skipped**. The four skips require JAX in the Python 3.14 host; the experiment itself ran in the isolated RouteB Python 3.12.10/JAX worker.
- Ruff on the changed CLI, runner, and tests: **PASS**.
- Current source-manifest identity: 425 files, SHA-256 `669a3b2f60fab1ba8d7ad5c44b2701421ea2bf0870dbeec223b389b441faeae7`; manifest verification passes.

## Decision

Keep the candidate as research evidence only. Do not open R3/R4 or claim an economic benefit. The result supports continuing R2 work on persistence and volatility clustering with a new, preregistered hypothesis and a never-consumed Train/Validation interval. The experiment scope is consumed and cannot be reused for further tuning.
