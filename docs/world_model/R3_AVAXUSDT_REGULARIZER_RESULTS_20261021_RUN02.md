# R3 AVAXUSDT Regularizer Ablation — Run 02 Results

**Status:** Completed, Validation evidence only  
**Date:** 2026-10-21  
**Scope:** Fresh AVAXUSDT fold beginning 2024-04-15; preregistration: [R3 run-02 preregistration](R3_AVAXUSDT_REGULARIZER_PRE_REGISTRATION_20261021_RUN02.md)

## Protocol and execution

- Arms: R2 regularizer lambda `0`, `0.01`, `0.05`.
- Seeds: `31`, `102`, `20261018`; E1 and E2 each completed 1,000 optimizer updates for every arm/seed (9 registered arm/seed attempts; 18 model fits).
- Scoring: 180 common Validation anchors, 15-step horizon, 32 paths per seed, paired block bootstrap with block length 12 and 2,000 replicates.
- Evidence is restricted to registered Train/Validation. The Test payload was not decoded; 23,040 post-Validation market rows remained sealed as timestamp-only coverage markers.
- This result does not certify economics, promote a candidate, or authorize R4–R6.

## Pooled paired Validation comparison

CRPS differences are candidate minus lambda 0; negative values favor the candidate. Both 95% intervals include zero.

| Comparison | Metric | Mean difference | 95% CI | p-value |
|---|---|---:|---:|---:|
| lambda 0.01 vs 0 | Return CRPS | -0.0000049149 | [-0.0000190027, 0.0000085830] | 0.487756 |
| lambda 0.01 vs 0 | Path log-RV CRPS | -0.0015715825 | [-0.0072867777, 0.0032736075] | 0.587206 |
| lambda 0.05 vs 0 | Return CRPS | 0.0000033224 | [-0.0000074959, 0.0000156686] | 0.574213 |
| lambda 0.05 vs 0 | Path log-RV CRPS | 0.0013588039 | [-0.0016272565, 0.0044219600] | 0.376312 |

## Interpretation

No regularizer setting shows a statistically reliable pooled Validation improvement over lambda 0. The small favorable point estimates for lambda 0.01 are inconclusive; lambda 0.05 has unfavorable pooled point estimates on both metrics. Keep lambda 0 as the reference setting for subsequent research. Do not describe either nonzero setting as superior based on this fold.

## Integrity and ledger

- Aggregate report SHA-256: `3b0abfc4c6cd190313e05b0352080b53c8e19db8ac9bd26f1962376350257f8d`.
- Aggregate report: `.artifacts/world_model/fresh-avaxusdt-20240415-r3-20261021/ensemble/run-02/r3_ablation_report.json`.
- Development completion receipt: `.artifacts/world_model/fresh-avaxusdt-20240415-r3-20261021/ensemble/run-02/research_completion_receipt.json`.
- Source dataset hash: `242843a04e2680380d8adb11cd59fff513666e9997d979d1a3a0a8c001b3f6f7`.
- Fold hash: `86e9bc4136f09975db6a1c81712d6336ecd47c0bc13afd08967cfbf053f6e108`.
- Runner SHA-256: `04bd2f9e7fe605f5c762f40e0d0c90b16d87e8e6973f62aa7bd951d83e7c3437`.
- Ledger check: 9 registered attempts, all latest attempt events `completed`; development result `completed` and bound to the aggregate report SHA.
- Completion receipt access key: `dda156be8b1e2b75088cd2c00cd36c5a184ee8875b252921eb6ce4ecb5a2b89b`.
