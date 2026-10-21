# R2 TRXUSDT risk-noise persistence 0.35 results — 2026-10-21

## Outcome

The registered fresh-window Train/Validation run completed on the frozen 2026-05-19 to 2026-08-21 source. Aggregate report SHA-256: `90a36cbd098e9266f9c9f7cfb106a25cc5465dea7687250e3b4b8bcbe32a9bdc`. Canonical report digest and completion receipt match; all six E1/E2 attempts have terminal completed events; pooled forecast files match their recorded SHA-256. The Train+Validation scope was read once through the registered reader. Test payload was not decoded.

The preregistered all-seeds R2 predictive gate **passed**:

| Seed | Selected E0 baseline | E2 − E0 15m path log-RV CRPS | 95% paired block CI | Holm p | Seed gate |
|---:|---|---:|---:|---:|---|
| 31 | E0_S2_STATIONARY_BLOCK_BOOTSTRAP | -0.06690 | [-0.09284, -0.04086] | 0.00550 | Pass |
| 102 | E0_S2_STATIONARY_BLOCK_BOOTSTRAP | -0.05775 | [-0.08024, -0.04192] | 0.00550 | Pass |
| 20261018 | E0_S2_STATIONARY_BLOCK_BOOTSTRAP | -0.06509 | [-0.08243, -0.04710] | 0.00550 | Pass |

Pooled E2 15-minute path log-RV CRPS was 0.32050 vs 0.38649 for the validation-selected E0, difference -0.06599 (95% CI [-0.08066, -0.04903], p=0.00050; 2,000 paired moving-block bootstrap replicates).

No seed showed significant Holm-corrected E2 harm to 15-minute return CRPS. All generated OHLCV paths were valid (100%). The persistence adjustment reduced generated absolute-return lag-1 autocorrelation to 0.1063 vs realized 0.0737, much closer than the preceding 0.65-persistence result (0.2107). Dispersion remains high: pooled E2 rolling-RV mean/p95 were 0.0001948/0.0004255 vs realized 0.0001456/0.0003160.

## Data and controls

- 112,320 Train+Validation bars; zero missing minutes, duplicate minutes, or availability violations.
- Footer-only audit selected row groups 0–77; no group crossed the exclusive read boundary. Test payload remained unread.
- Ledger completion status is `completed`; all six model attempts completed; aggregate and pooled artifact hashes independently verified.
- The data audit marks perpetual economics ineligible because instrument rules, fees and funding evidence are absent. This result is a predictive Validation gate only; it is not profitability, risk, or execution parity evidence.
- Frozen E2 values were persistence `0.35` and risk-conditioning strength `0.25`; E1 and the matched 1,000-update optimizer budget were unchanged.

## Next work

The E2 prerequisite for beginning R3 research is now satisfied. R3 remains a separate preregistered development phase: specify the hierarchical multi-timeframe causal clock, freeze an independent Train/Validation Fold and E3/E4 matched ablation before reading that interval, and keep Test/Holdout closed. Economic validation remains gated on contract rules, fees, funding and native execution parity. Full hierarchical R3 results do not follow from this R2 pass.

