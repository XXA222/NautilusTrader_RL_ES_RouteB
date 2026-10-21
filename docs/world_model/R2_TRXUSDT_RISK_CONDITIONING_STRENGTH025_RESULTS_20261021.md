# R2 TRXUSDT risk-conditioning strength 0.25 results — 2026-10-21

## Outcome

The registered Train/Validation run completed on the frozen 2026-02-11 to 2026-05-16 source. Aggregate report SHA-256: `a6a104163f12e062ea769956342295f5b5559cdc9c6c5326035acfdc1faf0219`. Its canonical digest verifies, the completion receipt matches, all six E1/E2 seed attempts have registered → started → completed events, and both pooled forecast files match their recorded SHA-256. The reader scope was Train+Validation only; Test payload was not decoded.

The preregistered all-seeds R2 gate **did not pass**. Pooled performance improved, but only one of three seeds passed the strict Holm-corrected per-seed threshold:

| Seed | Selected E0 baseline | E2 − E0 15m path log-RV CRPS | 95% paired block CI | Holm p | Seed gate |
|---:|---|---:|---:|---:|---|
| 31 | E0_S2_STATIONARY_BLOCK_BOOTSTRAP | -0.01650 | [-0.03427, +0.00130] | 0.13293 | Fail |
| 102 | E0_S2_STATIONARY_BLOCK_BOOTSTRAP | -0.02564 | [-0.05074, -0.00854] | 0.03698 | Pass |
| 20261018 | E0_S2_STATIONARY_BLOCK_BOOTSTRAP | -0.03037 | [-0.05828, -0.00842] | 0.05397 | Fail |

The pooled 15-minute path log-RV CRPS was 0.29198 for E2 vs 0.31649 for selected E0, difference -0.02451 (95% CI [-0.04241, -0.00888], p=0.0060; 2,000 paired moving-block bootstrap replicates). The result is encouraging pooled Validation evidence, but does not override the preregistered requirement that all three seeds pass.

There was no Holm-corrected E2 harm to 15-minute return CRPS. All generated OHLCV paths were valid (100%). Pooled E2 rolling-RV mean/p95 were 0.0002478/0.0005735 against realized 0.0002426/0.0005247. The mean dispersion now matches closely, while the 95th-percentile tail remains somewhat high. Generated absolute-return lag-1 autocorrelation was 0.2107 vs realized 0.0913, so persistence is now too strong for this window.

## Data and controls

- 112,320 Train+Validation bars; zero missing minutes, duplicate minutes, or availability violations.
- Footer-only audit selected row groups 0–77; no row group crossed the exclusive read boundary. Test payload remained unread.
- Ledger completion result is `completed`; six model attempts have terminal `completed` events; report and pooled artifact digests verified independently.
- Test/Holdout and economic certification remain out of scope. Data audit reports perpetual economics ineligible because instrument rule, fee and funding evidence are absent.
- Frozen configs used E2 `rollout_risk_noise_persistence=0.65`, `rollout_risk_conditioning_strength=0.25`, with unchanged E1 comparator and matched 1,000-update optimizer budgets.

## Next work

Keep R3–R6 gated. On a new disjoint Train/Validation Fold, test one change only: retain strength 0.25 and reduce AR(1) risk-noise persistence from 0.65 to 0.35, targeting the observed excess absolute-return persistence while preserving the improved rolling-RV mean. Preregister before reading data, keep three seeds and matched budgets, and retain the all-seed gate. Do not use this Validation window for further model selection.

