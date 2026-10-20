# R2 TRXUSDT path-CRPS paths-32 adjacent transfer — results

Date: 2026-10-20. Registered evidence scope: Train/Validation only; no Test/Holdout access and no economic certification.

## Decision

The paths-32 transfer did not pass the preregistered all-seed path-derived log-realized-volatility CRPS gate. Keep R2 unpromoted; keep R3-R6 gated. This is predictive validation evidence only.

## Frozen scope and integrity

- Instrument/source: Binance USD-M TRXUSDT perpetual, source interval `[2025-01-22, 2025-04-26)`; Train+Validation reader scope `[2025-01-22T00:01:00Z, 2025-04-10T00:01:00Z)`.
- Validation: 180 anchors, 120-minute spacing; three training seeds `[31, 102, 20261018]`; 32 forecast paths per seed and 96 pooled paths.
- Data audit: 112,320 Train+Validation bars; zero missing minutes, duplicates, or availability violations. Test payload decoded: false. `economic_certification`: false. Perpetual economics eligibility: false because the instrument rule is missing.
- Aggregate report SHA-256 recorded by runner: `4ee1be818935b9f21e711bbb29c80a3a473c2dceebdeb59c0ae2914748977ed0`.
- Per-seed report digests matched the aggregate report. SHA-256 for both pooled forecast files and all three E2 risk forecast files matched their manifest entries.
- Pooled forecast shape for E1 and E2: `[180, 96, 60, 5]`.

## Registered scores and gate

- Selected Validation baseline: E0 stationary block bootstrap.
- Path-derived 15-minute log-RV CRPS: E0 `0.234689`; E1 `0.212234`; E2 `0.248922`. E2 minus E0 was `+0.014233`, 95% CI `[-0.032769, +0.058097]`, paired block-bootstrap `p=0.535732` (2,000 replicates). The aggregate result does not show improvement; E1 scored better than E2.
- Per-seed E2-minus-selected-E0 path score: seed 31 `+0.090732` (CI `[+0.024493, +0.156747]`, p `0.006997`, adverse); seed 102 `+0.011815` (CI `[-0.038140, +0.057786]`, p `0.624188`, inconclusive); seed 20261018 `-0.024173` (CI `[-0.056376, +0.005098]`, p `0.120440`, favorable direction but inconclusive). No seed passed the preregistered CI/Holm gate, so the all-seed gate failed.
- The 15-minute return-CRPS guardrail passed: none of the three seeds had significant Holm-corrected E2 harm versus selected E0.
- Generation diagnostics: E2 generated rolling-RV mean/p95 `0.0004185 / 0.0007073` versus realized `0.0004810 / 0.0008875`; generated absolute-return lag-1 autocorrelation `-0.00162` versus realized `0.02608`. OHLCV and path validity rates were 100%.

## Follow-up

Do not promote this recipe or unlock R3-R6. Before another experiment, inspect the seed-31 adverse path score and under-dispersed generated volatility, then preregister one justified change on a fresh, non-overlapping scope. Preserve this run unchanged as a completed negative result.

## Artifacts

- Aggregate report: `.artifacts/world_model/fresh-trxusdt-20250122-path-crps-paths32-scale3-20261020/ensemble/run-02/multi_seed_fold_experiment_report.json`
- Pooled E1/E2 validation forecasts and three per-seed reports/risk forecasts: same `run-02/` directory.
- Preregistration and append-only execution record: `R2_TRXUSDT_PATH_CRPS_PATHS32_ADJACENT_PRE_REGISTRATION_20261020.md`.
