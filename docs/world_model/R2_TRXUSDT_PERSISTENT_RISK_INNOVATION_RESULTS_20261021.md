# R2 persistent risk innovation — Validation results (2026-10-21)

## Frozen experiment and integrity

- Source commit `d32f1fd6abf48b39fdddfa5dca12ceddeac7a0cf`; commit-bound RouteB code fingerprint `7f6178515e53dec7361e1a1434e06681430228fde92783581cb5937fa524e388`; source-manifest identity `3bfe998eb3caac70768f4941cf78dad76d354d18b126bc84dd4e8385520cb3b8`.
- Registered Fold hash `cd9113d0c1ea80e996e7f803a2e5e3e2922a062a7455424b17e98cae8eb1b67d`; dataset hash `d9c202e17488d5e4059347074aa23a8533e5d5a1aed1e1d0891b0dca146ab956`; source Parquet SHA-256 `358dce9c317f485de50123fde20d1283c5e8bb55fc9d02c2d2f44eb34ca18036`; completed marker SHA-256 `57bca13d146f70d44b0b3e322a4a99323ff1b0bd1757868dd9f998cd7405209b`.
- Report: `.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/ensemble/run-01/multi_seed_fold_experiment_report.json`; canonical report SHA-256 `d9c667ae3d4c5bfad326c7768d326519c34d0b1179fe319e1cdba22ba7ff3526`. The report digest recomputes; the completion receipt and `development_result` ledger row contain the same digest and status `completed`. All six E1/E2 seed attempts have terminal `completed` events.
- Seeds `[31,102,20261018]`, 1,000 updates per candidate per seed, 180 common Validation anchors, 32 paths per seed and 96 pooled paths. Train/Validation scope `[2025-11-06T00:01:00Z,2026-01-23T00:01:00Z)`; `test_payload_decoded=false`; registered read scope excludes Test. Pooled forecast files match their recorded SHA-256 values.
- Data audit: 112,320 Train/Validation bars, zero missing or duplicate minutes, zero availability violations. Market-model eligibility passed; perpetual-economics eligibility remains false because contract rules, fees and funding evidence are missing.

## R2 result

The preregistered 15-minute log-realized-volatility path CRPS gate failed. The selected Validation baseline was `E0_S0_ZERO_DRIFT_FIXED_VOL` (mean CRPS `0.29606`). E2 (`rollout_risk_noise_persistence=0.65`) scored `0.32154`, a worse E2-minus-E0 difference of `+0.02547`; its paired 95% moving-block bootstrap interval was `[-0.00393, +0.04840]`, with `p=0.05597`. It did not meet the required per-seed improvement rule.

Per seed, E2-minus-E0 15-minute path log-RV CRPS differences were:

- Seed 31: `+0.03362`, 95% CI `[+0.00410, +0.05986]`, Holm `p=0.06447`.
- Seed 102: `-0.00106`, 95% CI `[-0.02789, +0.02221]`, Holm `p=0.92754`.
- Seed 20261018: `+0.05322`, 95% CI `[+0.01983, +0.07760]`, Holm `p=0.00550` (significant harm to this path score).

There was no significant Holm-corrected E2 harm to 15-minute return CRPS. Pooled E2-minus-E0 return CRPS was `+1.51e-7`, 95% CI `[-2.88e-5, +9.23e-6]`, Holm `p=1.0`. Generated OHLCV bars and paths were valid for 100% of samples.

The AR(1) change moved generated absolute-return lag-1 autocorrelation to `0.1130`, versus realized `0.0806`; E1 was `0.0140`. E2 rolling-RV mean and p95 remained low at `0.0002797` and `0.0005632`, versus realized `0.0003329` and `0.0006645`. The intervention improved persistence but did not improve the primary path score and left dispersion understated.

This is not an R2 success, does not certify trading economics, and does not unlock R3-R6. Test/Holdout payloads were not read. Continue only with a new preregistration on a disjoint window, testing a single increase in geometric risk-conditioning strength while keeping AR(1) persistence fixed at 0.65.