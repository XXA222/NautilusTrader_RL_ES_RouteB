# R2 ETHUSDT Gaussian CRPS-objective results, 2026-10-19

Status: completed; primary promotion gate not met. R3/R4/R5/R6 remain closed.

## Frozen experiment and data

This preregistered experiment changed only E2's prior observation training score from Gaussian NLL to the closed-form Gaussian CRPS. The Gaussian head, risk scale 0.5, risk loss, auxiliary terms, optimizer, E1/E0 and evaluation protocol were unchanged. See [`R2_ETHUSDT_CRPS_OBJECTIVE_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_CRPS_OBJECTIVE_PRE_REGISTRATION_20261019.md).

- Source: 93 official Binance USD-M ETHUSDT daily archives (2021-01-02 through 2021-04-04); all archive SHA-256 values matched official `.CHECKSUM` sidecars. Prepared source has 133,920 bars.
- Train/Validation read scope: `[2021-01-02T00:01:00Z, 2021-03-20T00:01:00Z)`. The access and query-intent ledger had zero overlaps before reservation and the one-shot runner registered the scope.
- Three seeds (`31`, `102`, `20261018`), 1,000 updates per candidate/seed, 32 paths per seed, 180 Validation anchors, 96 pooled paths per candidate, 2,000 paired block-bootstrap replicates with 12-anchor blocks. Test was not decoded.
- No economic certification or Test-based model selection.

## Primary result

E0 S1 was the strongest statistical baseline at 15 minutes. Lower CRPS is better.

| Candidate | 15m return CRPS |
|---|---:|
| E0 S1 conditional-volatility baseline | 0.0032611095 |
| E1 GRU | 0.0032511383 |
| E0 S2 stationary block bootstrap | 0.0033534214 |
| E2 Gaussian CRPS objective, risk scale 0.5 | 0.0032632485 |

E2 minus strongest E0 S1 was `+0.0000021390` (paired 95% CI `[-0.0000520096, +0.0000839570]`, Holm-adjusted `p=1.0`). The point estimate is slightly worse and the interval crosses zero, so the registered gate fails. E2 beat the non-strongest S0 point estimate, but Holm-adjusted `p=0.23788`; it does not qualify. No promotion.

## Risk and rollout diagnostics

E2 native log-realized-volatility CRPS improved against the Validation-selected baseline at both 30 and 60 minutes in all three seeds after Holm correction (`p=0.0049975` for each seed/horizon). These secondary results do not change the primary decision.

Generated paths were valid. E2's absolute-return lag-1 autocorrelation was `0.02627`, versus `0.01954` for E1 and `0.10454` realized. The new objective increased persistence modestly relative to E1 on this scope, but it remained far below realized clustering. E2 rolling-RV mean/p95 were `0.0012836` / `0.0022853`, versus realized `0.0011907` / `0.0021147`.

## Integrity and provenance

- Aggregate SHA-256 `60c27227ce42f2df3a3fdcbc4bc0314113abe902b27545bd8c97ca83946ca617`; aggregate canonical and predictive-comparison digests verify.
- All three seed report digests verify. Pooled E1 SHA-256 `a5ae22be58fe12bbde324252265ad249c60da60b68409ddab3e469b1f34ab3b4`; pooled E2 `c4077546451ff146ecee8d9b90204d1ca8decfd1007d3637c8ff5c8519840366`; both shapes `[180, 96, 60, 5]` and match the report. All E2 seed configs record `forecast_objective="crps"`.
- Dataset SHA-256 `ffcd7f82770c8f29f0a440756a9596e9abe076141c456f35e626b742d6dd0a16`; marker `94340f12ed48c82f5dc208cc130bf93096169203ac08698dc47f9eb7f275947e`; Parquet SHA-256 `52a1405622c076999d16cc3d07f08f4bc354bb5f060ae91333ae1f97fdd79595`.
- Source preparation receipt SHA-256 `ba2349efe9e91da7398ba6d7a2337b84015a5dc20c88d62a5654a4a51ba53ed6`; Fold hash `8deaf2d4ec02315be089788631c295447ea4c477ffb8feffd9a6a75007f7630e`.
- Reservation label `routeb-fold-ethusdt-binance-601515-20210102-crps-objective-v1`; `includes_test=false`.
- Report: `.artifacts/world_model/fresh_ethusdt_20210102_crps_objective_ensemble_20261019/multi_seed_fold_experiment_report.json`.

The proper-score objective executes correctly and improves native medium-horizon risk scores, but this one independent 2021 scope did not improve primary return CRPS against the strongest baseline. R2 remains open and unpromoted.
