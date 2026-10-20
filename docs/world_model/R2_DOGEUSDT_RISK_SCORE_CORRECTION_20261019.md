# R2 fresh Validation: DOGEUSDT risk-score correction

Date: 2026-10-19
Scope: one-shot RouteB Train/Validation only; Test payload was not decoded.

## Frozen run

- Instrument: `DOGEUSDT-PERP.BINANCE`, Binance USD-M one-minute source.
- Dataset SHA-256: `e7b4a00fb596649d7efd573d5251febb5a430d7d3ff325596876a1ac4c6bacef`.
- Fold SHA-256: `ac7deae220e461655200952608657c5679590ed13389bc48b1da1b663049939f`.
- Seeds: `31`, `102`, `20261018`; 180 Validation anchors; 32 paths per seed; 96 pooled paths per model.
- Aggregate report SHA-256: `942cb345882d8471213f26b4532a6f4ab8076352a98cc47f70ba0d96d2e8fb24`.
- `test_payload_decoded` and `economic_certification` are both `false`.

The immutable run is under `.artifacts/world_model/fresh_dogeusdt_risk_ensemble_20261019_r2/`. Its ordinary path-based 15-minute return CRPS is E2 `0.0024440497` versus S0 `0.0024692805`; E2-minus-S0 is `-0.0000252309`, with paired 95% block-bootstrap CI `[-0.0000948175, 0.0000617983]` and Holm-adjusted `p=1.0`. The point estimate favors E2, but this comparison is **INCONCLUSIVE**.

## Risk-score correction

The first direct native-risk evaluation found an implementation defect in the Validation scorer. The training target's second continuous channel is `log(realized_volatility + 1e-8)`, while the scorer compared it with raw realized volatility. The emitted `realized_volatility_crps` and paired comparisons for that channel in the immutable DOGE report are invalid and must not be interpreted. The saved risk forecasts, model training, path-based predictive scores, return-channel risk scores, and barrier Brier scores are unaffected by this target-space mismatch.

The scorer now uses and labels the log-volatility target consistently, and baseline path samples are transformed to the same space before paired comparison. Regression coverage checks the centered Gaussian CRPS when the forecast mean matches the log-volatility target. Targeted RouteB tests pass (21 passed) and Ruff passes.

The corrected DOGE risk scores cannot be reconstructed under its consumed one-shot scope. The fresh AVAXUSDT one-shot corrected run is complete; its metrics and R2 decision are documented in [`R2_AVAXUSDT_CORRECTED_360_15_15_20261019.md`](R2_AVAXUSDT_CORRECTED_360_15_15_20261019.md). It did not pass the R2 promotion gate. This correction does not authorize R3/R4 or economic claims.
