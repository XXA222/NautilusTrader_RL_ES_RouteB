# R2 corrected risk Validation: AVAXUSDT

Date: 2026-10-19
Scope: one-shot RouteB Train/Validation only. Test and Holdout payloads were not decoded.

## Frozen run

- Instrument: `AVAXUSDT-PERP.BINANCE`, Binance USD-M one-minute source.
- Source preparation: [`B0_avaxusdt_source_preparation.json`](B0_avaxusdt_source_preparation.json).
- Dataset SHA-256: `a30ee04af57e722185c6d4dc6549225aec3301f3985199525f620bf57c34510b`.
- Marker SHA-256: `5096ac5efb362dfb7930eb88f8b13050c0a6452db98a701dba051a1ab9e616ba`.
- Fold SHA-256: `3fa82c7fdd468b0131a37a86d6ea95a7031cce3d409c64d24c1d0b3769e59038`.
- Seeds: `31`, `102`, `20261018`; 180 Validation anchors; 32 paths per seed; 96 pooled paths per model.
- Aggregate report SHA-256: `2bf033bc28304fb45751b6f9be7806921ac71baa8528cf33c3a6928dcc03473b`.
- Report flags: `test_payload_decoded=false`; `economic_certification=false`.

All artifacts and the immutable report are under `.artifacts/world_model/fresh_avaxusdt_risk_ensemble_20261019/`.

## R2 predictive gate

At 15 minutes, E2 path CRPS is `0.0038566243` and the best statistical baseline S0 is `0.0036998354`. The paired E2-minus-S0 difference is `+0.0001567889`, with 95% block-bootstrap CI `[+0.0000455565, +0.0002821152]`, raw `p=0.007996`, and Holm-adjusted `p=0.231884`. The point estimate and unadjusted interval favor S0; multiplicity-adjusted evidence does not pass the gate. R2 remains **NOT PROMOTED**.

## Corrected native risk scores

All three seed reports use `routeb.native-risk-scores.v2`, which identifies the continuous target channels as `log_return`, `log_realized_volatility`, `max_high_return`, and `min_low_return`. CRPS targets and statistical baseline samples use the same log-volatility space. The seed-level 15-minute log-RV differences versus the selected S0 path baseline were:

| Seed | E2 native minus S0 log-RV CRPS | 95% block-bootstrap CI | Holm p |
|---|---:|---:|---:|
| 31 | +0.01757 | [-0.04080, +0.07648] | 1.000 |
| 102 | +0.06385 | [+0.00294, +0.12603] | 0.133 |
| 20261018 | +0.39541 | [+0.28807, +0.50311] | 0.005 |

At 15 minutes, native risk-head return CRPS was worse than S0 in all seeds; E2-native-minus-S0 differences were `+0.00227`, `+0.00629`, and `+0.00239` for seeds 31, 102, and 20261018 respectively, each Holm-adjusted `p=0.005`. The native risk head does not currently add useful return-distribution skill. These observations are Validation evidence only and do not certify trading economics.

The immutable report's native-risk comparison row currently uses the generic label `target="realized_volatility"`; its `native_risk_scores` schema v2 and `continuous_target_space` field identify that scoring channel as log realized volatility. A follow-up metadata-only change now labels future comparison rows `log_realized_volatility` explicitly. The numeric scores in this report were already computed in log space.

## Decision

The corrected scoring defect is resolved and regression-tested, but this AVAX Validation does not pass the predictive gate. Continue R2 research on a new, pre-registered source scope and improve conditional volatility/extreme-risk calibration before considering R3/R4. No Test data was consumed, no market economics were inferred, and no R3-R6 work is authorized by this result.
