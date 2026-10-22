# R2 TRXUSDT Train path log-RV CRPS scale 4.0 — results

Date: 2026-10-20. Preregistered three-seed Validation transfer completed.

## Decision

Pooled Validation evidence favored the registered scale-4 E2 recipe, but the pre-registered all-seed gate failed: seed 102 had an interval crossing zero, and seed 31 missed the Holm-adjusted 0.05 threshold. Do not promote R2 or open R3–R6. The frozen recipe is a candidate for one adjacent transfer, not a promoted model.

## Scope and integrity

- Binance USD-M TRXUSDT 1m source-open scope: [2024-04-15T00:00Z, 2024-07-18T00:00Z); 94 official daily archives and 135,360 expected closed bars.
- Registered Train+Validation read scope: [2024-04-15T00:01Z, 2024-07-02T00:01Z); 112,320 bars, zero missing/duplicate minutes and zero availability violations.
- includes_test=false, test_payload_decoded=false, economic_certification=false. Test/Holdout stayed unread. Market data are model eligible; perpetual economics are blocked due to missing instrument rule.
- Aggregate canonical digest and all three seed report digests verified. Checkpoint, config, forecast, risk and pooled file hashes matched their reports. Pooled arrays are finite and shaped (180, 96, 60, 5). Generated OHLCV and path validity are 100%.
- Aggregate file SHA-256: 2b7b4919baebd52d8f52e136329d78a0b556e61531d1c7b33dec454b1f535755.
- Aggregate canonical report digest: 7be23e417ea38112ed2f781ec0ec2e1aa4da36ac9106b6d9e7209698e1d69d74.
- Artifacts: .artifacts/world_model/fresh-trxusdt-20240415-path-crps-scale4-20261020/ensemble/.

## Primary metric

Validation selected E0 stationary block bootstrap as the strongest baseline.

| Candidate | Mean 15-bar path log-RV CRPS |
|---|---:|
| E0 stationary block bootstrap | 0.253771 |
| E1 | 0.250860 |
| E2 path-CRPS scale 4.0 | 0.230824 |

Pooled E2-minus-E0 was -0.022947, with paired 95% block-bootstrap CI [-0.042399, -0.006394] and p=0.012494.

| Seed | E2 minus selected E0 | 95% CI | Holm p | Seed gate |
|---|---:|---:|---:|---|
| 31 | -0.022670 | [-0.044350, -0.002984] | 0.070465 | Fail: Holm |
| 102 | -0.019956 | [-0.043823, +0.000692] | 0.169915 | Fail: CI and Holm |
| 20261018 | -0.026418 | [-0.047432, -0.011003] | 0.011994 | Pass |

This is a meaningful pooled improvement, but only one of three seeds passed the registered conjunction; R2 remains unpromoted.

## Guardrails and diagnosis

E2 had no Holm-corrected significant 15-bar return-CRPS harm in any seed (adjusted p=1.0). Generated OHLCV/path validity remained 100%.

Generated rolling-RV mean/p95 were 0.000264/0.000457 versus realized 0.000250/0.000465. The scale-4 recipe brought both level and upper-tail width close on this interval. Absolute-return lag-1 autocorrelation remained weak (generated 0.00871 versus realized 0.10640), so volatility persistence is not yet captured.

## Frozen identities

- Dataset: cbab4476b674010700b81d579bbc097952feaaca193d3f8f84a986e6c10176d0
- Semantic Fold: a1dedfb313342c410ccd13bf93a82c413adaec186c176093ece5e664c88a2306
- Source fingerprint: 593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c
- Aggregate report: .artifacts/world_model/fresh-trxusdt-20240415-path-crps-scale4-20261020/ensemble/multi_seed_fold_experiment_report.json

The adjacent transfer below uses the same frozen recipe without parameter selection on this Validation interval.
