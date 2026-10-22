# R2 ETHUSDT risk-scale-0.5 transfer results, 2026-10-19

Status: completed; primary promotion gate not met. R3/R4/R5/R6 remain closed.

## Frozen experiment

This fixed transfer run applied the already registered Gaussian E2 recipe with `risk_prediction_scale=0.5` to a distinct July-October 2023 ETHUSDT interval. E1/E0, all training and forecast settings, and the primary promotion rule were unchanged. See [`R2_ETHUSDT_RISK05_TRANSFER_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_RISK05_TRANSFER_PRE_REGISTRATION_20261019.md).

- Source: 93 official Binance USD-M ETHUSDT daily archives (July 15 through October 15, 2023), 133,920 bars. All exchange `.CHECKSUM` sidecars were independently rechecked.
- Registered 60/15/15 Train/Validation scope: `[2023-07-15T00:01:00Z, 2023-09-30T00:01:00Z)`, 1-hour purge, 23-hour embargo. Test was not decoded.
- Three seeds (`31`, `102`, `20261018`), 1,000 optimizer updates per seed, 32 paths per seed, 180 Validation anchors, 96 pooled paths per candidate, 2,000 paired bootstrap replicates, 12-anchor blocks.
- No economic certification or Test-based model selection.

## Primary result

The strongest statistical baseline at 15 minutes was E0 S2, narrowly ahead of S1. Lower CRPS is better.

| Candidate | 15m return CRPS |
|---|---:|
| E0 S2 stationary block bootstrap | 0.0007362676 |
| E0 S1 conditional-volatility baseline | 0.0007368085 |
| E2 risk scale 0.5 | 0.0007201673 |
| E1 GRU | 0.0007298731 |

E2 minus strongest E0 S2 was `-0.0000161002` (paired 95% CI `[-0.0000342012, +0.0000086187]`, Holm-adjusted `p=1.0`). The point estimate favors E2 but the interval crosses zero, so the registered gate fails. E2 beat S0 with Holm `p=0.03998`, but S0 was not the strongest baseline and cannot control the promotion decision. E2 versus E1 was also inconclusive (difference `-0.0000097057`, CI `[-0.0000262394, +0.0000042364]`, Holm `p=1.0`). No promotion.

## Risk and rollout diagnostics

E2's native 30-minute log-realized-volatility score beat its selected baseline in all three seeds after Holm correction (`p=0.0049975`, `0.0049975`, `0.0299850`). At 60 minutes, two seeds passed (`p=0.0069965`, `0.0209895`); the third narrowly missed (`p=0.0529735`). These secondary results cannot override the primary gate.

Generated paths were valid but under-dispersed. Realized absolute-return lag-1 autocorrelation was `0.08602`; E1 generated `0.03928`, E2 `0.02277`. Realized rolling-RV mean/p95 were `0.0003582` / `0.0008416`; E2 generated `0.0002545` / `0.0005079`.

## Integrity and provenance

- Aggregate report SHA-256 `cf16497c06a8a97e6c4ab35be2019d9b2c8ac49168c3a65ac27cdb2f9b7abfd2`; aggregate and predictive-comparison digests verify.
- All three seed report digests verify. Pooled E1 SHA-256 `af2a56688e0bdd811c9c29777c85e597116a418c21a0166b49e2ce04cf6c63eb`; pooled E2 `d420a6f4b1cd3ef9af4df208b6d7b03468facc64e07704e893a5f5d0a6465bd2`; each shape `[180, 96, 60, 5]` and matches the report.
- Dataset SHA-256 `0f3e46703e879bcb6d1b7530181106754dbbaed21f5931612e04e2ca9812ee73`; marker `0a8562b563acafd212599fceee2da2bc8335ed568f2d3a77833e308469edcc46`; Parquet `befad181767cfe1f135ea4c725478b5ebe8e15cf962d557a56e50b21f7eed7e8`.
- Source preparation receipt SHA-256 `9589b6122399bd3e54ac679feebc9af0c05275565239f19028234c2ee3c7cc56`; Fold hash `7ca42d67f92b14b5306cd80ce0e5f81f1980ec99a3a897a3a7c7aeb89d71479a`.
- Reservation label `routeb-fold-ethusdt-binance-601515-20230715-risk05-transfer-v1`; `includes_test=false`.
- Report: `.artifacts/world_model/fresh_ethusdt_20230715_risk05_transfer_ensemble_20261019/multi_seed_fold_experiment_report.json`.

The risk-scale-0.5 point estimate transferred well, but the corrected primary comparison against S2 remains inconclusive and free-rollout persistence/tails remain low. The R2 gate stays closed.
