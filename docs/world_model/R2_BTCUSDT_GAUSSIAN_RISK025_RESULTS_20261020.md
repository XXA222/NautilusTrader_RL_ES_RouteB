# R2 BTCUSDT Gaussian rollout-risk strength-0.25 results

Status: the registered three-seed Validation experiment completed on 2026-10-20. The primary path-derived 15-minute log-RV CRPS gate failed in the harmful direction; one seed was significant after Holm correction. The Gaussian strength-0.25 recipe is not promoted. Test/Holdout remained unread; no economic claim is made.

## Scope and integrity

- Binance USD-M BTCUSDT 1-minute source `[2023-04-29T00:00:00Z, 2023-08-01T00:00:00Z)`: 94 archives, all official checksum sidecars and ZIP checks passed, 135,360 bars.
- Registered reader accessed Train+Validation `[2023-04-29T00:01:00Z, 2023-07-17T00:01:00Z)` only: 112,320 bars, zero missing or duplicate minutes. Test `[2023-07-17T00:01:00Z, 2023-08-01T00:01:00Z)` was not decoded.
- E2 used Gaussian prior, native risk loss scale `0.5`, geometric rollout risk-conditioning strength `0.25`, 1,000 updates, batch size 8, seeds 31, 102, 20261018.
- 180 Validation anchors, 32 paths/seed, 96 pooled paths; pooled shapes `[180,96,60,5]`. All paths and bars valid.
- Dataset hash `5eaaf5f0c96f034033c22fd880d91e515d5af11776ecf2c0186ff06b5b4ed64f`; marker SHA-256 `bf87685e9965bbdff0f645b6728315392935d88ce8b716faae00686885cefad3`; Parquet SHA-256 `8b132416f8d5186df82b85c757480979ad8f9ee3ca44f88943b99622efe729de`.
- Semantic Fold hash `d4124b2385c67a46164a5084e10d9775d032faef498193dd439155903d82c6d3`; aggregate report SHA-256 `a968ff500d242d9e548f6e6198b88cd48b3e0e6f2334391fe28a42be5e166d7b`.
- Independent aggregate/predictive digest, all three seed report digests, all per-seed forecast/risk files, and both pooled files verify. Test flags are false throughout.
- Immutable output: `.artifacts/world_model/fresh-btcusdt-20230429-gaussian-risk025-20261020/experiment/`.

## Primary path-derived 15-minute log-RV CRPS

The strongest Validation-selected E0 was `E0_S1_EWMA_VOL_TRAIN_INNOVATIONS` for all seeds. Lower CRPS is better; positive E2-minus-E0 is worse.

| Seed | E2 minus E0 | Paired 95% CI | Raw p | Holm p |
|---:|---:|---:|---:|---:|
| 31 | +0.070344 | [+0.004330, +0.131629] | 0.025487 | 0.050975 |
| 102 | +0.052854 | [-0.007380, +0.111280] | 0.084458 | 0.128936 |
| 20261018 | +0.101871 | [+0.033989, +0.166623] | 0.002999 | 0.005997 |
| Overall | +0.078299 | [+0.015114, +0.137527] | 0.011994 | — |

All seed differences are positive. Seeds 31 and 20261018 have intervals strictly in the harmful direction; seed 20261018 is Holm-significant. E2 mean CRPS `0.356265` was worse than E0 `0.277966` and E1 `0.253608`. The registered gate fails.

## Diagnostics and disposition

E2 generated rolling-RV mean/p95 were `0.000433`/`0.000789` versus realized `0.000379`/`0.000915`: mean volatility was high while the upper tail remained narrow. Generated absolute-return lag-1 correlation was approximately zero (`-0.00056`) versus realized `0.08062`. Strength `0.25` worsened the path score relative to zero conditioning on the prior scope; it did not restore volatility clustering. No R2 promotion; R3/R4/R5/R6 stay gated. Continue with a single lower-strength transfer candidate on a new ledger-clear scope. See preregistration.