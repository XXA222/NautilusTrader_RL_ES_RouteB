# R4 DOGEUSDT matched-objective E2/E3 comparison — 2026-10-22

## Decision

The registered three-seed comparison completed with all six candidate attempts terminal and the Test payload unopened. E3 did **not** pass the preregistered all-seed primary gate and was not promoted. Return-CRPS passed the no-harm gate. The result is predictive evidence for this single DOGEUSDT Train/Validation Fold; it does not establish profitability, generalization, or production readiness.

| Seed | E3 − E2 path log-RV CRPS | 95% bootstrap CI | Holm p | E3 − E2 return CRPS | Return 95% CI |
|---:|---:|---:|---:|---:|---:|
| 31 | -0.03203 | [-0.05594, -0.01033] | 0.002999 | -0.00004346 | [-0.00018615, 0.00010383] |
| 102 | +0.10361 | [0.06450, 0.14166] | 0.001499 | +0.00002435 | [-0.00015342, 0.00023670] |
| 20261018 | -0.05223 | [-0.08042, -0.02430] | 0.001499 | -0.00007448 | [-0.00024273, 0.00010263] |

Positive CRPS differences favor E2. E3 significantly improved the primary metric on seeds 31 and 20261018 but significantly worsened it on seed 102. This seed disagreement fails the frozen requirement that every seed's CI be below zero with Holm p below 0.05. All three return-CRPS intervals cross zero, so no corrected return-CRPS harm was found. Each seed produced all 60 Validation anchors with the frozen 128-candidate/32-accepted path protocol.

## Frozen scope and integrity

- Instrument: `DOGEUSDT-PERP.BINANCE`; Train 360 days, Validation 15 days, Test 15 days.
- Registered Train/Validation read scope: `[2024-08-11T00:01:00Z, 2025-08-24T00:01:00Z)`. Immediate ledger preflight recorded zero overlapping access rows and zero query-intent rows.
- The official Binance receipt covers 14 monthly archives; every archive passed its published SHA-256. The curated source contains 574,560 continuous minute events. Its 23,040 events at and after the frozen Validation cutoff are timestamp-only coverage markers; no Test OHLCV values were supplied to the worker or scorer.
- E2 completed 2,067 updates × 3 seeds; E3 completed 1,000 updates × 3 seeds. E3's multi-horizon risk loss, scale floor, barrier quantile, and weights were derived from the frozen E2 configuration by the matched-objective runner.
- Runtime: RouteB's isolated WSL Python 3.12.10/JAX 0.11.2 CUDA environment on `cuda:0`; DreamerV3 commit `e01491fad6434b2245a3b8ca201dd7faedcc458c`.
- Test/Holdout payload decoded: `false`. Economic certification: `false`. Automatic promotion: `false`.

## Reproducible identities

- Protocol SHA-256: `89de47f04f22d1183f85b8d6371b91c53bfa4f1c930104e80268c8c2edd4d84d`
- Fold config SHA-256: `7ae9b64e167d0e81a60d2b8cc3848afd137642bb5513fa85699be71c1083bc42`
- Fold hash: `6c1cbe344728dd6c3f0047347b7a1d2098bfc54baa9f8afdbe554760fd022a00`
- Dataset hash: `221082db3217db3bac63a6924952cc461a7c308dc169976401eff98021dff153`
- Marker SHA-256: `0d1ffb5a6ff927a96a9697f83febced1b01eb140f4fcd4a6c0358efb955a73f5`
- Runner SHA-256: `508be5a97e3a6caef67b9ab1ee70af7ea5669f2d16922353b806952918c353f8`
- Registered report SHA-256: `97f77f22f305020515fabd6be9bb4a83ecc8d220b2a9f5493eb0f8aed3ac96cd`; its canonical self-digest verifies.
- RouteB research ledger completion receipt: `registered/research_completion_receipt.json`; all six candidate attempts have terminal `completed` events.

Full artifacts are under `.artifacts/world_model/fresh-dogeusdt-r4-e2-e3-20261022/`; preregistration: [`R4_DOGEUSDT_E2_E3_BASELINE_PREREGISTRATION_20261022.md`](R4_DOGEUSDT_E2_E3_BASELINE_PREREGISTRATION_20261022.md).

## Exploratory diagnostic after the registered decision

This descriptive review uses only saved Train/Validation outputs and does not change the registered gate or authorize tuning on this Fold. All six runs completed their fixed budgets with finite diagnostics. E3 seed 102's final loss was 1.3565, versus 0.4945 for seed 31 and 1.3073 for seed 20261018; its gradient norm was finite (39.67). The loss terms are not a cross-seed selection criterion, so they do not explain the divergence by themselves.

At the 60 paired anchors, E3's primary CRPS was worse than E2 on 44 anchors for seed 102, while it was worse on 17 anchors for each of seeds 31 and 20261018. The largest seed-102 differences cluster on several mid-August anchors (maximum +0.3694), rather than one isolated outlier. This points to seed-dependent forecast behavior across a contiguous Validation regime, but not to a verified cause. A new, independently preregistered Fold is required to determine whether the seed sensitivity replicates.

## Next work

Do not tune on this consumed Fold or open Test/Holdout. Any additional model comparison needs a new frozen hypothesis and ledger-cleared Fold; WB4/WB5 and economics remain gated.
