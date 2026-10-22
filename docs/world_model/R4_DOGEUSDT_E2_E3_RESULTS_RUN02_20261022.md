# R4 DOGEUSDT matched-objective E2/E3 replication — run 02, 2026-10-22

## Decision

The registered temporal replication completed all six candidate/seed runs. E3 again failed the all-seed primary gate and was not promoted. Return-CRPS no-harm passed; the Test payload remained unopened. Across both independent DOGEUSDT periods, seed 102 significantly favored E2, while seeds 31 and 20261018 significantly favored E3. The repeat of seed 102's adverse direction strengthens the seed-sensitivity finding but does not identify its cause or establish general hierarchy performance.

| Seed | E3 − E2 path log-RV CRPS | 95% bootstrap CI | Holm p | E3 − E2 return CRPS | Return 95% CI |
|---:|---:|---:|---:|---:|---:|
| 31 | -0.06341 | [-0.09976, -0.02968] | 0.001499 | -0.00002023 | [-0.00013496, 0.00007576] |
| 102 | +0.03938 | [0.00444, 0.07450] | 0.027486 | +0.00002289 | [-0.00005584, 0.00009534] |
| 20261018 | -0.09101 | [-0.11334, -0.06700] | 0.001499 | -0.00003335 | [-0.00010939, 0.00004454] |

Positive CRPS differences favor E2. Seed 102's E3 disadvantage is significantly adverse after Holm correction; the other two seeds favor E3 with corrected significance. All return-CRPS CIs include zero. Each run used the frozen 60 anchors and 128-candidate/32-accepted path protocol.

## Scope and integrity

- Instrument: `DOGEUSDT-PERP.BINANCE`; Train/Validation/Test: 360/15/15 days.
- Train/Validation registered scope: `[2023-07-30T00:01:00Z, 2024-08-11T00:01:00Z)`. Both immediate and runner ledger preflights found zero overlapping access and query-intent rows.
- Fourteen official Binance monthly archives passed published SHA-256 checks. Source preparation retained 574,560 continuous events and sealed 23,040 post-Validation rows as timestamp-only markers.
- E2 completed 2,067 updates per seed; E3 completed 1,000 updates per seed. E3's risk auxiliary settings came from the frozen E2 model config.
- RouteB isolated WSL CUDA/JAX runtime: JAX 0.11.2 on `cuda:0`; pinned DreamerV3 `e01491fad6434b2245a3b8ca201dd7faedcc458c`.
- Test/Holdout payload decoded: `false`; economics and automatic promotion: `false`.

## Reproducible identities

- Protocol SHA-256: `0b6269a36d2b0f2e5e505768693fd16a0241de4bcefffab557f3288030d4308e`
- Fold config SHA-256: `70b6edcf5a40fcbec30d9eef3c5a82bf6b95a06192c31ee30df86d58fb869ae3`
- Fold hash: `18041776e9fd778ea3a02cbd70469d1497149a35d6bfbe1c0a9af02e4ebbc7c7`
- Dataset hash: `d8999c10b1bd93908351ccccb9b63f5cc7a3285195446d46928e6cfda1d7960d`
- Marker SHA-256: `2e220481f12e24b1e231aeae6b989bda1f3f2911b49591b5069b88662b40420a`
- Runner SHA-256: `508be5a97e3a6caef67b9ab1ee70af7ea5669f2d16922353b806952918c353f8`
- Registered report SHA-256: `9bf4b224702a023b0fb609bf105a20f8c89836c716fa8e80c1d4ba2973bc4208`; canonical self-digest verifies.
- Completion receipt: `.artifacts/world_model/fresh-dogeusdt-r4-e2-e3-run02-20261022/registered/research_completion_receipt.json`; all six attempts ended `completed`.

## Cross-period diagnostic

The first DOGEUSDT Fold's E3-minus-E2 primary differences were `-0.03203`, `+0.10361`, and `-0.05223` for seeds 31, 102, and 20261018. This run produced `-0.06341`, `+0.03938`, and `-0.09101` in the same order. Seed 102's E3 disadvantage replicated across both periods, while the other seeds favored E3. This remains a three-seed, one-instrument observation; it does not establish why seed 102 underperforms or justify changing the registered gate.

Do not tune or rerun either consumed Fold. Investigate the seed-specific training/forecast path in synthetic or newly preregistered independent work. WB4/WB5, Test/Holdout, economics, and production use remain gated. Preregistration: [`R4_DOGEUSDT_E2_E3_BASELINE_PREREGISTRATION_RUN02_20261022.md`](R4_DOGEUSDT_E2_E3_BASELINE_PREREGISTRATION_RUN02_20261022.md).
