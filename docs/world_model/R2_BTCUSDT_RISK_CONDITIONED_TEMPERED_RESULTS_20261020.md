# R2 BTCUSDT geometrically tempered risk-conditioning results

Status: the registered three-seed Validation experiment completed on
2026-10-20. The primary gate is **inconclusive and fails promotion**. No Test
or Holdout payload was decoded and no economic claim is made.

## Scope and integrity

- Binance USD-M BTCUSDT one-minute source interval:
  `[2022-07-21T00:00:00Z, 2022-10-23T00:00:00Z)`, 94 daily archives,
  135,360 bars. All official checksum sidecars passed.
- Registered Train/Validation reader range:
  `[2022-07-21T00:01:00Z, 2022-10-07T00:01:00Z)`. Test flags are false in
  every seed and the aggregate report.
- E2 used a Gaussian prior and geometrically tempered rollout risk conditioning
  strength `0.5`; 1,000 updates, batch size 8, seeds 31, 102, 20261018.
- Every seed generated 32 paths over 180 anchors. Forecasts used four-anchor
  chunks; pooled arrays have shape `[180,96,60,5]` and all values are finite.
- Dataset hash:
  `6b1d8088b5da6a68bd1c1db92f3c91e73198b24213e956de98fb5fb658e17917`.
- Marker SHA-256:
  `83b954a674d471a500f1ad7160330c8f530e318ee18303044d3f1fdf0ddd3d46`.
- Parquet SHA-256:
  `ad9d6643c3083198337633a60ec8b9a135b1b9edd3403322ad7bddb8dabea655`.
- Fold hash:
  `c67ce7a2a99750452a80e430d63e87ac2de6a0807f2c656d71d914e4c0678eca`.
- Aggregate digest verified:
  `91cfa4a179ba9d5cc1980e7ca01ffb605f4d1118defd4e8abd14909602b8f229`.
  All seed and predictive digests, all E1/E2 individual forecast files and
  both pooled arrays match their recorded SHA-256 values.
- Immutable outputs:
  `.artifacts/world_model/fresh-btcusdt-20220721-risk-conditioned-tempered-20261020/experiment/`.

## Primary path-derived 15-minute log-RV comparison

The Validation-selected strongest E0 was `E0_S1_EWMA_VOL_TRAIN_INNOVATIONS`
for all seeds. Lower CRPS is better; positive E2-minus-E0 is worse.

| Seed | E0 mean CRPS | E2 mean CRPS | E2 minus E0 | Paired 95% CI | Raw p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 0.293952 | 0.307556 | +0.013604 | [-0.038021, +0.072384] | 0.648175 | 1.000000 |
| 102 | 0.290881 | 0.286389 | -0.004492 | [-0.046578, +0.046240] | 0.855072 | 0.855072 |
| 20261018 | 0.300165 | 0.314118 | +0.013953 | [-0.025813, +0.060970] | 0.549725 | 0.549725 |

All three intervals cross zero and no Holm-adjusted p-value is below 0.05.
The gate fails closed; the lone small favorable point estimate in seed 102 is
not evidence of improvement. E1's point score was lower than E2 for all seeds.

## Diagnostics and next step

Geometric strength `0.5` brought the generated rolling-RV mean close to the
realized mean in all seeds (generated `0.000736`-`0.000746`, realized
`0.000711`). Generated p95 remained below realized p95 (`0.001106`-`0.001345`
versus `0.001479`). The output path distribution therefore remains too narrow
in the upper tail, while the primary 15m CRPS comparison is inconclusive.
The full-strength revision had substantially over-expanded the mean and p95;
tempering corrected the mean but did not establish the registered benefit.

This consumed scope will not be reused. The next hypothesis tests a Student-t
OHLCV prior with correctly variance-matched, geometrically tempered native-risk
conditioning to address the remaining tail-width mismatch, on a fresh
zero-overlap interval. Test from this run remains closed. See its
pre-registration before further data access.
