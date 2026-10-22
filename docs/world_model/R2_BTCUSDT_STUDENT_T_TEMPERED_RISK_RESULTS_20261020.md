# R2 BTCUSDT Student-t with tempered risk-conditioning results

Status: the registered three-seed Validation experiment completed on
2026-10-20. The primary gate **failed in the wrong direction in all seeds**.
No Test or Holdout payload was decoded; no promotion or economic claim is
made.

## Scope and integrity

- Binance USD-M BTCUSDT one-minute source interval:
  `[2022-10-23T00:00:00Z, 2023-01-25T00:00:00Z)`, 94 official daily
  archives, 135,360 bars. All 94 official checksum sidecars and ZIP integrity
  checks passed.
- The registered reader accessed Train and Validation only:
  `[2022-10-23T00:01:00Z, 2023-01-09T00:01:00Z)`. All report flags show
  `includes_test=false` and `test_payload_decoded=false`.
- E2 used a Student-t marginal prior (`df=5`), native risk loss scale `0.5`,
  and distribution-correct geometric rollout risk conditioning strength `0.5`;
  training used 1,000 updates and batch size 8 for seeds 31, 102, 20261018.
- Each seed produced 32 paths over 180 Validation anchors. Four-anchor
  chunking completed without OOM. Pooled arrays have shape `[180,96,60,5]`
  and all values are finite.
- Dataset hash:
  `fdce961ae187744e035f00d8e62c3464237c422e0f6ae589fce2efef0b391690`.
- Marker SHA-256:
  `0df169b268b41dc78b0dfa1d0373ea88708bd223bf5da10073c998da2d083cd0`.
- Curated Parquet SHA-256:
  `ba28c9d15e6585ff8f83420288d70e0305a4539f9d998f60b5ce500b262e0e06`.
- Fold hash:
  `ac56f635a524119d569095c885c55fddc2bded435b1d6ddd5f6f1e61937f6e7d`.
- Aggregate canonical digest verified:
  `36c6a9dae1894d82e30b346641300792e4c25f704e709bcc9989a91e11f5d487`.
  All seed/predictive digests, per-seed forecast files and pooled forecast
  SHA-256 values verify.
- Immutable outputs:
  `.artifacts/world_model/fresh-btcusdt-20221023-student-t-tempered-20261020/experiment/`.

## Primary path-derived 15-minute log-RV CRPS

The Validation-selected strongest E0 was
`E0_S1_EWMA_VOL_TRAIN_INNOVATIONS` in all seeds. Lower CRPS is better;
positive E2-minus-E0 is worse.

| Seed | E0 mean CRPS | E2 mean CRPS | E2 minus E0 | Paired 95% CI | Raw p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 0.391901 | 0.891766 | +0.499864 | [+0.315162, +0.601029] | 0.000500 | 0.005497 |
| 102 | 0.403701 | 0.899116 | +0.495415 | [+0.320113, +0.589670] | 0.000500 | 0.005497 |
| 20261018 | 0.395003 | 0.945908 | +0.550906 | [+0.378118, +0.645678] | 0.000500 | 0.005497 |

All three paired intervals are strictly above zero, and all Holm-adjusted
p-values are significant in the harmful direction. The primary rule fails.
E2's Validation mean score was also worse than E1 in all three seeds.

## Diagnostics and disposition

Across all seeds E2 generated rolling-RV mean about `0.000449`-`0.000473`
and p95 about `0.000718`-`0.000767`; realized mean and p95 were `0.000182`
and `0.000410`. The Student-t prior plus risk scaling substantially
over-expanded both typical volatility and the upper tail in this low-volatility
period. Native risk-head CRPS was mixed and cannot substitute for the failed
path-derived primary.

This scope is consumed and will not be reused. Student-t risk-conditioned
rollouts are not promoted. Continue R2 with an isolated Gaussian candidate or
a training-time path/native-risk calibration mechanism, registered against a
new scope. R3/R4/R5/R6 remain gated. Test remains unread.
