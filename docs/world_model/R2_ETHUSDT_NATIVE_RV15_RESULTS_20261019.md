# R2 ETHUSDT native 15-minute realized-volatility results, 2026-10-19

Status: completed; R2 promotion gate not met. The registered primary endpoint
improved in two of three seeds but did not meet the all-seed rule. R3/R4/R5/R6
remain gated.

## Frozen experiment and data

The fixed Gaussian E2 risk-head recipe (`risk_prediction_scale=0.5`) and the
15-minute native log-realized-volatility endpoint were preregistered before
downloading or reading this source scope. See
[`R2_ETHUSDT_NATIVE_RV15_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_NATIVE_RV15_PRE_REGISTRATION_20261019.md).

- Source: 94 Binance USD-M ETHUSDT one-minute archives, `[2021-10-10T00:00:00Z,
  2022-01-12T00:00:00Z)`. All official `.CHECKSUM` sidecars passed. The curated
  source has 135,360 bars.
- Dataset SHA-256 `b3480ea2fd36c3854853f872d4acfd4679a9502a3cce89222b0cb6e290e2d180`;
  marker `bd494f9939f4189fe90d7c3a4c7131628002774f5c6af62e3a420b50fbed42d1`;
  Parquet `56e553d5d713c2c73c055332ed7aad6311e8e927960f089db3f5020056c2d743`;
  preparation receipt `c2f72c227893a18eb6a30b878aef1e876a72e15cf504d95eac0e557ae6c3e76c`.
- Train/Validation read scope was
  `[2021-10-10T00:01:00Z, 2021-12-27T00:01:00Z)`. The one-shot reservation
  accepted it after zero-overlap access and query-intent preflights. Test and
  Holdout payloads were not decoded.
- Three seeds (`31`, `102`, `20261018`), 1,000 updates each for E1 and E2,
  180 Validation anchors, 32 paths per seed and 96 pooled paths per candidate.
  Paired comparisons used 2,000 bootstrap replicates and 12-anchor (24-hour)
  blocks. No economic result is claimed.

## Primary result

Lower CRPS is better. E0 S2 was the strongest Validation-selected baseline for
the 15-minute native log-RV comparison in all seeds.

| Seed | E2 minus E0 S2 | Paired 95% interval | Holm-adjusted p | Seed result |
|---:|---:|---:|---:|---|
| 31 | -0.048436 | [-0.074845, -0.023492] | 0.004998 | Pass |
| 102 | -0.025055 | [-0.054820, +0.009886] | 0.266867 | Inconclusive |
| 20261018 | -0.062269 | [-0.084878, -0.039145] | 0.004998 | Pass |

The registered rule required a strictly favorable interval and Holm-adjusted
`p < 0.05` in all three seeds. Seed 102 misses both conditions, so the primary
gate fails and R2 is not promoted.

The 15-minute native return CRPS was materially worse for E2 in all seeds:
E2-minus-baseline differences were `+0.002923`, `+0.003181`, and `+0.003582`;
all three paired intervals were above zero and Holm-adjusted p-values were
`0.004998`. This confirms that the risk-head gain is target-specific.

## Free-rollout diagnostics

Generated OHLCV paths were valid in all seeds, but free paths continued to
understate volatility:

| Metric | Realized | E2 generated, seed range |
|---|---:|---:|
| Absolute-return lag-1 autocorrelation | 0.03953 | 0.00098–0.01002 |
| 15-minute rolling RV mean | 0.000886 | 0.000661–0.000797 |
| 15-minute rolling RV p95 | 0.001820 | 0.001051–0.001270 |
| Valid path rate | 100% | 100% |

The native risk head can improve its log-RV density while the generated market
paths still fail to reproduce volatility persistence and tail spread. These
are separate model behaviors and both remain required for R2 acceptance.

## Integrity and provenance

- Aggregate report: `.artifacts/world_model/fresh_ethusdt_20211010_native_rv15_ensemble_20261019/multi_seed_fold_experiment_report.json`;
  SHA-256 `7a7351b45b23f5fafd44a1a98a9209a1df5eb6ef72ee5b4ab46301341f6ed9a5`.
- Aggregate canonical digest, nested predictive digest, all three seed report
  digests, six path-forecast artifacts, three native-risk artifacts and pooled
  E1/E2 files were independently recomputed and match their manifests.
- Pooled E1 SHA-256 `9e5e10a4571e48a98a59a4d637a82a43a65c62c4c949b3597c0c814d16f08bed`;
  pooled E2 `9d8a80cbaaff146dda9455d54d69fe72846279d1c7a75a058511b58c0a422830`;
  both shapes are `[180,96,60,5]`.
- Fold manifest hash `76530715d1a06a0d82ef196da3d11f1a4db54ed8e36c2abe2acf1a852cd90f26`;
  frozen config SHA-256
  `953adec0bd90c05b7ea01c8dda106a550bebf0c3354f41b24265bd89b8189660`.
- Implementation identity: RouteB commit `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`,
  source-code SHA-256
  `ed5d97f0a409f21182530e44afd220c734ca97bab374aa09f0a7ac70984cc89a`.
- The initial metadata-only preflight caught and corrected a stale inherited
  marker before reservation. It read no market rows and consumed no scope; the
  full correction record is in the preregistration.

Continue only with R2 work on a new preregistered hypothesis and unused data.
The result does not authorize R3/R4/R5/R6.
