# R2 ETHUSDT risk-loss scale 0.25 results, 2026-10-19

Status: completed; the preregistered R2 primary promotion gate was not met. R3/R4/R5/R6 remain closed.

## Frozen experiment

E2 retained its Gaussian causal prior head and all model, risk-target, feature, horizon, optimizer and rollout settings. The sole change was `risk_prediction_scale=0.25`, tested against unchanged E1/E0 baselines. Three seeds used 1,000 updates each, 32 paths per seed, 180 Validation anchors, 96 pooled paths per model, 2,000 paired block-bootstrap replicates and 12-anchor blocks. The preregistration is [`R2_ETHUSDT_RISK025_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_RISK025_PRE_REGISTRATION_20261019.md).

The source comprised 93 official Binance USD-M ETHUSDT daily archives, May 10 through August 10, 2022, and 133,920 bars. All official `.CHECKSUM` files were independently matched to the archives. The registered 60/15/15 Fold reserved Train/Validation `[2022-05-10T00:01:00Z, 2022-07-26T00:01:00Z)`; Test was not decoded. No economic certification was performed.

## Primary result

The strongest statistical baseline at 15 minutes was E0 S1. Lower CRPS is better.

| Candidate | 15m return CRPS |
|---|---:|
| E0 S1 conditional-volatility baseline | 0.0034747115 |
| E2 risk scale 0.25 | 0.0034760011 |
| E1 GRU | 0.0034881407 |

E2 minus E0 S1 was `+0.0000012896` (paired 95% CI `[-0.0001002373, +0.0000716310]`, Holm-adjusted `p=1.0`). E2 was not significantly better than E1 either: difference `-0.0000121396`, 95% CI `[-0.0000822053, +0.0000427929]`, Holm `p=1.0`. The point estimate against S1 is nearly tied, but the interval is wide and the registered gate fails. E2's apparent gain over S0 also fails multiplicity correction (Holm `p=0.47476`). No promotion.

## Native risk and free-rollout diagnostics

At risk scale 0.25, the secondary native log-realized-volatility scores no longer showed significant improvements at 30 or 60 minutes; all six per-seed Holm p-values were at least `0.1939`. This contrasts with the stronger secondary improvements at scale 0.5 and indicates the lower scale removed some of that benefit without passing the primary gate.

All generated paths were geometrically valid. Realized absolute-return lag-1 autocorrelation was `0.09402`; E1 generated `0.03010`, and E2 `0.00808`. Realized rolling-RV mean/p95 were `0.0012993` / `0.0025565`; E2 generated `0.0012334` / `0.0021495`. Temporal clustering and upper-tail spread remain understated.

## Integrity and provenance

- Aggregate report SHA-256 `45d7c49943791d273b9e6ddd5f112126d20f4771c721b38d063da10c097aec23`; aggregate and predictive-comparison digests verify.
- All three seed reports verify. Pooled E1 SHA-256 `8032f3c21135de7ff7c4a718b602e3293ad1ff24cc0a349928fd1eac89ceb024`; pooled E2 `3c414c700b7d8246abd602fa75dc9442baab87ef5392a87d2641bee39eb3d560`; both shape `[180, 96, 60, 5]` and match the report.
- Dataset SHA-256 `ad851c2853350679270e35922c92aa46377587de6e12a60e9ceedc2dfe7bbc20`; marker SHA-256 `e307bd67d3b21e4e6cf95b3ea9db2ad3cc40445f2bb96da78109a8ea29d28d50`; Parquet SHA-256 `d378b80c66218c770b4e178428c5e6530de64cac54f4e61b9ba4a730c2a45420`.
- Source-preparation receipt SHA-256 `d6200bb54d9eba876821a537e60ce181c67aa77b2d08ff0b991fa3e29c775baf`; Fold hash `544570a117b958da7b06d70c95b8de919d2621773b5eef206ec6e91cf48460d9`; E2 config SHA-256 `19aab79d97e6123e2e4f4b0c5611d1f8545c520e766575b590255a9f448e3953`.
- Reservation label: `routeb-fold-ethusdt-binance-601515-20220510-risk025-v1`; `includes_test=false`.
- Report: `.artifacts/world_model/fresh_ethusdt_20220510_risk025_ensemble_20261019/multi_seed_fold_experiment_report.json`.

The 0.25 coefficient nearly matched the S1 point estimate, but uncertainty remained too large for promotion and native risk gains faded. R2 remains open; no later-stage gate is opened by this run.
