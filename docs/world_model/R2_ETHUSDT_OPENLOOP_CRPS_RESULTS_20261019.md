# R2 ETHUSDT 15-step open-loop CRPS results, 2026-10-19

Status: completed; primary R2 promotion gate not met. R3/R4/R5/R6 remain gated.

## Scope and protocol

This preregistered run changed only E2's prior observation objective to a fixed 15-step open-loop Gaussian CRPS. The RSSM prior is rolled forward without feeding future observations; all other E2 components, E1/E0, data features, risk targets, optimizer, seeds, training budget, and evaluation protocol were frozen. See [`R2_ETHUSDT_OPENLOOP_CRPS_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_OPENLOOP_CRPS_PRE_REGISTRATION_20261019.md).

- Source: 94 Binance USD-M ETHUSDT 1-minute archives for `[2021-04-05T00:00:00Z, 2021-07-08T00:00:00Z)`. All official `.CHECKSUM` sidecars passed; prepared source has 135,360 bars.
- RouteB's one-shot registry reserved Train plus Validation `[2021-04-05T00:01:00Z, 2021-06-22T00:01:00Z)`. The read covered 112,320 bars. Test/Holdout were not decoded.
- Approved 60/15/15 scoring windows, with one-day gaps, one-hour purge and 23-hour embargo. The additional day in the 94-day source was used as Train warmup.
- Three seeds (`31`, `102`, `20261018`), 1,000 updates per candidate/seed, 32 paths per seed, 180 two-hour Validation anchors and 96 pooled paths per candidate. Paired block bootstrap used 2,000 replicates and 12-anchor blocks.
- Data audit: no duplicate or missing minutes, no availability violations; market-model eligible. Perpetual economics are not certified.

## Primary result

Lower 15-minute return CRPS is better. E0 S1 was the strongest statistical baseline on Validation.

| Candidate | 15m return CRPS |
|---|---:|
| E0 S0 fixed volatility | 0.0036559661 |
| E0 S1 EWMA volatility | 0.0034373483 |
| E0 S2 stationary block bootstrap | 0.0034487597 |
| E1 GRU | 0.0033056222 |
| E2 15-step open-loop CRPS | 0.0033467892 |

E2's point estimate beat E0 S1 by `-0.0000905591`. The paired 95% interval was `[-0.0002378924, -0.0000040603]`, but the reported bootstrap `p=0.1424288` and Holm-adjusted `p=1.0` do not pass the registered significance gate. E1 also had a lower point CRPS than E2; that E2-vs-E1 comparison was inconclusive (difference `+0.0000411669`, Holm `p=1.0`). Therefore, no R2 promotion is granted.

## Risk and rollout diagnostics

E2 native log-realized-volatility CRPS beat E0 S2 at both 30 and 60 minutes in each seed after Holm correction (`p=0.0049975` for all six comparisons). These are secondary results and do not override the primary decision.

Generated paths were valid (100% bar and path validity). Generated absolute-return lag-1 autocorrelation was `0.00815`, versus realized `0.04079` and E1 `0.02039`, so volatility clustering remains understated. E2 rolling-RV mean was `0.0013091` versus realized `0.0013681`; generated p95 was `0.0028919` versus realized `0.0025955`, indicating the upper tail was too wide on this validation scope.

## Integrity and provenance

- Aggregate report file SHA-256 `63650e26ff24323a6a1416c0600d22b83705532e216e91817182926b439c2659`; embedded aggregate digest `14f24181505efe332a8b5b04de2d9452ebe6e8fd4a5aa79014f42d500ed95725` and predictive-comparison digest `33002c9d56cdcf1d3ae512b94445871007a414996579f2a955cd4873c7a19dcf` verify in the RouteB Python 3.14.7 coordinator runtime.
- All three seed report digests verify. Pooled E1 SHA-256 `568d5f7cbd89915c411b934a19b38fbdc4cf58596c2a98849bb014a753b4371b` and E2 `638200a4b698b5e7511b922d1b5bf3b4c969282897805f12f9aad8f87f5cb208`; both shapes `[180, 96, 60, 5]` match the aggregate report.
- Source dataset hash `2933e7010fa51abc05db591e865467964e796c3913697824679200069bed472e`; marker SHA-256 `bbd6abb6151f7c9236a6c1f2891dc95208bcfea89a8414ce14ed59bdedf20b87`; Parquet SHA-256 `84336c634f8d456fffdb818bdd487a21c692a6f0d89af6d4746aee918a186f62`.
- Source-preparation receipt SHA-256 `00d71bccab61b8bb2e8bac6ae4be0c50561a041a3409e150ed8bba0d9d9f0189`; Fold SHA-256 `d3c9944fdb60edc300bf64d1c785c9270eea708e3f97491342a288928aa2d740`.
- Refreshed RouteB manifest: 443 files, identity SHA-256 `0ba9a41e11bfa9691ab50b5d5f98ecb3329657492e50e2bcef00ecdd76359f20`; verification passes.
- Full 15-step CPU synthetic train/checkpoint-resume/worker smoke passes; full `tests/world_model`: 138 passed, 5 JAX-only skips; targeted Ruff passes. These verify engineering behavior, not market skill or economics.
- Registered report records `test_payload_decoded=false`, `economic_certification=false`; no Test-based selection or economic claim.

Report: `.artifacts/world_model/fresh_ethusdt_20210405_openloop_crps_ensemble_20261019/multi_seed_fold_experiment_report.json`.
