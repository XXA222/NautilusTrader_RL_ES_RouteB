# R2 ETHUSDT risk-loss scale 0.5 results, 2026-10-19

Status: completed; the preregistered R2 primary promotion gate was not met. R3/R4/R5/R6 remain closed.

## Frozen experiment

This one-shot experiment tested whether the native risk objective was competing too strongly with next-bar forecast quality. E2 kept the Gaussian prior head and all model/feature/horizon/optimizer settings fixed; only `risk_prediction_scale` changed from 2.0 to 0.5. E1 and E0 baselines were unchanged. See [`R2_ETHUSDT_RISK05_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_RISK05_PRE_REGISTRATION_20261019.md).

- Source: official Binance USD-M ETHUSDT 1-minute klines, open times `[2022-02-01T00:00:00Z, 2022-05-05T00:00:00Z)`, 93 daily archives / 133,920 bars. Every official `.CHECKSUM` sidecar matched its archive.
- Fold: 60/15/15, 1-hour purge, 23-hour embargo. The one-shot project reservation covered Train/Validation `[2022-02-01T00:01:00Z, 2022-04-19T00:01:00Z)`. Test was not decoded.
- Three seeds (`31`, `102`, `20261018`), 1,000 updates per seed, 32 paths per seed, 180 two-hour Validation anchors, 96 pooled paths per candidate, 2,000 paired block-bootstrap replicates and 12-anchor blocks.
- No economic certification or Test-based selection was performed.

## Primary result

The strongest statistical baseline at 15 minutes was E0 S1. Lower CRPS is better.

| Candidate | 15m return CRPS |
|---|---:|
| E0 S1 conditional-volatility baseline | 0.0016327746 |
| E1 GRU | 0.0016390082 |
| E2 risk scale 0.5 | 0.0016427192 |

E2 minus E0 S1 was `+0.0000099446` (paired 95% CI `[-0.0000297960, +0.0000494486]`, Holm-adjusted `p=1.0`). E2 minus E1 was `+0.0000037110` (95% CI `[-0.0000292057, +0.0000422306]`, Holm `p=1.0`). Neither comparison passes the promotion rule. E2 did beat E0 S0 by `-0.0001126506` (95% CI `[-0.0001893538, -0.0000499490]`), but its Holm-adjusted `p=0.05547` narrowly misses 0.05 and S0 was not the strongest baseline. R2 is not promoted.

## Risk and rollout diagnostics

Native risk predictions retained useful log-realized-volatility scores. Against each seed's Validation-selected statistical baseline, all three seeds improved at both 30 and 60 minutes; Holm-adjusted p-values ranged from `0.0049975` to `0.0284858` across these six comparisons.

Free paths were valid, but temporal clustering remained weak. Realized absolute-return lag-1 autocorrelation was `0.08675`; E1 generated `0.02486` and E2 `0.00077`. Realized rolling-RV mean/p95 were `0.0007368` / `0.0015178`; E2 generated `0.0007047` / `0.0012964`. The level was close, but the tail and persistence remained low.

## Integrity and provenance

- Aggregate report SHA-256: `c53f9a1f9b1f27d7cb8d7823ac775254890de8201ba5dcfb3e3e762cd3cdec39`; aggregate and predictive-comparison digests verified.
- All three seed report digests verify. Pooled E1 SHA-256 `6d6ec32f1ed93a619546c3f51aac62a1d9ad67f3e3e49af0437abc5841aeddc3`; pooled E2 SHA-256 `33bb051d9e5c8c5c96d29eb31734fd598a690f1fb692375a1a13e6667b42c8d6`; both have shape `[180, 96, 60, 5]` and match the aggregate report.
- Dataset SHA-256: `a47aca409ff6e0db3e5c161030ac6b0443d984737a89b1954d48c3f16da13cf3`; marker SHA-256: `ad329c70d646c97f6fd22dee3ab68348150491af5020565b721d67bb624b6b2a`; Parquet SHA-256: `5b6448e26b5df396a58d3cf47f5c592e7a3bdc7a5f7059538c35fd84e443c5af`.
- Source preparation receipt SHA-256: `15b3d362050877104dc47943fc11a2788adac6b029fd3af587e70d74b6971020`; Fold hash `c73e849fcdab29ab61f252d7d8f99422cb10da94d66197aa9fb077e08d35867f`.
- E2 config SHA-256: `e64938514b7ed10ae375b2dc42e100100772a9f406deb8fbd50b299606ca60e9`. Reservation label: `routeb-fold-ethusdt-binance-601515-20220201-risk05-v1`.
- Report: `.artifacts/world_model/fresh_ethusdt_20220201_risk05_ensemble_20261019/multi_seed_fold_experiment_report.json`.

Reducing the risk-loss scale from 2.0 to 0.5 retained secondary risk gains but did not resolve primary 15-minute CRPS or the rollout-persistence deficit. The next R2 test is a single lower fixed scale on a fresh scope; no sweep and no R3/R4/R5/R6 promotion.
