# R2 ETHUSDT risk-scale-0.5 adjacent-period results, 2026-10-19

Status: completed; primary promotion gate not met. R3/R4/R5/R6 remain closed.

## Frozen experiment and data

This single preregistered run applied the unchanged Gaussian E2 recipe with `risk_prediction_scale=0.5` to the next chronological ETHUSDT interval following the July-October 2023 transfer. See [`R2_ETHUSDT_RISK05_ADJACENT_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_RISK05_ADJACENT_PRE_REGISTRATION_20261019.md).

- Source: 93 official Binance USD-M ETHUSDT daily archives (2023-10-16 through 2024-01-16); every archive SHA-256 matched its `.CHECKSUM` sidecar. Prepared source has 133,920 bars.
- Train/Validation read scope: `[2023-10-16T00:01:00Z, 2024-01-01T00:01:00Z)`. The access and query-intent ledger had zero overlaps before reservation and was authoritatively reserved by the one-shot runner.
- Three seeds (`31`, `102`, `20261018`), 1,000 optimizer updates per candidate/seed, 32 paths per seed, 180 Validation anchors, 96 pooled paths per candidate, 2,000 paired block-bootstrap replicates with 12-anchor blocks. Test was not decoded.
- No economic certification or Test-based model selection.

## Primary result

At 15 minutes, E0 S2 was the strongest statistical baseline. Lower CRPS is better.

| Candidate | 15m return CRPS |
|---|---:|
| E0 S2 stationary block bootstrap | 0.0014909569 |
| E0 S1 conditional-volatility baseline | 0.0014997719 |
| E1 GRU | 0.0014815420 |
| E2 risk scale 0.5 | 0.0014828516 |

E2 minus E0 S2 was `-0.0000081053` (paired 95% CI `[-0.0000539319, +0.0000161153]`, Holm-adjusted `p=1.0`). The point estimate favors E2, but the interval crosses zero and the registered gate fails. E1 had a slightly lower point CRPS than E2; its paired comparison with E2 is also inconclusive. No promotion.

## Risk and rollout diagnostics

E2 native log-realized-volatility CRPS improved versus the Validation-selected baseline at 30 minutes in all three seeds (Holm-adjusted `p=0.0049975` each), and at 60 minutes in two of three seeds (`p=0.0049975`, `0.0049975`, and `0.4457771`). These are secondary results and do not change the primary decision.

Generated paths were valid, but E2 remained under-dispersed. Realized absolute-return lag-1 autocorrelation was `0.09146`; E1 generated `0.04157`, E2 `0.00996`. Realized rolling-RV mean/p95 were `0.0006376` / `0.0012382`; E2 generated `0.0004975` / `0.0008813`.

## Integrity and provenance

- Aggregate report SHA-256 `d6b49d2af090f78ff283bbd73d7273fb1a0490bc4e26395bba47f49a9ca6f214`; aggregate canonical digest and predictive-comparison digest verify.
- All three seed report digests verify. Pooled E1 SHA-256 `41bb8e60d426be8ee6b044d203a7e3d122c01bd736dca6c8e798e6831b5af2a3`; pooled E2 `3387673818f82de11a388f1d2d429f096d31f270a4bf74d5f74415488fd1a58a`; both shapes are `[180, 96, 60, 5]` and match the report.
- Dataset SHA-256 `68e29f58f1fe3c81af4c0e741b8838e057d531eabb693ffd32fc6754ae987d37`; marker `6cf4ca844440d00fc9d848cd8124347408a530033132129dcb7a2dc8e2ba7e23`; Parquet SHA-256 `c40d0ae4cf77505dcc013478bd56e4fc520ba50a3b8f3a921fffb132d5d62c7f`.
- Source preparation receipt SHA-256 `e9387aaa3804848decc0dd4d818b619e2a4d20aec8fb330e5751db04503561b0`; Fold hash `e10ffb5b33e51a1997a8945eac7968986cee9c33ff868600fd529f15463683d0`.
- Reservation label `routeb-fold-ethusdt-binance-601515-20231016-risk05-adjacent-v1`; `includes_test=false`.
- Report: `.artifacts/world_model/fresh_ethusdt_20231016_risk05_adjacent_ensemble_20261019/multi_seed_fold_experiment_report.json`.

The adjacent-period result reproduces native medium-horizon risk gains but does not pass the return-CRPS promotion gate, and the free rollout still understates volatility persistence and tails. R2 remains open and unpromoted.
