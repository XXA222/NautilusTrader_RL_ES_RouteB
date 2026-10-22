# R2 ETHUSDT two-component Gaussian-mixture results, 2026-10-19

Status: completed; the preregistered R2 primary promotion gate was not met. R3/R4/R5/R6 remain closed.

## Frozen experiment

This one-shot experiment changed only the E2 causal prior OHLCV forecast head to a fixed two-component diagonal Gaussian mixture. Exact joint mixture NLL trained the head; free rollout sampled one shared component for each five-coordinate bar. Reconstruction and posterior auxiliary likelihoods remained Gaussian. The risk-weighted native risk head, E1, E0, optimizer, features and forecast protocol stayed frozen. See [`R2_ETHUSDT_MIXTURE_GAUSSIAN_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_MIXTURE_GAUSSIAN_PRE_REGISTRATION_20261019.md).

- Source: official Binance USD-M ETHUSDT 1-minute klines, open times `[2023-01-02T00:00:00Z, 2023-04-05T00:00:00Z)`, 93 daily archives / 133,920 bars. Each archive matched its official `.CHECKSUM` sidecar.
- Fold: 60/15/15; 1-hour purge and 23-hour embargo. The project ledger accepted one fresh reservation for Train/Validation `[2023-01-02T00:01:00Z, 2023-03-20T00:01:00Z)`. The Test interval was not decoded.
- Three training seeds (`31`, `102`, `20261018`), 1,000 updates per seed, 32 forecast paths per seed, 180 two-hour Validation anchors, 96 pooled paths per candidate, 2,000 paired block-bootstrap replicates, 12-anchor blocks.
- No economic certification or model selection was performed.

## Primary result

The lowest-scoring statistical baseline at 15 minutes was E0 S1. Lower CRPS is better.

| Candidate | 15m return CRPS |
|---|---:|
| E0 S1 conditional-volatility baseline | 0.0016401095 |
| E1 GRU | 0.0016491853 |
| E2 two-component mixture | 0.0016580219 |

E2 minus E0 S1 was `+0.0000179125` (paired 95% block-bootstrap CI `[-0.0000321407, +0.0000774650]`, Holm-adjusted `p=1.0`). E2 also did not improve over E1: difference `+0.0000088367`, 95% CI `[-0.0000255259, +0.0000442902]`, Holm `p=1.0`. The interval includes zero and the corrected p-value misses the frozen `0.05` threshold; the mixture head is not promoted. E2's 15m CRPS was also worse than the baseline point estimate.

## Risk and rollout diagnostics

The native risk head retained its log-realized-volatility gains: at both 30 and 60 minutes, each of the three seeds beat its validation-selected statistical baseline, with Holm-adjusted `p=0.0049975` per comparison. This secondary result does not override the primary R2 gate.

Free-rollout paths were valid, but volatility persistence remained weak. Realized absolute-return lag-1 autocorrelation was `0.12063`; E1 generated `0.05004`, while E2 generated `0.00410`. Realized rolling-RV mean/p95 were `0.0008134` / `0.0020140`; E1 generated `0.0006594` / `0.0012464`, and E2 generated `0.0007012` / `0.0013718`. The mixture moved the volatility level and upper tail toward realized values relative to E1, but did not recover temporal clustering.

## Integrity and provenance

- Aggregate experiment report SHA-256: `0cf6fe5fa52cc7db96e67e2f96718b56ba56fba6d5e726ceeff596c874d33e95`.
- Aggregate and predictive-comparison canonical digests verified; all three seed report digests verified.
- Pooled E1 forecast SHA-256: `ebe82fdbff96b0b7d30a17633b5ea65ba5efcbe8810671d377593c4feca6bbb8`; E2: `855011f6b1dd81d9325e6356dac1a6d7d716abba60d4025bb0378aed52b33cbc`. Both have shape `[180, 96, 60, 5]` and match the aggregate report.
- Dataset SHA-256: `e528d6509b953b0028deaf529e37f6e0b4db00ec3f2534f53b893e1bebd0d471`; source marker SHA-256: `0a4689a62dc7fd6eeda93f01b8bf20b3bd7a7522158cba23f7aa638b5c4341cf`; Fold hash: `b5bd6d26a5978fa3018e938617e34e1508e095ce30bf282f1aa6d6cdb3e6b7e4`.
- One-shot reservation label: `routeb-fold-ethusdt-binance-601515-20230102-mixture-gaussian-v1`; receipt states `includes_test=false`.
- Report: `.artifacts/world_model/fresh_ethusdt_20230102_mixture_ensemble_20261019/multi_seed_fold_experiment_report.json`.

The next R2 work should target the remaining temporal-clustering deficit with one new, preregistered and ledger-clean scope. Do not promote the mixture or open R3/R4/R5/R6 from this run.
