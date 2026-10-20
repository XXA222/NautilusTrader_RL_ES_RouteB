# R2 TRXUSDT stochastic risk/path alignment — pre-registration

Frozen on 2026-10-21 before downloading or decoding the new source interval. The preceding deterministic rollout-risk candidates either over-expanded volatility or remained under-dispersed, while the Train path log-RV objective sampled the unconditioned prior even when F1 inference used risk-conditioned scales. RouteB now uses the same scale construction in Train path CRPS and F1 rollout. The new candidate samples conditional one-minute log-RV per path and geometrically applies it to the return-channel scale. This is one structural hypothesis, not a strength, path-count, or loss-scale sweep.

## Frozen recipe and decision rule

- E2: RouteB Gaussian-prior RSSM; five-channel OHLCV objective; Train-only equal-weight sampled free-prior path log-RV CRPS at horizons `[15,30,60]`; path coefficient `3.0`; 16 Train paths; native risk loss scale `1.0`; stochastic rollout risk conditioning enabled at geometric strength `0.1` in both Train path sampling and F1 inference.
- E1: the unchanged fixed GRU comparator. Both candidates use 1,000 updates, batch size 8, learning rate 0.0003, report interval 50, and seeds `[31,102,20261018]`.
- Validation: 180 anchors at 120-minute spacing, 32 forecast paths per seed, 96 pooled paths, 4-anchor chunks, 2,000 paired moving-block bootstrap replicates, block length 12.
- Select among frozen E0 zero-drift, Train-innovation EWMA, and stationary block-bootstrap baselines by Validation mean 15-bar path log-RV CRPS.
- Full gate: every seed must have E2-minus-selected-E0 15-bar path log-RV CRPS 95% CI strictly below zero and Holm `p<0.05` in the fixed 11-test family; no seed may show significant Holm-corrected E2 harm to 15-bar return CRPS; generated OHLCV/path structures must be valid; generated rolling-RV mean/p95 and absolute-return persistence are reported as non-bypassable diagnostics. No Test/Holdout read, economic claim, or automatic promotion.

## Frozen data scope

- Binance USD-M TRXUSDT 1m source-open interval `[2025-04-26T00:00:00Z, 2025-07-29T00:00:00Z)`: 94 UTC days, 94 daily archives, 135,360 expected closed bars.
- 60/15/15 Fold with one-hour purge and 23-hour embargo. Train+Validation reader scope is `[2025-04-26T00:01:00Z, 2025-07-13T00:01:00Z)`.
- Train score `[2025-04-28T00:01:00Z, 2025-06-27T00:01:00Z)`; Validation score `[2025-06-28T00:01:00Z, 2025-07-13T00:01:00Z)`, warmup begins `2025-06-16T00:01:00Z`. Test score `[2025-07-14T00:01:00Z, 2025-07-29T00:01:00Z)`, warmup begins `2025-07-02T00:01:00Z`.
- Read-only RouteB ledger preflight on 2026-10-21 found zero overlapping `access` rows and zero overlapping `query_intent` rows for TRXUSDT `perpetual-market-v1` on exact Train+Validation nanoseconds `[1745625660000000000,1752364860000000000)`.
- Output root: `.artifacts/world_model/fresh-trxusdt-20250426-stochastic-risk-path-aligned-20261021/`.
- Download official daily archives only after this freeze; verify checksum sidecars and ZIP members; prepare and record source/Fold identities; repeat the read-only ledger query immediately before registration; exclude Test/Holdout.

## Frozen code and configurations

- RouteB source identity SHA-256 `49c51fa24d302dfa7734cdf9eac2ebbeee0cd63d131fc9edd71ae68f579d81df`; implementation base commit `79f44dc44a27c5c01c34470498a1db9901bcafc6`.
- Registered ensemble runner SHA-256 `9fc8c0a98579d5564e4abaad61a702694f6006e6c56118610aa3854100d78f7a`.
- E1 config `configs/routeb/fresh_trxusdt_20250426_e1_stochastic_risk_path_aligned_ensemble.json`; SHA-256 `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`.
- E2 config `configs/routeb/fresh_trxusdt_20250426_e2_stochastic_risk_path_aligned_ensemble.json`; SHA-256 `01b34cac6b072aaf67a602d9f13fc8dd02f8bcd2ccddbb4e32e65533e9daa1f0`.
- Worker lock SHA-256 `ab99ac8f831bc06dc6e408cf91090b69aca413df8f38cfcd2f28b77a6ea18c1b`; isolated Python 3.12.10 / JAX 0.11.2 worker.
- Synthetic evidence before freeze: the path objective produced finite nonzero gradients for the prior head (`0.5669276`), transition (`0.9956610`), and risk head (`1.4325235`); future-observation mutation invariance and fixed-seed stochastic rollout reproducibility passed. Six isolated JAX tests and the 141-test host world-model suite passed (six expected host-JAX skips).

No source, scoring, candidate recipe, or decision-rule changes are permitted after this freeze. Source/Fold identities and the final ledger preflight will be appended before the registered reader is invoked. If reservation fails, do not access Train/Validation.
