# R2 TRXUSDT stochastic risk/path alignment — fresh-window pre-registration

Frozen on 2026-10-21 before downloading or decoding the replacement market window. This is the same single structural hypothesis as the prior 2026-10-21 registration. The earlier Train+Validation interval was reserved, but the row-group reader failed closed at the Validation endpoint before decoding any row-group payload. That reserved interval will not be reused. This preregistration uses a disjoint interval and a source writer that splits row groups at the registered availability boundary.

## Frozen recipe and decision rule

- E2: RouteB Gaussian-prior RSSM; five-channel OHLCV objective; Train-only equal-weight sampled free-prior path log-RV CRPS at horizons `[15,30,60]`; path coefficient `3.0`; 16 Train paths; native risk loss scale `1.0`; stochastic rollout risk conditioning enabled at geometric strength `0.1` in both Train path sampling and F1 inference.
- E1: the unchanged fixed GRU comparator. Both candidates use 1,000 updates, batch size 8, learning rate 0.0003, report interval 50, and seeds `[31,102,20261018]`.
- Validation: 180 anchors at 120-minute spacing, 32 forecast paths per seed, 96 pooled paths, 4-anchor chunks, 2,000 paired moving-block bootstrap replicates, block length 12.
- Select among frozen E0 zero-drift, Train-innovation EWMA, and stationary block-bootstrap baselines by Validation mean 15-bar path log-RV CRPS.
- Full gate: every seed must have E2-minus-selected-E0 15-bar path log-RV CRPS 95% CI strictly below zero and Holm `p<0.05` in the fixed 11-test family; no seed may show significant Holm-corrected E2 harm to 15-bar return CRPS; generated OHLCV/path structures must be valid; generated rolling-RV mean/p95 and absolute-return persistence are reported as non-bypassable diagnostics. No Test/Holdout read, economic claim, or automatic promotion.

## Frozen data scope

- Binance USD-M TRXUSDT 1m source-open interval `[2025-08-01T00:00:00Z, 2025-11-03T00:00:00Z)`: 94 UTC days, 94 daily archives or monthly archives with checksum sidecars, 135,360 expected closed bars.
- 60/15/15 Fold with one-hour purge and 23-hour embargo. Train+Validation reader scope is `[2025-08-01T00:01:00Z, 2025-10-18T00:01:00Z)`.
- Train score `[2025-08-03T00:01:00Z, 2025-10-02T00:01:00Z)`; Validation score `[2025-10-03T00:01:00Z, 2025-10-18T00:01:00Z)`, warmup begins `2025-09-21T00:01:00Z`. Test score `[2025-10-19T00:01:00Z, 2025-11-03T00:01:00Z)`, warmup begins `2025-10-07T00:01:00Z`.
- Exact Train+Validation nanoseconds: `[1754006460000000000,1760745660000000000)`. This interval is disjoint from the earlier reserved `[1745625660000000000,1752364860000000000)`.
- Parquet writer boundary: split before the first event with `available_at_ns >= 1760745660000000000`. The registered reader must decode only fully contained row groups.
- Output root: `.artifacts/world_model/fresh-trxusdt-20250801-stochastic-risk-path-aligned-20261021/`.
- Download official Binance archives only after this freeze; verify checksum sidecars and ZIP members; prepare and record source/Fold identities; run the read-only ledger preflight immediately before the registered reader; exclude Test/Holdout.

## Frozen implementation and artifact identity

- Implementation change is limited to the row-group boundary writer and its Binance source-preparation option; candidate recipes and scoring rules are unchanged.
- E1 config: `configs/routeb/fresh_trxusdt_20250801_e1_stochastic_risk_path_aligned_ensemble.json`.
- E2 config: `configs/routeb/fresh_trxusdt_20250801_e2_stochastic_risk_path_aligned_ensemble.json`.
- Append the final source/Fold/config hashes, implementation commit and code identity after data preparation, then repeat the read-only ledger preflight. Do not invoke the registered reader if the preflight or identity checks fail.
