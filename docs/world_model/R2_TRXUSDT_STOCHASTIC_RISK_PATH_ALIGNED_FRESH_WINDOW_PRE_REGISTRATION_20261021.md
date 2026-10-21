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

## Append-only prepared data and pre-read identities

- Binance public monthly archive SHA-256 values: 2025-08 `b798348b476d5e0d9cf829a157252e3d5c5b7e85f35755da2427c3e9a0090e5b`; 2025-09 `e4c41fbacf8fe70b720d95f079384d1ab0fdbc7b15c8c83d9469bcfda6bad396`; 2025-10 `6e823ee97cad3ba2dc73f4260ab8f880585fef7c700b73d3ec4420448eb8cf4d`; 2025-11 `e7a0f53d4521e378e680504c3171b91262841e40e6573431bbc189fce6e721a8`. Source preparation revalidated each official `.CHECKSUM` sidecar and ZIP member.
- Source preparation SHA-256 `4573f9abc48c0bed0d61c841cccdc8185b704ca53fd19657b3447b70a16a74ca`; 135,360 contiguous closed bars, no missing or duplicate minute timestamps.
- Curated Parquet SHA-256 `0221c8e63227f5a5cfe9837f6aeae27fa89f4aede900b1c3b7ef113bc3b81d6c`; dataset hash `2cbafd76127a661054a3b2fbe92539ab6f07249ccf128812784e077ddf4562d5`; completed marker SHA-256 `1fe27ce49336df56595b552e30089dbf480fe3bd45387a0a3f1c8576aa4c151c`.
- Parquet has 94 row groups. A footer-only check found registered-scope groups 0–77 wholly contained in `[1754006460000000000,1760745660000000000)` and zero crossing groups. The writer split before event offset 112,320; first event beyond the reserved end has `available_at_ns=1760745661000000000`. No payload rows were decoded during this check.
- Fold hash `29e9251606e9190d5457c3da48a475c7f5b80107199c5723f7d8268dabc34314`; registered Fold config SHA-256 `e3c7346250354a296b8afda70a10b2db78ec20e64d3d262157b2717a38066970`.
- E1 config SHA-256 `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`; E2 config SHA-256 `01b34cac6b072aaf67a602d9f13fc8dd02f8bcd2ccddbb4e32e65533e9daa1f0`.
- RouteB source manifest identity SHA-256 `0e2b4199e43aed953e1a9e81027e135d1785aa216fd626936b067d268aa6bf17`; source commit `da89fdfa8fefcf4efec9f5ea30bbef8d1f282b39`; frozen world-model code fingerprint `8f8ad36767d77646d757a602e44be9aed8a4afbb1fae0049cc671576f49b708a`.
- Runner SHA-256 `9fc8c0a98579d5564e4abaad61a702694f6006e6c56118610aa3854100d78f7a`; worker environment declaration SHA-256 `04951e1986b75ece21e64f67774b8f820d20a5e0ab710a99a4bf2725e653161c`; lock SHA-256 `69584c518ce6fb3ce01ae7c4900b9f5d5c5bc522796399f72686042784ed0256`.
- Isolated worker preflight passed with NumPy 2.3.3, JAX 0.11.2, and CUDA device `cuda:0`.
