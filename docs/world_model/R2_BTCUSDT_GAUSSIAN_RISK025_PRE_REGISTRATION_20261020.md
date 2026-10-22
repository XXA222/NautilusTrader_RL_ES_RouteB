# R2 BTCUSDT Gaussian rollout-risk strength-0.25 pre-registration

Status: frozen on 2026-10-20 before downloading or decoding this source interval. This is one diagnostic candidate; no strength sweep is allowed.

## Hypothesis and fixed recipe

Recent BTC Validation runs show opposing calibration errors: full-strength Gaussian risk scaling significantly over-expanded path log-RV; geometric strength 0.5 improved typical rolling-RV level but was inconclusive and left tails low; Student-t strength 0.5 over-expanded both mean and p95; unconditioned Gaussian reduced expansion but remained under-dispersed and did not improve the primary CRPS. This run tests one intermediate geometric strength, `0.25`, while retaining the Gaussian prior and unchanged native risk objective. It asks whether partial state-conditional scaling can improve path-derived 15m log-RV CRPS and RV level without the prior over-expansion.

- E2: two-level RSSM, Gaussian prior, risk horizons `[1,5,15,30,60]`, native risk prediction loss scale `0.5`, rollout risk conditioning enabled at geometric strength `0.25`, 1,000 updates, batch size 8, seeds `31,102,20261018`.
- E1: frozen GRU baseline, 1,000 updates and batch size 8.
- E0: frozen zero-drift, Train-innovation EWMA and stationary block-bootstrap baselines; select the lowest Validation mean path-derived 15-minute log-RV CRPS before pairing.
- Forecast: 32 paths per seed; 180 Validation anchors at 120-minute intervals; 96 pooled paths; four-anchor forecast chunks; 2,000 paired moving-block bootstrap replicates with 12-anchor blocks.
- Primary gate: for each seed E2-minus-selected-E0 path-derived 15-minute log-RV CRPS must have its paired 95% interval strictly below zero and Holm-adjusted p below `0.05`. Lower CRPS is better.
- The Holm family remains 11 tests per seed: native return and native log-RV CRPS at horizons `[1,5,15,30,60]`, plus path-derived 15-minute log-RV CRPS.
- Test and Holdout remain unread. No economic claim or promotion is made unless the complete registered gate passes.

## Frozen scope and governance

- Source: Binance USD-M BTCUSDT 1-minute klines, open-time interval `[2023-04-29T00:00:00Z, 2023-08-01T00:00:00Z)` (94 daily archives).
- Train+Validation reader scope: `[2023-04-29T00:01:00Z, 2023-07-17T00:01:00Z)`; approved 60/15/15-day Fold with warmups, one-day purges, and embargo as declared in its frozen JSON.
- Test score interval: `[2023-07-17T00:01:00Z, 2023-08-01T00:01:00Z)`; never decoded.
- RouteB `.artifacts/research-access.sqlite` was inspected read-only immediately before registration: zero overlapping `access` and zero overlapping `query_intent` rows for BTCUSDT `perpetual-market-v1` over the Train+Validation scope.
- Output root: `.artifacts/world_model/fresh-btcusdt-20230429-gaussian-risk025-20261020/`.
- The source preparation must validate all official Binance `.CHECKSUM` sidecars and ZIP members. The registered reader checks marker, source identity, Fold hash, repository/source identity and exact Train+Validation window before decoding.

## Append-only execution record

Append exact source/config/Fold/code hashes before the registered reader runs. After the one-shot run, append aggregate and per-seed report/artifact hashes and the result. Any Test access or missing/invalid output fails closed.
## Append-only frozen execution identities

Frozen after Binance archive and source preparation and before the registered reader ran:

- Repository commit: `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`; source identity SHA-256: `4b1327326acd79bdfaf50da581bfb11367c6871c13818292571ba3772ed0ee8e`.
- E2 configuration SHA-256: `89751d4a1196f9f0c349e87c10b09cca8fd200d2d3e5fd73abea9bec449bab5a`.
- Fold configuration SHA-256: `96a193f0441afd6c81898eb73c73b78c965e8b2f50fb9e5d200399903a79f565`; semantic Fold hash: `d4124b2385c67a46164a5084e10d9775d032faef498193dd439155903d82c6d3`.
- Dataset snapshot hash: `5eaaf5f0c96f034033c22fd880d91e515d5af11776ecf2c0186ff06b5b4ed64f`; marker SHA-256: `bf87685e9965bbdff0f645b6728315392935d88ce8b716faae00686885cefad3`; curated Parquet SHA-256: `8b132416f8d5186df82b85c757480979ad8f9ee3ca44f88943b99622efe729de`.
- Source preparation verified all 94 official `.CHECKSUM` sidecars and ZIP members; curated 135,360 bars.
- Read-only RouteB ledger preflight found zero `access` and zero `query_intent` overlap for `[2023-04-29T00:01:00Z, 2023-07-17T00:01:00Z)`.

Output directory: `.artifacts/world_model/fresh-btcusdt-20230429-gaussian-risk025-20261020/experiment`.

## Append-only execution record

The three-seed Validation run completed. Aggregate report SHA-256 `a968ff500d242d9e548f6e6198b88cd48b3e0e6f2334391fe28a42be5e166d7b`; seed report SHA-256 values: seed 31 `fa921d202574e92d81dea6d69dbe33509a2930154184c91ff693e17bdacd90b7`, seed 102 `6323848f8711aceb9ac98db767fdfcc42f3df8efc24c1740f946a8560c1f6818`, seed 20261018 `f8cfdf62aea40bbd45c6a0777dc4cd312532bdf8f973b79846d909d9c6e289e8`. Pooled E1 SHA-256 `fd49cd1ecc45da6ca5b9a9d457afe516f29d7679e403843a9fd3e25e415ae921`; pooled E2 SHA-256 `3af3230758a633c70fbbd1ffad91fe335988cdd4c216979ec3fab6687b6a0bc1`. Aggregate and predictive digests and all seed/per-seed/pooled artifacts verify; Test remained unread. Primary path gate failed harmfully; see [`R2_BTCUSDT_GAUSSIAN_RISK025_RESULTS_20261020.md`](R2_BTCUSDT_GAUSSIAN_RISK025_RESULTS_20261020.md).
