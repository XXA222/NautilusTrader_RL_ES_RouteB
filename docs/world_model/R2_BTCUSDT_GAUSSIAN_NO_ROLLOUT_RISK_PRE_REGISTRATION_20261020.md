# R2 BTCUSDT Gaussian rollout-conditioning ablation pre-registration

Status: frozen before downloading or decoding the source interval on 2026-10-20. This is a diagnostic ablation, not a promotion candidate.

## Hypothesis and fixed comparison

After full-strength Gaussian conditioning and strength-0.5 Gaussian conditioning both failed their primary path-derived 15-minute log-realized-volatility (log-RV) comparison (the latter was inconclusive), and a Student-t prior with strength-0.5 significantly over-expanded the path tail, this run isolates the rollout adjustment. E2 retains the Gaussian prior and the same native multi-horizon risk training objective at scale 0.5, but disables risk-conditioned rollout scaling. If rollout scaling is the source of the path miscalibration, this fixed ablation should reduce Validation path-derived 15-minute log-RV CRPS versus the strongest Validation-selected E0 baseline and improve the generated rolling-RV distribution diagnostics.

No tuning or alternate distribution/conditioning sweep is allowed on this scope. The result is diagnostic and cannot promote R2 by itself.

- E2: RouteB two-level RSSM, Gaussian prior, risk horizons `[1,5,15,30,60]`, risk prediction loss scale `0.5`, `rollout_risk_conditioning=false`, strength `0.0`, 1,000 updates, batch size 8, seeds `31,102,20261018`.
- E1: frozen GRU baseline and optimizer budget.
- E0: frozen zero-drift, Train-innovation EWMA, and stationary block-bootstrap baselines; select the lowest Validation mean path-derived 15-minute log-RV CRPS before pairing.
- Forecast: 32 paths per seed, 180 anchors at 120-minute intervals, 96 pooled paths per candidate, 2,000 paired moving-block bootstrap replicates, 12-anchor blocks.
- Primary gate: for every seed, E2-minus-selected-E0 path-derived 15-minute log-RV CRPS must have a paired 95% interval strictly below zero and Holm-adjusted p below `0.05`. Lower CRPS is better.
- The Holm family remains 11 tests per seed: native return and native log-RV CRPS at horizons `[1,5,15,30,60]`, plus the path-derived 15-minute log-RV CRPS.
- Test and Holdout remain unread; no economic claim is made.

## Frozen source and access scope

- Source: Binance USD-M BTCUSDT 1-minute bars, open-time interval `[2023-01-25T00:00:00Z, 2023-04-29T00:00:00Z)` (94 daily archives).
- Registered reader scope: Train and Validation only, `[2023-01-25T00:01:00Z, 2023-04-14T00:01:00Z)`. The exact Fold score intervals retain the approved 60/15/15-day split, one-day purges, and warmup intervals.
- Test score scope: `[2023-04-14T00:01:00Z, 2023-04-29T00:01:00Z)`; never decode it.
- RouteB's independent `.artifacts/research-access.sqlite` was inspected read-only immediately before registration. The candidate Train+Validation range has zero overlapping BTCUSDT `access` rows and zero overlapping `query_intent` rows.
- Output root: `.artifacts/world_model/fresh-btcusdt-20230125-gaussian-no-rollout-risk-20261020/`.

## Integrity and result policy

The official Binance `.CHECKSUM` sidecars and ZIP members must verify before deterministic curation. The registered Fold reader must validate the dataset marker/hash, source bounds, frozen repository/source identity and exact Train+Validation scope before reading. Aggregate, seed, predictive, and pooled artifact hashes are independently checked after the run. Any missing integrity check, Test access, or invalid output fails closed. Append code/config/source/report identities and the result after the one-shot run; never overwrite the preregistration.

## Append-only frozen execution identities

Frozen after official-source preparation and before the registered Train/Validation reader ran:

- RouteB repository commit: cb7a3ec9d542ed48b9f182acbaac330fc90b3177.
- RouteB model/source identity SHA-256: 4b1327326acd79bdfaf50da581bfb11367c6871c13818292571ba3772ed0ee8e.
- Model implementation SHA-256: 0e463779338a7ff2238f7ba4dd4a41de72a8e032b09ae282c1502c501378fe05.
- Regression-test SHA-256: 9ea5d095e28d7423ef6a4254996b3139db60345a5e5a47b08e86b4a3a813fd11.
- E2 configuration SHA-256: $e2.
- Fold configuration SHA-256: $fold; semantic Fold hash: 63eb0971017b97eef3622e466e27b7cd72106be2e09f22050138d5e30575dc35.
- Dataset snapshot hash: d00cd88db1b3b623ce38cde6ab6a97a8dc4f6acee08e5f7bb788f8d40ee2e95a; marker SHA-256: ca05a6a393cb9f9f33994e491458a2b7d147dd69789385934d474996c122c4d7; Parquet file SHA-256: 2cba49ceb75ad86f162dd2c24bcbb50163ba74f40579a0206dd0602b93a9b65b.
- Source preparation verified 94 official .CHECKSUM sidecars and ZIP members; 135,360 one-minute bars cover the registered source.
- Read-only ledger preflight returned zero overlap for both access and query_intent over [2023-01-25T00:01:00Z, 2023-04-14T00:01:00Z).

Output directory: .artifacts/world_model/fresh-btcusdt-20230125-gaussian-no-rollout-risk-20261020/experiment.

## Append-only execution record

Completed the registered three-seed Validation experiment. Aggregate report SHA-256 `787f2ce05c2dfe38b937c854c7e49f7b9ed5bdf7bf3f5dbee7899d3ca726a75e`; seed report SHA-256 values: seed 31 `3dcb307ba9844ecf2b687686c959597ac2d471b74d2ec6caa6c64b1558ff9c19`, seed 102 `f437bf6cb6326aad911931e65fbda9bf0ec598d0eaaddae3f6d61aca59e7d245`, seed 20261018 `327d93b2f51b51cd51228a3c23997a063f6e231b71445396dc1c9e6816cf16a3`.

Pooled E1 artifact SHA-256 `a3d45dab1f37ba013cb0b596a2b49820779528367b6b43f27e116903a5847870`; pooled E2 artifact SHA-256 `6b7ee8609428ec1c20038dc5f875a468ac576064ef5e8441a45d23e1e7ba64c5`. Aggregate/predictive and all seed report digests, per-seed forecast/risk artifacts and pooled forecast hashes independently verify. All test-scope flags are false. Primary gate failed inconclusively; see [`R2_BTCUSDT_GAUSSIAN_NO_ROLLOUT_RISK_RESULTS_20261020.md`](R2_BTCUSDT_GAUSSIAN_NO_ROLLOUT_RISK_RESULTS_20261020.md).
