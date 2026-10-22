# R2 BTCUSDT Student-t prior with tempered native-risk rollout

Status: frozen on 2026-10-20 before downloading or decoding this source
interval. The two preceding full/tempered Gaussian risk-conditioned runs
showed that direct scale replacement can over-expand volatility and geometric
strength `0.5` corrected the mean but left the upper tail narrow. This single
candidate adds a heavier-tailed OHLCV prior while applying the native risk
head through the correct Student-t marginal absolute-return/standard-deviation
ratio. The previous Test intervals remain unread.

## Frozen hypothesis and candidates

The Student-t E2 prior with degrees of freedom `5.0`, combined with
geometrically tempered risk conditioning at strength `0.5`, will improve
Validation CRPS for 15-minute path-derived log realized volatility over the
strongest Validation-selected E0 baseline. Lower CRPS is better. Per-anchor
scores are computed for E0, E1 and E2, and the E0 with the lowest Validation
mean score is selected before pairing. No distribution, df, or strength sweep
is allowed.

- E2: unchanged two-level RSSM, Student-t marginal prior with df `5.0`, native
  risk horizons `[1,5,15,30,60]`, risk prediction loss scale `0.5`, rollout
  risk conditioning enabled, geometric strength `0.5`, 1,000 optimizer
  updates, batch size 8, seeds `31,102,20261018`.
- E1: unchanged GRU baseline and optimizer budget.
- E0: frozen zero-drift fixed-volatility, Train-innovation EWMA and stationary
  block-bootstrap baselines; select the lowest path-derived 15-minute log-RV
  mean CRPS from Validation only.
- Forecast: 32 retained paths per seed, 180 anchors at 120-minute intervals,
  96 pooled paths per candidate, 2,000 paired moving-block bootstrap
  replicates, 12-anchor (24-hour) blocks.
- Forecast worker memory bound: four ordered anchors per chunk; chunk seed is
  `request_seed + chunk_start_index`.

## Student-t scale mapping frozen in this candidate

The model's prior scale for Student-t is its marginal standard deviation. For
df `nu>2`, the variance-one Student-t absolute-return ratio is
`sqrt(nu-2) * Gamma((nu-1)/2) / (sqrt(pi) * Gamma(nu/2))`. The Gaussian log-RV
head's log-normal expectation of raw one-minute absolute return is divided by
that ratio to obtain the Student-t marginal standard deviation. The rollout
adjustment then applies the geometric factor
`exp(0.5 * log(target_std / prior_std))` to the gap/body channels. The frozen
Train feature scaler converts the adjusted channels to raw return units.
Mixture-Gaussian risk conditioning remains unsupported because it has no
single variance-preserving scale parameter.

## Primary rule and multiple testing

For each seed compare E2 versus the strongest Validation-selected E0 on
path-derived 15-minute
`log(sqrt(sum(close_log_return**2)) + 1e-8)` CRPS. The primary gate passes
only if all three E2-minus-E0 paired 95% intervals are strictly below zero
and every per-seed Holm-adjusted p-value is below `0.05`. The Holm family is
11 tests per seed: native return and native log-RV CRPS at horizons
`[1,5,15,30,60]`, plus path-derived 15-minute log-RV CRPS. Missing or invalid
outputs fail closed. Test and Holdout stay unread and no economic claim is
made.

## Data and access scope

- Binance USD-M BTCUSDT one-minute source interval:
  `[2022-10-23T00:00:00Z, 2023-01-25T00:00:00Z)`, 94 UTC days and 135,360
  expected bars.
- Proposed Train/Validation reader range:
  `[2022-10-23T00:01:00Z, 2023-01-09T00:01:00Z)`. Before downloading, the
  RouteB `access` and `query_intent` ledgers must show zero overlaps for
  BTCUSDT `perpetual-market-v1`; repeat the read-only check immediately before
  the authoritative one-shot reservation. If the registered runner rejects,
  do not decode Train or Validation.
- 60/15/15 Fold with one-hour purge and 23-hour embargo: Train warmup starts
  October 23 00:01; Train score `[2022-10-25T00:01:00Z,
  2022-12-24T00:01:00Z)`; Validation warmup starts December 13 00:01;
  Validation score `[2022-12-25T00:01:00Z, 2023-01-09T00:01:00Z)`; Test
  warmup starts December 30 00:01; Test score
  `[2023-01-10T00:01:00Z, 2023-01-25T00:01:00Z)`. Test is never decoded.
- Fold label:
  `routeb-fold-btcusdt-binance-601515-20221023-student-t-tempered-v1`.

## Frozen implementation

- Independent project:
  `D:\Program Files\NautilusTrader_RL_ES_RouteB`.
- E2 config:
  `configs/routeb/fresh_btcusdt_e2_risk05_conditioned_ensemble.json`.
- Model implementation: `packages/world_model/src/rles_world_model/routeb/jax_model.py`.
  Student-t conditioning is limited to finite `df>2`; it corrects the
  marginal-scale map for the distribution and uses strength `0.5`.
- Full host world-model suite and locked JAX CPU numerical smoke must pass
  before registration. Code, config, source, Fold and report hashes are
  appended below after source preparation and before the registered read.

## Append-only execution record

Source preparation completed before Train/Validation access. All 94 official
Binance archives passed their official `.CHECKSUM` and ZIP integrity checks
(5,578,654 bytes total); deterministic preparation produced 135,360 bars.
The immediate pre-registration read-only RouteB ledger preflight found zero
overlapping `access` and zero `query_intent` rows for BTCUSDT
`perpetual-market-v1` over `[2022-10-23T00:01:00Z, 2023-01-09T00:01:00Z)`.
The registered one-shot reader remains authoritative.

- Dataset hash:
  `fdce961ae187744e035f00d8e62c3464237c422e0f6ae589fce2efef0b391690`.
- Marker SHA-256:
  `0df169b268b41dc78b0dfa1d0373ea88708bd223bf5da10073c998da2d083cd0`.
- Curated Parquet SHA-256:
  `ba28c9d15e6585ff8f83420288d70e0305a4539f9d998f60b5ce500b262e0e06`.
- Fold config SHA-256:
  `bf46663917c8aae0b1a52909e0d85487bd408e8db8b4a8d1cccbd0d293d449f6`;
  semantic Fold hash:
  `ac56f635a524119d569095c885c55fddc2bded435b1d6ddd5f6f1e61937f6e7d`.
- E2 config SHA-256:
  `10be03b3d308a4bad7141be1cc62697a6fa20cf88c83a2bf029d516c7fc1335e`.
- Model implementation SHA-256:
  `0e463779338a7ff2238f7ba4dd4a41de72a8e032b09ae282c1502c501378fe05`;
  regression-test SHA-256:
  `9ea5d095e28d7423ef6a4254996b3139db60345a5e5a47b08e86b4a3a813fd11`.
- Repository commit: `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`;
  source-code identity SHA-256:
  `4b1327326acd79bdfaf50da581bfb11367c6871c13818292571ba3772ed0ee8e`.
- RouteB source-manifest identity SHA-256:
  `974fe685225eb8629aed22a7b16487c3510f617d396c399a8eccaec8002f9846`
  (451 files).

Output directory:
`.artifacts/world_model/fresh-btcusdt-20221023-student-t-tempered-20261020/experiment`.
Only Train and Validation will be read. Test score interval
`[2023-01-10T00:01:00Z, 2023-01-25T00:01:00Z)` remains unread.
