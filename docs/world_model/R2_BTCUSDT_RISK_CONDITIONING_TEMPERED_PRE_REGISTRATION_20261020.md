# R2 BTCUSDT geometrically tempered risk conditioning

Status: frozen on 2026-10-20 before downloading or decoding the new source
interval. This tests a new bounded conditioning strength after the previous
full-strength risk-scale replacement increased free-rollout volatility and
failed the registered path-derived 15-minute log-RV gate in all three seeds.
The prior registered Test rows remain unread.

## Hypothesis and fixed candidates

E2 uses the prior-native one-minute log-RV head to adjust free-rollout opening-
gap and close-body return scales by a geometric fraction of the full adjustment.
At strength `0.5`, the applied multiplier is `sqrt(full_multiplier)`, which
shrinks an over- or under-shoot toward the unconditioned OHLCV prior while
retaining state-conditional risk information. This is a single predeclared
candidate, not a strength sweep. It preserves RSSM architecture, Gaussian
forecast head, 1,000 update budget, optimizer, risk loss weight, seeds and
forecast schedule from the preceding registered experiment.

- E2: RSSM, Gaussian prior, native risk horizons `[1,5,15,30,60]`, risk loss
  scale `0.5`, rollout risk conditioning enabled with geometric strength `0.5`,
  1,000 optimizer updates, batch size 8, seeds `31,102,20261018`.
- E1: unchanged GRU baseline, same update budget.
- E0: frozen zero-drift fixed-volatility, Train-innovation EWMA and stationary
  block-bootstrap baselines; select the lowest Validation path-derived
  15-minute log-RV CRPS before paired comparison.
- 180 Validation anchors at 120-minute intervals; 32 forecast paths per seed;
  96 pooled paths per candidate; four anchors per worker chunk; chunk seed is
  `request_seed + chunk_start_index`; 2,000 paired moving-block replicates
  with 12-anchor (24-hour) blocks.

## Primary decision rule

For each seed, compare E2 versus the strongest Validation-selected E0 on
path-derived 15-minute `log(sqrt(sum(close_log_return**2)) + 1e-8)` CRPS.
Lower is better. Pass only if all three seed-level E2-minus-E0 95% confidence
intervals are strictly below zero and each Holm-adjusted p-value is below
0.05. Holm family comprises native return and native log-RV scores at five
horizons plus the path-derived 15-minute log-RV score (11 tests per seed).
Missing or invalid outputs fail closed. Test and Holdout payloads remain
unread; no trading-economic claim is made.

## Data scope and access control

- Binance USD-M BTCUSDT one-minute source interval:
  `[2022-07-21T00:00:00Z, 2022-10-23T00:00:00Z)`, 94 UTC days / 135,360 bars.
- Proposed Train/Validation reader interval:
  `[2022-07-21T00:01:00Z, 2022-10-07T00:01:00Z)`. It is the immediate next
  chronological interval after the previous frozen source window. The
  read-only preflight must find zero overlaps in both RouteB `access` and
  `query_intent` tables for BTCUSDT `perpetual-market-v1`; the one-shot reader
  is authoritative and must reject before payload access if any overlap exists.
- 60/15/15 Fold with one-hour purge and 23-hour embargo: Train warmup starts
  July 21 00:01; Train score `[2022-07-23T00:01:00Z, 2022-09-21T00:01:00Z)`;
  Validation warmup starts September 10 00:01; Validation score
  `[2022-09-22T00:01:00Z, 2022-10-07T00:01:00Z)`; Test warmup starts
  September 27 00:01; Test score `[2022-10-08T00:01:00Z,
  2022-10-23T00:01:00Z)`. Only Train and Validation may be read.
- New Fold label:
  `routeb-fold-btcusdt-binance-601515-20220721-risk-conditioned-tempered-v1`.

## Frozen implementation identity

- Independent RouteB checkout: `D:\Program Files\NautilusTrader_RL_ES_RouteB`.
- E2 candidate configuration:
  `configs/routeb/fresh_btcusdt_e2_risk05_conditioned_ensemble.json`.
- Geometric tempering is applied in
  `packages/world_model/src/rles_world_model/routeb/jax_model.py` through
  `RSSMConfig.rollout_risk_conditioning_strength`; default `1.0` preserves
  the old full adjustment when omitted. It is bounded in `[0,1]` and enters
  the checkpoint-bound optimizer recipe.
- Source manifest was refreshed after the parameter and regression test were
  added. Exact implementation, config, data, marker, Fold and report digests
  are appended below after source preparation and before the single registered
  Train/Validation read.

## Append-only execution record

Source preparation completed before any Train/Validation reader access: 94
official Binance archives passed their matching `.CHECKSUM` sidecars and ZIP
integrity checks (5,802,972 total bytes); the deterministic preparation
produced 135,360 bars. The immediately pre-registration read-only check again
found zero overlapping `access` and zero overlapping `query_intent` rows for
BTCUSDT `perpetual-market-v1` over `[2022-07-21T00:01:00Z,
2022-10-07T00:01:00Z)`. The registered one-shot reader remains authoritative.

- Dataset hash: `6b1d8088b5da6a68bd1c1db92f3c91e73198b24213e956de98fb5fb658e17917`.
- Marker SHA-256: `83b954a674d471a500f1ad7160330c8f530e318ee18303044d3f1fdf0ddd3d46`.
- Curated Parquet SHA-256: `ad9d6643c3083198337633a60ec8b9a135b1b9edd3403322ad7bddb8dabea655`.
- Fold config SHA-256: `8093535622f78a63445e5a6fa2d45880f9a3d9b8f37541460e8c677f2ea2c16c`;
  semantic Fold hash: `c67ce7a2a99750452a80e430d63e87ac2de6a0807f2c656d71d914e4c0678eca`.
- E2 config SHA-256:
  `6fbfe0a8411595a087af7dec9e7448095ee4aa282386af73a782362237aa1eb9`.
- Model source SHA-256:
  `c2a5dfe43d4eb31433e7574bd4d07acd5341cd8d33a2e5f002da46ae27e0f370`;
  model regression-test SHA-256:
  `e1a97c9d72cdb95fccc1a6f071e384dffb5d6033644d783ae797dfb57728ef0e`.
- Repository commit: `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`;
  frozen source-code identity SHA-256:
  `9924e2a78c28840dbc7697c1ad40245eb55612acb82bcc57929375be72750d44`.
- RouteB source-manifest identity SHA-256:
  `d9a0ce70685c59a6dd43c1192b262dbd4053c48d0a12ab890fb3539331b33154`
  (450 files).

Only Train and Validation may be opened by the registered one-shot runner.
The Test score interval `[2022-10-08T00:01:00Z, 2022-10-23T00:01:00Z)` is
still unread. Registration output directory:
`.artifacts/world_model/fresh-btcusdt-20220721-risk-conditioned-tempered-20261020/experiment`.
