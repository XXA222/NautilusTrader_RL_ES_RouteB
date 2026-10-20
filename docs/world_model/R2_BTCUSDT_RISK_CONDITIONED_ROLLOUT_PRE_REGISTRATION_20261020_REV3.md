# R2 BTCUSDT risk-conditioned rollout validation, revision 3

Status: frozen before downloading the new interval or reading its Train or
Validation rows. This revision adds the missing registered paired score for
path-derived log realized volatility. The preceding 2022-01-13 BTC run
completed its forecasts but its report only had descriptive path-RV scores;
it did not emit the preregistered per-seed 15-minute log-RV paired bootstrap
or include that test in the Holm family. That run is therefore diagnostic and
cannot pass or fail the primary rule. Its data, predictions, and report remain
unchanged. This revision uses a separate later chronological scope and the
same frozen model recipe.

## Frozen hypothesis and candidates

The E2 prior-only rollout, with OHLCV return scales conditioned on its own
one-minute native log-realized-volatility forecast, will improve Validation
CRPS for 15-minute path-derived log-realized-volatility over the strongest
Validation-selected E0 baseline. The target is
`log(sqrt(sum(close_log_return**2)) + 1e-8)` over the next 15 contiguous
one-minute bars. Lower CRPS is better. Per-anchor scores are produced for E0,
E1, and E2 in that target space; the E0 with the lowest Validation mean CRPS
is selected before the paired comparison. A 24-hour moving-block bootstrap is
then run independently for each seed. No cross-seed pooling can substitute for
the per-seed gate.

- E2: unchanged Gaussian two-level RSSM, risk prediction scale 0.5, risk
  horizons `[1,5,15,30,60]`, risk-conditioned rollout enabled, 1,000 optimizer
  updates, batch size 8, seeds `31,102,20261018`, from
  `configs/routeb/fresh_btcusdt_e2_risk05_conditioned_ensemble.json` (SHA-256
  `cbda420000d23258d7e813885c6117dd6e849a66ad92ebd11ccd11b89e9ef086`).
- E1: unchanged GRU baseline from
  `configs/routeb/fresh_ethusdt_e1_ensemble.json`, with the same optimizer
  budget.
- E0: frozen zero-drift fixed-volatility, Train-innovation EWMA, and
  stationary-block-bootstrap baselines. Select the lowest 15-minute
  path-derived log-RV CRPS from Validation only; do not sweep candidates or
  scales.
- Forecasts: 32 final paths per seed, four sampled candidates per retained
  path, 180 anchors every 120 minutes, 96 pooled paths per candidate, 2,000
  paired moving-block bootstrap replicates, 12-anchor (24-hour) blocks.
- Worker execution: four ordered anchors per chunk; chunk seed is
  `request_seed + chunk_start_index`, shared between E1 and E2. This fixed
  resource bound preserves anchor order, candidate count, and deterministic
  path generation.

## Primary decision rule and multiplicity

For each seed, compare E2 15-minute path-derived log-RV CRPS against the
strongest E0 Validation baseline. The primary prediction passes only if the
E2-minus-E0 95% interval is strictly below zero and Holm-adjusted p-value is
below 0.05 in all three seeds. Holm adjustment is across eleven comparisons
per seed: native return and native log-RV CRPS at 1/5/15/30/60 minutes, plus
the path-derived 15-minute log-RV CRPS. The per-seed report must include the
selected baseline, per-anchor score hashes, raw p-value, confidence interval,
and adjusted p-value. Missing or invalid per-seed output fails closed.

Secondary results, which cannot replace the primary gate, include native
and path return CRPS, other path RV horizons, barrier/extreme scores,
free-rollout absolute-return lag-1 autocorrelation, rolling-RV mean and p95,
invalid/rejected path rates, and provenance hashes. Test and Holdout payloads
remain unread. No trading-economic claim is allowed.

## Frozen data and Fold scope

- Binance USD-M BTCUSDT one-minute archives, open-time interval
  `[2022-04-18T00:00:00Z, 2022-07-21T00:00:00Z)`: 94 daily files and 135,360
  expected bars. Each daily archive must match its official `.CHECKSUM` before
  preparation.
- Read-only RouteB ledger preflight on 2026-10-20 found zero overlapping
  `access` and `query_intent` rows for BTCUSDT `perpetual-market-v1` and the
  proposed reader window `[2022-04-18T00:01:00Z, 2022-07-05T00:01:00Z)`. The
  registered one-shot reader remains authoritative; if it rejects, stop
  before reading Train or Validation.
- 60/15/15 Fold with one-hour purge and 23-hour embargo: Train warmup April 18
  00:01; Train score `[2022-04-20T00:01:00Z, 2022-06-19T00:01:00Z)`;
  Validation warmup June 8 00:01; Validation score `[2022-06-20T00:01:00Z,
  2022-07-05T00:01:00Z)`; Test warmup June 25 00:01; Test score
  `[2022-07-06T00:01:00Z, 2022-07-21T00:01:00Z)`. Only Train and Validation
  may be read. Test stays unread.
- New Fold label:
  `routeb-fold-btcusdt-binance-601515-20220418-risk-conditioned-v3`.
- Fold config SHA-256:
  `f54450019810c187b4a394b5224d70395c2288622732f431b09f8a31befd4378`.
- Preparation receipt SHA-256:
  `902f6681c47faa9448797cff97f1293ca25d93c5bee226db56737dc22ff479f9`;
  all 94 official Binance archive checksum sidecars passed.
- Dataset hash
  `8b09e1844b705f06015c0851e43933c0652d1ad15f30fd4c104cffb4121bb0c7`;
  marker SHA-256
  `b07f7cef4525d0fce19985433a72b505c54042f6f232bb810073402129df7900`;
  Parquet SHA-256
  `d7220bbe0c843a396e20ae69222cbaf8c09cba1db30bdb1c649a272064f89924`.
- RouteB source manifest verified 449 files, identity SHA-256
  `57add5cd21282a5128bcc85f28ec13f0e03b008327946d12b114b907f47705f4`.
- Candidate E2 config SHA-256:
  `cbda420000d23258d7e813885c6117dd6e849a66ad92ebd11ccd11b89e9ef086`.
- Runner SHA-256
  `aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9`;
  predictive evaluator SHA-256
  `e54d4d8621e37aed1ef829c19df2e286e3a447f19e58ca261ae80ac63ffaedd7`;
  fold evaluator SHA-256
  `a057abaf93bcc60ff53157a18bb49c6ccd8c2e962006ccee4af60ab73bd83971`;
  risk target/scorer SHA-256
  `3bd61a9e7b1b6b68708cce0999eb8e567969050ed2f67bc4764a0e88b6d5ffc4`;
  worker CLI SHA-256
  `aa3c04ec68e785c69e8d795fd4039cc6f5221c6e7674d86d332bce8e5eeaa607`;
  fold runner SHA-256
  `8c158057edf4c59c565362023f3a0ab072b3579431213557bcc098fb6df39657`;
  worker client SHA-256
  `b3ba1902a26fb1eae97b038d763138cd50811f4abfaf2a2e030707c8b1c9d129`;
  model SHA-256
  `cae4ea362f50c186b188e649e51656a9dad31a72caa786f67acbe76e25e67482`;
  train API SHA-256
  `cfbf8f412a47274f2b5ead4a5e3c02903e842dcd0536cc18c3ef31cd0e0281b0`.
- Immediate pre-reservation ledger preflight on 2026-10-20 again found zero
  BTCUSDT `access` and zero `query_intent` overlaps for
  `[2022-04-18T00:01:00Z, 2022-07-05T00:01:00Z)`.

## Frozen implementation identity

- Independent project: `D:\Program Files\NautilusTrader_RL_ES_RouteB`.
- New registered path-log-RV scorer:
  `packages/world_model/src/rles_world_model/routeb/risk_targets.py` and
  `packages/world_model/src/rles_world_model/routeb/evaluation.py`.
- Fold-level Holm integration:
  `packages/world_model/src/rles_world_model/routeb/fold_evaluation.py`.
- The January BTC source scope was already consumed and is not reused. This
  scope was selected as the immediate next 94-day UTC interval, without
  looking at any rows from it. No changes to model, scales, training budget,
  anchor schedule, or primary decision rule are permitted after registration.
