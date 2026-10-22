# R2 BTCUSDT risk-conditioned rollout validation, revision 2

Status: preregistered before downloading this source interval or reading its
Train/Validation rows. This is a fixed-recipe rerun on a separate chronological
scope after the first BTC attempt stopped before scoring from GPU memory
exhaustion. It does not promote R2 or authorize downstream stages.

## Frozen hypothesis and candidates

The E2 prior-only rollout, with its OHLCV return scales conditioned on its own
one-minute native log-realized-volatility forecast, will improve Validation
CRPS for 15-minute path-derived log-realized-volatility over the strongest
Validation-selected E0 baseline. The path target is
`log(sqrt(sum(close_log_return**2)) + 1e-8)` over the next 15 contiguous
one-minute bars. Lower CRPS is better. The conditioning rule is fixed in
`RSSMConfig.rollout_risk_conditioning`: recover expected raw realized variance
from the predicted log-RV Gaussian, convert it to expected absolute-return
scale, and apply the resulting multiplier only to opening-gap/body return
coordinates using frozen Train scaler standard deviations. Wick and volume
scales are unchanged; no future observation enters the rollout.

- E2: Gaussian two-level RSSM, risk prediction scale 0.5, risk horizons
  `[1,5,15,30,60]`, risk-conditioned rollout enabled, 1,000 optimizer updates,
  batch size 8, seeds `31,102,20261018`, from
  `configs/routeb/fresh_btcusdt_e2_risk05_conditioned_ensemble.json` (SHA-256
  `cbda420000d23258d7e813885c6117dd6e849a66ad92ebd11ccd11b89e9ef086`).
- E1: unchanged GRU baseline from
  `configs/routeb/fresh_ethusdt_e1_ensemble.json`; same 1,000-update budget.
- E0: frozen unconditional, volatility-conditioned, and stationary-block
  baselines; select the lowest 15-minute path log-RV CRPS using Validation
  only. No candidate or scale sweep is permitted.
- Forecast protocol: 32 final paths per seed, four sampled candidates per
  retained path, 180 anchors every 120 minutes, 96 pooled paths per candidate,
  2,000 paired moving-block bootstrap replicates, and 12-anchor (24-hour)
  blocks. To cap worker memory, process four ordered anchors per chunk; for
  chunk starting at anchor index `i`, use forecast seed
  `request_seed + i`. E1 and E2 share this schedule. This deterministic
  execution rule is frozen before reading this scope.

## Primary decision rule and multiplicity

For each seed, compare E2 15-minute path-derived log-RV CRPS against the
strongest E0 Validation baseline using paired 24-hour block bootstrap. The
primary prediction passes only if the E2-minus-E0 95% interval is strictly
below zero and Holm-adjusted p-value is below 0.05 in all three seeds. Holm
adjustment covers the preregistered native-risk return/log-RV family at
1/5/15/30/60 minutes and 15-minute path-derived log-RV. Secondary outcomes
include 15-minute path return CRPS, 1/5/30/60-minute path RV,
barrier/extreme scores, free-rollout absolute-return lag-1 autocorrelation,
rolling-RV mean and p95, invalid/rejected path rates, and provenance digests.
Secondary outcomes cannot replace a failed primary gate. Test and Holdout
payloads remain unread; no trading-economic claim is made.

## Frozen data and Fold scope

- Binance USD-M BTCUSDT one-minute archives, open-time interval
  `[2022-01-13T00:00:00Z, 2022-04-17T00:00:00Z)`, 94 daily files and 135,360
  expected one-minute bars. Verify each daily archive against its official
  `.CHECKSUM` sidecar before preparation.
- Read-only RouteB ledger preflight on 2026-10-19 found zero overlapping
  `access` and `query_intent` records for BTCUSDT `perpetual-market-v1` and
  read window `[2022-01-13T00:01:00Z, 2022-04-01T00:01:00Z)`. The registered
  one-shot reader remains authoritative; if it rejects, do not read
  Train/Validation rows.
- 60/15/15 Fold with a one-hour purge and 23-hour embargo: Train warmup starts
  January 13 00:01; Train score `[2022-01-15T00:01:00Z,
  2022-03-16T00:01:00Z)`; Validation warmup starts March 5 00:01; Validation
  score `[2022-03-17T00:01:00Z, 2022-04-01T00:01:00Z)`; Test warmup starts
  March 22 00:01; Test score `[2022-04-02T00:01:00Z,
  2022-04-17T00:01:00Z)`. Only Train and Validation may be read. Test stays
  unread.
- New Fold label:
  `routeb-fold-btcusdt-binance-601515-20220113-risk-conditioned-v2`.
- Fold config SHA-256:
  `2bb6a1fe8f9806ba278f9d70cf1995e9e078c7b8074bc30f34e215557fe38248`.
- Preparation receipt SHA-256:
  `78959f4260e4d1f02c36c55f41b7fff0d3568bb48feee4aefa842a15f7ce89f7`;
  all 94 Binance daily archive `.CHECKSUM` sidecars passed.
- Dataset hash
  `e0fe9b0976f0aa31318d09e7ff2e12164ba52f158548b03396c20584a0d23449`;
  marker SHA-256
  `099592fab7459add7b078666f85ab5df502db58721295911851a0de32e28476c`;
  Parquet SHA-256
  `7be2c5b626ed8311d0f6e94f71fdd8760d9af3d92bbc741b0de9377e0fa5178c`.
- RouteB source manifest verified 448 files, identity SHA-256
  `6d032dd1f4f75c8144bf7528cb892ac271df04a9e72eb53bc9f3bbceefa12869`.
- Candidate E2 config SHA-256:
  `cbda420000d23258d7e813885c6117dd6e849a66ad92ebd11ccd11b89e9ef086`.
- Runner SHA-256
  `aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9`;
  worker CLI SHA-256
  `aa3c04ec68e785c69e8d795fd4039cc6f5221c6e7674d86d332bce8e5eeaa607`;
  fold experiment SHA-256
  `8c158057edf4c59c565362023f3a0ab072b3579431213557bcc098fb6df39657`;
  worker client SHA-256
  `b3ba1902a26fb1eae97b038d763138cd50811f4abfaf2a2e030707c8b1c9d129`;
  model SHA-256
  `cae4ea362f50c186b188e649e51656a9dad31a72caa786f67acbe76e25e67482`;
  train API SHA-256
  `cfbf8f412a47274f2b5ead4a5e3c02903e842dcd0536cc18c3ef31cd0e0281b0`;
  native scorer SHA-256
  `128a769a100ae8be8a614c66dd2c326eab902792eb52c0bf549490594b96e985`.
- Immediate read-only ledger preflight on 2026-10-19 found zero overlapping
  BTCUSDT `access` rows and zero overlapping `query_intent` rows for
  `[2022-01-13T00:01:00Z, 2022-04-01T00:01:00Z)`.

## Frozen implementation identity

- Independent project: `D:\Program Files\NautilusTrader_RL_ES_RouteB`.
- The four-anchor chunking change is only an execution/memory bound. Each
  chunk preserves anchor order and output geometry; it changes no trained
  weights, target, scoring rule, or retained forecast count.
- A real locked-worker CPU B2 smoke completed an E2 forecast across six
  synthetic anchors split 4+2; it verified output shape `[6,2,3,5]`, finite
  OHLCV paths, and the chunk seed metadata.
- The failed 2021 scope remains marked as consumed and unscored. No checkpoint
  from that scope is reused. This revision has one new BTCUSDT interval and one
  fixed candidate recipe; no retraining on Validation, seed search, threshold
  tuning, or later scope search is permitted.
