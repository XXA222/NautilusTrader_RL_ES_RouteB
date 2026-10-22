# R2 BTCUSDT risk-conditioned rollout validation

Status: preregistered before downloading this source interval or reading its
Train/Validation rows. This is a fixed-recipe mechanism validation following
the ETHUSDT native-risk experiment. It does not promote R2 or authorize
downstream stages by itself. The run was stopped before scoring after the
worker exhausted GPU memory while creating batched forecasts; see the
operational failure record below. Its checkpoint is retained for audit and is
not reused on a different market interval.

## Frozen hypothesis and candidates

The E2 prior-only rollout, with OHLCV return scales conditioned on its own
one-minute native log-realized-volatility forecast, will improve Validation
CRPS for 15-minute path-derived log-realized-volatility over the strongest
Validation-selected E0 baseline. The path target is
`log(sqrt(sum(close_log_return**2)) + 1e-8)` over the next 15 contiguous
one-minute bars. Lower CRPS is better. The conditioning rule is fixed in
`RSSMConfig.rollout_risk_conditioning`: use the predicted log-RV Gaussian to
recover expected raw realized variance, convert to expected absolute-return
scale, and apply the resulting multiplier only to opening-gap/body return
coordinates using frozen Train scaler standard deviations. Wick and volume
scales are unchanged. No future observation enters the rollout.

- E2: Gaussian two-level RSSM, risk prediction scale 0.5, risk horizons
  `[1,5,15,30,60]`, risk-conditioned rollout enabled, 1,000 optimizer updates,
  batch size 8, seeds `31,102,20261018`, as frozen in
  `configs/routeb/fresh_btcusdt_e2_risk05_conditioned_ensemble.json` (SHA-256
  `cbda420000d23258d7e813885c6117dd6e849a66ad92ebd11ccd11b89e9ef086`).
- E1: unchanged GRU baseline from
  `configs/routeb/fresh_ethusdt_e1_ensemble.json`; all seeds share the same
  1,000-update budget.
- E0: existing frozen unconditional, volatility-conditioned, and
  stationary-block baselines; select the lowest 15-minute path log-RV CRPS
  using Validation only. No candidate or scale sweep is permitted.
- Forecasts: 32 paths per seed, 180 anchors sampled every 120 minutes, 96
  pooled paths per candidate, 2,000 paired moving-block bootstrap replicates,
  12-anchor (24-hour) blocks.

## Primary decision rule and multiplicity

For each seed, compare E2 15-minute path-derived log-RV CRPS against the
strongest E0 Validation baseline using paired 24-hour block bootstrap. The
primary prediction passes only if the E2-minus-E0 95% interval is strictly
below zero and Holm-adjusted p-value is below 0.05 in all three seeds. Holm
adjustment covers the preregistered native-risk return/log-RV family at
1/5/15/30/60 minutes and 15-minute path-derived log-RV. Secondary outcomes
include 15-minute path return CRPS, 1/5/30/60-minute path RV, barrier/extreme
scores, free-rollout absolute-return lag-1 autocorrelation, rolling-RV mean
and p95, invalid/rejected path rates, and provenance digests. Secondary
outcomes cannot replace a failed primary gate. No Test or Holdout payload may
be decoded, and no trading-economic claim may be made.

## Frozen data and Fold scope

- Binance USD-M BTCUSDT one-minute archives, open-time interval
  `[2021-10-10T00:00:00Z, 2022-01-12T00:00:00Z)`, 94 daily files and 135,360
  expected one-minute bars. Every archive must match its official `.CHECKSUM`
  sidecar before preparation.
- Read-only RouteB ledger preflight on 2026-10-19 found zero overlaps in both
  `access` and `query_intent` for market `BTCUSDT`, family
  `perpetual-market-v1`, proposed Fold read window
  `[2021-10-10T00:01:00Z, 2021-12-27T00:01:00Z)`. The one-shot registered
  reader remains authoritative; if it rejects, do not read Train/Validation
  rows.
- 60/15/15 Fold with one-hour purge and 23-hour embargo: Train warmup starts
  October 10 00:01; Train score `[2021-10-12T00:01:00Z,
  2021-12-11T00:01:00Z)`; Validation warmup starts November 30 00:01;
  Validation score `[2021-12-12T00:01:00Z, 2021-12-27T00:01:00Z)`; Test warmup
  starts December 17 00:01; Test score `[2021-12-28T00:01:00Z,
  2022-01-12T00:01:00Z)`. Only Train and Validation may be read. Test stays
  unread.
- The Fold config, source preparation receipt, dataset/marker/Parquet hashes,
  RouteB source-manifest identity, implementation hashes, and the immediate
  pre-reservation ledger preflight will be appended here before the single
  registered reader is invoked.

Frozen artifact identities recorded before registration:

- Fold config SHA-256:
  `1210d8948f90c9bed191c79838f7d42bad2c4a7a896ecb74de8a9f5f438ba075`.
- Preparation receipt SHA-256:
  `fb6544b67626682036d944cf82e9a32a03626aa83b187c470d58f93422899060`;
  94 official Binance daily archive checksum sidecars passed.
- Dataset hash `fcdc10cfd70808303dd11557b69326c60217d4456acf5a77a42fadab0152bab0`;
  marker SHA-256
  `eb6fc649b7e99459db257a26f11287895c72f4441ef7e37d27e289d5414d4dee`;
  Parquet SHA-256
  `3494e38410377fa8fca9d931f613c406fbb5af4ca1b7345005a1847a383c3b6a`.
- Source manifest verified 447 files, identity SHA-256
  `a41f37dc668bb0cf92135bb12a5e6855682a05e3dbbbc4721122b5e8e5c0f19c`.
- Candidate E2 config SHA-256:
  `cbda420000d23258d7e813885c6117dd6e849a66ad92ebd11ccd11b89e9ef086`.
- Runner SHA-256:
  `aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9`;
  model SHA-256
  `cae4ea362f50c186b188e649e51656a9dad31a72caa786f67acbe76e25e67482`;
  train/forecast API SHA-256
  `cfbf8f412a47274f2b5ead4a5e3c02903e842dcd0536cc18c3ef31cd0e0281b0`;
  worker CLI SHA-256
  `e28e48d46dc07ef16bc7ef626341fa30a6dcaab7822cbf4129af21e3c9aa2ebe`;
  native scorer SHA-256
  `128a769a100ae8be8a614c66dd2c326eab902792eb52c0bf549490594b96e985`.
- Immediate pre-reservation read-only preflight on 2026-10-19 found zero
  overlapping BTCUSDT `access` rows and zero overlapping `query_intent` rows
  for `[2021-10-10T00:01:00Z, 2021-12-27T00:01:00Z)`.

## Frozen implementation identity

- Independent project: `D:\Program Files\NautilusTrader_RL_ES_RouteB`.
- Runner: `qa/world_model/run_registered_fold_ensemble.py`.
- Risk-conditioned rollout implementation:
  `packages/world_model/src/rles_world_model/routeb/jax_model.py` and
  `packages/world_model/src/rles_world_model/routeb/jax_train.py`.
- Any integrity rejection before the reader runs must be documented; no model
  output or data rows may be inspected before the frozen identity is restored
  and reverified.
- This is one new BTCUSDT historical scope and one fixed candidate recipe. No
  retraining on Validation, repeated seed search, threshold tuning, or
  subsequent scope search is permitted. Existing v3 gates determine
  disposition after the result.

## Operational failure record

The registered Train/Validation read and seed 31 E1/E2 training completed.
The run stopped during its first batched
forecast request with JAX `RESOURCE_EXHAUSTED`: the GPU could not allocate an
additional 1.41 GiB. No forecast artifact, predictive score, or aggregate
result was emitted, and Test remained unread. This attempt is not scored and
does not count as evidence for or against the hypothesis.

The correction bounds worker forecast batches to four ordered anchors and
derives each chunk seed as `request_seed + chunk_start_index`; E1 and E2 share
that schedule. This avoids the original 180-anchor GPU allocation while
keeping deterministic paired forecasts. The correction changes no model
weights, feature/target definitions, training budget, or scoring rule. Because
the first scope was already consumed, the amended protocol runs only on a
separate, later, zero-overlap BTCUSDT interval. The failed scope's checkpoints
are not resumed or used for Validation scoring.
