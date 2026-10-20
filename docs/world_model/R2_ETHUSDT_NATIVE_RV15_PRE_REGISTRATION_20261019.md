# R2 ETHUSDT native 15-minute realized-volatility confirmation

Status: preregistered before downloading this source interval or reading its
Train/Validation rows. This is a fixed-recipe confirmation of the RouteB v3 R2
multi-horizon conditional-risk objective. It does not authorize R3/R4/R5/R6.

## Frozen hypothesis and candidates

The E2 prior risk head will improve Validation CRPS for 15-minute
log-realized-volatility over the strongest Validation-selected E0 statistical
baseline. The exact target is `log(sqrt(sum(close_log_return**2)) + 1e-8)` over
the next 15 contiguous one-minute bars. Lower CRPS is better. The primary
forecast is the checkpoint-bound E2 native risk distribution, not a statistic
chosen after inspecting the new scope. This target is selected before access to
this scope because v3 prioritizes conditional volatility, and earlier frozen
R2 experiments showed repeatable 30/60-minute native log-RV gains while their
15-minute return primary endpoint did not promote.

- E2: unchanged Gaussian two-level RSSM with `risk_prediction_scale=0.5`,
  `risk_horizons=[1,5,15,30,60]`, 1,000 updates and 8-example batches from
  `configs/routeb/fresh_ethusdt_e2_risk05_ensemble.json` (SHA-256
  `e64938514b7ed10ae375b2dc42e100100772a9f406deb8fbd50b299606ca60e9`).
- E1: unchanged GRU baseline from
  `configs/routeb/fresh_ethusdt_e1_ensemble.json` (SHA-256
  `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`).
- E0: the existing frozen unconditional, volatility-conditioned, and
  stationary-block baselines; select the lowest 15-minute log-RV CRPS using
  Validation only. No candidate or scale sweep is permitted.
- Seeds: `31,102,20261018`; 32 generated paths per seed; 180 anchors sampled
  every 120 minutes; 96 pooled paths per candidate; 2,000 paired moving-block
  bootstrap replicates with 12-anchor (24-hour) blocks.

## Primary decision rule and multiplicity

For each seed, compare E2 native 15-minute log-RV CRPS against the strongest
E0 Validation baseline using paired 24-hour block bootstrap. The primary
prediction passes only if the E2-minus-E0 95% interval is strictly below zero
and the Holm-adjusted p-value is below 0.05 in all three seeds. Holm adjustment
is applied across the preregistered native-risk family (return and log-RV at
1/5/15/30/60 minutes); other endpoints cannot replace the primary result.
Failure or inconclusive evidence leaves R2 unpromoted and the downstream gates
closed.

Secondary, reported without overriding the primary rule: native 1/5/30/60m
log-RV and return scores; 15m path-derived return and realized-volatility
CRPS; barrier/extreme scores; free-rollout absolute-return lag-1
autocorrelation; rolling-RV mean and p95; invalid/rejected path rates; and
per-seed plus aggregate provenance digests. No Test or Holdout payload may be
decoded, and no trading-economic claim may be made.

## Frozen data and Fold scope

- Binance USD-M ETHUSDT one-minute daily archives, open-time interval
  `[2021-10-10T00:00:00Z, 2022-01-12T00:00:00Z)`: 94 daily files and 135,360
  expected one-minute bars. Every archive must match its official `.CHECKSUM`
  sidecar before preparation.
- Read-only RouteB ledger preflight on 2026-10-19 found zero overlaps in both
  `access` and `query_intent` for market `ETHUSDT`, family
  `perpetual-market-v1`, proposed read window
  `[2021-10-10T00:01:00Z, 2021-12-27T00:01:00Z)`. The one-shot registered reader
  remains authoritative; if it rejects, do not read Train/Validation rows.
- 60/15/15 Fold: Train warmup begins October 10 00:01; Train score
  `[2021-10-12T00:01:00Z, 2021-12-11T00:01:00Z)`; Validation warmup begins
  November 30 00:01; Validation score
  `[2021-12-12T00:01:00Z, 2021-12-27T00:01:00Z)`; Test warmup begins December
  17 00:01; Test score `[2021-12-28T00:01:00Z, 2022-01-12T00:01:00Z)`. Preserve
  the one-hour purge and 23-hour embargo. Only Train and Validation may be
  read; Test stays unread.
- Fold config:
  `configs/routeb/fresh_ethusdt_20211010_native_rv15_60_15_15_fold.json`
  (SHA-256 `953adec0bd90c05b7ea01c8dda106a550bebf0c3354f41b24265bd89b8189660`).
- Preparation receipt `.artifacts/world_model/fresh-ethusdt-20211010-native-rv15-20261019/source/curated/source_preparation.json`
  SHA-256 `c2f72c227893a18eb6a30b878aef1e876a72e15cf504d95eac0e557ae6c3e76c`;
  all 94 source archives matched official Binance `.CHECKSUM` sidecars.
- Prepared dataset hash `b3480ea2fd36c3854853f872d4acfd4679a9502a3cce89222b0cb6e290e2d180`,
  marker SHA-256 `bd494f9939f4189fe90d7c3a4c7131628002774f5c6af62e3a420b50fbed42d1`,
  and Parquet SHA-256
  `56e553d5d713c2c73c055332ed7aad6311e8e927960f089db3f5020056c2d743`.
- Before registering the Fold, RouteB's source manifest verified 445 files
  with identity SHA-256
  `44e24b7958263cf204192de0400846a76167f8bd951831173423c96f41fbefa1`.
- Read-only ledger preflight immediately before reservation for
  `[2021-10-10T00:01:00Z, 2021-12-27T00:01:00Z)` again found zero overlapping
  `access` records and zero overlapping `query_intent` records. The single
  registered reader remains the final authority.

## Frozen implementation identity

- RouteB checkout is the independent
  `D:\Program Files\NautilusTrader_RL_ES_RouteB` environment; no other Route
  or main-project checkout is a target.
- Runner `qa/world_model/run_registered_fold_ensemble.py` SHA-256:
  `aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9`.
- Native scorer `packages/world_model/src/rles_world_model/routeb/fold_evaluation.py`
  SHA-256:
  `128a769a100ae8be8a614c66dd2c326eab902792eb52c0bf549490594b96e985`.
- RouteB source-manifest identity before the registered read:
  `44e24b7958263cf204192de0400846a76167f8bd951831173423c96f41fbefa1`
  (445 files; manifest verification passed after writing the frozen Fold).
- Integrity correction before reservation: the first read-only runner preflight
  found that the new Fold had inherited the prior dataset's marker SHA. It
  rejected the registration before the reader, no Train/Validation rows were
  read, no output directory was created, and no scope was reserved. The marker
  was replaced with the verified new Parquet marker, then Fold/source hashes
  were recomputed and the source manifest reverified. This did not change the
  data, boundaries, models, target, or statistical procedure.
- This is one new chronological scope and one fixed candidate recipe. No
  retraining on this Validation, repeated seed search, threshold tuning, or
  subsequent scope search is allowed. Existing v3 gates determine the
  disposition after the result.
