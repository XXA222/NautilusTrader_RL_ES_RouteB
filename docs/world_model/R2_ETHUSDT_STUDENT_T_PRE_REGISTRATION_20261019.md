# R2 pre-registration: variance-matched Student-t prior forecast head

Date frozen: 2026-10-19, before downloading or reading the proposed market payload. This experiment follows the completed long-horizon weighting run and targets its observed under-dispersion and weak free-rollout volatility tails.

## Hypothesis and single model change

A diagonal Gaussian prior-to-observation head produces too few large OHLCV-coordinate innovations in F1 free rollout. Replace only E2's causal prior forecast likelihood and sampling distribution with a Student-t distribution at fixed `nu=5`. Parameterize the model output as location plus marginal standard deviation; convert standard deviation to Student-t scale using `sqrt((nu-2)/nu)` in the density and sampler so the learned scale retains the same variance interpretation. Keep the posterior auxiliary and reconstruction likelihoods Gaussian, and keep the multi-horizon risk head, its loss scale/weights, RSSM dimensions, E1, E0 baselines, optimizer recipe, features and F1 protocol unchanged.

The fixed hypothesis is that the heavier-tailed prior observation distribution improves 15-minute return CRPS against the strongest statistical baseline while reducing free-rollout volatility-tail error. The primary score alone controls promotion; secondary tail diagnostics cannot override it. This is one fixed distribution-family comparison, not a degrees-of-freedom or loss-weight sweep.

## Frozen scope and protocol

- Instrument/source: ETHUSDT USD-M perpetual, Binance 1-minute klines; source-open interval `[2022-10-01T00:00:00Z, 2023-01-02T00:00:00Z)`, 93 days / 133,920 expected bars. Use daily archives October 1 through January 1 inclusive.
- Before downloading, the RouteB registry was checked read-only. The proposed Train/Validation event-read interval `[2022-10-01T00:01:00Z, 2022-12-17T00:01:00Z)` has no overlapping ETH access record or query intent. The final one-shot preflight remains authoritative; if it rejects, do not read the payload.
- Approved 60/15/15 Fold: Train warmup October 1 00:01; Train score October 2 through December 1 00:01; Validation warmup November 21 00:01; Validation score December 2 through December 17 00:01; Test warmup December 7 00:01; Test score December 18 through January 2 00:01. Keep the established one-day gaps, one-hour purge, and 23-hour embargo. Test remains undecoded.
- E1/E2 share seeds `31, 102, 20261018`, 1,000 optimizer updates per seed, 32 forecast paths per seed, 180 two-hour Validation anchors, 96 pooled paths per candidate, and 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Primary measure: 15-minute return CRPS against the strongest statistical baseline selected on Validation. Promotion requires a paired 95% interval favoring E2 and Holm-adjusted p < 0.05 under the existing registered comparison family. Report comparison against E1 separately.
- Secondary: 30/60-minute native log-realized-volatility CRPS/Brier, free-rollout absolute-return lag-1 autocorrelation, rolling realized-volatility mean/p95, path validity, and scale calibration. No Test/Holdout decoding, economics claim, threshold search, or additional candidate.

## Frozen implementation delta

- Fold config: `configs/routeb/fresh_ethusdt_20221001_student_t_60_15_15_fold.json`.
- E1: `configs/routeb/fresh_ethusdt_e1_ensemble.json` unchanged.
- E2: `configs/routeb/fresh_ethusdt_e2_student_t_ensemble.json`; only new model setting is `forecast_distribution="student_t"` and fixed `forecast_student_t_df=5.0`. All other settings match the previous risk-weighted E2 configuration.
- Implement and verify the distribution loss, variance-matched F1 sampler, checkpoint identity, worker transport and metrics on synthetic inputs before fetching Binance archives. Do not change the frozen hypothesis after inspecting Validation.
- Source preparation must persist the official exchange archive checksums, source marker/hash, Parquet hash, Fold hash and config hashes here before the registered reader accesses Train/Validation. Registration is one-shot and Test is never decoded.

Any failed primary gate leaves R3/R4/R5/R6 closed. The scope is consumed after registration and cannot be reused for tuning.


## Frozen source and implementation identity

The checksum-verified source preparation completed after the hypothesis and exact interval above were frozen. The RouteB reader has not yet reserved or read Train/Validation.

- 93 daily archives (2022-10-01 through 2023-01-01) and the exchange-published `.CHECKSUM` sidecars passed SHA-256 validation; total archive bytes: 5,782,700.
- Curated source: `.artifacts/world_model/fresh-ethusdt-20221001-student-t-60-15-15-20261019/source/market.parquet`; 133,920 continuous bars.
- Dataset SHA-256: `fad492bb174357be17cab66970b8e8f0e815ba13d80a417e5ac77a8fa530ed5f`.
- Source marker SHA-256: `eea1f54f83bf8199daedff79171b011ed98a9430e5ecc95222f1aa25b1b197ab`.
- Parquet file SHA-256: `6651b2df555df887415117589dc83525ce016241f7760f8e1410d94f51fea21a`.
- Fold manifest hash: `2c9e1c745b0b89a12b742017e2f9dd668bff1eca5cf55d5d251c9cbd1da91743`.
- Fold config SHA-256: `589abf134667cbeeeae08a2ca544aaae409d54bc72ea61d7d6412c453e69c100`.
- E1 config SHA-256: `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`.
- E2 Student-t config SHA-256: `51a13b10f29681b3ebca3f40a59a9a81909d48d109d2013c85f45c0dfeefbf41`.
- Source preparation receipt SHA-256: `738eeed116d85a609ce650f81c6d38f7eceeec510b62de78feb74d7b7acf132c`.
- RouteB source manifest: 431 files, SHA-256 `fef20ce83f8a86ad79a1588dd7640c1f933a2b27ecbc535a8f457145830a38ab`; manifest verification passes.
- Before market download, the Student-t implementation passed 136 world-model tests (5 JAX-only host skips), Ruff, and the synthetic one-shot CPU/JAX train, checkpoint, batched F1 forecast, prior-risk export, and report path. Receipt: [`B2_student_t_cpu_smoke_20261019.json`](B2_student_t_cpu_smoke_20261019.json). This is mechanism evidence only.

The exact registered reader preflight remains the final authority. It must reject if the project ledger now has any overlapping access or query intent; if it does, stop without Train/Validation access.
