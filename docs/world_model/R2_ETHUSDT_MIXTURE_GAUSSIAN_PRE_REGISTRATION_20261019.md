# R2 pre-registration: fixed two-component Gaussian-mixture prior forecast head

Date frozen: 2026-10-19, before downloading or reading the proposed Binance market payload. The experiment follows the completed variance-matched Student-t and long-horizon risk-head experiments. Student-t reduced some native-risk errors but failed the primary predictive promotion gate, so this experiment changes one registered dimension: the E2 causal prior observation head.

## Hypothesis and single model change

A single Gaussian prior forecast head may average distinct next-bar regimes into a weak conditional forecast. Replace only E2's causal prior OHLCV forecast head with a fixed two-component diagonal Gaussian mixture. A single categorical component is shared across the five OHLCV feature coordinates for each predicted bar. Train with exact joint mixture negative log likelihood; in F1, sample a component from the predicted logits and then sample the five coordinates from that component. Export normalized component weights, component means/scales, and the correctly computed marginal mean and standard deviation. Keep reconstruction and posterior auxiliary likelihoods Gaussian. Keep the risk head and its normalized 1/5/15/30/60-minute weighting, model dimensions, E1, E0 baselines, feature pipeline, optimizer budget, and F1 path protocol unchanged.

The fixed hypothesis is that this prior distribution improves 15-minute return CRPS against the strongest statistical baseline selected on Validation, and improves calibrated free-rollout dispersion. The primary score controls promotion; secondary diagnostics cannot override it. This is one fixed two-component configuration, with no component-count, temperature, loss-weight, or seed sweep.

## Frozen scope and protocol

- Instrument/source: ETHUSDT USD-M perpetual, Binance 1-minute klines; source-open interval `[2023-01-02T00:00:00Z, 2023-04-05T00:00:00Z)`, 93 days / 133,920 expected bars. Download daily archives January 2 through April 4 inclusive.
- Before freezing this registration, the RouteB project registry was read in read-only mode. Proposed Train/Validation read interval `[2023-01-02T00:01:00Z, 2023-03-20T00:01:00Z)` had zero overlapping ETHUSDT `perpetual-market-v1` access records and zero overlapping query intents. The final one-shot preflight remains authoritative; if it rejects, do not read Train/Validation.
- Approved 60/15/15 Fold: Train warmup January 2 00:01; Train score January 3 through March 4 00:01; Validation warmup February 22 00:01; Validation score March 5 through March 20 00:01; Test warmup March 10 00:01; Test score March 21 through April 5 00:01. Preserve the established one-hour purge and 23-hour embargo. Test remains undecoded.
- E1/E2 share seeds `31, 102, 20261018`, 1,000 optimizer updates per seed, 32 forecast paths per seed, 180 two-hour Validation anchors, 96 pooled paths per candidate, and 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Primary measure: 15-minute return CRPS against the strongest statistical baseline selected on Validation. Promotion requires the paired 95% interval to favor E2 and Holm-adjusted p < 0.05 under the registered comparison family. Report E2 versus E1 separately.
- Secondary: 30/60-minute native log-realized-volatility CRPS/Brier, free-rollout absolute-return lag-1 autocorrelation, rolling realized-volatility mean/p95, path validity, and scale calibration. No Test/Holdout decoding, economics claim, threshold search, or additional candidate.

## Frozen implementation delta

- E1: `configs/routeb/fresh_ethusdt_e1_ensemble.json` unchanged.
- E2: `configs/routeb/fresh_ethusdt_e2_mixture_gaussian_ensemble.json`; preserve the risk-weighted E2 configuration and change only `forecast_distribution` to `mixture_gaussian`. The two-component prior head is fixed in code; other outputs/losses remain unchanged.
- Before any market download, the mixture implementation passed focused tests, Ruff, and isolated WSL/JAX CPU synthetic end-to-end validation covering training, checkpoint reload, F1 path sampling, next-distribution component export, batched forecasts, risk artifact export, and one-shot E0/E1/E2 fold orchestration. Receipt: [`B2_mixture_gaussian_cpu_smoke_20261019.json`](B2_mixture_gaussian_cpu_smoke_20261019.json). This is mechanism evidence only, not market-performance evidence.
- RouteB code and configuration identities will be refreshed and appended below before the registered reader is allowed to access Train/Validation. Source archive checksums, curated source marker/hash, Parquet hash, and Fold hash will also be recorded before that reader runs.
- The scope is consumed once the registered reader reserves it. A failed primary gate leaves R3/R4/R5/R6 closed. Do not decode Test or reuse this scope for tuning.

## Frozen implementation identity (before payload download)

- RouteB repository HEAD: `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`.
- RouteB code identity SHA-256 (the exact identity supplied to the registered-reader reservation): `e1e08a244a8e95e3c3dfd26c1c1e15765a0ec0d6114cc322a257afd63df656db`.
- E1 config SHA-256: `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`.
- E2 fixed-mixture config SHA-256: `cb7d44ad81717a3666f8eb3bbf1ea174a644bd7c367f0711102c68c43832dc0f`.
- Synthetic mixture smoke receipt SHA-256: `c3124a77ffa3e9e362ec95acfa73ca243310a21a035ce4fc09887e5e00c49e7a`.
- Source manifest: 433 files, identity SHA-256 `3b906e45695f433af46bd2cb68a2e493bc9d4d3145e7f96e3f9e4b1cf6abe03e`; `qa/source_manifest.py --verify` passes.
- Pre-download checks: `ruff` passes; `143` host tests pass and `5` tests requiring host JAX are skipped; isolated Python 3.12/JAX 0.11.2 CPU synthetic end-to-end smoke passes. No Train/Validation or Test payload has been decoded.

## Frozen data and Fold identity (before registered reader access)

- Downloaded all 93 official Binance USD-M daily archives (January 2 through April 4, 2023) and exchange-published `.CHECKSUM` files. All sidecars were rechecked against the archive bytes; archive payload total: 5,799,864 bytes. Canonical archive receipt manifest SHA-256: `e8d524e2a2dc0480cb41b9e6e744c6d08a73a16eee35988f54f47d0808eedcbf`.
- Curated source: `.artifacts/world_model/fresh-ethusdt-20230102-mixture-60-15-15-20261019/source/market.parquet`; 133,920 continuous bars; source-open interval `[2023-01-02T00:00:00Z, 2023-04-05T00:00:00Z)`.
- Dataset SHA-256: `e528d6509b953b0028deaf529e37f6e0b4db00ec3f2534f53b893e1bebd0d471`.
- Source marker SHA-256: `0a4689a62dc7fd6eeda93f01b8bf20b3bd7a7522158cba23f7aa638b5c4341cf`.
- Parquet file SHA-256: `695ad16a622cc3582b1a84bf2d6b39db0c2e8924620cd751eb8036151ea6b15b`.
- Source preparation receipt SHA-256: `7c23ca5aeee8b5aedf73a03ce070139920fc2a5f88f2a3ebc725a84150248f04`.
- Registered Fold config: `configs/routeb/fresh_ethusdt_20230102_mixture_60_15_15_fold.json`; config SHA-256 `f238a6fb6527eb4dcf008eeaa9d3b3cf29148b7082ab3eb0800f552aac24e104`; Fold hash `b5bd6d26a5978fa3018e938617e34e1508e095ce30bf282f1aa6d6cdb3e6b7e4`.
- `FoldManifest` schema `0.4.0` and the registered 60/15/15 interval, purge, embargo, minute alignment, source footer and official archive sidecars all pass pre-read integrity checks. No registered Train/Validation reader has run yet; the one-shot reservation remains pending and will be the final scope authorization check.
