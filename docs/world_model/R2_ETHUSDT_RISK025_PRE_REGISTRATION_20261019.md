# R2 pre-registration: native-risk loss scale 0.25

Date frozen: 2026-10-19, before downloading or reading the market payload. The completed risk-scale-0.5 run retained native 30/60-minute log-realized-volatility gains but missed the primary return-CRPS gate by a small margin against the strongest baseline. This next fixed experiment reduces only the same auxiliary risk-loss coefficient once more.

## Hypothesis and single parameter change

Keep E2's Gaussian prior OHLCV head and the complete architecture, risk horizons/weights, target definitions, E1, E0, optimizer, features, three-seed budget and F1 protocol unchanged. Change only `risk_prediction_scale` from 2.0 to `0.25` (the registered risk-scale-0.5 configuration is the immediate implementation reference). The hypothesis is that a weaker auxiliary risk loss restores primary 15-minute return CRPS while leaving enough native-risk learning to improve 30/60-minute log-realized-volatility scores. This is one fixed setting, not a sweep; no follow-up setting will be chosen from this Validation result.

## Frozen scope and protocol

- Instrument/source: ETHUSDT USD-M perpetual, Binance 1-minute klines; source-open interval `[2022-05-10T00:00:00Z, 2022-08-11T00:00:00Z)`, 93 days / 133,920 expected bars. Download daily archives May 10 through August 10 inclusive.
- Before freezing, the RouteB registry was read in read-only mode. Train/Validation read interval `[2022-05-10T00:01:00Z, 2022-07-26T00:01:00Z)` had zero overlapping ETHUSDT `perpetual-market-v1` accesses and zero overlapping query intents. The one-shot registered reader is authoritative; if it rejects, do not read Train/Validation.
- Approved 60/15/15 Fold: Train warmup May 10 00:01; Train score May 11 through July 10 00:01; Validation warmup June 30 00:01; Validation score July 11 through July 26 00:01; Test warmup July 6 00:01; Test score July 27 through August 11 00:01. Preserve the established one-hour purge and 23-hour embargo. Test remains undecoded.
- E1/E2 share seeds `31, 102, 20261018`, 1,000 optimizer updates per seed, 32 forecast paths per seed, 180 two-hour Validation anchors, 96 pooled paths per candidate, and 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Primary measure: 15-minute return CRPS against the strongest statistical baseline selected on Validation. Promotion requires a paired 95% interval favoring E2 and Holm-adjusted p < 0.05. E2 versus E1 is reported separately.
- Secondary: 30/60-minute native log-realized-volatility CRPS/Brier, absolute-return lag-1 autocorrelation, rolling-RV mean/p95, and path validity. No Test/Holdout decoding, economics claim, threshold search or additional candidate.

## Frozen implementation delta

- E1 remains `configs/routeb/fresh_ethusdt_e1_ensemble.json`.
- E2: `configs/routeb/fresh_ethusdt_e2_risk025_ensemble.json`, derived from the Gaussian risk-scale-0.5 configuration by changing only `risk_prediction_scale=0.25`.
- No model source change. The installed isolated RouteB worker and fixed source identity are reused. Source checksums, curated marker/hash, Parquet hash and Fold hash will be recorded here before the registered reader accesses Train/Validation.
- Scope is consumed once reservation succeeds. A failed primary gate keeps R3/R4/R5/R6 closed. Do not decode Test or reuse this scope.

## Frozen code/config identity (before payload download)

- RouteB HEAD `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`; source identity SHA-256 `e1e08a244a8e95e3c3dfd26c1c1e15765a0ec0d6114cc322a257afd63df656db`.
- E1 config SHA-256 `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`.
- E2 Gaussian risk-scale-0.25 config SHA-256 `19aab79d97e6123e2e4f4b0c5611d1f8545c520e766575b590255a9f448e3953`.
- Source manifest: 436 files, identity SHA-256 `f31f619e46e45394773635f2a1c6eeae47d560b83fcef3c9815a029a05a2a1ae`; verification passes.
- The model config validates with Gaussian forecast distribution and risk scale 0.25. No real-market payload has been downloaded or read for this iteration.

## Frozen source/Fold identity (before registered reader access)

- Downloaded 93 official Binance USD-M ETHUSDT daily archives (May 10 through August 10, 2022), total 5,959,379 bytes. Every exchange `.CHECKSUM` sidecar was independently rechecked. Canonical archive receipt manifest SHA-256: `a11c5a7b7d7b480111d2ba0ea4ae35d1dd96e1f606729cad499bcb67513a59a2`.
- Curated source `.artifacts/world_model/fresh-ethusdt-20220510-risk025-60-15-15-20261019/source/market.parquet`: 133,920 continuous bars; source-open interval `[2022-05-10T00:00:00Z, 2022-08-11T00:00:00Z)`.
- Dataset SHA-256 `ad851c2853350679270e35922c92aa46377587de6e12a60e9ceedc2dfe7bbc20`; source marker SHA-256 `e307bd67d3b21e4e6cf95b3ea9db2ad3cc40445f2bb96da78109a8ea29d28d50`; Parquet SHA-256 `d378b80c66218c770b4e178428c5e6530de64cac54f4e61b9ba4a730c2a45420`.
- Source preparation receipt SHA-256 `d6200bb54d9eba876821a537e60ce181c67aa77b2d08ff0b991fa3e29c775baf`.
- Fold config `configs/routeb/fresh_ethusdt_20220510_risk025_60_15_15_fold.json`, config SHA-256 `e5492ff05805ddcdff2e6a8d9b23daf48e7e4e97d3a668440111a9ba648ba0bc`; Fold hash `544570a117b958da7b06d70c95b8de919d2621773b5eef206ec6e91cf48460d9`.
- Fold schema 0.4.0, 60/15/15 boundaries, purge, embargo, source footer and all archive sidecars pass. The one-shot Train/Validation reader has not run; Test remains excluded.
