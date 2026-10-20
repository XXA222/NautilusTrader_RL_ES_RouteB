# R2 pre-registration: reduce native-risk loss scale to 0.5

Date frozen: 2026-10-19, before downloading or reading this market interval. The previous ETHUSDT long-risk and distribution experiments found repeatable 30/60-minute native log-realized-volatility gains but no primary 15-minute return CRPS promotion. This experiment tests whether the auxiliary risk objective is competing too strongly with the primary next-bar forecast objective.

## Hypothesis and single parameter change

Keep the E2 forecast head Gaussian and keep the model, features, risk horizons, horizon weights, E1, E0 baselines, optimizer, training budget, and F1 protocol fixed. Change only `risk_prediction_scale` from the risk-weighted E2 reference value `2.0` to `0.5`. The hypothesis is that reducing the auxiliary risk loss contribution improves 15-minute return CRPS versus the strongest statistical baseline while retaining useful 30/60-minute log-realized-volatility scores. No scale sweep or additional E2 candidate is allowed.

## Frozen scope and protocol

- Instrument/source: ETHUSDT USD-M perpetual, Binance 1-minute klines; source-open interval `[2022-02-01T00:00:00Z, 2022-05-05T00:00:00Z)`, 93 days / 133,920 expected bars. Download daily archives February 1 through May 4 inclusive.
- Before freezing, the RouteB registry was checked read-only. Train/Validation read interval `[2022-02-01T00:01:00Z, 2022-04-19T00:01:00Z)` had zero overlapping ETHUSDT `perpetual-market-v1` access records and zero overlapping query intents. The registered reader's one-shot preflight is authoritative; if it rejects, do not read Train/Validation.
- Approved 60/15/15 Fold: Train warmup February 1 00:01; Train score February 2 through April 3 00:01; Validation warmup March 24 00:01; Validation score April 4 through April 19 00:01; Test warmup March 30 00:01; Test score April 20 through May 5 00:01. Preserve the established one-hour purge and 23-hour embargo. Test remains undecoded.
- E1/E2 share seeds `31, 102, 20261018`, 1,000 optimizer updates per seed, 32 forecast paths per seed, 180 two-hour Validation anchors, 96 pooled paths per candidate, and 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Primary measure: 15-minute return CRPS against the strongest statistical baseline selected on Validation. Promotion requires a paired 95% interval favoring E2 and Holm-adjusted p < 0.05 under the existing registered comparison family. E2 versus E1 is reported separately.
- Secondary measures: 30/60-minute native log-realized-volatility CRPS/Brier, F1 absolute-return lag-1 autocorrelation, rolling realized-volatility mean/p95 and path validity. No Test/Holdout decoding, economics claim, threshold search, or extra candidate.

## Frozen implementation and provenance

- E1: `configs/routeb/fresh_ethusdt_e1_ensemble.json` unchanged.
- E2: `configs/routeb/fresh_ethusdt_e2_risk05_ensemble.json`; derived from the Gaussian risk-weight-2 configuration with only `risk_prediction_scale=0.5` changed. The config is validated against `RSSMConfig` before any payload download.
- No model source change is made for this configuration-only experiment. The existing RouteB source identity and isolated JAX 0.11.2 worker are reused. The prior mixture experiment passed the full synthetic E1/E2 worker pipeline and 143 host tests passed with 5 host-JAX skips.
- Source archive checksums, curated source marker/hash, Parquet hash, and Fold hash will be recorded here before the registered reader accesses Train/Validation. The scope is consumed once reservation succeeds. A failed primary gate leaves R3/R4/R5/R6 closed.

## Frozen code and configuration identity (before payload download)

- RouteB HEAD: `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`; RouteB source identity SHA-256: `e1e08a244a8e95e3c3dfd26c1c1e15765a0ec0d6114cc322a257afd63df656db`.
- E1 configuration SHA-256: `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`.
- Frozen Gaussian E2 `risk_prediction_scale=0.5` configuration SHA-256: `e64938514b7ed10ae375b2dc42e100100772a9f406deb8fbd50b299606ca60e9`.
- Source manifest: 434 files, identity SHA-256 `22afb1150041cd860bdfd7df38924331712a8aaee26c2cb6489614f656c51de5`; manifest verification passes.
- Configuration validates as a Gaussian E2 model with risk scale 0.5, 1,000 steps and batch size 8. No real-market payload has been downloaded or read for this iteration.

## Frozen source and Fold identity (before registered reader access)

- Downloaded all 93 official Binance USD-M daily archives (February 1 through May 4, 2022) and exchange-published `.CHECKSUM` sidecars. Every sidecar was rechecked against archive bytes; archive total: 5,859,312 bytes. Canonical archive receipt manifest SHA-256: `787482fe29732da18daae3972b933b28e102233a386f80ee1759f3a71a401010`.
- Curated source: `.artifacts/world_model/fresh-ethusdt-20220201-risk05-60-15-15-20261019/source/market.parquet`; 133,920 continuous bars; source-open interval `[2022-02-01T00:00:00Z, 2022-05-05T00:00:00Z)`.
- Dataset SHA-256: `a47aca409ff6e0db3e5c161030ac6b0443d984737a89b1954d48c3f16da13cf3`.
- Source marker SHA-256: `ad329c70d646c97f6fd22dee3ab68348150491af5020565b721d67bb624b6b2a`.
- Parquet SHA-256: `5b6448e26b5df396a58d3cf47f5c592e7a3bdc7a5f7059538c35fd84e443c5af`.
- Source preparation receipt SHA-256: `15b3d362050877104dc47943fc11a2788adac6b029fd3af587e70d74b6971020`.
- Fold config: `configs/routeb/fresh_ethusdt_20220201_risk05_60_15_15_fold.json`; config SHA-256 `e9e8be5df4417ad253cba7ec7c57b2cd8743a55a326472f980c64a6f55e0c1fc`; Fold hash `c73e849fcdab29ab61f252d7d8f99422cb10da94d66197aa9fb077e08d35867f`.
- Supported `FoldManifest` schema 0.4.0, 60/15/15 boundaries, purge, embargo, source footer and all Binance sidecars pass pre-read validation. The registered Train/Validation reader has not run yet; Test remains excluded.
