# R2 BTCUSDT Gaussian rollout-risk strength-0.1 transfer pre-registration

Status: frozen on 2026-10-20 before downloading or decoding this source interval. This is a single lower-strength transfer check, not a search.

## Hypothesis and fixed recipe

The registered zero-conditioning Gaussian diagnostic was close to the E0 path score but remained under-dispersed. Geometric strength `0.25` then worsened path log-RV CRPS on a different BTC interval, with positive E2-minus-E0 in all seeds and significant harm in one seed. This transfer fixes the risk-conditioning strength at `0.1`, a near-unconditioned adjustment, to test whether the risk signal can be introduced without the harmful expansion observed at `0.25`. No parameter sweep or post-hoc change is allowed.

- E2: same two-level RSSM, Gaussian prior, native risk horizons `[1,5,15,30,60]`, risk prediction scale `0.5`, rollout risk conditioning enabled with geometric strength `0.1`, 1,000 updates, batch size 8, seeds `31,102,20261018`.
- E1: fixed GRU baseline, 1,000 updates, batch size 8.
- E0: frozen zero-drift, Train-innovation EWMA and stationary block-bootstrap baselines; select strongest on Validation mean path-derived 15-minute log-RV CRPS.
- 180 Validation anchors at 120-minute intervals; 32 paths/seed; 96 pooled paths; four-anchor chunks; 2,000 paired moving-block bootstrap replicates with 12-anchor blocks.
- Primary gate: each seed's paired 95% CI for E2-minus-selected-E0 15-minute path log-RV CRPS must be strictly below zero and Holm-adjusted p below `0.05`. The 11-test Holm family is unchanged.
- Test and Holdout stay unread. No economic claim or R2 promotion follows unless the complete registered gate passes.

## Frozen data and access scope

- Binance USD-M BTCUSDT 1-minute source interval `[2023-08-01T00:00:00Z, 2023-11-03T00:00:00Z)` (94 daily archives).
- Train+Validation read `[2023-08-01T00:01:00Z, 2023-10-19T00:01:00Z)` using the approved 60/15/15-day Fold with declared warmup, purge, and embargo.
- Test score `[2023-10-19T00:01:00Z, 2023-11-03T00:01:00Z)` is never decoded.
- Read-only RouteB ledger preflight immediately before registration found zero overlapping `access` and zero `query_intent` rows for BTCUSDT `perpetual-market-v1` on Train+Validation.
- Output root: `.artifacts/world_model/fresh-btcusdt-20230801-gaussian-risk01-20261020/`.
- Official sidecars and ZIP integrity must pass; exact RouteB code, config, source, and Fold identities are appended before the registered reader. Append report and artifact hashes after the one-shot run.
## Append-only frozen execution identities

Frozen after source preparation and before the registered reader ran:

- Repository commit `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`; RouteB source identity SHA-256 `4b1327326acd79bdfaf50da581bfb11367c6871c13818292571ba3772ed0ee8e`.
- E2 configuration SHA-256 `61d09b69d82700822471abc4a47f44118518a3682f5926592e280ab2724c2a37e`.
- Fold configuration SHA-256 `43f30384866d4400ff7ee8dca8af28190f01c7fab64bc47bd519f21602943275`; semantic Fold hash `2fa5e4d9adc00f003296beb4220297fb8f8f67ca65cadbefa341157c5cb38286`.
- Dataset hash `edb8996498f9636a74f9d18b8d4d1794053dea88a9a7e901bd2186f29b2f63ea`; marker SHA-256 `07027ea80189516165063051b24c60d9af4d24697a9181ad935e0edc90697235`; Parquet SHA-256 `e6df1054917cbd83a47e6b4f34e645b4cffb01bc80d7354d36c8e64cca13f665`.
- Source preparation verified all 94 Binance `.CHECKSUM` sidecars and ZIP members; 135,360 bars curated.
- Read-only ledger preflight found zero `access` and zero `query_intent` overlaps for `[2023-08-01T00:01:00Z, 2023-10-19T00:01:00Z)`.

Output directory: `.artifacts/world_model/fresh-btcusdt-20230801-gaussian-risk01-20261020/experiment`.

## Append-only execution record

The registered three-seed Validation run completed. Aggregate report SHA-256 `1ebfd273379220c14a88ee5a7a947e72ccb84fad82d05f564bf9d435f9f794de`; seed report SHA-256 values: seed 31 `98fc0789b1c00c37f62528a7cc2c9f5312920bffa392d2a708c5efcb46139731`, seed 102 `9fe94587d3263a0f0b4ca1d2631bd9b8f81390c806de74ea72335c9582a8e704`, seed 20261018 `3dcb5658cd7003afec25eba1373ab6c862ef46c6a85368c8b2014ed08de513d7`. Pooled E1 SHA-256 `7bde504d1a3e47c5f867b25627fb6ee33a62dd52f09f41ae1e9e06f51cc3cddf`; pooled E2 SHA-256 `878a9b7b49727d936939ca457a73adcc9843f9d6181dd23d31d0825742f09da3`. All report/artifact digests verify and Test remained unread. The primary rule passed in two of three seeds but failed the all-seed promotion criterion; see [`R2_BTCUSDT_GAUSSIAN_RISK01_RESULTS_20261020.md`](R2_BTCUSDT_GAUSSIAN_RISK01_RESULTS_20261020.md).
