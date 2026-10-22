# R2 TRXUSDT train-time path log-RV CRPS pre-registration

Status: frozen before downloading or decoding this source interval. This is one fixed candidate, not a sweep.

## Hypothesis and fixed recipe

The preceding BTCUSDT rollout-risk experiments improved some seed-level path log-RV scores but did not pass the all-seed Holm gate; free rollouts still underrepresented volatility persistence. The latest full-size synthetic CUDA smoke established that an additive, Train-only free-prior path-derived 15-minute log-RV CRPS objective executes while retaining the five-channel OHLCV NLL. This experiment tests that single training-time objective on a fresh market and untouched interval.

- E2: default two-level RouteB RSSM (deter 256, hidden 128, stochastic 16x16 categorical state), Gaussian prior, risk prediction scale 1.0, rollout risk conditioning disabled; retain all five OHLCV prior-NLL channels and add Train-only free-prior sampled 15-minute path log-RV CRPS with objective weight 1.0, horizon 15 bars, 4 paths per training anchor.
- E1: fixed probabilistic GRU baseline.
- E1 and E2 each receive 1,000 optimizer updates, batch size 8, seeds [31,102,20261018], learning rate 0.0003; no tuning or candidate substitution.
- E0: frozen zero-drift, Train-innovation EWMA and stationary block-bootstrap baselines; select the strongest on Validation mean path-derived 15-minute log-RV CRPS.
- 180 Validation anchors at 120-minute intervals; 32 forecast paths per model seed; 96 pooled paths; four-anchor chunks; 2,000 paired moving-block bootstrap replicates with 12-anchor blocks.
- Primary gate: for each of the three seeds, E2-minus-selected-E0 path-derived 15-minute log-RV CRPS must have a paired 95% CI strictly below zero and Holm-adjusted p below 0.05, within the existing 11-test Holm family.
- Guardrail: E2 must not have statistically significant Holm-corrected harm on 15-minute return CRPS; all generated OHLCV and path-structure checks must pass.
- Test/Holdout remain unread. No economic claim, R2 promotion, or R3-R6 gate change follows unless every registered criterion passes. Any failed seed/guardrail is a failed promotion gate.

## Frozen source, split, and access scope

- Binance USD-M TRXUSDT one-minute klines, open-time interval [2023-01-01T00:00:00Z, 2023-04-05T00:00:00Z), 94 UTC days, 94 expected daily archives and 135,360 expected bars.
- Approved 60/15/15-day Fold with one-hour purge and 23-hour embargo.
- Train+Validation reader scope: [2023-01-01T00:01:00Z, 2023-03-20T00:01:00Z).
- Train score: [2023-01-03T00:01:00Z, 2023-03-04T00:01:00Z); Validation score: [2023-03-05T00:01:00Z, 2023-03-20T00:01:00Z).
- Test score: [2023-03-21T00:01:00Z, 2023-04-05T00:01:00Z); Test/Holdout payload must not be decoded.
- The read-only RouteB access/query-intent registry preflight found no overlap for this symbol and source interval; repeat immediately before registration and record exact counts.
- Verify every official .CHECKSUM sidecar and ZIP member before curation. Freeze source, E1/E2 configs, Fold semantics and identities before the registered reader.
- Output root: .artifacts/world_model/fresh-trxusdt-20230101-path-log-rv-crps-20261020/.

## Execution controls

Use only the RouteB repository and its isolated JAX workers. Run the registered multi-seed Fold runner with includes_test=false; capture test_payload_decoded=false. Independently verify the aggregate, predictive, per-seed, forecast, risk and pooled artifact digests after completion. Do not inspect, score, decode or tune against Test/Holdout.

## Append-only frozen identities

Exact source-preparation receipt, dataset hash, marker and Parquet hashes, access-registry counts, E1/E2/Fold/runner hashes, semantic Fold hash and isolated worker identity will be appended after archive validation and before the one-shot registered reader. No model or evaluation choices may change after registration.

### Append-only frozen candidate configuration hashes

- E1 config SHA-256: e3f979c910c8e4d4f3c70e11475516c5c5b2b86b1be1d2e222799d416aea2c38
- E2 config SHA-256: 565a7baeca10a3e971313013accfc7cb398a7de90bdee9ba8913ae3a21606d4d

### Append-only data, source, Fold and pre-read identities

- Read-only access ledger counts immediately before registration: access rows = 0; query_intent rows = 0 for TRXUSDT perpetual-market-v1 and Train+Validation scope [2023-01-01T00:01:00Z, 2023-03-20T00:01:00Z).
- access_rows: 0
- query_intent_rows: 0
- train_validation_scope_ns: [1672531260000000000, 1679270460000000000]
- archive_count: 94
- download_manifest_sha256: 3dbce4d0c89741879b85f6af9e055c25f3a43b88ac9e93f39639d1f34fbde582
- source_preparation_sha256: 1abeaaa01f1473b7755a8965d54fa6ce7396ecc002f0d89a2b07fb2631813dc0
- dataset_hash: ebecf510039357bd2607f2a7b63489ec95af9cf854bac1fbe55469aa87878c29
- marker_sha256: 9ffb31fa9b4e3a3f6315f5e532c849c5250e36da8c1a653361afbd4e38a88463
- parquet_sha256: 34e403975e9739f4024b685cb9ee61b0dd7e7fdf329a7a083b93e950d8f3caf9
- registered_fold_config_sha256: 4ce58957a6ec6bd309e48659bb14fc3cedac5efeb2f8d6036dd9004fcd05ba5c
- semantic_fold_hash: 98e80b4d1490f440d8968c9fa6a0e9a8f1a3b7668cdbf86930ec9b658b353570
- e1_config_sha256: e3f979c910c8e4d4f3c70e11475516c5c5b2b86b1be1d2e222799d416aea2c38
- e2_config_sha256: 565a7baeca10a3e971313013accfc7cb398a7de90bdee9ba8913ae3a21606d4d
- runner_sha256: aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9
- repository_commit: cb7a3ec9d542ed48b9f182acbaac330fc90b3177
- source_code_sha256: 0563d9b8e97468baaa7ddf8c820efbac9fb090de178f992bff824b49966c4558
