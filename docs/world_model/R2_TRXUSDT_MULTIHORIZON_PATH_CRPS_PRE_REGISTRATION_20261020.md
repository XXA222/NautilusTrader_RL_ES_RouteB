# R2 TRXUSDT multi-horizon Train path log-RV CRPS pre-registration

Status: frozen before downloading or decoding this source interval. One fixed follow-up hypothesis; no search.

## Hypothesis and fixed recipe

The previous fixed 15-minute Train-only sampled path log-RV CRPS objective improved the E2 mean score over the selected E0 S1, but every seed's paired interval crossed zero; E1 scored better. The full-size synthetic CUDA check confirms that one sampled free-prior rollout can be scored at 15, 30 and 60 minutes while preserving prior OHLCV NLL. This experiment tests that single multi-horizon objective on the next non-overlapping TRXUSDT interval.

- E2: default RouteB RSSM (deter 256, hidden 128, stochastic 16x16 categorical state), Gaussian prior; keep five-channel OHLCV NLL; add equal-weight Train-only free-prior sampled path log-RV CRPS at 15/30/60 bars. Use 4 paths per anchor, anchor spacing 15 bars, and reuse each free-prior sample up to 60 bars. Keep risk prediction scale 1.0 and rollout risk conditioning disabled.
- E1: fixed probabilistic GRU baseline.
- E1 and E2: 1,000 optimizer updates, batch 8, learning rate 0.0003, seeds [31,102,20261018], report interval 50. No tuning or candidate substitution.
- E0: frozen zero-drift, Train-innovation EWMA and stationary block-bootstrap baselines; select the strongest on Validation mean path-derived 15-minute log-RV CRPS.
- 180 Validation anchors at 120-minute intervals; 32 forecast paths per model seed; 96 pooled paths; four-anchor chunks; 2,000 paired moving-block bootstrap replicates with 12-anchor blocks.
- Primary gate: each seed's paired 95% CI for E2-minus-selected-E0 15-minute path log-RV CRPS must be strictly below zero and Holm-adjusted p below 0.05 within the existing 11-test family.
- Guardrail: no statistically significant Holm-corrected E2 harm on 15-minute return CRPS; generated OHLCV and path structure must be valid.
- Test/Holdout remain unread. No economic claim, R2 promotion, or R3-R6 gate change follows unless every registered criterion passes.

## Frozen source, split, and access scope

- Binance USD-M TRXUSDT one-minute source open-time interval [2023-04-05T00:00:00Z, 2023-07-08T00:00:00Z): 94 UTC days, 94 expected daily archives, 135,360 expected bars.
- Approved 60/15/15-day Fold with one-hour purge and 23-hour embargo.
- Train+Validation reader scope: [2023-04-05T00:01:00Z, 2023-06-22T00:01:00Z).
- Train score: [2023-04-07T00:01:00Z, 2023-06-06T00:01:00Z); Validation score: [2023-06-07T00:01:00Z, 2023-06-22T00:01:00Z).
- Test score: [2023-06-23T00:01:00Z, 2023-07-08T00:01:00Z); Test/Holdout payload must not be decoded.
- Read-only RouteB ledger preflight must show zero overlapping access and query_intent rows for TRXUSDT perpetual-market-v1 over Train+Validation before download and again immediately before registration.
- Verify all Binance checksum sidecars and ZIP members before curation. Freeze exact configs, source/Fold identities and worker environment before the registered reader.
- Output root: .artifacts/world_model/fresh-trxusdt-20230405-multihorizon-path-crps-20261020/.

## Execution controls

Use only RouteB and its isolated JAX worker. Run the registered multi-seed Fold runner with Test excluded. Independently verify aggregate, predictive, seed, checkpoint, forecast, risk and pooled artifact digests; retain test_payload_decoded=false.

## Append-only frozen identities

Candidate config hashes, code/runner identities, read-only registry counts, downloaded archive manifest, curated dataset and marker hashes, Fold config and semantic hash will be appended before the one-shot registered reader. Model and evaluation choices remain frozen.

### Append-only frozen code/config identity and initial read-only preflight

- Access rows = 0 and query_intent rows = 0 for the exact Train+Validation scope.
- read_only_access_rows: 0
- read_only_query_intent_rows: 0
- train_validation_scope_ns: [1680652860000000000, 1687392060000000000]
- e1_config_sha256: e3f979c910c8e4d4f3c70e11475516c5c5b2b86b1be1d2e222799d416aea2c38
- e2_config_sha256: b29f574cb598c1df48e91fd9e92b540deefa043c948dfc31b8e7d9f63fb9db0a
- runner_sha256: aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9
- repository_commit: cb7a3ec9d542ed48b9f182acbaac330fc90b3177
- source_code_sha256: 35af314d1b39eaf276d4d770b3f4630427376ec8a5197392fb42153c55f98107

### Append-only source, Fold and final pre-read identities

- Final read-only ledger preflight before the registered reader: access rows = 0; query_intent rows = 0 for TRXUSDT perpetual-market-v1.
- second_read_only_access_rows: 0
- second_read_only_query_intent_rows: 0
- train_validation_scope_ns: [1680652860000000000, 1687392060000000000]
- archive_count: 94
- download_manifest_sha256: 1105aa6e411f6d659be3cbbd59bf3f2eb82594752c5afaff6a770e4e1ebb1d05
- source_preparation_sha256: 51ca6e07c8fe9ef553dfd860f332a6362a968e03bfcfdeb36fd8af7fd0c5ec8a
- dataset_hash: 4950c128145e4526ccc8ab60b96ab56b199c1acede1e188de970da6707b04345
- marker_sha256: 7b85f4afe1caba1ee6928b56800669cceaa555137f8a8f584258ccd72ff5c2f6
- parquet_sha256: 2ed633ce9af859bc8b710a165f214bef1e7529a77bebaeb6c61f7f54656183fb
- registered_fold_config_sha256: 2e491356eba70cdf3cbf3d2d746005a3997e7b9e00bd264af043fa90aba157c0
- semantic_fold_hash: e31c58c6c7e50352027c30e13fd6e8eddd676c47c85ed280df16f74b22818692
- runner_sha256: aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9
- repository_commit: cb7a3ec9d542ed48b9f182acbaac330fc90b3177
- source_code_sha256: 35af314d1b39eaf276d4d770b3f4630427376ec8a5197392fb42153c55f98107

### Append-only registered technical failure

The one-shot reader completed against the frozen Train+Validation scope, then seed 31 stopped before predictive scoring because E2 checkpoint JSON identity used a tuple/list mismatch for path_log_rv_crps_horizons. E1 training/Validation and E2 training/checkpoint completed; E2 forecast did not start successfully. Registry state after failure: one recorded access row and zero query_intent rows. Test/Holdout remained unread. Root cause, corrective code identity and focused regression evidence are recorded in [R2_TRXUSDT_MULTIHORIZON_PATH_CRPS_BLOCKED_20261020.md](R2_TRXUSDT_MULTIHORIZON_PATH_CRPS_BLOCKED_20261020.md). This interval is consumed and will not be retried after the code identity changed.
