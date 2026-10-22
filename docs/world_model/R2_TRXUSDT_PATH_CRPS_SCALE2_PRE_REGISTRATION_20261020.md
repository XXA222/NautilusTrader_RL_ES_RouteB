# R2 TRXUSDT Train path-log-RV CRPS weight 2.0 — pre-registration

Status: frozen before downloading or decoding the source interval. This is a single-variable follow-up to the inconclusive paths-16 run. The observed generated rolling-RV p95 remained below realized values. Increase only the Train-only sampled path log-RV CRPS coefficient from 1.0 to 2.0 to test whether stronger distributional consistency reduces free-rollout under-dispersion without harming return CRPS.

## Frozen hypothesis and recipe

- E2 is the RouteB RSSM Gaussian-prior model with five-channel OHLCV NLL, equal-weight Train-only sampled free-prior path log-RV CRPS at 15/30/60 bars, 16 paths, and path_log_rv_crps_scale=2.0. Relative to the completed paths-16 experiment, only this coefficient changes from 1.0 to 2.0. Keep risk loss scale 1.0 and rollout risk conditioning disabled.
- E1 is the same fixed GRU comparator; no change.
- Both candidates: 1,000 updates, batch 8, learning rate 0.0003, report interval 50, seeds [31,102,20261018].
- Validation: 180 anchors at 120-minute spacing, 32 forecast paths per seed (96 pooled), 4-anchor forecast chunks, 2,000 paired moving-block bootstrap replicates, block length 12.
- Baselines: frozen E0 zero-drift, Train-innovation EWMA, and stationary block bootstrap. Select the strongest baseline by Validation mean 15m path log-RV CRPS.
- Primary gate: for every seed, E2-minus-selected-E0 15m path log-RV CRPS 95% paired CI strictly below zero and Holm p<0.05 in the existing 11-test family. Guardrails: no Holm-corrected significant E2 harm to 15m return CRPS; generated OHLCV and paths structurally valid. No Test/Holdout read, no economics claim, no automatic stage promotion.

## Frozen data scope

- Binance USD-M TRXUSDT 1m source-open interval [2024-01-12T00:00:00Z, 2024-04-15T00:00:00Z), 94 UTC days, 94 daily archives, 135,360 expected closed bars.
- Fold: 60/15/15 days; 1h purge and 23h embargo. Train+Validation reader scope [2024-01-12T00:01:00Z, 2024-03-30T00:01:00Z).
- Train score [2024-01-14T00:01:00Z, 2024-03-14T00:01:00Z); Validation score [2024-03-15T00:01:00Z, 2024-03-30T00:01:00Z), with 12-day warmup beginning 2024-03-03T00:01:00Z. Test score [2024-03-31T00:01:00Z, 2024-04-15T00:01:00Z), with warmup beginning 2024-03-19T00:01:00Z.
- The read-only RouteB ledger preflight for TRXUSDT perpetual-market-v1 and the exact Train+Validation interval returned zero access and zero query_intent overlaps on 2026-10-20. The one-shot reader's immediate preflight remains authoritative.
- Output root: .artifacts/world_model/fresh-trxusdt-20240112-path-crps-scale2-20261020/.
- Download and verify all official daily sidecars and ZIP members only after this pre-registration. Prepare source/Fold identities and repeat the read-only ledger check before any registered Train/Validation read. Exclude Test/Holdout.

## Frozen implementation and configs

- RouteB code source fingerprint: 593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c; base commit cb7a3ec9d542ed48b9f182acbaac330fc90b3177.
- Registered runner SHA-256: aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9.
- E1 config: configs/routeb/fresh_trxusdt_20240112_e1_pathcrps_paths16_ensemble.json; SHA-256 07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8.
- E2 config: configs/routeb/fresh_trxusdt_20240112_e2_pathcrps_paths16_scale2_ensemble.json; SHA-256 21c627d04df35a8e81f45b147d9e45a9b54091c057bd2085e543dd188c98eab2.
- JAX worker lock SHA-256: ab99ac8f831bc06dc6e408cf91090b69aca413df8f38cfcd2f28b77a6ea18c1b; isolated Python 3.12.10 / JAX 0.11.2 CUDA worker.

No source, model, training, scoring or decision-rule changes are authorized within this frozen run.

## Append-only verified source and Fold identities before registered access

{
  "archives": 94,
  "dataset_hash": "e1bc66ea3f2043f9590b0990a8765f1fa792b145e21db928e2428e7df5ba32c2",
  "download_manifest_sha256": "e1fd92174938741cc9ec1710491305b358f6ffce64ae6cd6e7cdc37098d05c97",
  "final_read_only_preflight": {
    "access": 0,
    "query_intent": 0
  },
  "fold_hash": "34360fb85e58fbb6cee43851a6d055e1b3441b426bc73d87e6304b6f1ca5616b",
  "marker_sha256": "cbc043cf1a6f3fe4553895acdb10160fa79c5a29591c24e848d3597e429bcffd",
  "parquet_sha256": "4591a0ec5206ed200a43acbf4b7f135def00ded73ddef0a6afa13af28c6362d2",
  "reader_scope_ns": [
    1705017660000000000,
    1711756860000000000
  ],
  "registered_fold_config_sha256": "f36c95b806c938b34d61008ab745fd781acda6ecc6770f6f0f48a20c1ea2ff01",
  "source_preparation_sha256": "031d178ecd628bccc1adddca38564f7ea29de089758b3bc8cafe167bd6815206"
}

## Append-only completion record — 2026-10-20

The registered scale-2 run completed at .artifacts/world_model/fresh-trxusdt-20240112-path-crps-scale2-20261020/ensemble/. E2-minus-selected-E0 15-bar path log-RV CRPS was -0.019916 (95% CI [-0.092988,+0.081086], p=0.651174); all three seed CIs crossed zero and Holm p-values were 0.458771, 0.638181, and 0.764118. E1 outscored E2. The return guardrail had no corrected significant harm, but generated rolling-RV p95 (0.000649) remained below realized (0.001385). All report and artifact hashes verified; Test/Holdout remained unread. R2 is not promoted. Full result: [R2_TRXUSDT_PATH_CRPS_SCALE2_RESULTS_20261020.md](R2_TRXUSDT_PATH_CRPS_SCALE2_RESULTS_20261020.md).
