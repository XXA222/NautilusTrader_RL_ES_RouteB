# R2 TRXUSDT Train path log-RV CRPS weight 4.0 — pre-registration

Status: frozen before downloading or decoding this source interval. The preceding scale-2 transfer retained favorable but inconclusive point estimates and persistent rolling-RV under-dispersion. Test a single higher Train-only path log-RV CRPS coefficient of 4.0, holding paths at 16 and all model/runtime/evaluation settings fixed. This is one fixed transfer, not a sweep.

## Frozen recipe and decision rule

- E2: RSSM with Gaussian prior, five-channel OHLCV objective, equal-weight Train-only free-prior sampled path log-RV CRPS at horizons [15,30,60], paths=16, coefficient=4.0, native risk loss scale=1.0, rollout risk conditioning disabled.
- Only difference from the previous scale-2 recipe is path_log_rv_crps_scale: 2.0 -> 4.0. No other tuning.
- E1 is the same fixed GRU comparator.
- Both: 1,000 updates, batch 8, learning rate 0.0003, report interval 50; seeds [31,102,20261018].
- Validation: 180 anchors at 120-minute intervals, 32 paths/seed, 96 pooled; four-anchor chunks; 2,000 paired moving-block bootstrap replicates, block length 12.
- E0 baselines: zero-drift, Train-innovation EWMA, stationary block bootstrap. Choose the strongest using Validation mean 15-bar path log-RV CRPS.
- Primary per-seed gate: E2-minus-selected-E0 15-bar path log-RV CRPS 95% CI strictly below zero and Holm p<0.05 in the fixed 11-test family for every seed. Guardrails: no significant Holm-adjusted E2 harm to 15-bar return CRPS and generated OHLCV/path structure valid. Test/Holdout excluded; no economic claim or stage promotion by this run.

## Frozen data scope

- Binance USD-M TRXUSDT 1m source-open interval [2024-04-15T00:00:00Z, 2024-07-18T00:00:00Z), 94 UTC days, 94 daily archives, 135,360 expected closed bars.
- Fold 60/15/15, one-hour purge, 23-hour embargo. Train+Validation read scope [2024-04-15T00:01:00Z, 2024-07-02T00:01:00Z).
- Train score [2024-04-17T00:01:00Z, 2024-06-16T00:01:00Z); Validation score [2024-06-17T00:01:00Z, 2024-07-02T00:01:00Z), warmup begins 2024-06-05T00:01:00Z. Test score [2024-07-03T00:01:00Z, 2024-07-18T00:01:00Z), warmup begins 2024-06-21T00:01:00Z.
- Read-only RouteB registry preflight on 2026-10-20: zero access overlaps and zero query_intent overlaps for TRXUSDT perpetual-market-v1 and the exact Train+Validation interval. The one-shot registered reader remains authoritative.
- Output: .artifacts/world_model/fresh-trxusdt-20240415-path-crps-scale4-20261020/.
- Download official daily archives only after this frozen registration; validate sidecars and ZIP members, prepare source/Fold identity, repeat ledger preflight, and exclude Test/Holdout from model access.

## Frozen code and config identities

- RouteB source SHA-256 593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c; commit cb7a3ec9d542ed48b9f182acbaac330fc90b3177.
- Registered ensemble runner SHA-256 aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9.
- E1 config: configs/routeb/fresh_trxusdt_20240415_e1_pathcrps_paths16_ensemble.json; SHA-256 07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8.
- E2 config: configs/routeb/fresh_trxusdt_20240415_e2_pathcrps_paths16_scale4_ensemble.json; SHA-256 d67ef003d7b130ef94e042f56996ee76c8a588f1efe4352306bcd3c9ad9e5ae4.
- JAX worker lock SHA-256 ab99ac8f831bc06dc6e408cf91090b69aca413df8f38cfcd2f28b77a6ea18c1b; isolated Python 3.12.10 / JAX 0.11.2 CUDA worker.

No source, config, scoring or decision-rule changes after freeze.

## Append-only source and Fold identities before registered access

{
  "archives": 94,
  "dataset_hash": "cbab4476b674010700b81d579bbc097952feaaca193d3f8f84a986e6c10176d0",
  "download_manifest_sha256": "7802349b94c0b0492985e168351b3f57d6fcb468bee6d6ac21d1b82a2ba83f54",
  "final_read_only_preflight": {
    "access": 0,
    "query_intent": 0
  },
  "fold_config_sha256": "16c78cbbec1ea5406918c94ca6dcb95f9e6e8d06dc532d118db2317e014d87ef",
  "fold_hash": "a1dedfb313342c410ccd13bf93a82c413adaec186c176093ece5e664c88a2306",
  "marker_sha256": "84e5646aa22059ab970d4ff53f39ea09cbd254cd34a10bbe0824ced11a2d88c6",
  "parquet_sha256": "666ab26bf8c2ab5d90866e11a2b9ef7b3ead80a41912b696eb6337454bf1efba",
  "reader_scope_ns": [
    1713139260000000000,
    1719878460000000000
  ],
  "source_preparation_sha256": "a678ddfab0f7f208bbec3b715472ac314fc93044da20399d1af29cc9683fe327"
}

## Append-only completion record — 2026-10-20

Scale-4 three-seed run completed at .artifacts/world_model/fresh-trxusdt-20240415-path-crps-scale4-20261020/ensemble/. Pooled E2-minus-E0 15-bar path log-RV CRPS was -0.022947 (95% CI [-0.042399,-0.006394], p=0.012494). Seed 20261018 passed its full gate; seed 31 missed Holm p=0.070465, and seed 102's CI crossed zero. Thus the all-seed promotion gate failed. Return guardrail showed no corrected significant harm. Generated RV mean/p95 (0.000264/0.000457) approached realized (0.000250/0.000465), while generated absolute-return persistence remained low. Test/Holdout unread; integrity hashes passed. R2 stays unpromoted. See [R2_TRXUSDT_PATH_CRPS_SCALE4_RESULTS_20261020.md](R2_TRXUSDT_PATH_CRPS_SCALE4_RESULTS_20261020.md).
