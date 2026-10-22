# R2 TRXUSDT fixed path-CRPS scale-4 adjacent transfer — pre-registration

Frozen before downloading or decoding this interval. The preceding fresh transfer had a favorable pooled 15-bar path log-RV CRPS result, with one of three seeds passing the full gate; the upper-tail level approached realized but temporal clustering remained weak. This run transfers the exact scale-4 recipe unchanged to the next adjacent source interval. It performs no parameter selection or tuning.

## Frozen recipe and decision rule

- E2: same RouteB RSSM Gaussian-prior model, five-channel OHLCV objective, Train-only equal-weight sampled free-prior path log-RV CRPS at horizons [15,30,60], paths=16, path coefficient=4.0, native risk loss scale=1.0, rollout risk conditioning disabled.
- E1 unchanged fixed GRU comparator. Both models use 1,000 updates, batch size 8, learning rate 0.0003, report interval 50; seeds [31,102,20261018].
- Validation uses 180 anchors at 120-minute spacing, 32 paths per seed, 96 pooled paths, 4-anchor chunks, 2,000 paired moving-block bootstrap replicates and block length 12.
- Select among frozen E0 zero-drift, Train-innovation EWMA, and stationary block-bootstrap baselines by Validation mean 15-bar path log-RV CRPS.
- Primary gate per seed: E2-minus-selected-E0 15-bar path log-RV CRPS 95% CI strictly below zero and Holm p<0.05 in the fixed 11-test family for every seed. Guardrails: no significant Holm-corrected 15-bar return-CRPS harm; generated OHLCV and paths valid. No Test/Holdout read, no economic claim, no automatic promotion.

## Frozen data scope

- Binance USD-M TRXUSDT 1m source-open interval [2024-07-18T00:00:00Z, 2024-10-20T00:00:00Z), 94 UTC days, 94 daily archives, 135,360 expected closed bars.
- 60/15/15 Fold, one-hour purge, 23-hour embargo. Train+Validation reader scope [2024-07-18T00:01:00Z, 2024-10-04T00:01:00Z).
- Train score [2024-07-20T00:01:00Z, 2024-09-18T00:01:00Z); Validation score [2024-09-19T00:01:00Z, 2024-10-04T00:01:00Z), warmup begins 2024-09-07T00:01:00Z. Test score [2024-10-05T00:01:00Z, 2024-10-20T00:01:00Z), warmup begins 2024-09-23T00:01:00Z.
- Read-only RouteB ledger preflight found zero access and zero query_intent overlaps for TRXUSDT perpetual-market-v1 and the exact Train+Validation scope. The registered one-shot reader remains authoritative.
- Output root: .artifacts/world_model/fresh-trxusdt-20240718-path-crps-scale4-adjacent-20261020/.
- Download official archives after freeze; validate sidecars and ZIPs, prepare and record source/Fold identities, repeat the read-only ledger check, then exclude Test/Holdout.

## Frozen code and configurations

- RouteB source fingerprint 593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c; commit cb7a3ec9d542ed48b9f182acbaac330fc90b3177.
- Runner SHA-256 aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9.
- E1 config: configs/routeb/fresh_trxusdt_20240718_e1_pathcrps_paths16_ensemble.json; SHA-256 07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8.
- E2 config: configs/routeb/fresh_trxusdt_20240718_e2_pathcrps_paths16_scale4_ensemble.json; SHA-256 d67ef003d7b130ef94e042f56996ee76c8a588f1efe4352306bcd3c9ad9e5ae4.
- Worker lock SHA-256 ab99ac8f831bc06dc6e408cf91090b69aca413df8f38cfcd2f28b77a6ea18c1b; isolated Python 3.12.10/JAX 0.11.2 CUDA worker.

This adjacent run is a fixed transfer of scale 4.0, not another coefficient search.

## Append-only verified source and Fold identities before registered access

{
  "archives": 94,
  "dataset_hash": "f3a75f7f54a085938dcdc5313c5f48234385494f2e2e06b30e4c29e895e96ade",
  "download_manifest_sha256": "4d9c8f275f87f45496dd3e8af6f11efc4a596c830374ccfe25e3485750e096a7",
  "final_read_only_preflight": {
    "access": 0,
    "query_intent": 0
  },
  "fold_config_sha256": "b32e3fd1cbf994df31f49972d798467a73daca644e88a0f2105452d276fb3cec",
  "fold_hash": "036b09a8a048eadd0f905951f1abd1c4eedca153ba5b7a0d145d4fe9cac27dc1",
  "marker_sha256": "737898fd1ac37b5de90e21922ed9e753bf7742aa1b2a60fb0ccc2545104ddd7e",
  "parquet_sha256": "a7ddabebab27582d2ad4f0420eb3ce4d13a66f37f889dd65b007ae7b34e0b330",
  "reader_scope_ns": [
    1721260860000000000,
    1728000060000000000
  ],
  "source_preparation_sha256": "39339ca36588e4bacb34cabebaa77c760b44b4e341582021b6230e2118fdb8e5"
}

## Append-only completion record — 2026-10-20

The fixed scale-4 transfer on fresh TRXUSDT [2024-07-18, 2024-10-20) completed on seeds 31, 102 and 20261018. E2 beat the Validation-selected S2 baseline on pooled 15-bar path log-RV CRPS (`0.252385` vs `0.306793`, difference `-0.054408`, 95% CI `[-0.093431,-0.023968]`, p `0.001499`); all three per-seed Holm gates passed. However, seed 102 failed the registered 15-bar return-CRPS guardrail (harm `+0.000126349`, 95% CI `[+0.000055336,+0.000202628]`, Holm p `0.031984`). The full gate fails; R2 stays unpromoted and R3-R6 remain gated. E2 path/bar validity was 100%, but generated absolute-return lag-1 autocorrelation (`0.003779`) remained below realized (`0.051493`). Test/Holdout were not decoded; all report and pooled/risk hashes verified. See [R2_TRXUSDT_PATH_CRPS_SCALE4_ADJACENT_RESULTS_20261020.md](R2_TRXUSDT_PATH_CRPS_SCALE4_ADJACENT_RESULTS_20261020.md).
