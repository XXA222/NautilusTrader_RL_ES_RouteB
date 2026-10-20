# R2 TRXUSDT multi-horizon path log-RV CRPS with 16 training paths — pre-registration

Status: frozen before downloading/decoding the source interval. This tests one follow-up mechanism after the previous multi-horizon run's pooled estimate was favorable but inconclusive and one of three seeds failed: increase only the Train objective's Monte Carlo path count from 4 to 16 to reduce sampled-score/gradient noise and test whether gains become more stable.

## Frozen hypothesis and recipe

- E2: same RouteB RSSM architecture, Gaussian prior, five-channel OHLCV NLL, equal-weight Train-only free-prior sampled path log-RV CRPS at 15/30/60 bars, risk prediction scale 1.0, rollout risk conditioning disabled. **Only method change from the last completed TRX experiment: `path_log_rv_crps_paths=16` instead of 4.** No other tuning.
- E1: fixed probabilistic GRU comparator, unchanged.
- E1/E2: 1,000 updates, batch 8, learning rate 0.0003, report interval 50, seeds `[31,102,20261018]`.
- Validation: 180 anchors at 120-minute spacing; 32 forecast paths per seed; 96 pooled; four-anchor forecast chunks; 2,000 paired moving-block bootstrap replicates with block length 12.
- E0: frozen zero-drift, Train-innovation EWMA and stationary block-bootstrap baselines; select strongest on Validation mean 15m path log-RV CRPS.
- Primary gate: for each seed, E2-minus-selected-E0 15m path log-RV CRPS 95% paired interval strictly below zero and Holm p<0.05 in the existing 11-test family. Guardrails: no significant Holm-corrected E2 harm to 15m return CRPS; generated OHLCV/path structure valid. Test/Holdout unread; no economic claim or automatic stage promotion.

## Frozen data scope

- Binance USD-M TRXUSDT 1-minute source-open interval `[2023-10-10T00:00:00Z, 2024-01-12T00:00:00Z)`, 94 UTC days, 94 daily archives, 135,360 expected closed bars.
- Fold: 60/15/15 days; 1h purge; 23h embargo. Train+Validation reader scope `[2023-10-10T00:01:00Z, 2023-12-27T00:01:00Z)`.
- Train score `[2023-10-12T00:01:00Z, 2023-12-11T00:01:00Z)`; Validation score `[2023-12-12T00:01:00Z, 2023-12-27T00:01:00Z)` with warmup `[2023-11-30T00:01:00Z, 2023-12-12T00:01:00Z)`; Test score `[2023-12-28T00:01:00Z, 2024-01-12T00:01:00Z)` with warmup beginning `2023-12-16T00:01:00Z`.
- Initial read-only ledger preflight for TRXUSDT perpetual-market-v1: zero overlapping access rows and zero query-intent rows.
- Output root `.artifacts/world_model/fresh-trxusdt-20231010-path-crps-paths16-20261020/`.
- Download all daily archives only after this preregistration; validate each Binance sidecar and ZIP member. Prepare curated source and append exact data/Fold identities, then perform another read-only ledger preflight before the one-shot registered reader. Exclude Test/Holdout.

## Frozen code/config/runtime identity

- RouteB base commit `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`; source code SHA-256 `593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c`.
- Runner SHA-256 `aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9`.
- E1 config SHA-256 `e3f979c910c8e4d4f3c70e11475516c5c5b2b86b1be1d2e222799d416aea2c38`; E2 paths-16 config SHA-256 `1abae0c394c4b0b10101372a98b554ccb30b7b3da093b4e4502e9bf51d6d3387`.
- Worker lock SHA-256 `ab99ac8f831bc06dc6e408cf91090b69aca413df8f38cfcd2f28b77a6ea18c1b`; isolated Python 3.12.10/JAX 0.11.2 CUDA worker.

Source/data/Fold hashes will be appended before registered access. Do not modify code, configs or scoring after this freeze.

## Append-only verified source and Fold identities before registered access

```json
{
  "archives": 94,
  "dataset_hash": "e273b372543285281ef57b10c5ec9fd19d4dba3bb5e33bf4a8d46ca7da3dc9bf",
  "download_manifest_sha256": "d10d9a3ae76f10ffa9216cb344030ee378368787513014f22a36d0da696fcc06",
  "footer_bounds": {
    "max_ts_event_ns": 1705017600000000000,
    "min_ts_event_ns": 1696896060000000000
  },
  "marker_sha256": "cb355367567fe6b153eff6b5880d5d865d620863542138ea658966192d6ac47f",
  "parquet_sha256": "220c32aa7fca48aa8bd595933d7f5019ec4df788a4301ce012164ac9236fd5f5",
  "reader_scope_ns": [
    1696896060000000000,
    1703635260000000000
  ],
  "registered_fold_config_sha256": "7076e7f9b9b73848e61c8f8e51d44eb36685b791bdf732de31df66b68b92f264",
  "second_read_only_preflight": {
    "access": 0,
    "query_intent": 0
  },
  "semantic_fold_hash": "60e5a532f75b0cabe3517bdc47617c4885305f933510714fe5b17cc9a46c8ef6",
  "source_code_identity": {
    "repository_commit": "cb7a3ec9d542ed48b9f182acbaac330fc90b3177",
    "source_code_sha256": "593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c"
  },
  "source_identity": {
    "events": 135360,
    "file_sha256": "220c32aa7fca48aa8bd595933d7f5019ec4df788a4301ce012164ac9236fd5f5",
    "marker_sha256": "cb355367567fe6b153eff6b5880d5d865d620863542138ea658966192d6ac47f",
    "schema_version": "0.4.1",
    "snapshot_hash": "e273b372543285281ef57b10c5ec9fd19d4dba3bb5e33bf4a8d46ca7da3dc9bf"
  },
  "source_preparation_sha256": "c854bbeca16aa98c6493153d575278e6651de5e8c668c8193516470fb335fcd1"
}
```

## Append-only completion record — 2026-10-20

Completed at .artifacts/world_model/fresh-trxusdt-20231010-path-crps-paths16-20261020/ensemble/. The pre-registered all-seed gate failed: pooled E2-minus-E0 15-bar path log-RV CRPS was -0.007879 (95% CI [-0.022809,+0.013310], p 0.388306); none of the seeds passed the CI/Holm gate and seed 20261018 reversed direction. Return-CRPS guardrail showed no corrected significant harm; generated OHLCV/path validity was 100%. Test/Holdout remained unread. Aggregate and per-seed canonical digests plus checkpoint/config/forecast/risk/pooled file hashes verified. R2 remains unpromoted. Full results: [R2_TRXUSDT_PATH_CRPS_PATHS16_RESULTS_20261020.md](R2_TRXUSDT_PATH_CRPS_PATHS16_RESULTS_20261020.md).
