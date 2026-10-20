# R2 TRXUSDT path-CRPS scale-3 adjacent transfer — pre-registration

Frozen before downloading or decoding this source interval. The preceding scale-4 transfer passed the 15-bar path log-RV CRPS gate on all three seeds, but failed its return-CRPS guardrail for seed 102. This next run tests one intermediate coefficient on a fresh adjacent interval to assess whether path-RV skill can be retained without corrected 15-bar return-CRPS harm. This is one fixed transfer, not a sweep; no parameter will be selected from this interval.

## Frozen recipe and decision rule

- E2: same RouteB RSSM Gaussian-prior model and five-channel OHLCV objective, Train-only equal-weight sampled free-prior path log-RV CRPS at horizons [15,30,60], paths=16, coefficient=3.0, native risk loss scale=1.0, rollout risk conditioning disabled.
- The sole recipe change versus the adjacent scale-4 experiment is `path_log_rv_crps_scale: 4.0 -> 3.0`. E1 remains the same fixed GRU comparator.
- Both candidates: 1,000 updates, batch size 8, learning rate 0.0003, report interval 50; seeds [31,102,20261018].
- Validation: 180 anchors at 120-minute spacing, 32 paths per seed, 96 pooled paths, 4-anchor chunks, 2,000 paired moving-block bootstrap replicates, block length 12.
- Select among frozen E0 zero-drift, Train-innovation EWMA, and stationary block-bootstrap baselines by Validation mean 15-bar path log-RV CRPS.
- Full gate: every seed must have E2-minus-selected-E0 15-bar path log-RV CRPS 95% CI strictly below zero and Holm p<0.05 in the fixed 11-test family; no seed may show significant Holm-corrected E2 harm to 15-bar return CRPS; generated OHLCV/path structures must be valid. No Test/Holdout read, no economic claim, and no automatic promotion.

## Frozen data scope

- Binance USD-M TRXUSDT 1m source-open interval [2024-10-20T00:00:00Z, 2025-01-22T00:00:00Z), 94 UTC days, 94 daily archives, 135,360 expected closed bars.
- 60/15/15 Fold, one-hour purge, 23-hour embargo. Train+Validation reader scope [2024-10-20T00:01:00Z, 2025-01-06T00:01:00Z).
- Train score [2024-10-22T00:01:00Z, 2024-12-21T00:01:00Z); Validation score [2024-12-22T00:01:00Z, 2025-01-06T00:01:00Z), with warmup beginning 2024-12-10T00:01:00Z. Test score [2025-01-07T00:01:00Z, 2025-01-22T00:01:00Z), with warmup beginning 2024-12-26T00:01:00Z.
- Read-only RouteB registry preflight on 2026-10-20 queried `.artifacts/research-access.sqlite` in SQLite read-only mode. It found zero overlapping `access` rows and zero overlapping `query_intent` rows for TRXUSDT `perpetual-market-v1` on the exact Train+Validation scope above. The registered one-shot reader remains authoritative.
- Output root: `.artifacts/world_model/fresh-trxusdt-20241020-path-crps-scale3-20261020/`.
- Download official daily archives only after this freeze; validate Binance sidecars and ZIP members, prepare and record source/Fold identities, repeat the read-only ledger query immediately before the registered reader, and exclude Test/Holdout.

## Frozen code and configurations

- RouteB source fingerprint `593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c`; commit `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`.
- Registered ensemble runner SHA-256 `aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9`.
- E1 config `configs/routeb/fresh_trxusdt_20241020_e1_pathcrps_paths16_ensemble.json`; SHA-256 `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`.
- E2 config `configs/routeb/fresh_trxusdt_20241020_e2_pathcrps_paths16_scale3_ensemble.json`; SHA-256 `0416fb35c72895cfc903b02f48b8bf345b073604a599c4c4c32282d1a6875a99`.
- Worker lock SHA-256 `ab99ac8f831bc06dc6e408cf91090b69aca413df8f38cfcd2f28b77a6ea18c1b`; isolated Python 3.12.10 / JAX 0.11.2 CUDA worker.

No source, scoring, or decision-rule changes are permitted after freeze. Append verified archive, source, Fold, final-preflight and registered-run identities before scoring. If the reader rejects the reservation, do not read Train/Validation.

## Append-only verified source and Fold identities before registered access

- Downloaded 94 Binance official daily archives and verified each official `.CHECKSUM` sidecar and ZIP member. Download manifest SHA-256: `153a5cc9157fc29b3065626a1e2b332f913a964137dc23e43f12b66e6b0e3bc2`.
- Source preparation SHA-256: `abd7405ed4937d9fe9b27a5daeb5a508e8f90fdd087499aa1d6eb3b7c6c7237a`; curated Parquet SHA-256: `9b7239de7ab53a4cb6e168a5bdd855fc2c2f41f78a7aabfa3a5d76d3a77fbc70`.
- Dataset hash: `e5e90419c00cc2cb9ecb697c60b0a66cb000c7f138c9c29f3125f66e6a2e30e0`; marker SHA-256: `cb805d401e3e47125d6a5ffa8793a80a4cde1124e1bc20137f348f401ae022f5`; all 135,360 expected bars are continuous and pass source preparation checks.
- Fold hash: `e612cd789c39cbe3121185245aa5d9227b6861116042593b527a8e489354542d`; Fold config SHA-256: `786297faefab13554b6017af7da01dd0d36e9dd6193bfd5e2b91abd4e5a95b30`.
- Final read-only RouteB ledger preflight on 2026-10-20 for Train+Validation `[1729382460000000000, 1736121660000000000)` ns returned access rows `0` and query_intent rows `0` for TRXUSDT `perpetual-market-v1`. The one-shot registered reader remains authoritative.



## Append-only runner preflight correction before market access

The first runner invocation stopped before source registration because its output target was mistakenly pre-created; the runner requires its exact target to be absent while its parent exists. It wrote only `ensemble/runner_failure.json` and stopped before the registered reader, so no Train/Validation access or query intent was recorded. The failed preflight artifact is preserved. Repeat the read-only RouteB ledger query before resuming; use fresh output target `.artifacts/world_model/fresh-trxusdt-20241020-path-crps-scale3-20261020/ensemble/attempt-2/`.

## Append-only completion record — 2026-10-20

The fixed scale-3 transfer on fresh TRXUSDT [2024-10-20, 2025-01-22) completed on seeds 31, 102 and 20261018. E2 pooled path-derived 15-bar log-RV CRPS beat selected S2 (`0.233027` vs `0.283873`, difference `-0.050846`, 95% CI `[-0.099325,-0.017006]`, p `0.016992`), but seed 102 was inconclusive (difference `-0.007204`, CI `[-0.061248,+0.032775]`, Holm p `1.0`), so the all-seed primary gate failed. The 15-bar return-CRPS guardrail passed on all seeds. E1 scored better than E2 on average (`0.201562` vs `0.233027`); E2 volatility persistence remained low. Aggregate and per-seed digests, pooled forecast archives and risk forecast hashes verified with the RouteB Python 3.14 runtime; Test/Holdout were not decoded. R2 remains unpromoted and R3-R6 remain gated. See [R2_TRXUSDT_PATH_CRPS_SCALE3_ADJACENT_RESULTS_20261020.md](R2_TRXUSDT_PATH_CRPS_SCALE3_ADJACENT_RESULTS_20261020.md).
