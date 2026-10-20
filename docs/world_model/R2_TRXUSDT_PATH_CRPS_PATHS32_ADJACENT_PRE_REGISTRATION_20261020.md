# R2 TRXUSDT path-CRPS paths-32 adjacent transfer — pre-registration

Frozen before downloading or decoding this source interval. The adjacent scale-3/paths-16 transfer passed the 15-bar return-CRPS guardrail but missed the all-seed path-RV gate because seed 102 was inconclusive; E1 also outscored E2 on average. This transfer holds the coefficient at 3.0 and doubles only the number of sampled Train paths from 16 to 32, reducing Monte Carlo noise in the sampled path objective. It is one fixed transfer, not a sweep; this interval will not be used to select a different path count.

## Frozen recipe and decision rule

- E2: same RouteB RSSM Gaussian-prior model, five-channel OHLCV objective, Train-only equal-weight sampled free-prior path log-RV CRPS at horizons [15,30,60], coefficient=3.0, native risk loss scale=1.0, rollout risk conditioning disabled; training paths=32.
- Sole recipe change from the adjacent scale-3 experiment: `path_log_rv_crps_paths: 16 -> 32`. E1 remains the same fixed GRU comparator. Both candidates use 1,000 updates, batch 8, learning rate 0.0003, report interval 50; seeds [31,102,20261018].
- Validation: 180 anchors at 120-minute spacing, 32 forecast paths per seed, 96 pooled paths, 4-anchor chunks, 2,000 paired moving-block bootstrap replicates, block length 12.
- Select among frozen E0 zero-drift, Train-innovation EWMA and stationary block-bootstrap baselines by Validation mean 15-bar path log-RV CRPS.
- Full gate: every seed must have E2-minus-selected-E0 15-bar path log-RV CRPS 95% CI strictly below zero and Holm p<0.05 in the fixed 11-test family; no seed may show significant Holm-corrected E2 harm to 15-bar return CRPS; generated OHLCV/path structures must be valid. No Test/Holdout read, no economic claim, no automatic promotion.

## Frozen data scope

- Binance USD-M TRXUSDT 1m source-open interval [2025-01-22T00:00:00Z, 2025-04-26T00:00:00Z), 94 UTC days, 94 daily archives, 135,360 expected closed bars.
- 60/15/15 Fold, one-hour purge, 23-hour embargo. Train+Validation reader scope [2025-01-22T00:01:00Z, 2025-04-10T00:01:00Z).
- Train score [2025-01-24T00:01:00Z, 2025-03-25T00:01:00Z); Validation score [2025-03-26T00:01:00Z, 2025-04-10T00:01:00Z), warmup begins 2025-03-14T00:01:00Z. Test score [2025-04-11T00:01:00Z, 2025-04-26T00:01:00Z), warmup begins 2025-03-30T00:01:00Z.
- Read-only RouteB registry preflight on 2026-10-20 queried `.artifacts/research-access.sqlite` in SQLite read-only mode. It found zero overlapping `access` and zero overlapping `query_intent` rows for TRXUSDT `perpetual-market-v1` on the exact Train+Validation scope. The registered one-shot reader remains authoritative.
- Output root: `.artifacts/world_model/fresh-trxusdt-20250122-path-crps-paths32-scale3-20261020/`.
- Download official daily archives only after this freeze; verify sidecars and ZIP members; prepare and record source/Fold identities; repeat the read-only ledger query immediately before registration; exclude Test/Holdout.

## Frozen code and configurations

- RouteB source fingerprint `593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c`; commit `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`.
- Registered ensemble runner SHA-256 `aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9`.
- E1 config `configs/routeb/fresh_trxusdt_20250122_e1_pathcrps_paths16_ensemble.json`; SHA-256 `07c90c08b898314723e939ef46b12c77d54e3b95fd956a240e02ccbb4ded2fd8`.
- E2 config `configs/routeb/fresh_trxusdt_20250122_e2_pathcrps_paths32_scale3_ensemble.json`; SHA-256 `d7f6e5d9b42dca872fd6a14242f5272d9833dbcfb0191fc07e2010f312e13393`.
- Worker lock SHA-256 `ab99ac8f831bc06dc6e408cf91090b69aca413df8f38cfcd2f28b77a6ea18c1b`; isolated Python 3.12.10 / JAX 0.11.2 CUDA worker.

No source, scoring, or decision-rule changes are permitted after freeze. Append verified archive, source, Fold and final-preflight identities before registered access. If the reader rejects the reservation, do not access Train/Validation.

## Append-only verified source and Fold identities before registered access

- Downloaded 94 official Binance daily archives. The downloader validated each exchange `.CHECKSUM` sidecar; source preparation rechecked every archive checksum and ZIP member. Download manifest SHA-256: `fe7c7b74c9c618fbb3d60245944f65686817bae2c44adb1dd7d220032fdd211f`.
- Source preparation SHA-256: `7b7cf6ccc7efb788135cd659a99b0d7aab54e306804b32cbf19b3a4440560b11`; curated Parquet SHA-256: `7fa84bf0eeb722c991656a8889eebedc5065454dab2b076fcde0ae6f80a76150`.
- Dataset hash: `436e513c537ce7fee3a354908461cf255058caddecf2a7f3553817e885571f49`; marker SHA-256: `df3a5006c76d8ba99a43701f02d32048eee15f4ed25ff51d7979fe72c20f0ae5`; all 135,360 expected bars are continuous and pass source checks.
- Fold hash: `f7851bf711a46caabaee61d2cdffbf483bb0e422ea57cc48f8ab8ca428311083`; Fold config SHA-256: `6cbc5d5bd094ea26f32ffc2231a2d76458c75b1170cad8f7359562393b087dd1`.
- Final read-only RouteB ledger preflight on 2026-10-20 for Train+Validation `[1737504060000000000, 1744243260000000000)` ns returned access rows `0` and query_intent rows `0` for TRXUSDT `perpetual-market-v1`. The one-shot reader remains authoritative.

## Append-only configuration-cap correction before market access

The first path-32 runner invocation rejected the frozen configuration because `RSSMConfig` capped Train path samples at 16. It stopped during configuration validation, before the registered reader; the attempt folder contains only `runner_failure.json`. A read-only ledger recheck still found zero overlapping access and zero query_intent rows. RouteB now accepts 2–32 Train path samples, with a boundary regression assertion for 32 accepted and 33 rejected; the focused test passed (1 passed). The frozen candidate recipe and scope are unchanged. Updated RouteB source fingerprint: `f1ea982b8fa2019f1353762502a507ee2ee4a39b9350d3b01e58775f18df1aa2`; base commit remains `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`. Repeat the same one-shot registered reader with fresh output target `.artifacts/world_model/fresh-trxusdt-20250122-path-crps-paths32-scale3-20261020/ensemble/run-02/`.


## Append-only completed result — 2026-10-20

The preregistered paths-32 scale-3 transfer on fresh TRXUSDT [2025-01-22, 2025-04-26) completed with three seeds and 96 pooled validation paths. E2 path-derived 15-minute log-RV CRPS was 0.248922 versus selected E0 S2 0.234689 (difference +0.014233, 95% CI [-0.032769,+0.058097], p 0.535732); E1 scored 0.212234. Seed 31 was significantly adverse, seed 102 inconclusive, and seed 20261018 favorable but inconclusive; no seed passed the preregistered Holm/CI gate. The 15-minute return-CRPS guardrail passed all seeds. Data audit found 112,320 bars with zero missing/duplicate/availability errors; Test/Holdout stayed unread. Pooled forecast, per-seed report and risk forecast hashes matched. Generated RV mean/p95 and absolute-return persistence remained below realized. R2 stays unpromoted and R3-R6 gated. See [R2_TRXUSDT_PATH_CRPS_PATHS32_ADJACENT_RESULTS_20261020.md](R2_TRXUSDT_PATH_CRPS_PATHS32_ADJACENT_RESULTS_20261020.md).
