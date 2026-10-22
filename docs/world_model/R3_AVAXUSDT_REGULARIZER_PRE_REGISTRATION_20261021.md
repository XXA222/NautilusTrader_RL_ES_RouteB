# R3 AVAXUSDT latent-redundancy regularizer pre-registration — 2026-10-21

Frozen before downloading or decoding this experiment's market payload. This is the v3 architecture's R3 regularizer ablation, not the R4 hierarchical E3/E3b/E4 comparison.

## Prerequisite and hypothesis

The registered TRXUSDT R2 persistence-0.35 experiment passed its preregistered all-seed Validation gate. R3 now tests the auxiliary projector/cross-correlation loss as a small, rejectable ablation. The hypothesis is that a weak nonzero weight can lower redundant projected representation structure without materially harming free-prior path forecasts. The OHLCV output head remains active in all arms. No result is promoted automatically.

## Frozen intervention

- Arms: `lambda={0.0, 0.01, 0.05}`; `alpha=0.005`; fixed-interval mask sampling every 60 latent steps; target projection view stop-gradient; identical projector graph and parameter count in every arm, including lambda zero.
- E1 GRU and E2 RSSM use the same Train examples, three seeds `[31, 102, 20261018]`, 1,000 optimizer updates, batch size 8, learning rate `0.0003`, gradient clip `100`, and report interval 50. E2 otherwise uses the passed R2 recipe from the 2026-10-21 TRX persistence-0.35 run.
- Each seed/arm uses identical forecast RNG seed (`20261018 + model_seed`), 32 paths, identical ordered Validation anchors at 120-minute spacing, 15-minute maximum scored horizon, and identical context/volatility protocol.
- Primary forecast comparison: paired moving-block bootstrap of path-derived 15-minute log-RV CRPS, lambda 0.01 and 0.05 each versus lambda zero. Guardrail: paired 15-minute return CRPS on exactly the same anchors, seeds and paths. Freeze 12-anchor blocks and 2,000 bootstrap replicates.
- Representation evidence: report per-seed final-window `rep_kl`, latent/projected prediction and target variance, identity loss, off-diagonal redundancy loss, condition bound, and batch stability. These diagnostics are descriptive; the regularizer is retained only if the representation evidence improves and neither forecast metric shows a preregistered statistically significant harm. Otherwise lambda zero remains the default. This experiment does not authorize R4-R6.

## Fresh independent Fold and access boundary

- Instrument: `AVAXUSDT-PERP.BINANCE`; source: official Binance USD-M monthly 1-minute klines.
- Train score: `[2024-01-01T00:01:00Z, 2024-03-01T00:01:00Z)` (60 UTC days).
- Validation score: `[2024-03-02T00:01:00Z, 2024-03-17T00:01:00Z)` (15 days).
- Test score: `[2024-03-18T00:01:00Z, 2024-04-02T00:01:00Z)` (15 days). One-day Train/Validation and Validation/Test gaps are reserved. Test/Holdout remains unread.
- Registered reader may decode only the Fold's Train warmup/score and Validation rows, with exclusive end `2024-03-17T00:01:00Z`. The runner verifies source footer coverage for the Fold but does not decode Test rows.
- Read-only ledger preflight on 2026-10-21 found zero overlapping `access` and `query_intent` rows for AVAXUSDT over Train/Validation, Test, and the full Fold. Repeat the exact preflight immediately before registration/read.
- Source archives, sidecar checksums, curated marker, dataset hash, feature hash, and final Fold manifest will be appended after archive verification and before the registered reader is invoked.
- Planned output root: `.artifacts/world_model/fresh-avaxusdt-20240101-r3-20261021/`.

## Frozen decision and limits

The registered report is Validation evidence only. No Test/Holdout read, economics certification, profitability claim, automatic promotion, or authorization for hierarchy or trading work follows from this ablation. Any source/footer, access-boundary, causal-label, common-anchor, hash, training, or bootstrap failure terminates the one-shot interval; do not retry the consumed scope.

## Append-only source and Fold freeze — 2026-10-21

- Official Binance USD-M monthly archives passed their official companion SHA-256 checks:
- AVAXUSDT-1m-2024-01.zip: 1e8a7fd2013d83971ad3b6892128c6fedfcace2e170d6bbeab560abce9ba59ef (1577236 bytes).
- AVAXUSDT-1m-2024-02.zip: f66958652e77169ecc29f879556a185de70278b2d20d321e650cdd73b438d8a5 (1452186 bytes).
- AVAXUSDT-1m-2024-03.zip: 886fc94b45ace4c19596aa245837425043cb8d121201d112bdd5d3b77f33341c (1617408 bytes).
- AVAXUSDT-1m-2024-04.zip: 24a388e74d5be57cac665d6e8b3ff71002ec0cfd2c6ad95a4099e9f6ba0aa602 (1513266 bytes).
- Curated source-preparation report SHA-256: e68429deb1280fe49e8c2cb64f7db3f09bcb5474c86052f756c76ac2b0f261bd.
- Curated Parquet SHA-256: 4715dada40ebec5c933098cc95650856884b8c42328c67040d739e4bf6ce4d91; immutable dataset hash: 69447e3adb18527a8fb18a81f31da82170fa07c4bb49e02fb6ebe920367985f4; marker SHA-256: b40ec3569f56a39e675f7aa384f57f2ba7653e9240aa82b82611efe8db363e35.
- Prepared timestamp bounds are 1704067260000000000 through 1712016000000000000. The row-group boundary at 1710633661000000000 separates Train/Validation from all later rows.
- 23040 post-Validation rows are timestamp-only coverage_marker events. The preparation report confirms test_ohlcv_values_excluded=true; no Test/embargo OHLCV values are retained.
- Fold score windows are Train [2024-01-01T00:01:00Z,2024-03-01T00:01:00Z), Validation [2024-03-02T00:01:00Z,2024-03-17T00:01:00Z), and unread Test [2024-03-18T00:01:00Z,2024-04-02T00:01:00Z). Purge is 1 hour and embargo 23 hours.
- Frozen Fold hash: 2c57dfaf4872ecd6818bd0761697fb10940bf74eab6bf224fd04940c7068b204; dataset hash and feature hash are recorded in the Fold JSON.
- Frozen candidate configs and SHA-256:
- configs/routeb/r3_avaxusdt_20240101_60_15_15_fold.json: 4267935d237772954c10d17daa5a3a953e7142a6bc010d3fb4b9b19fe71f7a4c.
- configs/routeb/r3_avaxusdt_e1.json: 579a6b89ceb9d66d8c4c04aacb957bb584825cd8ca8b59cab1ffc915782f01d4.
- configs/routeb/r3_avaxusdt_e2.json: bcb677c7f2070c47aaa92c86430ecf0fad431c2e9b0d532e0bd5b326eb0c4c08.
- Registered R3 runner SHA-256: e3de8728c2c7194f3cb42a51a062de5d56ae95126a31be2c6f37068e8df0a8a3.
- The read-only RouteB access/query-intent preflight was repeated immediately before source preparation and remained zero-overlap across the complete Fold. Repeat it again immediately before registered reader execution.

## Append-only run-01 termination — 2026-10-21

The run-01 Train/Validation scope [2024-01-01T00:01:00Z,2024-03-17T00:01:00Z) was formally reserved once and is consumed. Seed 31 completed 1,000 E1/E2 updates and produced its validation paths, but run_fold_experiment stopped in S0/S1/S2 scoring because the frozen Fold incorrectly set validation.warmup_start_ns equal to validation.score_start_ns; the baseline requires at least 61 context bars. All nine candidate attempts were terminally failed, no aggregate result/completion receipt was issued, and this interval must not be retried or used for R3 metrics. The failure record is under .artifacts/world_model/fresh-avaxusdt-20240101-r3-20261021/ensemble/run-01/runner_failure.json. The next pre-registration fixes the Fold geometry by explicitly assigning the one-day Train/Validation and Validation/Test gaps as warmup context where permitted.


## Append-only ledger reconciliation — 2026-10-21

After the failure audit, all nine run-01 candidate attempts were verified terminally failed. The exact runner_failure.json SHA-256 was recorded as a failed development result, and research_failure_receipt.json now binds that digest to the consumed run-01 access key. No successful completion result exists for that Fold.
