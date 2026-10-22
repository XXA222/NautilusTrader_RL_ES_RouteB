# RouteB R4 ADAUSDT single-scale versus hierarchy comparison

Status: protocol and date scope frozen before downloading or decoding these market rows.

## Purpose

Run a cross-instrument replication of the AVAX hierarchy result. Compare the existing E2 single-scale RSSM system with the E3 three-timescale feature hierarchy. E2 includes its existing multi-horizon risk-head supervision, so this is a system-level comparison; it does not isolate architecture alone. The result is research evidence and cannot promote a model automatically.

## Fresh source and Fold

- Instrument: `ADAUSDT-PERP.BINANCE`, official Binance USD-M 1-minute klines.
- Train score: `[2023-08-01T00:01:00Z, 2024-07-26T00:01:00Z)`; 360 days.
- Validation score: `[2024-07-27T00:01:00Z, 2024-08-11T00:01:00Z)`; 15 days.
- Test score: `[2024-08-12T00:01:00Z, 2024-08-27T00:01:00Z)`; 15 days, OHLCV sealed to timestamp-only markers.
- Train and Validation context use two warmup days. Train warmup begins `2023-07-30T00:01:00Z`; Validation warmup begins `2024-07-25T00:01:00Z`. Purge is one hour and embargo is 23 hours.
- Registered data-read scope is `[2023-07-30T00:01:00Z, 2024-08-11T00:01:00Z)`. The source open-time range is `[2023-07-25T00:00:00Z, 2024-08-27T00:00:00Z)` so it retains both causal warmups and complete timestamp coverage for sealed Test markers.
- The existing ADAUSDT access in the RouteB ledger begins in September 2025. A fresh read-only access and query-intent overlap check is mandatory immediately before reservation; any overlap stops the run.
- Only validation OHLCV before the frozen Validation end may be decoded. No Test or Holdout OHLCV value may enter the model worker or evaluator.

## Frozen comparison protocol

The machine-readable controls are in [`r4_adausdt_e2_e3_baseline_protocol.json`](../../configs/routeb/r4_adausdt_e2_e3_baseline_protocol.json). E2 and E3 use seeds 31, 102, and 20261018, 1,000 updates, 1,024-minute windows, 32-minute burn-in, batch size 1, and identical optimizer settings. E2 sequence windows use stride 256 and Train-only scaler/risk targets. E3 follows the already exercised R4 `e3_features` architecture and bounded forecast batching.

Both models use the same 60 Validation anchors, each six hours apart, the same 15-minute target horizon, and a budget of 128 forecast candidates per anchor. E3 uses the frozen ordered seed range; E2 uses the worker's frozen request-seed and anchor-chunk rule with 4x over-generation. Candidate counts match, but random draws are not common-random-number paired. Invalid OHLCV paths are rejected without clipping; the first 32 valid candidates are scored, and a seed fails if any anchor has fewer than 32 valid paths. Primary scoring is path log-realized-volatility CRPS, with return CRPS as the no-harm guardrail. Paired bootstrap uses 12-anchor blocks and 2,000 replicates. Holm correction covers the three per-seed primary comparisons.

This is a cross-instrument replication and a system-level comparison. If the strict all-seed primary gate or the return guardrail fails, E3 remains unpromoted. The Test interval remains sealed for any future preregistered work.

## Frozen controls

- Protocol JSON: `configs/routeb/r4_adausdt_e2_e3_baseline_protocol.json`.
- Dates, seeds, model variants, optimizer budgets, anchor grid, metrics, and decision gates are frozen before downloading the source archives.
- Archive checksums, sealed-source identities, derived Fold hash, protocol digest, source-preparation digest, runner digest, and code identity will be recorded before registered Train/Validation access.
- A registered attempt is terminal. Failures consume this scope and cannot be retried on these timestamps.

## Sealed-source identities frozen before registered access

- Protocol JSON SHA-256: `74484987959f3b4bd72573a99ecd4fdbee5cbc421432a382c49013d3f4dbd7b`
- Registered Fold config SHA-256: `bcfb63cce06850f90ccf7010d8085ef5f292ba6bf7b0d49c4ae26d4d8d68bca9`
- Fold hash: `bbadee4333ccc9bc3a43b032032714923d3a4bf2493516f668d10eb0772c4d56`
- Dataset hash: `4cd497b1eb4ae773cb8ebbf2f866941c31afe111f41d0e2657d5e325e51666bb`
- Marker SHA-256: `e00300266d9ffb47d42074746c70634edb0aa9e8afec485888eff6821f44f088`
- Official archive receipt SHA-256: `985c096b5159b2823fddcbc07bcaf0155106c10dea3c732b91cb8d1ba6db5451`
- Sealed source-preparation SHA-256: `a8f0f923abf6f42183c24b7ddaabb7c976cc97d345aa316259767834f4176e1a`
- Registered runner SHA-256: `dcaf3096e10ccd8c92ab8c1145dd83385dda87d16e7d8cdc4d6c3899120752fd`
- Test coverage consists of 23,040 timestamp-only markers; no Test OHLCV payload is stored in the curated source.

## Boundary correction before registered access

The first runner invocation rejected the Fold before ledger registration or market-row access because Validation warmup had been assigned the 2023 year instead of 2024. A second invocation stopped during the worker runtime preflight because symlink resolution selected the base Python instead of the isolated JAX environment. Both stopped before ledger registration and market-row access. The frozen Train, Validation, and Test scoring windows, model recipes, seeds, metrics, and decision gates did not change. The Fold correction updates only the Validation context boundary and resulting Fold/read-scope identity; the worker correction preserves the RouteB venv interpreter path. The official archives and sealed curated source had already been prepared; no registered Train/Validation replay, model score, or Test OHLCV value had been read. The corrected hashes above are final before the registered read.
