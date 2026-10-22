# R3 AVAXUSDT latent-redundancy regularizer pre-registration — run 02 — 2026-10-21

Frozen after the run-01 configuration failure and before downloading this disjoint source interval. Run-01 data and artifacts are not included in candidate fitting, scoring, or model selection.

## Hypothesis and frozen candidate protocol

This is the architecture-v3 R3 regularizer ablation; R4 is the hierarchical E3/E3b/E4 comparison. The R2 TRXUSDT persistence-0.35 all-seed Validation gate passed. R3 tests whether a weak nonzero projector/cross-correlation auxiliary loss improves representation diagnostics without statistically significant harm to free-prior path forecasts. The OHLCV forecast head remains active. The run is rejectable and never automatically promoted.

- Arms: lambda={0.0, 0.01, 0.05}, alpha=0.005, fixed-interval mask sampling every 60 latent steps, stopped-gradient target view, same projector graph/parameter count in all arms.
- E1 GRU and E2 RSSM: same new Train coverage, seeds [31, 102, 20261018], 1,000 optimizer updates, batch 8, learning rate 0.0003, gradient clip 100, report interval 50. E2 otherwise uses the passed TRXUSDT persistence-0.35 / risk-conditioning-strength-0.25 recipe.
- Each arm/seed uses the same forecast RNG seed 20261018 + model_seed, 32 paths, shared ordered Validation anchors at 120-minute spacing, 15-minute scored horizon, identical context and volatility windows.
- Primary: paired 12-anchor moving-block bootstrap of path-derived 15-minute log-RV CRPS, lambda 0.01 and 0.05 versus lambda zero, 2,000 replicates. Guardrail: paired 15-minute return CRPS on exactly the same paths, seeds and anchors.
- Report per-seed final-window rep_kl, latent/projected prediction and target variance, identity loss, off-diagonal redundancy loss, condition bound and batch stability. These are diagnostics, not an automatic acceptance gate. Keep lambda zero unless a nonzero arm has improved redundancy diagnostics and neither forecast metric has statistically significant paired harm.

## Independent Fold and causal warmup geometry

- Instrument/source: AVAXUSDT-PERP.BINANCE, Binance USD-M 1-minute klines.
- Train score: [2024-04-15T00:01:00Z, 2024-06-14T00:01:00Z) (60 UTC days), Train warmup starts at Train score start.
- Validation warmup: [2024-06-14T00:01:00Z, 2024-06-15T00:01:00Z) (the reserved one-day gap, 1,440 context bars). Validation score: [2024-06-15T00:01:00Z, 2024-06-30T00:01:00Z) (15 days).
- Test warmup: [2024-06-30T00:01:00Z, 2024-07-01T00:01:00Z); Test score: [2024-07-01T00:01:00Z, 2024-07-16T00:01:00Z) (15 days). Test warmup and Test score stay unread.
- Purge 1 hour + embargo 23 hours exactly fill each one-day gap. Reader scope ends exclusively at 2024-06-30T00:01:00Z; all post-Validation OHLCV values are replaced with timestamp-only coverage_marker records.
- Full Fold interval [2024-04-15T00:01:00Z, 2024-07-16T00:01:00Z). Read-only RouteB ledger preflight on 2026-10-21 found zero overlapping access and zero overlapping query_intent rows for this full interval; repeat immediately before registration.
- Expected source open interval [2024-04-15T00:00:00Z, 2024-07-16T00:00:00Z). Freeze source ZIP sidecar hashes, curated marker, dataset hash, feature hash, exact Fold hash and configuration digests after source preparation but before registered reading.
- Planned output: .artifacts/world_model/fresh-avaxusdt-20240415-r3-20261021/ensemble/run-02/.

## Stop conditions and limits

Any ledger overlap, footer crossing, missing validation warmup, source identity mismatch, invalid timestamp coverage, failed training, inconsistent anchors, hash mismatch or scoring error consumes this one-shot scope. Do not rerun it. No Test/Holdout read, economics certification, profitability claim, automatic promotion, or R4-R6 authorization follows from this Validation ablation.

## Append-only source and Fold freeze — 2026-10-21

- Official Binance monthly archive SHA-256:
  - AVAXUSDT-1m-2024-04.zip: 24a388e74d5be57cac665d6e8b3ff71002ec0cfd2c6ad95a4099e9f6ba0aa602 (1513266 bytes).
  - AVAXUSDT-1m-2024-05.zip: 5c3722a3631ecf7d8f326fa15a4a8c3f9c118b5ca7ee76e0fd72dfa8edd6308d (1518471 bytes).
  - AVAXUSDT-1m-2024-06.zip: 2d5faf93b680029c5230cf48f7fda9b6aba1bdc56fd9664bbe951d1a8f810406 (1459660 bytes).
  - AVAXUSDT-1m-2024-07.zip: fb9f3ace6b96ffe0b1848989ed4aa76f231e580f1b66aee2356387f4aa873ba5 (1527879 bytes).
- Source-preparation report SHA-256: 94e1b2c2627229f8c8bdb31713db42eb9ffda59b5ee6c5ea41aaaa0fb6b5611d.
- Curated Parquet SHA-256: 6254cf6b922fa4358a8011f152a564816fc1c83f1cdfd66eeabaf3b3c6922d5f; dataset hash: 242843a04e2680380d8adb11cd59fff513666e9997d979a1d3a0a8c001b3f6f7; marker SHA-256: 372a4487cb2c20f44dad68cfa3d633984f25bd244e8a3b603e2769edee2f558b.
- Source timestamp footer: 1713139260000000000 through 1721088000000000000; Train/Validation rowgroup boundary availability: 1719705661000000000.
- 23040 post-Validation rows are coverage markers with no OHLCV values; test_ohlcv_values_excluded is true.
- Fold hash: 86e9bc4136f09975db6a1c81712d6336ecd47c0bc13afd08967cfbf053f6e108. Validation warmup is exactly 1,440 bars (one day); both score gaps are exactly purge plus embargo.
- Fold JSON, E1 config, and E2 config SHA-256:
  - configs/routeb/r3_avaxusdt_20240415_60_15_15_fold.json: 39d72b4cf535e3ca4f132bc59064b57bb7ac6617ad824cec2998f43b9c1b0401.
  - configs/routeb/r3_avaxusdt_e1.json: 579a6b89ceb9d66d8c4c04aacb957bb584825cd8ca8b59cab1ffc915782f01d4.
  - configs/routeb/r3_avaxusdt_e2.json: bcb677c7f2070c47aaa92c86430ecf0fad431c2e9b0d532e0bd5b326eb0c4c08.
- Registered run-02 runner SHA-256: 04bd2f9e7fe605f5c762f40e0d0c90b16d87e8e6973f62aa7bd951d83e7c3437.
- Read-only full-Fold RouteB ledger preflight remained zero-overlap immediately before source preparation; the registered runner will repeat it immediately before reservation/read.
