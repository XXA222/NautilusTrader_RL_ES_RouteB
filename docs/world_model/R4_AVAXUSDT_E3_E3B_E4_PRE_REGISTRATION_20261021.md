# RouteB R4 AVAXUSDT hierarchy comparison — preregistration

Status: frozen candidate protocol; no R4 market rows have been read.

## Question and decision rule

Compare architecture-v3 candidates E3 (minute RSSM with closed H/D features), E3b (independent M/H/D RSSMs with token fusion), and E4 (conditional H→M and D→H/M RSSM transitions). This is a research comparison, not a production promotion or economic test. E4 is retained for follow-up only if its 15-minute path log-RV CRPS improves over both simpler variants on every training seed after the paired block-bootstrap gate, with no statistically significant return-CRPS harm. Otherwise prefer E3 as the simpler research path; do not promote any candidate from this single Fold.

## One-shot disjoint Fold

- Instrument/source: AVAXUSDT-PERP.BINANCE, Binance USD-M 1-minute kline archives.
- Train score: `[2022-01-01T00:01:00Z, 2022-12-27T00:01:00Z)` (360 days); Train causal warmup begins `2021-12-30T00:01:00Z`.
- Validation score: `[2022-12-28T00:01:00Z, 2023-01-12T00:01:00Z)` (15 days); Validation warmup begins `2022-12-26T00:01:00Z` and is read-only.
- Test score: `[2023-01-13T00:01:00Z, 2023-01-28T00:01:00Z)` (15 days). The Fold source will replace every OHLCV payload at or after Validation end with timestamp-only coverage markers. The R4 reader ends at Validation end and does not decode Test or Holdout values.
- Purge is 1 hour and embargo 23 hours at each transition. The common registered read scope is `[2022-12-26T00:01:00Z, 2023-01-12T00:01:00Z)`, covering the earlier Validation context start and excluding Test score.
- Official source open-time interval: `[2021-12-26T00:00:00Z, 2023-01-28T00:00:00Z)`. This supplies the complete fold plus the two-day Validation warmup. Archive ZIP and official checksum sidecars are recorded before curated preparation.
- A fresh RouteB ledger overlap scan is mandatory immediately before reservation. The scan is read-only and includes every existing AVAX perpetual access and query-intent row.

## Frozen models and training budget

The exact machine-readable controls are in [`r4_avaxusdt_hierarchy_protocol.json`](../../configs/routeb/r4_avaxusdt_hierarchy_protocol.json). E3/E3b/E4 use the same minute/hour/day RSSM sizes, same three ordered training seeds, same 1,000 optimizer updates, same 1,024-minute windows, 32-window burn-in, same visible context limits, learning rate, gradient clip, source rows, scaler moments, and checkpoint protocol. One window is sampled per update. Per-run reports retain sampled start digests, unique sampled starts, and the non-overlapping scored-window capacity; those counts are sampling diagnostics, not claims of independent market observations.

Each candidate/seed has its own registered ledger attempt. No early stopping or Validation model selection is allowed. Only the E3, E3b, and E4 variant changes between candidate configurations; the R3 auxiliary loss stays at zero.

## Validation scoring

- Fixed common anchors occur every 360 minutes, beginning at Validation score start; this yields 60 anchors. Each anchor uses the same 1,024-minute causal context and a 15-minute static-slow-context prior horizon. The fixed grid avoids future hour/day closes during the rollout.
- Each training seed emits 32 paths at each anchor using the frozen shared forecast-seed list. Unsafe decoded OHLCV paths are rejected; this protocol requires all 32 paths to be valid at every scored anchor or the registered run fails.
- Primary: empirical CRPS of 15-minute path-derived log realized volatility. Comparisons are E4 minus E3 and E4 minus E3b, paired on all common anchors. Guardrail: empirical CRPS of 15-minute close-to-close return on the same paths.
- Paired moving-block bootstrap uses 12 consecutive anchors and 2,000 replicates. Holm correction is applied to the six per-seed primary E4-versus-E3/E3b tests. The all-seed E4 gate requires every corrected per-seed primary test to pass with a 95% interval strictly below zero and no significant return-CRPS harm.
- Report per-seed and pooled scores, generated-path validity, parameter counts, optimizer updates, unique sampled windows, non-overlapping window capacity, device, training wall time, configuration/source/fold/runner hashes, and completion receipts.

## Stop conditions and limits

Any ledger overlap, source/checksum/footer mismatch, incomplete Validation warmup, invalid/missing anchor, Test payload decode, unsafe path count, failed optimizer update, model/checkpoint identity mismatch, inconsistent common paths, or scoring/hash failure consumes this one-shot Fold. A failed scope is recorded as failed and must not be retried on the same timestamps. This run cannot certify profitability, economics, native execution, or production readiness; it does not authorize future Test/Holdout reads.
