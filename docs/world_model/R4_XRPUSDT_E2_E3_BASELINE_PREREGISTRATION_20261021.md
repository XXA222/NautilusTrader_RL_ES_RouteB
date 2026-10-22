# RouteB R4 XRPUSDT single-scale versus hierarchy comparison

Status: protocol and date scope frozen before downloading or decoding these market rows.

## Purpose and prior attempt status

The ADA attempt on the 2023–2024 scope was terminally recorded as failed after the E2 CUDA worker ran out of memory during a 1,024-step backpropagation window. It will not be retried. Synthetic-only CUDA stress checks showed the same E2 model completes 2,000 updates with a 512-step window. The BNB cross-market comparison completed all six runs. Two seeds favored E3 on path log-RV CRPS and one seed was inconclusive with a positive point estimate, so the strict gate failed and E2 remains the default. This XRP Fold is a fresh replication using the unchanged frozen E2/E3 recipes.

This remains a system-level comparison: E2 includes its established multi-horizon risk-head supervision, so it does not isolate architecture alone. No result can auto-promote a model or certify profitability.

## Fresh XRP Fold and source boundary

- Instrument: `XRPUSDT-PERP.BINANCE`, official Binance USD-M 1-minute klines.
- Train score: `[2023-08-01T00:01:00Z, 2024-07-26T00:01:00Z)`; 360 days.
- Validation score: `[2024-07-27T00:01:00Z, 2024-08-11T00:01:00Z)`; 15 days.
- Test score: `[2024-08-12T00:01:00Z, 2024-08-27T00:01:00Z)`; OHLCV values sealed to timestamp-only markers.
- Train warmup begins `2023-07-30T00:01:00Z`; Validation warmup begins `2024-07-25T00:01:00Z`.
- Purge is one hour; embargo is 23 hours. Registered read scope is `[2023-07-30T00:01:00Z, 2024-08-11T00:01:00Z)`.
- Official source open-time range is `[2023-07-25T00:00:00Z, 2024-08-27T00:00:00Z)` to retain both causal warmups and sealed Test marker coverage.
- The read-only XRP ledger/query-intent scan found no overlap with this proposed scope. The runner repeats the scan immediately before its one-time reservation.
- No Test or Holdout OHLCV payload is available in the curated source, worker, evaluator, or comparison report.

## Frozen model and compute protocol

Machine-readable controls are in [`r4_xrpusdt_e2_e3_baseline_protocol.json`](../../configs/routeb/r4_xrpusdt_e2_e3_baseline_protocol.json). Training seeds are 31, 102, and 20261018. Both models use batch size 1, burn-in 32, learning rate 0.0003, gradient clip 100, and the same Train-only minute scaler.

- E2 uses the existing single-scale RSSM and risk heads with a 512-minute window, 2,067 updates, and stride 256. At 480 scored targets per update, its total budget is 992,160 Train targets.
- E3 uses the previously exercised `e3_features` hierarchy with a 1,024-minute window and 1,000 updates. At 992 scored minute targets per update, its budget is 992,000 Train targets.
- E2’s shorter sequence limits reverse-mode memory. The total number of Train targets is matched within 0.02%; each run records its actual sequence/window configuration and device.

Both models use the same 60 Validation anchors six hours apart and a 15-minute horizon. Each anchor has a candidate budget of 128 paths and 32 accepted safe paths. E3 uses the frozen ordered path-seed range; E2 uses the worker’s frozen request seed and anchor-chunk rule with 4x over-generation. Candidate counts match, but the random draws are not paired by common random numbers. Invalid OHLCV paths are rejected without clipping. A seed fails if any anchor produces fewer than 32 safe paths.

Primary scoring is path log-realized-volatility CRPS, with return CRPS as the no-harm guardrail. Paired bootstrap uses 12-anchor blocks and 2,000 replicates. Holm correction covers the three per-seed primary comparisons. Promotion requires every seed’s primary 95% CI to stay below zero after Holm adjustment and no return-CRPS harm.

## Data and attempt policy

The official monthly archive checksums, sealed-source identity, derived Fold hash, protocol/config/preregistration digests, runner digest, and code identity are recorded before the registered Train/Validation read. Any failure after reservation consumes this XRP scope; it cannot be retried on these timestamps. The prior ADA failure and BNB comparison use separate instruments.


## XRP replication scope

The candidate interval is checked against the RouteB access ledger immediately before one-time reservation. The protocol uses the same dates, compute budgets, Validation anchors, scoring, and Test sealing as the completed BNB comparison. This remains a system-level comparison because E2 includes its multi-horizon risk-head supervision. No market archives are downloaded or decoded until this protocol and this preregistration are saved.

## Frozen source identities before registered access

The 14 official monthly archives passed their published SHA-256 checks. The curated source has 574,560 continuous close events; 23,040 Test rows contain timestamp-only coverage markers.

- Protocol SHA-256: `1e1b95c5d93a69342e8fa6597b6ec68198c66bb3344fc1db31a6dd996e6ccd13`.
- Archive receipt SHA-256: `ac62bd50fb37efc519b6b12360117e3efa28a9f142d868177045eef6da2bcc95`.
- Curated dataset hash: `46cef5c531cd205fc92a79988b2b32280ab189d48f808210698e710089531929`.
- Curated marker SHA-256: `7285b34353aaea7dcc6b7374dbdc1206d14db1befaa673d2bcc99fc08161ab15`.
- Fold hash: `6086817db18eb8a1b9de4204997838a2d03fb437b2ee4f9d509f7c35147dca73`.
- Fold config SHA-256: `0540721ec24975beffafce1f4deb7c46d3b1cd3b07bf33b5aa81a774cc33ee01`.
- Source preparation SHA-256: `0980bab4353611e99db3495db475d6a027c0c7e8e09724e6d508ff2eb70b94a2`.

