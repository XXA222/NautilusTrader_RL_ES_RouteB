# RouteB R4 BNBUSDT single-scale versus hierarchy comparison

Status: frozen before downloading the BNBUSDT source archives.

## Purpose and prior attempt status

The ADA attempt on the 2023–2024 scope was terminally recorded as failed after the E2 CUDA worker ran out of memory during a 1,024-step backpropagation window. It will not be retried. Synthetic-only CUDA stress checks showed the same E2 model completes 2,000 updates with a 512-step window. This BNB Fold tests the frozen memory-bounded E2 recipe against the existing E3 hierarchy on new market data.

This remains a system-level comparison: E2 includes its established multi-horizon risk-head supervision, so it does not isolate architecture alone. No result can auto-promote a model or certify profitability.

## Fresh BNB Fold and source boundary

- Instrument: `BNBUSDT-PERP.BINANCE`, official Binance USD-M 1-minute klines.
- Train score: `[2023-08-01T00:01:00Z, 2024-07-26T00:01:00Z)`; 360 days.
- Validation score: `[2024-07-27T00:01:00Z, 2024-08-11T00:01:00Z)`; 15 days.
- Test score: `[2024-08-12T00:01:00Z, 2024-08-27T00:01:00Z)`; OHLCV values sealed to timestamp-only markers.
- Train warmup begins `2023-07-30T00:01:00Z`; Validation warmup begins `2024-07-25T00:01:00Z`.
- Purge is one hour; embargo is 23 hours. Registered read scope is `[2023-07-30T00:01:00Z, 2024-08-11T00:01:00Z)`.
- Official source open-time range is `[2023-07-25T00:00:00Z, 2024-08-27T00:00:00Z)` to retain both causal warmups and sealed Test marker coverage.
- The read-only BNB ledger/query-intent scan found no overlap with this proposed scope. The runner repeats the scan immediately before its one-time reservation.
- No Test or Holdout OHLCV payload is available in the curated source, worker, evaluator, or comparison report.

## Frozen model and compute protocol

Machine-readable controls are in [`r4_bnbusdt_e2_e3_baseline_protocol.json`](../../configs/routeb/r4_bnbusdt_e2_e3_baseline_protocol.json). Training seeds are 31, 102, and 20261018. Both models use batch size 1, burn-in 32, learning rate 0.0003, gradient clip 100, and the same Train-only minute scaler.

- E2 uses the existing single-scale RSSM and risk heads with a 512-minute window, 2,067 updates, and stride 256. At 480 scored targets per update, its total budget is 992,160 Train targets.
- E3 uses the previously exercised `e3_features` hierarchy with a 1,024-minute window and 1,000 updates. At 992 scored minute targets per update, its budget is 992,000 Train targets.
- E2’s shorter sequence limits reverse-mode memory. The total number of Train targets is matched within 0.02%; each run records its actual sequence/window configuration and device.

Both models use the same 60 Validation anchors six hours apart and a 15-minute horizon. Each anchor has a candidate budget of 128 paths and 32 accepted safe paths. E3 uses the frozen ordered path-seed range; E2 uses the worker’s frozen request seed and anchor-chunk rule with 4x over-generation. Candidate counts match, but the random draws are not paired by common random numbers. Invalid OHLCV paths are rejected without clipping. A seed fails if any anchor produces fewer than 32 safe paths.

Primary scoring is path log-realized-volatility CRPS, with return CRPS as the no-harm guardrail. Paired bootstrap uses 12-anchor blocks and 2,000 replicates. Holm correction covers the three per-seed primary comparisons. Promotion requires every seed’s primary 95% CI to stay below zero after Holm adjustment and no return-CRPS harm.

## Data and attempt policy

The official monthly archive checksums, sealed-source identity, derived Fold hash, protocol/config/preregistration digests, runner digest, and code identity are recorded before the registered Train/Validation read. Any failure after reservation consumes this BNB scope; it cannot be retried on these timestamps. The prior failed ADA access does not overlap this BNB instrument scope.

## Pre-run implementation amendment and source freeze

Before any registered Train/Validation read, the protocol added `forecast_context_length: 1024` to make the shared 60-anchor forecast context explicit; the E2 and E3 optimizer recipes and data split were unchanged. The BNB runner was updated to validate this preregistration's frozen-status wording and use that explicit field. The 14 official monthly archives passed their published SHA-256 checks. The curated source contains 574,560 continuous close events; 23,040 Test rows carry timestamp-only coverage markers and no OHLCV values.

- Protocol SHA-256: `96ae10b6a7cbdd8adb6a0c1442c49f204e8790b67b09c0507bc0ca855d38062d`.
- Archive receipt SHA-256: `ca576db6d8097272b6bf1935df945dd05c3c13559dd4606a01d82d56a8fa1d53`.
- Curated dataset hash: `73051c11f09d7731bbc2954b0539ab2d27d8b84b31cb5c47398c471e5a487c8b`.
- Curated marker SHA-256: `fb438cebe34e64ae49311418fd8b56058bcce5b0ae5f088993ea0ae165b08840`.
- Fold hash: `bc94a0e87818474fd96ddd7d90564131610e1b8cc9665e0cf2bbf6bca2f26292`.
- Fold config SHA-256: `d32f057a806cf8b72ac3ee7d58755c8507cf170520460723a1c69e4b30c557ce`.
- Source preparation SHA-256: `034f75fb399e5e03e14c35e5385a9776fe0ebc5923578ab9485c21baae5f2e47`.
