# RouteB R4 DOGEUSDT matched-objective E2/E3 replication — run 02

Status: protocol and date scope frozen before downloading or decoding these market rows.

## Hypothesis and scope

The first matched-objective DOGEUSDT Fold produced mixed seed directions: two seeds favored E3 and one significantly favored E2. This fixed-protocol replication tests whether that seed disagreement persists in an earlier independent time interval. It does not change architecture, optimizer budgets, targets, or promotion criteria. It is not a parameter search and cannot certify profitability or auto-promote a model.

- Instrument: `DOGEUSDT-PERP.BINANCE`; official Binance USD-M Futures 1-minute klines.
- Train score: `[2023-08-01T00:01:00Z, 2024-07-26T00:01:00Z)` (360 days).
- Validation score: `[2024-07-27T00:01:00Z, 2024-08-11T00:01:00Z)` (15 days).
- Test score: `[2024-08-12T00:01:00Z, 2024-08-27T00:01:00Z)` (15 days). OHLCV at/after the Validation cutoff is timestamp-only coverage.
- Warmup starts: Train `2023-07-30T00:01:00Z`; Validation `2024-07-25T00:01:00Z`. Purge is 60 minutes; embargo is 1,380 minutes.
- Registered read scope: `[2023-07-30T00:01:00Z, 2024-08-11T00:01:00Z)`. Read-only ledger preflight found zero overlapping access and zero overlapping query-intent rows. This scope ends exactly where the later DOGEUSDT Fold's scope begins; the half-open windows do not overlap. The registered runner must repeat the check before reservation.
- Source open interval: `[2023-07-25T00:00:00Z, 2024-08-27T00:00:00Z)`, exactly 14 monthly archives. Only official Binance archives and published SHA-256 checksums are allowed.
- Test/Holdout OHLCV must not be decoded by source preparation, training, evaluation, or reporting.

## Frozen E2/E3 protocol

Controls are defined in [`r4_dogeusdt_e2_e3_run02_baseline_protocol.json`](../../configs/routeb/r4_dogeusdt_e2_e3_run02_baseline_protocol.json). This is a fixed-protocol temporal replication of run 01, not a new candidate sweep.

- Compare `R4_E2_SINGLE_SCALE` against `R4_E3_HIERARCHY`; ordered training seeds 31, 102, and 20261018.
- E2: 2,067 updates per seed, 512-minute windows, 480 targets/update (992,160 total). E3: 1,000 updates per seed, 1,024-minute windows, 992 targets/update (992,000 total). Both use batch size 1, burn-in 32, learning rate 0.0003, and grad clip 100.
- E3's multi-horizon risk auxiliary loss, floor, barrier quantile, and horizon weights derive directly from the frozen E2 settings. Do not independently retune.
- Use 60 shared Validation anchors six hours apart, horizon 15 minutes, 128 candidates and 32 accepted paths per anchor; 8-path inference chunks. Primary is path log-RV CRPS; return CRPS is a no-harm guardrail. Use 12-anchor bootstrap blocks, 2,000 replicates, and Holm correction over the three seed-level primary tests.
- Promotion requires every seed's primary CI strictly below zero, every Holm p below 0.05, and no return-CRPS harm. Otherwise do not promote E3. No economic certification or automatic promotion.

## One-time attempt and source policy

Date scope and protocol are frozen before downloads. Download receipts retain archive names, URLs, file sizes, and checksums. Source preparation seals values after the Validation cutoff. Final Fold/protocol/runner digests must verify before registered Train/Validation reads. After reservation, any failure consumes this interval permanently; no retry and no Test/Holdout inspection.

## Frozen controls before registered access

- Protocol JSON SHA-256: `0b6269a36d2b0f2e5e505768693fd16a0241de4bcefffab557f3288030d4308e`
- Registered Fold config SHA-256: `70b6edcf5a40fcbec30d9eef3c5a82bf6b95a06192c31ee30df86d58fb869ae3`
- Registered runner SHA-256: `508be5a97e3a6caef67b9ab1ee70af7ea5669f2d16922353b806952918c353f8`
- Expected Train targets: E2 992,160; E3 992,000. Test/Holdout payload decode: `false`.

## Sealed source identity before registered access

All 14 official monthly archives passed their published SHA-256 checks. Preparation retained 574,560 continuous minute events. The 23,040 rows after the Validation cutoff are timestamp-only markers; their OHLCV values were excluded. This source identity was frozen before the registered runner reads Train or Validation.

- Archive receipt SHA-256: `947fe6444e646ef07880ce0734edc46fcb624e82454b3647c578da2c8c9bf923`
- Curated dataset hash: `d8999c10b1bd93908351ccccb9b63f5cc7a3285195446d46928e6cfda1d7960d`
- Curated marker SHA-256: `2e220481f12e24b1e231aeae6b989bda1f3f2911b49591b5069b88662b40420a`
- Fold hash: `18041776e9fd778ea3a02cbd70469d1497149a35d6bfbe1c0a9af02e4ebbc7c7`
- Source preparation SHA-256: `f6a1fda1c4e98514f86a1435674f95e1845593f30e9615397a9f44d2aad8795f`
- Source event interval: `[2023-07-25T00:01:00Z, 2024-08-27T00:00:00Z]`; OHLCV payload at and after the Validation cutoff was excluded.
