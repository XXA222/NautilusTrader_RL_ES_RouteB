# RouteB R4 AVAXUSDT hierarchy comparison — run 02 preregistration

Status: frozen before downloading or decoding the run-02 market rows.

## Reason for a new run

The first R4 Fold (read interval 2021-12-26 through 2023-01-12) was consumed and recorded failed after a generated path exceeded the OHLCV codec's physical wick limit. That Fold will not be retried. Run 02 uses a different, earlier read interval and retains the codec's reject-invalid-path rule; it does not clip or repair model outputs.

## One-shot disjoint Fold

- Instrument/source: AVAXUSDT-PERP.BINANCE, official Binance USD-M 1-minute kline monthly archives.
- Train score: `[2020-11-26T00:01:00Z, 2021-11-21T00:01:00Z)` (360 days); Train warmup begins `2020-11-24T00:01:00Z`.
- Validation score: `[2021-11-22T00:01:00Z, 2021-12-07T00:01:00Z)` (15 days); Validation warmup begins `2021-11-20T00:01:00Z`.
- Test score: `[2021-12-08T00:01:00Z, 2021-12-23T00:01:00Z)` (15 days). OHLCV values at and after Validation end are sealed to timestamp-only markers.
- Purge: 1 hour; embargo: 23 hours. Registered read scope: `[2020-11-24T00:01:00Z, 2021-12-07T00:01:00Z)`, which covers the Train warmup and all Validation context and does not overlap the consumed run-01 scope. The source begins earlier to retain the complete Validation warmup marker history.
- Source open-time interval: `[2020-11-20T00:00:00Z, 2021-12-23T00:01:00Z)`. Archive checksums and exact source hash are frozen before registered reads.
- A fresh read-only AVAX perpetual ledger overlap scan is mandatory immediately before reservation.

## Frozen training and path protocol

Model variants, sizes, training seeds, 1,000 updates, window length 1,024, burn-in 32, scaler rules, anchor grid, 15-minute horizon, scoring metrics, bootstrap, and decision gates match the v1 protocol. The machine-readable controls are in [`r4_avaxusdt_hierarchy_run02_protocol.json`](../../configs/routeb/r4_avaxusdt_hierarchy_run02_protocol.json).

At each fixed Validation anchor, each trained model uses the same ordered pool of 128 frozen path seeds and keeps the first 32 candidates that decode safely. It evaluates all 128 candidates, rejects paths that violate codec constraints without clipping, and fails if fewer than 32 safe paths remain at any anchor. The report records total and per-anchor rejection counts. This rule is fixed before the run and applies equally to E3, E3b, and E4.

The statistical comparison remains paired on the same 60 anchor timestamps. Empirical CRPS uses the first 32 safe paths per anchor. The paired moving-block bootstrap uses 12 consecutive anchors and 2,000 replicates; Holm correction applies to the six per-seed primary tests. Promotion remains research-only and requires the same strict all-seed gates as run 01.

## Data boundary and stop conditions

No Test or Holdout OHLCV payload is decoded. Any ledger overlap, source/checksum mismatch, incomplete warmup, missing anchor, insufficient safe candidate paths, optimizer failure, invalid checkpoint, scoring failure, or report/hash failure consumes this run-02 Fold and is recorded as failed. It cannot be retried on the same timestamps. This comparison is not economic or production certification.

## Frozen source and run-control digests

The ordered official archive names and per-archive SHA-256 values are in the source receipt. Before registered access, the aggregate controls were checked as follows:

- Protocol JSON SHA-256: `e4cd492657d0cb662d1308fdb6ba6d0bb86e468484610d835977d2c3070d89ab`
- Registered Fold config SHA-256: `f69e9fcca1ccbd2bb5d264df5b27b34898be10d7537df8578c69e0d58bfbbf22`
- Fold hash: `7959f22680bb5d6b5c0f76428f3b39201353e2238565e9a395847100f70e62a0`
- Official archive receipt SHA-256: `dbe99e2b95f690c740f9b0500b6094a65a34e0d079ffd08d807a4421a17a75d6`
- Curated source preparation SHA-256: `18efcda238d17e7a05d166d888f363da080d7a22041ba93db6479a0162863be4`
- Curated snapshot hash: `8bb41785cc3137e1055be0363f6ce121e89ab6d2f36b6c1b96e24af89f31f193`
- Marker manifest SHA-256: `10d2cc7a454bd9695ad09bc0eb6492653c6d25200b49176d7c29424610966590`

The runner records the preregistration document digest together with the source, Fold, protocol, runner, and code identities in the run receipt/report.
