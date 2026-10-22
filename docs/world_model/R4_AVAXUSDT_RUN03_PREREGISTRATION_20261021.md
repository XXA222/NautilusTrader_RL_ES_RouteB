# RouteB R4 AVAXUSDT hierarchy comparison — run 03 preregistration

Status: frozen before downloading or decoding the run-03 market rows.

## Reason and disjointness

Run 01 failed on an unsafe generated OHLCV wick. Run 02 failed after E3 completed because a 128-seed JAX forecast batch exceeded the available GPU memory. Both reads were registered as failed and their timestamps are consumed. Run 03 uses a separate later Fold and bounded inference batches. Its intended read scope ends before the next known registered AVAX access interval.

## One-shot Fold and source bounds

- Instrument/source: AVAXUSDT-PERP.BINANCE, official Binance USD-M 1-minute klines.
- Train score: `[2024-08-01T00:01:00Z, 2025-07-27T00:01:00Z)` (360 days); Train warmup begins `2024-07-30T00:01:00Z`.
- Validation score: `[2025-07-28T00:01:00Z, 2025-08-12T00:01:00Z)` (15 days), with two-day causal context.
- Test score: `[2025-08-13T00:01:00Z, 2025-08-28T00:01:00Z)` (15 days). Its OHLCV payloads are sealed to timestamp-only markers.
- Purge: 1 hour; embargo: 23 hours. Registered scope: `[2024-07-30T00:01:00Z, 2025-08-12T00:01:00Z)`.
- Official source open-time interval: `[2024-07-30T00:00:00Z, 2025-08-28T00:00:00Z)`.
- The runner repeats a read-only ledger overlap check immediately before reservation. Any overlap stops the attempt.

## Frozen training and path protocol

Use the exact run-02 protocol JSON: E3/E3b/E4, the same three training seeds, 1,000 optimizer updates, 1,024-minute windows, burn-in 32, 60 fixed Validation anchors, 32 accepted paths per anchor, 128 ordered candidate path seeds, 15-minute horizon, 12-anchor block bootstrap, and 2,000 replicates. At each anchor, all 128 forecast candidates are evaluated in batches of eight to bound device memory. Unsafe paths are rejected without clipping; the first 32 safe paths are retained. Fewer than 32 valid candidates at any anchor fails the Fold.

Metrics, paired comparisons, Holm correction, and research-only decision gates match run 02. Test and Holdout values are not read or decoded. No automatic promotion, profitability certification, or production readiness claim follows from this Fold.

## Frozen digests

- Protocol JSON SHA-256: `e4cd492657d0cb662d1308fdb6ba6d0bb86e468484610d835977d2c3070d89ab`
- Fold config SHA-256: `a4caf1e9f6cad5024e81850d0a0d194feca5e1f9ff6482eee85869b70aed51ae`
- Fold hash: `3ef84b2879ad7dac3e76ce315cb44672bfb213239ef2076fbdf15e25f5e19a4d`
- Official archive receipt SHA-256: `49353e18c904b9706d5b0200d7a7851d2e4b62aa09a27bc7212f9989baa46cb9`
- Curated source preparation SHA-256: `7b3f12e776a3ee97fb5586a24aeff45b0279da1480e83670863ba01476fddb06`
- Curated snapshot hash: `c2820b9f7a200791accf50833e1196a697d830f09c6de4839c8e20634c27880d`
- Marker manifest SHA-256: `7064051e31617ddcbec1bfa3cd6c0101be3ca79822945560e5a9b1292b49049f`
- Runner SHA-256: `651c0546800e29596e2078f21b8fdefcec30fa7537d6b819b348eb702f30de80`
- Chunked forecast implementation SHA-256: `39896efbd6f9e2f146866b0bcdb5b8f81aa6ad95d0bba776e77433d9819ef218`

The registered receipt records this document's final digest, the source and config digests, code identity, and the runtime report hashes.
