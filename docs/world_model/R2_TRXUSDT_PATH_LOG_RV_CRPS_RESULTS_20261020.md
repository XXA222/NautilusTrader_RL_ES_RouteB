# R2 TRXUSDT Train-Time Path Log-RV CRPS Results

Status: COMPLETED; R2 PROMOTION GATE FAILED. Test/Holdout remained unread.

## Frozen protocol and integrity

- 94 official Binance USD-M TRXUSDT 1-minute archives for [2023-01-01, 2023-04-05), 135,360 bars; all Binance checksum sidecars and ZIP integrity checks passed.
- Registered 60/15/15 Fold; Train+Validation read scope [2023-01-01T00:01:00Z, 2023-03-20T00:01:00Z). RouteB read-only ledger preflight: zero overlapping access rows and zero overlapping query_intent rows before the one-shot registered reader.
- Dataset hash ebecf510039357bd2607f2a7b63489ec95af9cf854bac1fbe55469aa87878c29; Fold hash 98e80b4d1490f440d8968c9fa6a0e9a8f1a3b7668cdbf86930ec9b658b353570.
- Three seeds [31, 102, 20261018], 1,000 updates per candidate, batch 8, 180 Validation anchors, 32 paths per seed, 96 pooled paths, 2,000 paired block bootstrap replicates.
- Aggregate report canonical digest: 3f866abf92b2bec50bbfd7d5b222fb9f71096e7913d5519f598e5b770b9c6664; aggregate file SHA-256: bd7f33f53939c293cb218ac89d3c713ae6c97c76ad557596e2ed61620e49b868. Predictive digest: ffa4661ca8166f292f064274e5df0d3cc5ceaf5cb8a5cfb6ab24babc7ddbe89e.
- Pooled E1 artifact SHA-256 e2a32f1d299cbe63df42109376db8e53aea5fa7d5c6f29ae3c10c6860989072a; E2 artifact SHA-256 657fc5fb985ce08a497f0a6028e4913027e44a7797a9ad83959d1bbe0c6ce8cc. Independent digest verification passed for aggregate and predictive reports, all three per-seed reports/checkpoints and both pooled forecast artifacts. Aggregate and per-seed Test/Holdout decoded flags were false.

## Predictive result

The Validation-selected statistical baseline was E0 S1 Train-innovation EWMA. Mean 15-minute path-derived log-RV CRPS was 0.283269 for E0 S1, 0.264673 for E2, and 0.238129 for E1. Aggregate E2-minus-E0 S1 was -0.018597, paired 95% CI [-0.044814, +0.007603], p=0.169915; the interval crossed zero. All three seed intervals crossed zero and Holm-adjusted p-values were 0.214893, 0.730635 and 0.226887. The registered all-seed gate therefore failed, and E2 also did not match the E1 path score.

The 15-minute return-CRPS guardrail showed no significant E2 harm: aggregate E2-minus-E0 S1 +0.00004805, 95% CI [-0.00004228, +0.00018748], Holm p=1.0. Generated OHLCV bar and path validity were both 100%. Generated 15-minute rolling-RV mean/p95 (0.000544/0.000955) and absolute-return lag-1 autocorrelation (0.0133) remained below realized (0.000761/0.001590 and 0.0585).

## Decision

The additive Train-only sampled path log-RV CRPS objective is functional and structurally valid, but this single frozen transfer does not establish statistically reliable improvement. Do not promote R2 and keep R3-R6 gated. The next development step is a multi-horizon path-consistency objective with 15/30/60-minute Train targets, followed by a single frozen evaluation on a non-overlapping interval. This is a new hypothesis; this result will not be used for parameter selection or Test/Holdout scoring.
