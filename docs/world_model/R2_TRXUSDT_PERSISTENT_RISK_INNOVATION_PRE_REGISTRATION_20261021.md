# R2 persistent risk innovation — fresh-window pre-registration

Frozen on 2026-10-21 before downloading or decoding the new market interval. The preceding stochastic risk/path alignment run completed with valid generated paths but did not pass its forecast-skill gate. E2 produced a lower Validation path log-RV CRPS mean than the selected baseline, but the confidence interval crossed zero on every seed; generated rolling-RV mean/p95 and absolute-return persistence were below the observed diagnostics. This follow-up tests one structural change on an untouched time interval. It does not reuse the prior Validation observations for model selection.

## Frozen hypothesis and protocol

- Hypothesis: independent per-minute log-RV draws weaken return-magnitude persistence. Apply a stationary Gaussian AR(1) transform to standardized risk innovations, `epsilon_t = 0.65 * epsilon_(t-1) + sqrt(1 - 0.65^2) * eta_t`, with an independent standard-normal `eta_t` and a fresh initial draw per path.
- E2 uses the existing Gaussian RSSM and Train-only multi-horizon path log-RV CRPS at `[15,30,60]`, coefficient `3.0`, 16 Train paths, native risk loss scale `1.0`, and risk-conditioning strength `0.1`. The AR(1) risk-noise transformation is applied in both Train path scoring and F1 rollout, preserving distributional alignment. The single frozen new parameter is `rollout_risk_noise_persistence=0.65`.
- E1 remains the unchanged fixed GRU comparator. Both candidates use 1,000 updates, batch size 8, learning rate 0.0003, report interval 50 and seeds `[31,102,20261018]`.
- Validation: 180 anchors at 120-minute spacing, 32 paths per seed, 96 pooled paths, 4-anchor chunks, 2,000 paired moving-block bootstrap replicates, block length 12.
- Same gate as the preceding pre-registration: each seed must show E2-minus-selected-E0 15-minute path log-RV CRPS 95% CI strictly below zero and Holm `p<0.05` in the fixed 11-test family; no seed may show significant Holm-corrected E2 harm to 15-minute return CRPS; generated OHLCV paths must be valid; rolling-RV mean/p95 and absolute-return persistence remain required diagnostics. No Test/Holdout read, economic claim or automatic promotion.

## Frozen data scope

- Binance USD-M TRXUSDT 1m source-open interval `[2025-11-06T00:00:00Z,2026-02-08T00:00:00Z)`: 94 UTC days and 135,360 expected closed bars.
- Approved 60/15/15 Fold with one-hour purge and 23-hour embargo. Train+Validation reader scope `[2025-11-06T00:01:00Z,2026-01-23T00:01:00Z)` (`[1762387260000000000,1769126460000000000)`). Train score starts 2025-11-08T00:01Z; Validation score ends 2026-01-23T00:01Z; Test score is `[2026-01-24T00:01:00Z,2026-02-08T00:01:00Z)`.
- Read-only ledger preflight on 2026-10-21 returned `access_overlap=0` and `query_intent_overlap=0` for the exact Train+Validation nanoseconds above. The prior reservation ends 2025-10-18T00:01Z, so this interval is disjoint.
- Parquet writer must split before the first event whose `available_at_ns >= 1769126460000000000`; the registered reader remains the only operation that decodes Train/Validation row groups.
- Output root: `.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/`.
- Download official Binance archives only after this freeze. Verify published checksum sidecars, ZIP contents, continuous bars, source and Fold identities; repeat read-only ledger preflight immediately before registered access. Do not read Test/Holdout.

## Implementation and identity freeze

- Implement the AR(1) transform as a shared risk-noise helper used by Train path scoring and F1 rollout, with default persistence `0.0` retaining existing behavior. Reject persistence outside `[0,0.99]`, and reject nonzero persistence unless risk conditioning and risk sampling are both enabled.
- Add focused checks for shape handling, persistence boundaries, deterministic seeded rollout, positive path-CRPS gradients through prior/transition/risk heads, and Train/F1 transformation parity. These tests validate mechanism only; the registered Validation gate remains decisive.
- Append implementation commit, RouteB source fingerprint, candidate/Fold config hashes, worker identity, prepared-source hashes and final ledger preflight after data preparation and before the registered reader. No additional modeling changes after that freeze.
