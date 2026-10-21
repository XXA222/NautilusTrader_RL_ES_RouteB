# R2 persistent risk conditioning strength 0.25 — fresh-window pre-registration

Frozen on 2026-10-21 after the preceding registered experiment completed and before downloading or decoding this new market interval. The AR(1) persistence intervention improved generated absolute-return persistence but left RV dispersion low and failed the primary path log-RV CRPS gate. The next experiment changes one candidate parameter only, on data wholly after the preceding Test interval.

## Frozen hypothesis and protocol

- Hypothesis: increase the geometric blend from the RSSM one-minute risk forecast into the generated return scale from `rollout_risk_conditioning_strength=0.10` to `0.25` to reduce the observed under-dispersion. Keep the stationary standardized log-RV AR(1) parameter fixed at `rollout_risk_noise_persistence=0.65` and use the same transformed risk noise in Train path scoring and F1 rollout.
- E2 retains the Gaussian RSSM, Train-only equal-weight sampled free-prior path log-RV CRPS at horizons `[15,30,60]`, coefficient `3.0`, 16 Train paths, native risk loss scale `1.0`, risk-conditioned sampling, and the same optimizer recipe. The sole model-configuration change versus the preceding registered R2 run is conditioning strength `0.25` instead of `0.10`.
- E1 remains the unchanged fixed GRU comparator. Both candidates use 1,000 updates, batch size 8, learning rate `0.0003`, report interval 50, and seeds `[31,102,20261018]`.
- Validation remains 180 anchors at 120-minute spacing, 32 paths per seed, 96 pooled paths, 4-anchor chunks, 2,000 paired moving-block bootstrap replicates, and block length 12.
- Decision rule remains unchanged: each seed must show E2-minus-selected-E0 15-minute path log-RV CRPS 95% CI strictly below zero and Holm `p<0.05` in the fixed 11-test family; no seed may show significant Holm-corrected E2 harm to 15-minute return CRPS; generated OHLCV paths must be valid; rolling-RV mean/p95 and absolute-return persistence remain required diagnostics. No Test/Holdout read, economic claim, or automatic promotion.

## Frozen data scope

- Binance USD-M TRXUSDT 1m source-open interval `[2026-02-11T00:00:00Z,2026-05-16T00:00:00Z)`: 94 UTC days and 135,360 expected closed bars.
- Approved 60/15/15 Fold with one-hour purge and 23-hour embargo. Train+Validation reader scope `[2026-02-11T00:01:00Z,2026-04-30T00:01:00Z)` (`[1770768060000000000,1777507260000000000)`). Train score starts 2026-02-13T00:01Z; Validation score ends 2026-04-30T00:01Z; Test score is `[2026-05-01T00:01:00Z,2026-05-16T00:01:00Z)`.
- Read-only RouteB ledger preflight on 2026-10-21 returned `access_overlap=0` and `query_intent_overlap=0` for exact Train+Validation scope `[1770768060000000000,1777507260000000000)`. This source interval begins after the preceding registered Test interval ends on 2026-02-08T00:00Z; the intervening Test payload remains unread.
- Parquet writer must split before the first event whose `available_at_ns >= 1777507260000000000`; the registered reader remains the only operation that decodes Train/Validation Parquet row groups. Test/Holdout payloads remain excluded.
- Output root: `.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/`.
- Download official Binance archives only after this freeze. Verify checksum sidecars, ZIP contents, continuous bars, source and Fold identities; repeat the read-only ledger preflight immediately before registered access.

## Frozen candidate and source identity

- E1 config: `configs/routeb/fresh_trxusdt_20260211_e1_persistent_risk_strength025_ensemble.json` (unchanged fixed GRU recipe).
- E2 config: `configs/routeb/fresh_trxusdt_20260211_e2_persistent_risk_strength025_ensemble.json`; `rollout_risk_noise_persistence=0.65`, `rollout_risk_conditioning_strength=0.25`.
- The pre-download access/query-intent counts are zero as stated above. Final commit-bound RouteB source fingerprint, source/Fold/config hashes, isolated worker identity, footer-only audit, and repeated immediate ledger preflight will be written to the experiment's `pre_read_identity.json` after data preparation and before the registered reader. No modeling or scoring changes are allowed after this freeze.

## Previous-window outcome informing this hypothesis

The prior fresh-window R2 run (2025-11-06 through 2026-02-08) did not meet its path-score gate. Pooled E2-minus-E0 15-minute path log-RV CRPS was `+0.02547` (95% CI `[-0.00393,+0.04840]`, `p=0.05597`), with Holm-corrected significant path-score harm on seed 20261018. Generated E2 rolling-RV mean/p95 were `0.0002797/0.0005632`, below realized `0.0003329/0.0006645`. Generated absolute-return lag-1 autocorrelation was `0.1130` versus realized `0.0806`. The new window tests whether a stronger risk-to-return-scale blend addresses dispersion while retaining the improved persistence. Full previous-run evidence is recorded in `R2_TRXUSDT_PERSISTENT_RISK_INNOVATION_RESULTS_20261021.md`.