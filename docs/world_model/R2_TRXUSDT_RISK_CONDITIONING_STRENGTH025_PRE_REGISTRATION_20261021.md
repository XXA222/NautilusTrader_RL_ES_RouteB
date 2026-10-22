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
## Prepared-source verification (2026-10-21)

- Registered Fold semantic hash: `e2cb5dc62cc55beb03e322fe5d281561fa07067700180c30536983bb749d1aca`; Fold config SHA-256: `24b79a47267c2b7358bf79367bd3881a9277e645cbf972d90b308a3966670419`.
- E1 config SHA-256: `e3f979c910c8e4d4f3c70e11475516c5c5b2b86b1be1d2e222799d416aea2c38`; E2 config SHA-256: `64194210210e5183d114c1ce1c5437c890a43c0e7e82fb340c93a042caf6ad0f`.
- Binance download receipt SHA-256: `068368647419333876100ab80993c834a5043228540763a47c5f87daf9c396c9`; curated Parquet SHA-256: `deb82fcf5a7e4f7ba3e872e53f0662441501f88af3c018000677a3fcd66d1691`; completion marker SHA-256: `fcbeaa361e912d926b53d3cebcd83db151f948cc45980428dd429fe7d8f6c5d7`; source-preparation receipt SHA-256: `5b7f4962a471d2e10dd09c49891dc377cbc3376e3e7faf676f18b2b7b9f2a8d6`.
- Source semantic identity is schema `0.4.1`, snapshot hash `528bdc786707272888248e088cc364198a570db24b7b5fc6b2a997db2947a378`, with 135,360 events. Archives: 2026-02 `37ad3848b0497dfa0d1283d432ff9be8602408beb1443057d7b5406389c59947`; 2026-03 `7e0027c87d8f955458692cc277d532e7020c7649e2b429f80cf717da407242ab`; 2026-04 `b44781c6e24c5aad44168b62a0a2232c8903523053793796c50aa54c564e0d56`; 2026-05 `06e00dda918f0fa9200617ed5a27beac4a292925c3cc74623c756ddd6adff8ed`.
- Footer-only audit: 94 row groups total; reader scope selects groups 0–77; no group crosses the Train+Validation end; group 78 begins at `available_at_ns=1777507261000000000`, one second after the exclusive reader boundary. No Test payload was decoded.
- E1/E2 optimizer configurations compare equal: 1,000 updates, batch 8, learning rate `0.0003`, report interval 50. The frozen E2 values are persistence `0.65` and risk-conditioning strength `0.25`.
- Immediate read-only ledger recheck returned `access_overlap=0`, `query_intent_overlap=0`. Worker preflight returned NumPy `2.3.3`, JAX `0.11.2`, device `cuda:0`.
- Refreshed source manifest: 504 tracked source files, identity `8509e6a65e8d73aeb27795cd6ee8a1973360f645ef117aec9172afa0c2e0a802`.
- The final commit-bound code fingerprint and repeated just-before-reader ledger check are recorded in `pre_read_identity.json` before running the registered reader.



## Registered-run outcome (2026-10-21)

The all-seeds R2 gate failed. Pooled 15m path log-RV CRPS favored E2 by 0.02451 (95% CI [-0.04241,-0.00888], p=0.0060), but per-seed Holm gates passed only for seed 102 (p=0.03698); seed 31 p=0.13293 and seed 20261018 p=0.05397. E2 rolling-RV mean matched realized closely, while p95 was somewhat high and generated absolute-return lag-1 autocorrelation was 0.2107 versus realized 0.0913. All paths were valid and no corrected 15m return-CRPS harm was detected. Test was not decoded. Full outcome and artifact verification are in `docs/world_model/R2_TRXUSDT_RISK_CONDITIONING_STRENGTH025_RESULTS_20261021.md`. The next permitted R2 follow-up is a fresh disjoint Fold testing persistence 0.35 with strength fixed at 0.25.

