# R2 TRXUSDT multi-horizon Train path log-RV CRPS pre-registration — 2023-07-08 scope

Status: frozen before downloading or decoding this source interval. This is one fixed follow-up run of the failed-before-forecast 2023-04-05 scope, using the checkpoint identity fix already validated by six JAX-specific tests. The failed scope is consumed and will not be reused.

## Fixed hypothesis and recipe

The prior TRXUSDT 2023-01-01 scope showed a favorable but inconclusive E2-vs-E0 S1 point estimate at 15-minute path log-RV CRPS; E1 remained the strongest validation candidate. Its preregistered 2023-04-05 multi-horizon follow-up failed before E2 Validation forecasts because JSON checkpoint identity compared tuple/list representations for `path_log_rv_crps_horizons`. The implementation is corrected and the checkpoint roundtrip regression is now covered. This experiment asks whether equal-weight Train-only sampled free-prior path log-RV CRPS at 15/30/60 minutes improves E2 relative to the strongest frozen Validation statistical baseline on the next untouched TRX interval.

- E2: default RouteB RSSM (deter 256, hidden 128, categorical stochastic state 16x16), Gaussian prior; retain five-channel OHLCV NLL and add equal-weight Train-only free-prior sampled path log-RV CRPS at 15, 30 and 60 bars. Four sampled paths per anchor; anchors spaced 15 bars; reuse each free-prior path through minute 60. Keep risk prediction scale 1.0 and rollout risk conditioning disabled.
- E1: fixed probabilistic GRU comparator. E0: frozen zero-drift, Train-innovation EWMA and stationary block-bootstrap baselines; select strongest by Validation mean 15-minute path log-RV CRPS.
- E1/E2: 1,000 optimizer updates, batch 8, learning rate 0.0003, report interval 50, seeds `[31, 102, 20261018]`; no tuning or candidate substitution.
- Validation: 180 anchors at 120-minute spacing; 32 forecast paths per model seed; 96 pooled paths; four-anchor chunks; 2,000 paired moving-block bootstrap replicates, 12-anchor blocks.
- Primary gate: each seed's E2-minus-selected-E0 15m path log-RV CRPS paired 95% CI is strictly below zero and Holm-adjusted p<0.05 in the existing 11-test family.
- Guardrails: no statistically significant Holm-corrected harm on 15m return CRPS; generated OHLCV and path structure valid. Test/Holdout unread. No economic conclusion or automatic stage promotion.

## Frozen source and split

- Binance USD-M TRXUSDT one-minute source open-time interval: `[2023-07-08T00:00:00Z, 2023-10-10T00:00:00Z)`, 94 UTC days, 94 daily archives, expected 135,360 closed bars.
- Fold: 60/15/15 days, one-hour purge, 23-hour embargo. Train+Validation reader scope `[2023-07-08T00:01:00Z, 2023-09-24T00:01:00Z)`; Test begins `2023-09-25T00:01:00Z` and remains excluded.
- Train score `[2023-07-10T00:01:00Z, 2023-09-08T00:01:00Z)`; Validation score `[2023-09-09T00:01:00Z, 2023-09-24T00:01:00Z)` with 12-day warmup beginning `2023-08-28T00:01:00Z`; Test score `[2023-09-25T00:01:00Z, 2023-10-10T00:01:00Z)` with warmup beginning `2023-09-13T00:01:00Z`.
- Ledger preflight before download and again before the registered reader must show zero overlapping access and query-intent records for `(TRXUSDT, perpetual-market-v1)` and the Train+Validation scope.
- Output: `.artifacts/world_model/fresh-trxusdt-20230708-multihorizon-path-crps-20261020/`.
- Verify all Binance checksum sidecars and ZIP members, curated source identity, Fold footer bounds, source code/config identities, and worker environment before the one-shot registered reader. `includes_test=false`; Test/Holdout payload must not be decoded.

## Frozen implementation identities

- RouteB repository base commit: `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`.
- RouteB source fingerprint: `593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c`.
- Registered-fold runner SHA-256: `aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9`.
- E1 config SHA-256: `e3f979c910c8e4d4f3c70e11475516c5c5b2b86b1be1d2e222799d416aea2c38`.
- E2 config SHA-256: `b29f574cb598c1df48e91fd9e92b540deefa043c948dfc31b8e7d9f63fb9db0a`.
- Worker lock SHA-256: `ab99ac8f831bc06dc6e408cf91090b69aca413df8f38cfcd2f28b77a6ea18c1b`; Python 3.12.10/JAX 0.11.2.
- Initial read-only registry preflight: access rows 0; query-intent rows 0; exact Train+Validation scope `[1688774460000000000, 1695513660000000000)` ns.

Dataset, marker, download manifest, registered Fold and semantic identities will be appended before the reader after verified source preparation. No source/config/method changes after registration.

## Append-only source/Fold identities frozen before registered read

- Second read-only ledger preflight immediately before registration: access rows 0; query-intent rows 0.
- Binance archive count 94; download manifest SHA-256 `75f088bf3af86cd9e63ba4db0f0687aef4bdb5642a66a16c5eb8bfc48577b394`.
- Source preparation report SHA-256 `47a605d3b627567f55d18c8ec6ab2b4706bc344e7929381c5f06e7b1ca1edf34`; dataset hash `46ce6923d4053637f18cdd907d8dc4073f4a8a807433f7658f504f2e4b90fe3a`; marker SHA-256 `8c47d4854a553c489bf9366c19ee6d0e3d69758acc91ffdcfab43ae2beab18a0`; Parquet SHA-256 `751f5b59717cb5a27e63df5fe683d302062b66599aeaddeb7ca9ec79ab721f80`.
- Registered Fold config SHA-256 `9df15efcdb60c18626de9082de7449f6032eea295ab2704fc1b548ace4b4773b`; semantic Fold hash `f3f97ace8194d6f784d985fee7e6d42ee5dacfe4a29f90945c810c07a2eea80b`.
- Curated source bounds: `[1688774460000000000, 1696896000000000000]` ns. Reader scope: `[1688774460000000000, 1695513660000000000)` ns; Test remains excluded.
- Candidate config identities, code fingerprint, Runner and worker lock match the pre-download frozen values above.

### Final pre-registration metadata checks

The route uses RouteB's isolated Linux Python 3.14 host environment to access PyArrow and launch the separate Python 3.12/JAX CUDA worker. Read-only marker and row-group footer checks pass; source identity matches the prepared 135,360-bar Parquet and the second ledger preflight remains access=0/query_intent=0. Final WSL-mounted registered Fold config SHA-256: 580ed13f5df553bda42a046e520650017418b0a06655a78ad443f79e3a72fcc8; semantic Fold hash remains 3f97ace8194d6f784d985fee7e6d42ee5dacfe4a29f90945c810c07a2eea80b. Earlier failed launch attempts stopped before any row payload decode or reservation.

## Append-only completed-result record

- Completed: three seeds, 1,000 updates per candidate, 180 Validation anchors, 96 pooled paths. Aggregate report canonical SHA-256 941fb7c9f14a5cb6a4ab0bbb07be601f8e593edb2efccd71f6b4d70cd88ae014; file SHA-256 dca25db0965fe1fbc3333a0e5d8ac3f59ce0edb12efd5b5265120b1e8979a2ef.
- Primary gate NOT PASSED. Seed 31 difference +0.001743 (95% CI [-0.023012,+0.033626], Holm 1.0); seed 102 -0.036625 (CI [-0.061300,-0.010901], Holm 0.02399); seed 20261018 -0.034168 (CI [-0.057267,-0.006440], Holm 0.04398). Pooled difference -0.022346 (CI [-0.042868,+0.002877], p 0.05847).
- Test/Holdout remained unread; Train/Validation exact overlap count is one access reservation, zero query intents; Test access/query-intent counts both zero. Integrity verification passed for aggregate, all three seed reports, checkpoints, configurations, training sequences, forecast/risk files and pooled artifacts.
- Result detail: [R2_TRXUSDT_MULTIHORIZON_PATH_CRPS_RESULTS_20261020_NEXT.md](R2_TRXUSDT_MULTIHORIZON_PATH_CRPS_RESULTS_20261020_NEXT.md).
