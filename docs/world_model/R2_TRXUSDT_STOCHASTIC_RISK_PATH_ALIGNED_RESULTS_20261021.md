# R2 stochastic risk/path alignment — Validation results (2026-10-21)

## Frozen experiment and integrity

- RouteB source commit: `da89fdfa8fefcf4efec9f5ea30bbef8d1f282b39`; source code fingerprint: `8f8ad36767d77646d757a602e44be9aed8a4afbb1fae0049cc671576f49b708a`.
- Fold: `29e9251606e9190d5457c3da48a475c7f5b80107199c5723f7d8268dabc34314`; dataset: `2cbafd76127a661054a3b2fbe92539ab6f07249ccf128812784e077ddf4562d5`.
- Aggregate report SHA-256: `abd20be126c252f7f41f31188f2405f1762e214e66c7f4c98716be5bdff2e656`; recomputed canonical report hash and completion receipt both match. The ledger records the access as completed. All six candidate/seed attempts have terminal `completed` events.
- Three seeds `[31,102,20261018]`, 1,000 updates per model, 180 common Validation anchors, 32 paths per seed and 96 pooled paths per candidate. Read scope `[2025-08-01T00:01:00Z,2025-10-18T00:01:00Z)`; `test_payload_decoded=false`.
- Market data integrity passed: 112,320 Train/Validation bars; zero missing/duplicate minutes and zero availability violations. Market-model eligible is true; perpetual economics eligible is false because instrument rule/fee/funding evidence is missing.

## R2 gate result

The selected baseline was `E0_S0_ZERO_DRIFT_FIXED_VOL`. Pooled E2 15-minute path log-RV CRPS was `0.25730` versus baseline `0.27187`, a favorable mean difference of `-0.01457`; its paired 95% block-bootstrap interval was `[-0.04301, 0.03772]` and `p=0.4798`. Each seed had a favorable mean difference, but all three intervals crossed zero; Holm-adjusted p-values were `1.0`, `1.0`, and `0.4838`. The preregistered significance gate therefore failed.

E2 showed no significant Holm-corrected 15-minute return-CRPS harm. Generated bars and paths were valid for 100% of samples. However, E2 generated rolling realized-volatility mean/p95 were `0.000296/0.000498`, versus observed `0.000385/0.000850`; generated absolute-return lag-1 autocorrelation was `-0.0023`, versus observed `0.0650`. This points to low dispersion and weak volatility persistence. The report makes no economic claim, does not certify R2, and does not unlock R3-R6.

## Next registered work

Use a fresh TRXUSDT interval `[2025-11-06T00:00:00Z,2026-02-08T00:00:00Z)` with the same Fold and scoring protocol. The read-only RouteB ledger preflight for the resulting Train+Validation scope `[1762387260000000000,1769126460000000000)` returned zero overlapping `access` and zero overlapping `query_intent` rows. Before data download, freeze a single persistence hypothesis: AR(1)-correlated standardized log-RV innovations with `rho=0.65`, applied identically to the Train path-CRPS sampler and F1 rollout. Keep E1, seeds, optimizer budget, validation anchors, forecast path count and statistical gate unchanged.
