# R2 pre-registration: ETHUSDT conditional risk emphasis

Date frozen: 2026-10-19, revised before any successful RouteB registration or Train/Validation payload read. Revision 2 uses an older non-overlapping ETH interval after the initial interval was rejected by the access ledger.

## Hypothesis

The LTCUSDT 1,000-update Validation rollout had lower absolute-return lag-1 autocorrelation (0.0035 vs 0.0690 realized), lower rolling-volatility mean (0.000945 vs 0.001032), and lower rolling-volatility p95 (0.001708 vs 0.002115). Test one fixed training change: raise E2's multi-horizon prior-risk loss multiplier from 1.0 to 2.0 so latent states receive a stronger gradient from the already-trained return, log-volatility, excursion, and barrier targets. All other model/training settings remain identical, including 1,000 updates, seeds and forecast protocol. E1 remains unchanged.

## Frozen validation protocol

- Instrument/data: ETHUSDT USD-M 1m, UTC source-open interval [2024-01-10 00:00, 2024-04-12 00:00), expected 133,920 bars. This falls inside the registry-verified unused interval [2024-01-02 00:01, 2024-07-16 00:00).
- One-shot registered Fold: approved 60/15/15 protocol. Train warmup begins 2024-01-10 00:01 UTC; Train scores 2024-01-11 00:01 through 2024-03-11 00:01 (60d); Validation warmup begins 2024-03-01 00:01, Validation scores 2024-03-12 00:01 through 2024-03-27 00:01 (15d); Test warmup begins 2024-03-17 00:01, and the unconsumed Test interval is not decoded. Purge and embargo remain 1h and 23h.
- Seeds 31, 102, 20261018; 32 paths per seed; 180 two-hour Validation anchors; 96 pooled paths per candidate.
- Primary measure: 15-minute return CRPS against the strongest statistical baseline selected within Validation. Promotion requires the predeclared paired block-bootstrap confidence interval to favor E2 and Holm-adjusted p < 0.05 under the existing report protocol.
- Secondary mechanism diagnostics: native prior log-realized-volatility CRPS/Brier and free-rollout volatility-clustering/tail diagnostics. They cannot override a failed primary gate.
- Test/Holdout payloads are not decoded. No economic or trading certification is part of this experiment.

## Frozen inputs

- Final Fold config: `configs/routeb/fresh_ethusdt_2024_60_15_15_fold.json`, bound to dataset SHA-256 `7aafd059f086d9cc1651c171282de59a48a0c71c7f6b068209cf4b9b76c783cb` and marker `c762f4e5ec1d09a19474ecf46c0be1f8739df89b60f78d636a04fd07b334dee9`.
- Feature spec: `configs/routeb/fresh_ethusdt_feature_spec.json`.
- E1: `configs/routeb/fresh_ethusdt_e1_ensemble.json`.
- E2 candidate: `configs/routeb/fresh_ethusdt_e2_risk2_ensemble.json` (`risk_prediction_scale=2.0`).
- Source preparation: [`B0_ethusdt_202401_source_preparation.json`](B0_ethusdt_202401_source_preparation.json). The 93 in-window daily archives and checksum sidecars are under `.artifacts/world_model/fresh-ethusdt-20240110-60-15-15-20261019/archives/`; an extra April 12 daily file was downloaded but excluded from this source interval.

This is a single preregistered setting, not a Validation-tuned sweep. A failed gate leaves R3/R4 closed and requires a new hypothesis and never-consumed scope for another experiment.

## Initial interval rejected before payload access

The first proposed 2025-09-20 through 2026-10-18 ETH interval was submitted to the registered-fold runner and rejected because it overlaps prior ETHUSDT scope `04b91130966dbb9770e10e921e480adbac58c2af3697adcc774ea3f9101dc009` (2025-08-28 through 2025-10-27) and additional legacy intervals. The registry rejected before registering the candidate or reading Train/Validation payloads. The source preparation for that broad interval is retained but will not be used for this experiment. Revision 2 moves to the unused January-April 2024 interval above. Its curated source contains 133,920 bars with the recorded dataset and marker hashes.