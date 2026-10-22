# R2 BTCUSDT Gaussian rollout-risk strength-0.1 adjacent transfer pre-registration

Status: frozen on 2026-10-20 before downloading or decoding the source interval. This is one fixed adjacent transfer check; it is not a search.

## Hypothesis and fixed recipe

The prior frozen strength-0.1 experiment improved the primary 15-minute path-derived log-RV CRPS over selected E0 for two of three seeds, but the third seed was inconclusive and the all-seed promotion gate failed. E1 had a lower mean CRPS. Freeze the same candidate unchanged and evaluate once on the immediately adjacent BTC interval to determine whether the partial gain transfers. Do not tune parameters or substitute another market after seeing outcomes.

- E2: same two-level RSSM, Gaussian prior, native risk horizons `[1,5,15,30,60]`, risk prediction scale `0.5`, rollout risk conditioning enabled with geometric strength `0.1`, 1,000 updates, batch size 8, seeds `31,102,20261018`.
- E1: fixed GRU baseline, 1,000 updates, batch size 8.
- E0: frozen zero-drift, Train-innovation EWMA and stationary block-bootstrap baselines; select the strongest on Validation mean path-derived 15-minute log-RV CRPS.
- 180 Validation anchors at 120-minute intervals; 32 paths/seed; 96 pooled paths; four-anchor chunks; 2,000 paired moving-block bootstrap replicates with 12-anchor blocks.
- Primary gate: each seed's paired 95% CI for E2-minus-selected-E0 15-minute path log-RV CRPS must be strictly below zero and Holm-adjusted p below `0.05`, within the existing 11-test Holm family.
- Test/Holdout stay unread. No economic claim or R2 promotion follows unless every registered criterion passes. A single inconclusive seed is a failure of the all-seed promotion gate.

## Frozen source and access scope

- Binance USD-M BTCUSDT 1-minute source interval `[2023-11-03T00:00:00Z, 2024-02-05T00:00:00Z)` (94 daily archives; expected 135,360 bars).
- Approved 60/15/15-day Fold; Train+Validation read scope `[2023-11-03T00:01:00Z, 2024-01-21T00:01:00Z)`.
- Test score scope `[2024-01-21T00:01:00Z, 2024-02-05T00:01:00Z)` is never decoded.
- Read-only ledger preflight immediately before registration found zero overlapping `access` and zero overlapping `query_intent` rows for BTCUSDT `perpetual-market-v1` over the Train+Validation read scope.
- Archive `.CHECKSUM` sidecars and ZIP members must pass before curation. Append exact RouteB source/config/Fold and dataset identities before the registered reader.
- Output root: `.artifacts/world_model/fresh-btcusdt-20231103-gaussian-risk01-adjacent-20261020/`.

## Execution controls

Use the RouteB repository and its isolated JAX worker only. Run the one-shot registered Fold reader with `includes_test=false`; capture `test_payload_decoded=false` and verify report, per-seed artifact and pooled hashes independently after completion. Do not read or score Test/Holdout.

## Append-only frozen execution identities (before the registered reader)

- RouteB repository commit `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`; RouteB source identity SHA-256 `4b1327326acd79bdfaf50da581bfb11367c6871c13818292571ba3772ed0ee8e`.
- Multi-seed runner `qa/world_model/run_registered_fold_ensemble.py` SHA-256 `aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9`.
- E1 configuration SHA-256 `c5c1b11c5bca78fd11e7e24f8e4a574d38b9937788dd9e5338183a5d9d9c0aaa`.
- E2 configuration SHA-256 `17345d7e9ccb29d784cadaeb8fede763206f20b03651368da53a4acdead9f8c5`.
- Registered Fold configuration SHA-256 `2575110f65854fd47ce7474a6efad8d203f06b45da734962643a31521d2703dc`; semantic Fold hash `b284f278c94e713c5e686698a1d6ef9ab11eb79a0acc4705112763b0077d5607`.
- Dataset hash `bae840314107789817f66ae416498e6378b820084574d6ad7a203ce3ab9adf60`; marker SHA-256 `479f36c0d824c3958a95939b0169c0f25219d62ac29e9e84a4d95b2825e529ae`; Parquet SHA-256 `1ce97fbad648aa3611ef70363c5862c98f8a4914246b5e893e94d79e4d23ce89`.
- Download manifest SHA-256 `3188617963fa52ceb90d654a833b08034339d02fecfed15202edfb13fb465832`; RouteB source-preparation receipt SHA-256 `bedc9e35d036cd70b1bb12a3146ff773db77d470e4d69121a793de609b18aca2`. All 94 archive checksums and ZIPs passed; the official RouteB preparation utility independently reproduced the exact event count, dataset, marker, and Parquet hashes.
- Final read-only registry preflight immediately before registration: zero overlapping BTCUSDT `perpetual-market-v1` `access` rows and zero overlapping `query_intent` rows for `[2023-11-03T00:01:00Z, 2024-01-21T00:01:00Z)`.
- Isolated worker: JAX `0.11.2`, device `CudaDevice(id=0)`. Pinned DreamerV3 checkout commit `e01491fad6434b2245a3b8ca201dd7faedcc458c`.


## Append-only setup failure before any registered read

The first runner invocation at the frozen output path was blocked before reservation because the isolated host process omitted `packages/orchestrator/src` from `PYTHONPATH` (`ModuleNotFoundError: rles_orchestrator`). Its `runner_failure.json` reports `registered_read_started=false`. Read-only ledger inspection after the failure found zero matching BTCUSDT access and query-intent records. No Train/Validation payload was decoded and no candidate/configuration/data choices changed. Retry uses the same frozen protocol in a new output directory under this registration root.

## Append-only execution record

The retry completed with one registration and no Test read. Initial runner setup failure was recorded above and happened before registration. Final aggregate report digest `2f05d47b295cb3cc6cd3ec22de8c45106cc164d881fa0c4d0b0c99515e1287e8`; aggregate report file SHA-256 `98a01e0a869a28c596dcdef69193fdc583e9ffb3c861788082c5e76411cff7f3`. Seed report digests: 31 `5035aef69c1110ba47dfd51600f892082ab44d898bda51a092652509c72f52f1`, 102 `3dede8d3ac0c2c6a3c6bd5a29d84fc21af914fe5ceaa12a7950042213a749d6c`, 20261018 `a1e5c1f9b96fd1f62fae4974b317861cc8fe8bcc515db412c9ef96e4873258ab`. Independent validation confirmed every per-seed E1/E2 forecast artifact, E2 native-risk artifact, both pooled files, aggregate/predictive digests, and unread-Test flags. Seed 31's Holm p was `0.087956`; the all-seed promotion rule failed. See [`R2_BTCUSDT_GAUSSIAN_RISK01_ADJACENT_RESULTS_20261020.md`](R2_BTCUSDT_GAUSSIAN_RISK01_ADJACENT_RESULTS_20261020.md).

Result record SHA-256: `d4b92a7fcfc35432e71389be72ee38089a24207d1f27dc6c0b671ff55360e662`. The registered strength-0.1 transfer did not meet the all-seed Holm gate; next development focus recorded in V3_IMPLEMENTATION_STATUS.md and README_STAGE.md is a training-time path-consistency objective.
