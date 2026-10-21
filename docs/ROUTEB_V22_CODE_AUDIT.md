# RouteB v2.2 implementation audit — refreshed 2026-10-18

## Baseline and ownership

- Main project HEAD: `80194a587444bf4a11f06f6e273723e5754f2483`.
- RouteB source is additive under `packages/world_model/src/rles_world_model/routeb/`.
- DreamerV3 remains pinned at `e01491fad6434b2245a3b8ca201dd7faedcc458c`; the upstream checkout is clean and unpatched.
- RouteB accesses registered market data through the shared ResearchRegistry and orchestrator adapters. It has no imports from RouteA packages.

## Current verified WB0-R / WB1-R / WB2 engineering

| Requirement | Evidence | Result |
|---|---|---|
| Locked Windows CPU learner | Python 3.12.10, JAX/JAXlib 0.11.2, pinned upstream | PASS |
| Locked WSL2 CUDA learner | Full 78-pin lock; Python 3.12.3; JAX/JAXlib 0.11.2; RTX 5070 enumerated as `cuda:0` | PASS |
| RSSM gradient updates | Four optimizer updates on synthetic OHLCV windows; finite reported losses/gradients | PASS mechanism only |
| True checkpoint resume | Resumed vs uninterrupted model and Optax arrays compared exactly | PASS on CPU and CUDA |
| Forecast seed stability | Direct forecasts and worker IPC repeated with the same seed and compared array-by-array | PASS on CPU and CUDA |
| Forecast IPC/support | One-step distribution, three-step path, F1 worker path and valid OHLCV geometry | PASS mechanism only |
| E0/E1/E2 pipeline wiring | One-shot synthetic Fold orchestration, common Train artifact/budget, checksummed forecasts/report | PASS mechanism only |
| Source/access boundary | Registered Fold reader, Train-only export, validation-only scoring, Test payload exclusion | Covered by focused project regression |

The current focused cross-layer regression passes **207 tests**, with 3 JAX-only tests skipped in the main Python 3.14 environment. A separate CUDA-targeted run passes **47 tests** across the RouteB additive, E1 and quality-v2 modules with no skips. CPU, CUDA smoke, and CUDA test evidence is in `docs/world_model/B2_train_resume_cpu_smoke.json`, `docs/world_model/B2_cuda_smoke.json`, and `docs/world_model/B2_cuda_test_suite.json`. The CUDA mechanism smoke was rerun on 2026-10-18 and passed; its four-update toy fit took 4.40 seconds with 608 MiB peak live allocation, descriptive measurements rather than performance guarantees.

## Remaining gates and work

1. **B0 data / real E0-E1-E2 Validation: BLOCKED.** The current inventory has no registered fresh-fold configuration and no eligible row-group-scoped 0.4.1 source interval outside consumed/reserved registry ranges. Existing 0.4.0 snapshots do not satisfy the reader contract. No real market payload was opened and Test/Holdout remain unopened.
2. Run the frozen `run-registered-fold` workflow only after a new approved 60/15/15 or 360/15/15 Fold is available. It reserves one fresh scope, reads Train/Validation once, trains E1/E2 with matched budget and scores common Validation anchors. No result exists yet for market forecasting skill or calibration.
3. **E3/E4 hierarchy stays gated.** The causal multi-timeframe clock/data plumbing exists, but production hierarchical RSSM training, F2 model rollout, and the E3-vs-E4 ablation have not passed the prerequisite E2 market Validation gate. Do not interpret prototypes or synthetic tests as that result.
4. **WB4/WB5 remain future gated work.** Strategy imagination, calibrated ensemble, full Native execution/account/funding parity, and economic WFO have no completed evidence. The imagined account is not an authoritative native backtest.
5. Do not make a profitability, risk, or forecast-skill claim from these synthetic runs. Do not read Test/Holdout to unblock development.

## Environment and lock

`packages/world_model/environment/uv.lock` remains the Windows CPU environment. `packages/world_model/requirements-world-model-wsl-cuda.lock` now combines its exact training stack with the CUDA 13 runtime pins (78 exact package versions); its current SHA-256 is `e15b296ff2575272ad9f8e67c6a03dc74b2af57205321f363bf29b4e01c1243d`. The WSL venv lives in the ignored `packages/world_model/environment/.venv-wsl/` directory.

## Status update — 2026-10-21

RouteB now has a completed real-data Train/Validation audit on a fresh TRXUSDT Fold. Integrity checks found 112,320 scoped bars, no missing/duplicate minutes and no availability violations. The R2 multi-seed run completed and its report and ledger receipt verify. E2 had a lower mean 15-minute path log-RV CRPS than the selected baseline, but every seed's interval crossed zero and the preregistered Holm significance criterion failed. Generated volatility dispersion and absolute-return persistence were low. The R2 gate remains closed; R3-R6 remain gated. A focused follow-up is preregistered on disjoint 2025-11-06 through 2026-02-08 data to test AR(1)-correlated risk innovations with identical Train/F1 sampling.
