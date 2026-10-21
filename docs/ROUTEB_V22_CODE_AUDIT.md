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

1. **B0 data / R2 predictive evidence: IN PROGRESS, gate closed.** RouteB now has multiple registered fresh-fold Train/Validation runs using row-group-scoped 0.4.1 data. The newest AR(1) risk-innovation experiment completed on TRXUSDT, but failed its preregistered path log-RV skill gate. Test/Holdout remain unopened; perpetual economics remain ineligible without contract rules, fees, and funding evidence.
2. Continue R2 only on a new disjoint, preregistered 60/15/15 or 360/15/15 Fold. Keep E1/E2 budgets matched, read Train/Validation once, and evaluate common Validation anchors. Do not use a failed Validation window for another modeling selection.
3. **E3/E4 hierarchy stays gated.** The causal multi-timeframe clock/data plumbing exists, but production hierarchical RSSM training, F2 model rollout, and the E3-vs-E4 ablation have not passed the prerequisite E2 market Validation gate. Do not interpret prototypes or synthetic tests as that result.
4. **WB4/WB5 remain future gated work.** Strategy imagination, calibrated ensemble, full Native execution/account/funding parity, and economic WFO have no completed evidence. The imagined account is not an authoritative native backtest.
5. Do not make a profitability, risk, or forecast-skill claim from these synthetic runs. Do not read Test/Holdout to unblock development.

## Environment and lock

`packages/world_model/environment/uv.lock` remains the Windows CPU environment. `packages/world_model/requirements-world-model-wsl-cuda.lock` now combines its exact training stack with the CUDA 13 runtime pins (78 exact package versions); its current SHA-256 is `e15b296ff2575272ad9f8e67c6a03dc74b2af57205321f363bf29b4e01c1243d`. The WSL venv lives in the ignored `packages/world_model/environment/.venv-wsl/` directory.

## Status update — 2026-10-21

RouteB completed a second fresh TRXUSDT Train/Validation audit with 112,320 scoped bars, no missing/duplicate minutes and no availability violations. The AR(1) risk-innovation R2 report and completion receipt verify; Test payload was not decoded. Pooled 15-minute path log-RV CRPS was worse than E0 by 0.02547 (95% CI [-0.00393, 0.04840], p=0.05597), and seed 20261018 showed Holm-corrected significant path-score harm. AR(1) moved generated absolute-return lag-1 autocorrelation to 0.113 versus realized 0.081, while rolling-RV mean/p95 remained low. The R2 gate remains closed; R3-R6 remain gated. The next proposed fresh-window test changes only risk-conditioning strength and retains AR(1) persistence at 0.65. See `docs/world_model/R2_TRXUSDT_PERSISTENT_RISK_INNOVATION_RESULTS_20261021.md`.

## Status update — 2026-10-21, risk-conditioning strength 0.25

The follow-up fresh Fold completed and passed integrity checks (aggregate SHA-256 `a6a104163f12e062ea769956342295f5b5559cdc9c6c5326035acfdc1faf0219`, six candidate attempts completed, completion receipt recorded, pooled forecast digests verified, Test payload unopened). Pooled 15m path log-RV CRPS improved from 0.31649 (selected E0) to 0.29198 (E2), paired difference -0.02451, 95% CI [-0.04241,-0.00888], p=0.0060. However, the preregistered per-seed Holm gate passed only on seed 102 (p=0.03698); seed 31 p=0.13293 and seed 20261018 p=0.05397. The strict all-seed R2 gate remains closed. E2 generated rolling-RV mean closely matched realized (0.0002478 vs 0.0002426), but p95 remained higher (0.0005735 vs 0.0005247), and absolute-return lag-1 autocorrelation overshot (0.2107 vs 0.0913). No corrected return-CRPS harm was found and all generated paths were valid. Perpetual economics remain ineligible without contract rules, fees, and funding evidence. R3–R6 remain gated. Next, on a new disjoint preregistered Fold, keep strength 0.25 and test persistence 0.35. Full evidence: `docs/world_model/R2_TRXUSDT_RISK_CONDITIONING_STRENGTH025_RESULTS_20261021.md`.

## Status update — 2026-10-21, risk-noise persistence 0.35

The third fresh TRXUSDT Train/Validation Fold completed with aggregate SHA-256 `90a36cbd098e9266f9c9f7cfb106a25cc5465dea7687250e3b4b8bcbe32a9bdc`. Digest, pooled artifacts, completion receipt and six terminal attempt events independently verify; 112,320 scoped bars had no missing/duplicate minutes or availability violations; Test payload remained unopened. The preregistered all-seeds R2 predictive gate passed: E2-minus-selected-E0 15m path log-RV CRPS was -0.06690, -0.05775, and -0.06509 for seeds 31, 102 and 20261018, respectively; all 95% CIs were strictly below zero and Holm p=0.00550. Pooled difference was -0.06599 (95% CI [-0.08066,-0.04903], p=0.00050). No corrected 15m return-CRPS harm was found and all paths were valid. Persistence reduction brought generated absolute-return autocorrelation (0.1063) closer to realized (0.0737), but rolling-RV mean/p95 remain above realized (0.0001948/0.0004255 vs 0.0001456/0.0003160). R3 hierarchical model research can now be preregistered on an independent Fold; this R2 result does not pass R3 or economic gates. Test/Holdout stay closed and perpetual economics stay blocked without contract rules, fees, funding and native parity. Full evidence: `docs/world_model/R2_TRXUSDT_RISK_PERSISTENCE035_RESULTS_20261021.md`.
