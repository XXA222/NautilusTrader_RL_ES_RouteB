# Route B v3.0 Implementation Status

Date: 2026-10-19
Engineering baseline named by the design: `80194a587444bf4a11f06f6e273723e5754f2483`
Route B checkout: `codex/routeb-ci-integrity` at `cb7a3ec`
Pinned DreamerV3 source: `e01491fad6434b2245a3b8ca201dd7faedcc458c`

**Overall engineering acceptance: PARTIAL.** R0/R1 mechanisms and R2 implementation have passing synthetic/runtime evidence. Historical registered results include many failed gates; the latest independent TRXUSDT persistence-0.35 Fold passed its three-seed R2 predictive gate, allowing R3 research preregistration but not universal R2 promotion. E3/E3b/E4 have distinct synthetic-trainable variants and a checkpoint-bound F2 runner that advances actual JAX latent carries from generated multi-scale bars. No registered real-data hierarchy comparison, Native parity, or strategy acceptance is claimed.

## Latest registered BTCUSDT path-risk result

The BTCUSDT `[2022-04-18, 2022-07-21)` risk-conditioned rollout experiment
completed all three seeds with four-anchor forecast chunking and no OOM. Its
new per-seed path-derived 15-minute log-RV CRPS comparison failed in the wrong
direction for all seeds: E2-minus-E0 was `+0.129895`, `+0.090897`, and
`+0.157759`; each paired 95% interval was above zero and Holm-adjusted p-values
were `0.005497`, `0.013993`, and `0.005497`. E2 generated rolling RV mean/p95
(`0.001292`/`0.003011`) exceeded realized (`0.000919`/`0.001819`). All report
and forecast hashes verify; Test stayed undecoded. The next BTC interval
tested a single geometrically tempered strength-0.5 candidate. Generated RV
means matched realized closely, but p95 stayed low and the primary path log-RV
gate was inconclusive in all seeds (Holm p `1.0`, `0.855`, `0.550`); it did
not pass promotion. All report, per-seed and pooled hashes verify; Test stayed
unread. A Student-t prior with variance-correct tempered risk conditioning is
the next frozen hypothesis, on a new scope. See
[`R2_BTCUSDT_RISK_CONDITIONED_ROLLOUT_RESULTS_20261020_REV3.md`](R2_BTCUSDT_RISK_CONDITIONED_ROLLOUT_RESULTS_20261020_REV3.md)
and
[`R2_BTCUSDT_RISK_CONDITIONED_TEMPERED_RESULTS_20261020.md`](R2_BTCUSDT_RISK_CONDITIONED_TEMPERED_RESULTS_20261020.md),
[`R2_BTCUSDT_RISK_CONDITIONING_TEMPERED_PRE_REGISTRATION_20261020.md`](R2_BTCUSDT_RISK_CONDITIONING_TEMPERED_PRE_REGISTRATION_20261020.md),
and the new Student-t candidate's
[`pre-registration`](R2_BTCUSDT_STUDENT_T_TEMPERED_RISK_PRE_REGISTRATION_20261020.md).

## Delivered work packages

| Work package | Result | Evidence |
|---|---|---|
| R0: runtime and training integrity | Complete. Route B's WSL worker uses its own Python 3.12.10 runtime and locked JAX 0.11.2 dependencies. Four optimizer updates, save/resume parity, and deterministic free rollout pass on CPU and CUDA with the v3 risk head included. | CPU/CUDA smoke reports exact model and optimizer equality. It also hashes each sampled batch order and records each stateless JAX PRNG seed; resumed and uninterrupted runs match these values. Worker forecast and distribution exports pass. |
| R1: prior-to-observation alignment | Complete. `jax_model.py` transitions each posterior state through DreamerV3's prior before scoring the next bar. Both OHLCV and risk heads use that prior state; posterior next-bar NLL remains a separately weighted auxiliary objective. | [`B2_prior_risk_gradient_smoke.json`](B2_prior_risk_gradient_smoke.json) verifies finite nonzero gradients for prior and risk heads and transitions, and confirms future-observation mutation does not alter the aligned prior. The checkpoint-bound alignment API emits anchor/target times, mask, prior/posterior means/scales and categorical logits; CPU/CUDA smoke writes 93 aligned rows (72 valid) and verifies the artifact checksum. |
| R2: multi-horizon risk and comparisons | Train sequence NPZ v3 carries gap-aware 1/5/15/30/60-minute return, log realized-volatility, high/low excursion, and barrier targets plus per-anchor/per-horizon validity masks. Prior risk forecasts export checkpoint-bound Gaussian means/scales and barrier logits. Scoring compares forecasts and observations in matching target spaces, including log realized volatility. | Synthetic CPU/CUDA risk training and export pass. LINK, DOGE, corrected AVAX and LTC three-seed Validation runs are complete. None establishes primary 15m return CRPS improvement. LTC E2 is slightly better than S0 (`-0.00001606`, 95% CI `[-0.00008251,+0.00005456]`, Holm `p=1.0`) after 1,000 updates; its free rollouts understate volatility clustering/tails. The original DOGE log-volatility scores were invalid due to the target-space mismatch and are documented in [`R2_DOGEUSDT_RISK_SCORE_CORRECTION_20261019.md`](R2_DOGEUSDT_RISK_SCORE_CORRECTION_20261019.md); corrected scoring and regression-tested target labels are now used. R2 remains open for conditional risk improvement; R3/R4 remain closed. See [`R2_LTCUSDT_1000STEP_360_15_15_20261019.md`](R2_LTCUSDT_1000STEP_360_15_15_20261019.md). |
| Fresh-fold data governance | Complete. Added RouteB-owned one-shot fresh-scope reservations, marker/hash and Parquet-footer time-bound preflight, and row-group scoped Fold reads. The registry rejects any overlapping prior access or formal query intent. | The five RouteB fresh-fold failures from the first full-suite run were fixed; the RouteB fresh-fold and research-access regression tests pass (75 tests). |
| CI acceptance and KL weighting | Added explicit RouteB suite execution to PR/nightly/release CI and a pinned Python 3.12.10/JAX CPU worker smoke. Corrected hierarchy KL weights and added E3/E3b/E4 causal tests. | Workflow YAML parses; the latest local JAX suite and three-variant CPU smoke pass. This is engineering evidence only; no registered hierarchy promotion is claimed. |

## Verification run

- Previous full workspace run before this R2 training integration: **885 passed, 8 skipped, 0 failed**. The full suite was not rerun after this integration; the updated world-model suite below covers the changed package and callers.
- `pytest tests/world_model -q`: **136 passed, 5 skipped** after adding the approved 60/15/15 gap, risk-horizon weighting, checkpoint JSON round-trip, and aggregate-report digest regressions. The five skips require JAX in the Python 3.14 host; four JAX-specific hierarchy/risk-weight checks pass in the isolated RouteB Python 3.12.10 worker, and the synthetic weighted-E2 train/checkpoint/reload/forecast smoke passes.
- JAX skip resolution: the RouteB Python 3.14 host suite reports **138 passed, 6 JAX-only skips**; all six exact skipped node IDs pass in RouteB's locked Python 3.12.10/JAX 0.11.2 CPU worker (**6 passed**). Added those node IDs to the dedicated JAX CI job so the combined verification does not silently omit them. The host runtime remains intentionally JAX-free; this is an environment split, not a missing RouteB dependency.
- Latest preregistered R2 native-risk validation: ETHUSDT `[2021-10-10, 2022-01-12)` completed on three seeds. Native 15m log-RV CRPS passed in seeds 31 and 20261018, but seed 102 was inconclusive (E2-minus-E0 `-0.02506`, 95% CI `[-0.05482,+0.00989]`, Holm `p=0.26687`), so the preregistered all-seed gate failed. E2 native 15m return CRPS was worse across all seeds; generated-path volatility persistence and tails remained low. All report, seed, predictive and pooled hashes verify; Test was not decoded. See [`R2_ETHUSDT_NATIVE_RV15_RESULTS_20261019.md`](R2_ETHUSDT_NATIVE_RV15_RESULTS_20261019.md).
- `tests/world_model/test_routeb_predictive_evaluation.py`, `test_routeb_fold_evaluation.py`, and `test_routeb_e0.py`: **21 passed**.
- Pinned worker CPU smoke: **PASS**, Python 3.12.10, JAX/JAXlib 0.11.2, 4 updates, 92 valid targets per risk horizon, Train-fitted horizon barriers, finite risk loss, parameter/optimizer/sampler/PRNG resume parity, prior alignment artifact, deterministic forecasts and worker IPC.
- Updated B2 CPU smoke: **PASS**, including direct and worker-exported prior risk means/scales/barrier logits with shapes `[3,5,4]`, `[3,5,4]`, and `[3,5,2]`.
- Synthetic CLI/Fold worker smoke: **PASS**, including batch prior risk artifact `[16,5,4]`, valid positive scales, 5 configured risk horizons, and report SHA-256. Synthetic only; it is not market skill evidence.
- Fresh CI-equivalent worker environment: `uv sync --project packages/world_model/environment --locked --python 3.12.10` installed 61 locked packages under RouteB `.tmp`; `uv pip check` passes. The exact CI prior/gradient smoke and B2 CPU smoke both **PASS** in that environment; B2 completed 4 updates and matched 2+2 resumed against uninterrupted parameters, optimizer state, sampler order and PRNG.
- Pinned worker CUDA smoke: **PASS**, Python 3.12.10, JAX/JAXlib 0.11.2, `cuda:0`, 4 updates, 92 valid targets per risk horizon, Train-fitted horizon barriers, finite risk loss, parameter/optimizer/sampler/PRNG resume parity, prior alignment artifact, deterministic forecasts and worker IPC. Peak live allocation was 268,638,976 bytes.
- Prior/risk gradient and causal mutation smoke: **PASS**, synthetic mechanism only. Free prior rollouts at horizons 1/5/15/60 have valid shapes and finite values. Constant-input loss and latent-feature variance are finite; this is a negative-case diagnostic, not a real-market collapse or quality claim. Hierarchy KL weight gradients are `[2.0, 3.5]`, confirming independent unit-slope weighting.
- CI-equivalent static checks: workspace MyPy **184 source files, no issues**; Ruff `packages ci qa tests` **PASS**; root and worker `uv lock --check` **PASS**; CI workflow YAML parses.
- Current RouteB source identity: 428 files, SHA-256 `94735e0299d979db0073fef35c2f7faf1e84015f41bee96f8321b22fd900aebe`; manifest verification passes. The source-manifest walker now checks generated-directory exclusions before stat calls so inaccessible WSL environment symlinks do not break refresh.

## Research gates and scope

The LINKUSDT, DOGEUSDT, AVAXUSDT, and LTCUSDT runs used separate RouteB one-shot Train/Validation scopes and did not decode Test or Holdout payloads. LINK's primary 15m return CRPS is INCONCLUSIVE with the point estimate favoring S0. DOGE's point estimate favors E2, but its paired CI includes zero and Holm `p=1.0`. AVAX's point estimate favors S0; its paired CI is above zero, but Holm-adjusted `p=0.232` does not pass the family-wise gate. LTC's 1,000-update E2 point estimate favors E2 by `-0.00001606`, but its paired CI includes zero and Holm `p=1.0`; rollout diagnostics show weaker volatility clustering and lower upper volatility tails than observed. The AVAX direct risk head also underperforms the statistical baseline on its 15m return score across all three seeds; corrected log-volatility scores are valid but do not establish a consistent improvement. The first DOGE native log-volatility score was invalid because the scorer compared log-RV forecasts with raw RV observations. No performance or economic benefit is claimed.

The v3.0 handoff prioritizes R0/R1/R2 and gates R3/R4/R5/R6 on R2 evidence. Continue R2 work on conditional volatility and extreme-risk forecast quality with a fresh scope for each registered experiment. BTCUSDT, SOLUSDT, XRPUSDT, BNBUSDT, ADAUSDT, LINKUSDT, DOGEUSDT, AVAXUSDT, and LTCUSDT are represented in the RouteB access ledger and cannot be reopened. R2 regularizer ablations, strict three-level conditional RSSM, synthetic slow-scale feedback, Native execution parity, and Actor-Critic/WFO remain closed until the documented gates are met. A first 2025-09 to 2026-10 ETHUSDT Fold attempt was rejected by the project ledger before payload access because it overlapped legacy ETH scopes. The ledger-verified ETHUSDT 2024-01-10 to 2024-04-12 60/15/15 experiment completed with 94 checksum-verified Binance archive files, 133,920 prepared bars, three seeds, 1,000 updates per candidate and seed, and Test excluded. E2 risk weight 2.0 slightly improved primary 15m CRPS over the strongest statistical baseline, but the paired interval crossed zero and Holm p was 1.0; it did not pass promotion. Rollout volatility remained understated, and volatility clustering was weaker than E1. The aggregate digest was canonically sealed and independently verified; seed reports and pooled forecast hashes pass. See [`R2_ETHUSDT_RISK2_60_15_15_RESULTS_20261019.md`](R2_ETHUSDT_RISK2_60_15_15_RESULTS_20261019.md). Revision 1 of the long-horizon risk weighting experiment registered an unused 2024 scope and completed seed-31 training but failed when JSON checkpoint identity compared list/tuple weights; that scope is consumed and produced no forecasts. The checkpoint recipe is fixed and the new weighted-E2 synthetic train/reload/forecast smoke passes. Revision 2 completed on the ledger-verified ETHUSDT 2023-04-13 to 2023-07-15 interval. Its 15m E2 CRPS point estimate beat E0 S1, and the unadjusted 95% interval favored E2, but Holm-adjusted p was 0.342; the primary gate fails. E2 free rollouts remained more under-dispersed and had weaker volatility clustering than E1. Test stayed undecoded. See [`R2_ETHUSDT_LONG_HORIZON_RISK_RESULTS_20261019_REV2.md`](R2_ETHUSDT_LONG_HORIZON_RISK_RESULTS_20261019_REV2.md). The previous aggregate LTC report checksum discrepancy is documented in its result note; seed-level report checksums pass. The 18 CI golden requirements and their current evidence/deferred items are itemized in [`V3_CI_GOLDEN_COVERAGE.md`](V3_CI_GOLDEN_COVERAGE.md).

## Runtime isolation

- Host workspace environment remains at `D:\Program Files\NautilusTrader_RL_ES_RouteB\.venv` (Python 3.14.7).
- Windows CPU worker: `packages/world_model/environment/.venv-win-cpu` (Python 3.12.10, JAX/JAXlib 0.11.2). The host's Python 3.14 intentionally remains free of JAX; invoking it directly caused the apparent missing-package report.
- WSL CUDA worker: `packages/world_model/environment/.venv-wsl` (Python 3.12.10, JAX/JAXlib 0.11.2).
- From PowerShell, run `qa/world_model/check_routeb_jax.ps1` to compile and execute a small JAX operation in both isolated workers. The Windows worker is CPU-only; CUDA training stays in WSL.
- Its interpreter installation is stored under Route B's ignored `.tmp/uv-python`; installed packages remain inside `.venv-wsl`.
- The installed DreamerV3 checkout is pinned separately under Route B's ignored `.tmp/dreamerv3-route-b`; upstream source was not modified.
- UV's download cache and the physical GPU are shared host resources. Package installation and imports use the Route B worker environment and Route B checkout.


The subsequent frozen ETHUSDT 2022-10-01 to 2023-01-02 Student-t prior-head experiment also completed. At the primary 15m return CRPS, E2 was worse than strongest E0 S1 by `+0.00003712` (95% CI `[+0.00000517,+0.00006403]`, Holm p `0.23188`). The native E2 risk head did significantly improve 30/60m log-realized-volatility CRPS across all three seeds (Holm p `0.0049975` per comparison), while free-rollout volatility clustering remained weak. R2 remains NOT PROMOTED, so R3/R4/R5/R6 remain gated. Continue with a new fixed distribution-family hypothesis and unconsumed scope; see [`R2_ETHUSDT_STUDENT_T_RESULTS_20261019.md`](R2_ETHUSDT_STUDENT_T_RESULTS_20261019.md).

## Latest R2 distribution result: Gaussian mixture

The ETHUSDT 2023-01-02 to 2023-04-05 two-component prior-mixture experiment completed on three seeds. E2 failed the primary 15m return CRPS gate: `0.0016580219` versus E0 S1 `0.0016401095`, E2-minus-S1 `+0.0000179125`, 95% CI `[-0.0000321407,+0.0000774650]`, Holm p `1.0`. Native 30/60m log-volatility risk scores improved in all seeds, but generated volatility clustering remained weak. Aggregate, predictive and per-seed report digests and pooled forecast hashes verify; Test remained undecoded. R2 is still not promoted. See [`R2_ETHUSDT_MIXTURE_GAUSSIAN_RESULTS_20261019.md`](R2_ETHUSDT_MIXTURE_GAUSSIAN_RESULTS_20261019.md) and the report under `.artifacts/world_model/fresh_ethusdt_20230102_mixture_ensemble_20261019/`.

## Latest R2 risk-loss scale result

The ETHUSDT 2022-02-01 to 2022-05-05 risk-scale-0.5 experiment completed. E2 failed the primary 15m gate: `0.0016427192` versus E0 S1 `0.0016327746`, difference `+0.0000099446`, paired 95% CI `[-0.0000297960,+0.0000494486]`, Holm p `1.0`. Its improvement against S0 was significant before correction but Holm p `0.05547`; it does not pass. Native 30/60m log-RV gains persisted, but generated volatility clustering stayed weak. Integrity digests pass and Test was not decoded. See [`R2_ETHUSDT_RISK05_RESULTS_20261019.md`](R2_ETHUSDT_RISK05_RESULTS_20261019.md).

## Latest R2 risk-loss scale result: 0.25

The ETHUSDT risk-scale-0.25 run completed on a new 2022-05-10 to 2022-08-11 source interval. E2 nearly tied E0 S1 at 15m CRPS (`0.0034760011` vs `0.0034747115`), but its paired CI crossed zero and Holm p was `1.0`. The lower scale also removed the significant native 30/60m log-RV gains observed at 0.5. Free-rollout clustering and upper tails remained understated. All report digests pass; Test was not decoded. R2 is still not promoted. See [`R2_ETHUSDT_RISK025_RESULTS_20261019.md`](R2_ETHUSDT_RISK025_RESULTS_20261019.md).

## Latest R2 risk-scale transfer result

Risk scale 0.5 transferred to ETHUSDT July-October 2023 and completed on three seeds. E2's point estimate beat the strongest statistical baseline (S2) at 15m, `0.0007201673` versus `0.0007362676`, but its 95% interval crossed zero and Holm p was `1.0`. The native 30m risk head passed in all seeds and 60m in two seeds; free-rollout persistence and upper tails stayed low. The one-shot report digests pass; Test was not decoded. R2 remains unpromoted. See [`R2_ETHUSDT_RISK05_TRANSFER_RESULTS_20261019.md`](R2_ETHUSDT_RISK05_TRANSFER_RESULTS_20261019.md).

## Latest ETHUSDT R2 transfer evidence

The risk-scale-0.5 recipe was confirmed once on the adjacent 2023-10-16 to 2024-01-17 ETHUSDT interval. Its 15m E2 CRPS point estimate beat strongest E0 S2 (`0.0014828516` vs `0.0014909569`), but the paired interval crossed zero and Holm p was `1.0`; R2 promotion remains blocked. E2 native 30m log-RV scores passed in all three seeds and 60m in two, while free rollouts remained under-dispersed. All provenance digests verify and Test was not decoded. Continue within R2; R3/R4/R5/R6 remain gated. See [`R2_ETHUSDT_RISK05_ADJACENT_RESULTS_20261019.md`](R2_ETHUSDT_RISK05_ADJACENT_RESULTS_20261019.md).

## Latest R2 loss-alignment experiment

The Gaussian CRPS training objective ran on the unused ETHUSDT 2021-01-02 to 2021-04-05 scope. E2 15m CRPS was slightly worse than strongest E0 S1 and inconclusive after paired bootstrap/Holm (`+0.0000021390`, CI `[-0.0000520096,+0.0000839570]`, Holm p `1.0`). Native 30/60m log-RV scores improved across all three seeds; Test stayed undecoded. The objective, gradient path, full world-model suite (138 passed, 5 JAX-host skips), aggregate/seed reports, and pooled artifacts verify. R2 remains unpromoted; R3/R4/R5/R6 remain gated. See [`R2_ETHUSDT_CRPS_OBJECTIVE_RESULTS_20261019.md`](R2_ETHUSDT_CRPS_OBJECTIVE_RESULTS_20261019.md).

## Latest R2 evidence: 15-step open-loop CRPS

The three-seed ETHUSDT 2021-04-05 to 2021-07-08 Validation experiment completed with E2 open-loop CRPS at fixed horizon 15. Its 15m return CRPS point estimate beat the strongest statistical baseline, but the registered bootstrap/Holm gate failed (`p=0.1424`, Holm `p=1.0`); E1 retained a lower point estimate. E2 native 30/60m log-realized-volatility scores improved in all seeds, but free-rollout volatility clustering remained below realized and the 95th-percentile rolling volatility was over-wide. The report and all three seed digests plus pooled artifacts verify; Test was not decoded. The experiment confirms that the open-loop objective executes end to end, but does not establish R2 promotion or authorize R3-R6. See [`R2_ETHUSDT_OPENLOOP_CRPS_RESULTS_20261019.md`](R2_ETHUSDT_OPENLOOP_CRPS_RESULTS_20261019.md).

## Latest R2 adjacent confirmation: open-loop CRPS

The 15-step E2 open-loop CRPS recipe was tested once on the adjacent ETHUSDT July-October 2021 interval. E2 failed the primary gate: 15m return CRPS `0.0031071156` versus strongest statistical baseline E0 S1 `0.0030902426`, with E2-minus-S1 `+0.0000168730`, paired 95% CI crossing zero and Holm `p=1.0`. The result does not replicate the previous favorable point estimate. Native volatility risk results were mixed, and free rollouts understate realized clustering and spread. Aggregate/predictive/seed digests and pooled hashes verify; Test remained unread. No R2 promotion or R3-R6 authorization follows. See [`R2_ETHUSDT_OPENLOOP_CRPS_ADJACENT_RESULTS_20261019.md`](R2_ETHUSDT_OPENLOOP_CRPS_ADJACENT_RESULTS_20261019.md).

## Latest registered BTCUSDT Student-t rollout result

The Student-t tempered-risk run on `[2022-10-23, 2023-01-25)` completed its three-seed Validation experiment and failed the primary path-derived 15m log-RV CRPS gate in the harmful direction for every seed (E2-minus-E0 `+0.499864`, `+0.495415`, `+0.550906`; all Holm p `0.005497`). Generated rolling-RV mean/p95 were materially above realized. Aggregate, per-seed, predictive and pooled hashes verify; Test was not decoded. Student-t risk-conditioned rollouts are not promoted. See [`R2_BTCUSDT_STUDENT_T_TEMPERED_RISK_RESULTS_20261020.md`](R2_BTCUSDT_STUDENT_T_TEMPERED_RISK_RESULTS_20261020.md).

## Current R2 diagnostic in progress

A single Gaussian rollout ablation is preregistered on the fresh BTCUSDT interval `[2023-01-25, 2023-04-29)`: retain the Gaussian prior and native risk-head training loss at scale 0.5, while disabling risk-conditioned rollout scaling. The RouteB ledger preflight returned zero overlapping access and query-intent rows for Train+Validation `[2023-01-25T00:01, 2023-04-14T00:01)`. Ninety-four official Binance archives passed checksums and ZIP checks; 135,360 bars were curated into an isolated RouteB source. The three-seed run is pending; Test remains unread. No R2 promotion or R3-R6 gate change is implied. See [`R2_BTCUSDT_GAUSSIAN_NO_ROLLOUT_RISK_PRE_REGISTRATION_20261020.md`](R2_BTCUSDT_GAUSSIAN_NO_ROLLOUT_RISK_PRE_REGISTRATION_20261020.md).

## Latest R2 Gaussian rollout ablation

The BTCUSDT `[2023-01-25, 2023-04-29)` three-seed Validation run disabled rollout risk scaling while retaining the Gaussian prior and risk-head loss scale 0.5. The path-derived 15m log-RV comparison was inconclusive in all seeds (E2-minus-E0 `+0.030780`, `-0.013814`, `+0.001799`; all Holm p `1.0`). E2 rolling-RV mean/p95 stayed below realized (`0.000426/0.000809` vs `0.000499/0.001089`), with absolute-return lag-1 correlation `0.0090` vs `0.1221`. This removes the severe over-expansion from the Student-t candidate but leaves volatility persistence unresolved. All aggregate, predictive, seed and forecast artifact digests verify; Test remained unread. R2 is not promoted. See [`R2_BTCUSDT_GAUSSIAN_NO_ROLLOUT_RISK_RESULTS_20261020.md`](R2_BTCUSDT_GAUSSIAN_NO_ROLLOUT_RISK_RESULTS_20261020.md).

## Current R2 Gaussian strength-0.25 diagnostic

The next diagnostic is preregistered for BTCUSDT `[2023-04-29, 2023-08-01)`, with Train+Validation `[2023-04-29T00:01, 2023-07-17T00:01)`. RouteB access and query-intent ledger preflight returned zero overlaps. The frozen E2 recipe uses a Gaussian prior and native risk loss scale `0.5`, with geometric rollout risk-conditioning strength `0.25`; no strength sweep is permitted. Official source archives are being prepared before the registered reader runs. See [`R2_BTCUSDT_GAUSSIAN_RISK025_PRE_REGISTRATION_20261020.md`](R2_BTCUSDT_GAUSSIAN_RISK025_PRE_REGISTRATION_20261020.md).

## Latest R2 Gaussian rollout-risk strength-0.25 result

The BTCUSDT `[2023-04-29, 2023-08-01)` experiment completed on three seeds. Path-derived 15m log-RV CRPS worsened against the selected E0 in all seeds (`+0.070344`, `+0.052854`, `+0.101871`); seed 20261018 was Holm-significant (`p=0.005997`), and seed 31's interval was harmful but narrowly missed the adjusted threshold (`p=0.050975`). Generated mean rolling-RV was high while the p95 stayed low; absolute-return persistence was near zero versus realized `0.0806`. Aggregate, predictive, seed and artifact hashes verify; Test stayed unread. No promotion. See [`R2_BTCUSDT_GAUSSIAN_RISK025_RESULTS_20261020.md`](R2_BTCUSDT_GAUSSIAN_RISK025_RESULTS_20261020.md).

## Current R2 Gaussian strength-0.1 transfer

A single lower-strength Gaussian transfer recipe is preregistered on BTCUSDT `[2023-08-01, 2023-11-03)`. Train+Validation `[2023-08-01T00:01, 2023-10-19T00:01)` had zero overlapping RouteB access or query-intent rows. The candidate retains the Gaussian prior and native risk loss scale `0.5`, with geometric rollout risk-conditioning strength `0.1`. Data preparation and a fixed three-seed run are next; no sweep is permitted. See [`R2_BTCUSDT_GAUSSIAN_RISK01_PRE_REGISTRATION_20261020.md`](R2_BTCUSDT_GAUSSIAN_RISK01_PRE_REGISTRATION_20261020.md).

## Latest R2 Gaussian rollout-risk strength-0.1 transfer

The BTCUSDT `[2023-08-01, 2023-11-03)` transfer completed. E2-minus-selected-E0 path-derived 15m log-RV CRPS was negative in all seeds (`-0.107479`, `-0.087662`, `-0.045661`); seeds 31 and 102 passed Holm (`0.005497` each), but seed 20261018 remained inconclusive (95% CI `[-0.105167,+0.016583]`, Holm `0.146427`). E2 average CRPS `0.275592` remained above E1 `0.244568`; realized volatility persistence was also stronger than generated persistence. All integrity hashes verify and Test stayed unread. R2 is not promoted; an adjacent fixed-strength transfer is next. See [`R2_BTCUSDT_GAUSSIAN_RISK01_RESULTS_20261020.md`](R2_BTCUSDT_GAUSSIAN_RISK01_RESULTS_20261020.md).

## Latest R2 adjacent Gaussian strength-0.1 transfer

The fixed strength-0.1 BTCUSDT recipe was evaluated on the adjacent `[2023-11-03, 2024-02-05)` scope. E2-minus-selected-E0 path-derived 15m log-RV CRPS was favorable in all seeds (`-0.036284`, `-0.083410`, `-0.081016`); seed 31's CI was below zero but Holm p `0.087956` missed the registered `0.05` threshold. Seeds 102 and 20261018 passed individually. Pooled E2 CRPS was `0.204828` vs E0 `0.277485`, but rolling-RV mean/p95 and absolute-return persistence remained below realized. Aggregate, predictive, seed, path, risk and pooled integrity checks all pass; Test was not decoded. Keep R2 unpromoted and R3-R6 gated. Next focus is a registered training-time path-consistency objective, not another rollout-strength sweep. See [`R2_BTCUSDT_GAUSSIAN_RISK01_ADJACENT_RESULTS_20261020.md`](R2_BTCUSDT_GAUSSIAN_RISK01_ADJACENT_RESULTS_20261020.md).



## Latest R2 result: TRXUSDT train-time path log-RV CRPS

The frozen three-seed TRXUSDT [2023-01-01, 2023-04-05) evaluation completed with verified report/checkpoint/pool digests and Test/Holdout unread. E2's path-derived 15m log-RV CRPS beat selected E0 S1 by 0.018597 on average, but the paired 95% CI crossed zero [-0.044814, +0.007603], p=0.169915; every per-seed Holm gate failed. E1 scored better than E2. Generated OHLCV/path structure remained 100% valid; volatility persistence remained underrepresented. R2 is not promoted and R3-R6 remain gated. Next: implement and synthetically verify a single multi-horizon Train-only path CRPS term (15/30/60m), then freeze a new non-overlapping validation interval. See [R2_TRXUSDT_PATH_LOG_RV_CRPS_RESULTS_20261020.md](R2_TRXUSDT_PATH_LOG_RV_CRPS_RESULTS_20261020.md).

## RouteB checkpoint compatibility fix

The first registered multi-horizon TRXUSDT attempt on [2023-04-05, 2023-07-08) stopped before scoring: checkpoint JSON converted the new horizon tuple to a list while strict forecast identity comparison retained the tuple. E1 training/Validation and E2 training completed; E2 forecast and all scoring stopped before output. One Train+Validation access was recorded; Test/Holdout remained unread. optimizer_recipe() now canonicalizes path_log_rv_crps_horizons with the other sequence-valued configuration fields. The focused recipe and full checkpoint roundtrip tests pass, as does Ruff. The corrected source identity is 593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c. The consumed interval will not be retried; the multi-horizon candidate needs a fresh frozen scope. See [R2_TRXUSDT_MULTIHORIZON_PATH_CRPS_BLOCKED_20261020.md](R2_TRXUSDT_MULTIHORIZON_PATH_CRPS_BLOCKED_20261020.md).

- JAX test entry point: run `qa/world_model/run_routeb_jax_tests.ps1` from PowerShell to execute the six JAX-dependent world-model tests in the isolated Python 3.12.10 CPU worker. Running the same test files directly with the Python 3.14 host will intentionally skip JAX-only cases.
- Runtime verification: `qa/world_model/check_routeb_jax.ps1` passed on Windows CPU and WSL CUDA; both report JAX/JAXlib 0.11.2 and the JIT sum-of-squares result 1240.0. The six isolated JAX tests passed (6 passed).


## 2026-10-20 TRXUSDT multi-horizon path objective result

A preregistered TRXUSDT [2023-07-08, 2023-10-10) Train/Validation comparison completed on three seeds. The E2 multi-horizon 15/30/60m Train-only free-prior path log-RV CRPS objective scored pooled 15m mean 0.235569 versus selected E0 S2 0.257915, but pooled difference -0.022346 has 95% CI [-0.042868,+0.002877] and p 0.05847; seed 31 also failed the all-seed gate. No promotion. E1 mean was 0.212183. OHLCV/path validity was 100%; E2 free paths remained under-dispersed and volatility clustering was weak. Test/Holdout remained unread. Aggregate canonical digest 941fb7c9f14a5cb6a4ab0bbb07be601f8e593edb2efccd71f6b4d70cd88ae014; see [the result report](R2_TRXUSDT_MULTIHORIZON_PATH_CRPS_RESULTS_20261020_NEXT.md).


## 2026-10-20 TRXUSDT paths-16 result

The preregistered 16-path Train-objective follow-up on [2023-10-10, 2024-01-12) did not pass the all-seed gate. Pooled E2-minus-E0 15-bar path log-RV CRPS was -0.007879 (95% CI [-0.022809,+0.013310], p 0.388306); no seed passed the registered CI/Holm requirement, and seed 20261018 reversed direction. Return-CRPS guardrail had no corrected significant harm. Generated OHLCV/path validity was 100%, while generated rolling-RV p95 remained below realized. Test/Holdout was not read; report and artifact digests verified. R2 remains unpromoted and R3–R6 gated. See [the result report](R2_TRXUSDT_PATH_CRPS_PATHS16_RESULTS_20261020.md).

## 2026-10-20 TRXUSDT path-CRPS scale-2 result

A fresh [2024-01-12, 2024-04-15) transfer raised only the Train path log-RV CRPS coefficient from 1 to 2. Pooled E2-minus-selected-E0 15-bar path log-RV CRPS was -0.019916 (95% CI [-0.092988,+0.081086], p 0.651174); all three seed gates failed and E1 outscored E2. No corrected significant return-CRPS harm occurred. Generated OHLCV/path validity was 100%, but generated rolling-RV p95 remained below realized. Test/Holdout stayed unread and all report/artifact hashes passed. R2 remains unpromoted; R3–R6 stay gated. See [the result report](R2_TRXUSDT_PATH_CRPS_SCALE2_RESULTS_20261020.md).

## 2026-10-20 TRXUSDT path-CRPS scale-4 result

On the fresh [2024-04-15, 2024-07-18) scope, pooled E2-minus-E0 15-bar path log-RV CRPS was -0.022947 (95% CI [-0.042399,-0.006394], p 0.012494). Only seed 20261018 passed the full gate; seed 31 missed Holm and seed 102 was inconclusive. The all-seed gate failed, so R2 remains unpromoted. Return-score guardrail passed. E2 generated RV mean/p95 approached realized, but absolute-return persistence remained weak. Test/Holdout stayed unread and all report/artifact hashes verified. A fixed adjacent transfer is next. See [the result report](R2_TRXUSDT_PATH_CRPS_SCALE4_RESULTS_20261020.md).

## Append-only completion record — 2026-10-20

The fixed scale-4 transfer on fresh TRXUSDT [2024-07-18, 2024-10-20) completed on seeds 31, 102 and 20261018. E2 beat the Validation-selected S2 baseline on pooled 15-bar path log-RV CRPS (`0.252385` vs `0.306793`, difference `-0.054408`, 95% CI `[-0.093431,-0.023968]`, p `0.001499`); all three per-seed Holm gates passed. However, seed 102 failed the registered 15-bar return-CRPS guardrail (harm `+0.000126349`, 95% CI `[+0.000055336,+0.000202628]`, Holm p `0.031984`). The full gate fails; R2 stays unpromoted and R3-R6 remain gated. E2 path/bar validity was 100%, but generated absolute-return lag-1 autocorrelation (`0.003779`) remained below realized (`0.051493`). Test/Holdout were not decoded; all report and pooled/risk hashes verified. See [R2_TRXUSDT_PATH_CRPS_SCALE4_ADJACENT_RESULTS_20261020.md](R2_TRXUSDT_PATH_CRPS_SCALE4_ADJACENT_RESULTS_20261020.md).

## Append-only completion record — 2026-10-20

The fixed scale-3 transfer on fresh TRXUSDT [2024-10-20, 2025-01-22) completed on seeds 31, 102 and 20261018. E2 pooled path-derived 15-bar log-RV CRPS beat selected S2 (`0.233027` vs `0.283873`, difference `-0.050846`, 95% CI `[-0.099325,-0.017006]`, p `0.016992`), but seed 102 was inconclusive (difference `-0.007204`, CI `[-0.061248,+0.032775]`, Holm p `1.0`), so the all-seed primary gate failed. The 15-bar return-CRPS guardrail passed on all seeds. E1 scored better than E2 on average (`0.201562` vs `0.233027`); E2 volatility persistence remained low. Aggregate and per-seed digests, pooled forecast archives and risk forecast hashes verified with the RouteB Python 3.14 runtime; Test/Holdout were not decoded. R2 remains unpromoted and R3-R6 remain gated. See [R2_TRXUSDT_PATH_CRPS_SCALE3_ADJACENT_RESULTS_20261020.md](R2_TRXUSDT_PATH_CRPS_SCALE3_ADJACENT_RESULTS_20261020.md).


## Append-only completed result — 2026-10-20

The preregistered paths-32 scale-3 transfer on fresh TRXUSDT [2025-01-22, 2025-04-26) completed with three seeds and 96 pooled validation paths. E2 path-derived 15-minute log-RV CRPS was 0.248922 versus selected E0 S2 0.234689 (difference +0.014233, 95% CI [-0.032769,+0.058097], p 0.535732); E1 scored 0.212234. Seed 31 was significantly adverse, seed 102 inconclusive, and seed 20261018 favorable but inconclusive; no seed passed the preregistered Holm/CI gate. The 15-minute return-CRPS guardrail passed all seeds. Data audit found 112,320 bars with zero missing/duplicate/availability errors; Test/Holdout stayed unread. Pooled forecast, per-seed report and risk forecast hashes matched. Generated RV mean/p95 and absolute-return persistence remained below realized. R2 stays unpromoted and R3-R6 gated. See [R2_TRXUSDT_PATH_CRPS_PATHS32_ADJACENT_RESULTS_20261020.md](R2_TRXUSDT_PATH_CRPS_PATHS32_ADJACENT_RESULTS_20261020.md).

## 2026-10-21 E3/E3b/E4 and F2 engineering smoke

Added distinct `e3_features`, `e3b_fusion`, and `e4_conditional` training variants. E4 injects prior-available day state into hourly transitions and hour/day states into minute posterior and prior transitions; its prior directly predicts the next OHLCV-coordinate target. Static-context forecasts refuse horizons crossing a UTC hour/day close. The checkpoint-bound `rollout_hierarchical_f2_checkpoint()` loads Train-identity-bound parameters, initializes all three RSSM carries from the observed prefix, generates valid minute OHLCV, re-encodes closed synthetic hour/day bars with frozen scale-specific Train scalers, and updates JAX slow carries. On synthetic Train-only data, E3, E3b and E4 passed both full-filter posterior and day/hour coincident-close cases; E4 also passed slow-only. Full-filter prior/posterior logits differed on each generated minute. Checkpoint identity includes dataset and all three scaler fingerprints. All variants trained with finite loss; E4 produced deterministic one-step prior samples; interrupted/resumed parameters and Optax state matched exactly. Isolated runtime: Python 3.12.10, JAX 0.11.2, CPU. The full host world-model suite passed **154 tests / 10 JAX-only skips**; the isolated JAX suite passed **12 tests**, plus the integrated R3 CPU loss/gradient smoke. Test/Holdout and market data were not accessed in those synthetic checks. No registered R4/E3/E3b/E4 Validation run or predictive claim exists.

## 2026-10-21 R3 latent redundancy objective

Implemented the v3-design R3 auxiliary loss in `r2_regularizer.py` and wired it into the RSSM training objective. Fixed-interval masked rows are projected only after sampling to reduce adjacent-minute dependence and dense-layer cost; frozen `alpha=0.005` and `lambda` is configurable but defaults to zero. The zero-lambda control retains the same projector graph/parameter count and receives the same diagnostics. Diagnostics include effective rows, view variance, an O(d²) capped Gershgorin condition bound, both loss terms, and per-report-window loss mean/std/CV across batches. A Validation-only paired scorer compares each R3 arm against lambda zero on common path log-RV and return CRPS. Synthetic CPU tests verify finite nonzero gradients, exact objective deltas at λ=0.01/0.05, paired-score alignment and a two-update Train-only optimizer run. The full host suite is **155 passed / 10 JAX-only skipped**; the isolated JAX suite is **14 passed**, plus model/optimizer smoke. No real-data R3 preregistration or result exists.

## Append-only completion record — R3 regularizer Validation, 2026-10-21

R3 run-02 completed on fresh AVAXUSDT [2024-04-15, 2024-07-16), with 180 common Validation anchors and three lambdas (0/0.01/0.05) × seeds 31/102/20261018. All nine registered attempts completed; both E1 and E2 completed 1,000 updates per arm/seed on CUDA. The pooled paired CRPS deltas versus lambda 0 were inconclusive for both nonzero settings: lambda 0.01 return `-0.0000049149` (95% CI `[-0.0000190027,+0.0000085830]`, p `0.487756`) and path log-RV `-0.0015715825` (CI `[-0.0072867777,+0.0032736075]`, p `0.587206`); lambda 0.05 return `+0.0000033224` (CI `[-0.0000074959,+0.0000156686]`, p `0.574213`) and path log-RV `+0.0013588039` (CI `[-0.0016272565,+0.0044219600]`, p `0.376312`). No setting is promoted. The development result receipt and all nine latest ledger events verify as completed. Test payload remained unopened; 23,040 post-Validation market rows were sealed; economic certification remains false. Full result: [R3_AVAXUSDT_REGULARIZER_RESULTS_20261021_RUN02.md](R3_AVAXUSDT_REGULARIZER_RESULTS_20261021_RUN02.md). The historical statement above that no real-data R3 result exists was accurate when written and is superseded by this append-only record.

## 2026-10-21 Train/Validation hierarchy boundary repair

Added a separately scoped Validation tensor builder that reuses the exact three Train-fitted scalers and records the Validation score interval. A Train-checkpoint-bound Validation prior API now loads checkpoint identity and parameters from Train tensors only, consumes Validation tensors only as causal context, and rejects anchors/targets outside the registered score window. The Train hierarchy builder now accepts an explicit score interval: leading replay rows may form causal warmup but are excluded from minute/hour/day scaler fitting and all loss masks; optimizer windows start at the first scored minute with two visible daily states. The synthetic B3 JAX smoke now exercises an actual 15-step Train-checkpoint-to-Validation forecast. RouteB isolated CPU JAX smoke passed on `cpu:0`; three model variants remained distinct (1,248 / 3,158 / 3,478 parameters), and F2 synthetic feedback checks remained green. Focused hierarchy/F2 tests passed 17/17; complete `world_model` host run passed 159 with 10 JAX skips, and isolated JAX run passed 154 with the CLI module omitted because its separate Gymnasium dependency is not in the world-model environment. These remain engineering checks only. No R4 registered real-data comparison has been run.

## 2026-10-21 R4 registered hierarchy comparison — run 03

The earlier R4 preparation note in the code audit is superseded. Run 01 stopped on an unsafe generated wick; run 02 stopped after E3 completed because one 128-seed forecast batch exceeded GPU memory. Both scopes were recorded failed and not retried. Inference was changed to bounded batches of eight and passed synthetic CPU smoke. The third, disjoint registered Fold completed nine model/seed runs, 1,000 updates each, 60 common Validation anchors, and 32 valid paths per anchor. Run-03 report SHA-256: `6625d54282413a8ec2f97f3e006584fb86bb9879bed7e40db07dcf178dfa0a32`; Test/Holdout payloads remained sealed. E4 failed the all-seed path-log-RV CRPS gate against both simpler baselines, while the return-CRPS guardrail passed. No hierarchy promotion; prefer E3 pending multi-Fold and cross-instrument checks against the recurrent baseline. See [R4 run-03 results](R4_AVAXUSDT_RUN03_RESULTS_20261021.md).


## Append-only completion record — matched-objective R4 E2/E3 DOGEUSDT, 2026-10-22

The frozen objective-matched E2/E3 comparison completed on fresh DOGEUSDT `[2024-08-13, 2025-08-24)` Train/Validation scores using seeds 31, 102, and 20261018. E2 completed 2,067 updates per seed; E3 completed 1,000 updates per seed with auxiliary risk configuration derived from E2. E3-minus-E2 path log-RV CRPS was -0.03203 (95% CI [-0.05594,-0.01033], Holm p 0.002999), +0.10361 (CI [+0.06450,+0.14166], p 0.001499), and -0.05223 (CI [-0.08042,-0.02430], p 0.001499). The mixed seed directions fail the strict all-seed gate. Return-CRPS no-harm passed; there is no hierarchy promotion or economic claim. All six attempts completed and Test payload was not decoded. Full report and hashes: [R4 DOGEUSDT results](R4_DOGEUSDT_E2_E3_RESULTS_20261022.md).


## Append-only completion record — matched-objective R4 DOGEUSDT temporal replication, run 02, 2026-10-22

The fixed-protocol E2/E3 replication completed on the earlier disjoint DOGEUSDT Fold `[2023-08-01, 2024-08-27)` score windows (360/15/15 days). All six candidate/seed attempts completed, all 14 official archives passed checksums, and the 23,040 post-Validation rows remained timestamp-only. E3-minus-E2 path log-RV CRPS was -0.06341 (95% CI [-0.09976,-0.02968], Holm p 0.001499) for seed 31, +0.03938 (CI [+0.00444,+0.07450], p 0.027486) for seed 102, and -0.09101 (CI [-0.11334,-0.06700], p 0.001499) for seed 20261018. Return-CRPS no-harm passed, but the all-seed gate failed again. Across both disjoint DOGEUSDT periods, seed 102 significantly favored E2 and the other seeds favored E3. This demonstrates repeatable seed-direction sensitivity for these two folds, not its cause or general skill. No promotion, economics, or production claim. Report digest: `9bf4b224702a023b0fb609bf105a20f8c89836c716fa8e80c1d4ba2973bc4208`; see [run-02 results](R4_DOGEUSDT_E2_E3_RESULTS_RUN02_20261022.md).
