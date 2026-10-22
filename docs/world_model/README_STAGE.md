# Route B World Model — B0/B1 and WM-0..WM-6

This folder records the first Route B engineering increment. It does not claim
that the market model is trained, calibrated, economically useful, or ready for
trading.

## Implemented

- `packages/world_model` defines immutable 1m market frames and replay contracts,
  projects canonical `rles_data` events, and filters visible bars at an explicit
  availability barrier.
- B0 audits timestamp gaps and duplicates. S0 uses zero drift with Train-only
  fixed volatility; S1 uses EWMA volatility with Train-only standardized
  innovations; S2 samples stationary-bootstrap paths with geometric block
  lengths and scores their one-step marginal. All three use empirical one-step
  log-return CRPS.
- Validation scoring uses the Fold's causal warmup and score windows. Scaling
  statistics and bootstrap source returns come only from Train. The current B0
  baseline is one-step only; it does not establish multi-step path fidelity.
- `rles world-model ...` is routed through the existing CLI. `audit-data
  --registered-train-config` reserves a Train scope before reading. The B0
  baseline command accepts `--registered-fold-config`: it verifies the frozen
  0.4.1 marker and full dataset hash, records a one-shot fresh scope spanning
  Train through Validation, and only then decodes row groups. It rejects any
  prior overlapping access or formal query intent. Before reserving, it checks
  that 0.4.1 row-group/footer time metadata reaches the Fold Test end; no Test
  rows are decoded. It writes
  paired `B0_data_audit.json` and `baselines.json` into a new report directory,
  so both artifacts share one read and Fold identity. The reservation and reports
  include the Git commit plus a deterministic SHA-256 over the full recursive
  `rles_world_model` source tree (including `routeb/`) and workspace lock files.
  Only approved 60/15/15 and
  360/15/15 Fold durations are accepted. Unregistered JSON replay is accepted
  only when explicitly marked `synthetic_test_only`.
- B1 keeps the JAX learner in a separate Python 3.12.10 environment under
  `packages/world_model/environment/`. The main Python 3.14.7 environment has no
  JAX dependency. DreamerV3 source is checked out in ignored `.tmp` at the exact
  design commit; its patch manifest is empty. The adapter calls upstream RSSM
  `observe` and `imagine` with an all-zero `market_dummy` action and never
  forwards policy action values.
- **WM-0/WM-1 engineering:** `features.py`, `temporal.py`, and `dataset.py`
  provide a frozen `market_features_v1` schema, complete UTC 1h/1d aggregation,
  availability barriers, 15-minute/24-hour/30-day warmups, Train-score-only
  robust scalers, and invertible next-minute OHLCV targets. The Train builder
  stops at `train_end_ns`, including when a wider Fold replay also contains
  Validation rows. `learner.py` trains the pinned discrete DreamerV3 RSSM with
  a low-rank joint Gaussian reconstruction term, exact-zero hurdle logits,
  Dreamer dynamics/representation KL, and constant zero market action. Its
  learned 1m/1h/1d encoders use bounded sigmoid gates; no policy action, account,
  position, PnL, or future funding field is accepted.
- **WM-2/WM-3 state components:** `scales.py` implements a GRU updated only by
  complete contiguous UTC hours, a fixed 30-day D0 rolling-statistics state,
  and availability-masked cross-scale fusion helpers. The Train sequence
  includes the deterministic frozen hourly GRU state (32 values) and D0 state
  (8 values) beside the four raw/scaled feature groups, for 79 model inputs.
  The GRU transition weights use the versioned deterministic seed in the feature
  hash; downstream RSSM encoders and gates are learned. A separate hourly
  auxiliary head is trained only at complete UTC hour boundaries against the
  next complete hourly candle; its loss is masked when the target falls beyond
  Train.
- **WM-4/WM-6 engineering:** posterior inference has incremental and sequence
  APIs with a shared per-minute RNG schedule. Prior forecasting requires one
  observed market carry and defaults to 128 independent paths; the decoder
  samples a joint low-rank Gaussian plus zero-mass hurdles, then decodes each
  path into legal 1m candles and only complete UTC aggregate candles. The
  deployment API seeds an independent aggregator and slow-state snapshot per
  path from the causal 30-day history; generated complete hours update each
  path's hourly GRU, and generated complete days update its D0 state. Updated
  slow states condition subsequent prior decoder emissions. Forecast
  target support is explicitly clipped to log-return `[-0.1, 0.1]`, wick
  `[0, 2]`, and log-volume `[0, 30]` before OHLCV decoding. `runtime.py` exposes
  an action-free prior-only inference API. `bundle.py` stores parameter and
  optimizer PyTrees as checksummed NPZ arrays plus a compatibility manifest;
  it rejects existing targets, object arrays, wrong source/feature/config
  identity, and mismatched tensor structures. The CPU worker smoke covers an
  optimizer step, 79-input inference, incremental parity, 128 x 120 prior
  samples, generated-hour aggregation, inference and training-state resume.
- **WM-5 status:** B0 S0/S1/S2 mechanisms and the new world-model Train feature
  pipeline operate on synthetic sequences. A clean 0.4.1 fresh Fold is still
  unavailable, so no real-market RSSM fitting, WFO, ablation, calibration,
  performance ranking, or forecast skill conclusion has been run. CPU smoke is
  an engineering test, not a market-model evaluation.
- B-L contract exploration adds opt-in `execution_schema_version=0.5.0`
  DTOs under `rles_contracts`, an OHLC lifecycle prototype under
  `rles_gym_bridge`, and a separate `NativeLimitSubmitter` using Nautilus public
  Strategy order APIs. The native adapter binds frozen instrument tick/step and
  unit rules, translates GTC LIMIT and CANCEL intents, tracks pending IDs, and
  leaves order/account facts to the native engine. SIM regression observed a
  native pending LIMIT and its cancellation acknowledgement. An explicit expiry
  maps to Nautilus GTD and is processed on the native clock. CANCEL intents are
  scheduled with native strategy-clock alerts at their effective timestamps;
  replay requires that timestamp to align with a complete native batch.
  SIM/OHLC comparisons now cover cancellation before a later touch (L-04) and
  fill before a later cancellation (L-05), 30 untouched pending GTC bars
  without economic events (L-03), and marketable-at-open TAKER fill price and
  quantity (L-07). A cancel timestamp inside an unordered bar path fails
  closed (L-06), and off-tick price, off-step quantity, or below-minimum
  notional orders are rejected before native submission (L-10). `EngineSession`
  exposes a separate `advance_limit()` path that reserves the intent identity,
  applies it at the decision barrier, and rejects replay batches before its
  effective timestamp. It enforces exact tick,
  quantity-step, minimum-quantity and minimum-notional rules; keeps GTC orders
  pending; fills whole quantities at the frozen limit price; tags marketable
  open fills as taker; selects the earliest known cancel/expiry event (expiry
  wins an exact timestamp tie); and fails closed on activation/cancel/expiry
  races that OHLC cannot order. A marketable open fill precedes later
  intrabar cancellations. A cancellation strictly earlier than activation
  resolves before that bar's trigger, even when both timestamps fall inside
  the same OHLC bar. It does not alter the 0.4.0 MARKET plan. The native adapter
  is not yet connected to the OHLC lifecycle, historical Gym/planner,
  account/funding parity harness, or fast simulator.

## Verify

## Route boundary

The RouteB package code has no RouteA package imports. Three archived 2026-10-16
plans retain absolute RouteA artifact paths as historical provenance; the
Train-signal, forecast, and zero-target runners and their preparation scripts
now reject plan, model, market, rules, and output paths outside their own
workspace before reading or writing them. These legacy plans therefore cannot
reuse RouteA artifacts from this checkout.

From the repository root (the source package paths are included by the project
environment when installed):

```powershell
uv sync --project packages/orchestrator --extra application --group dev
uv run --project packages/orchestrator --extra application rles world-model audit-data --synthetic-replay-json <synthetic-replay.json>
uv run --project packages/orchestrator --extra application rles world-model audit-data --registered-train-config <registered-train-config.json>
uv run --project packages/orchestrator --extra application rles world-model baselines --registered-fold-config <fresh-fold-config.json> --report-dir <new-report-dir>
uv run --project packages/orchestrator --extra application --group dev pytest -q tests/world_model tests/test_limit_execution_contract.py tests/test_research_access.py tests/test_curated_window_read.py
.venv-world-model/Scripts/python.exe qa/world_model/b1_cpu_smoke.py --checkout .tmp/dreamerv3 --output .tmp/world-model-b1-cpu-smoke.json
packages/world_model/environment/.venv/Scripts/python.exe qa/world_model/wm1_cpu_smoke.py --checkout .tmp/dreamerv3 --output .tmp/world-model-wm1-cpu-smoke.json
```

The `synthetic_test_only` replay CLI does not confer research-data authorization.
For production baselines, create a config that freezes the curated marker, full
snapshot hash, approved `FoldManifest`, instrument, and a stable
`routeb-fold-*` label. The registered-fold reader records source and Fold
identity before decoding any payload. Its fresh reservation is one-shot; a
failed attempt remains consumed and cannot be rerun under a new output path.
Only row-group-scoped 0.4.1 data is accepted. Do not use Test/Holdout or any
previously consumed interval to create a new Route B result.

## Current gate

- **B0 engineering:** PASS for contract, causal barriers, gap rejection, CRPS,
  S0/S1/S2 mechanics, fresh one-shot scope registration, marker/Fold identity
  binding, footer-only Test coverage checks, Test payload exclusion, repository
  and source-code identity, and paired registered baseline report generation.
- **B0 data/baseline evidence:** PASS for registered 0.4.1 Train/Validation
  reads and real market baseline scoring. Fresh ETHUSDT folds from 2024-01,
  2024-04 and 2023-04 were registered and evaluated without decoding Test.
  These runs do not promote E2; continue only with a new hypothesis on an
  unconsumed scope.
- **B1 CPU:** PASS reproduced from this checkout using Python 3.12.10, JAX
  0.11.2, and DreamerV3 commit `e01491fad6434b2245a3b8ca201dd7faedcc458c`.
  Same-seed action-independence and NPZ state roundtrip passed; the ignored
  environment and source checkout remain local setup artifacts.
- **WM-0..WM-4/WM-6 synthetic engineering:** PASS on the focused synthetic
  suite and pinned CPU worker. This verifies causal feature and scaler plumbing,
  complete-period state updates, Train-window isolation, one optimizer step,
  79-feature RSSM input, fixed-seed incremental/batch posterior parity, 128
  independent 120-minute prior paths, candle legality/hour aggregation,
  recursive generated slow-state updates, and checksummed optimizer/model
  bundle resume. It does not verify CUDA speed, multi-seed stability, or
  forecast accuracy.
- **WSL2/CUDA:** The system WSL Python still has no JAX; RouteB now uses its
  own locked 78-package environment at
  `packages/world_model/environment/.venv-wsl`. The pinned CUDA 13/JAX 0.11.2
  worker ran the synthetic E2 RSSM fit, exact model/Optax checkpoint-resume
  parity, forecast and worker IPC on `cuda:0`; see `B2_cuda_smoke.json`. This
  verifies the CUDA learner mechanism, not throughput, market skill, or economics.
- **B2 engineering:** Synthetic CPU/CUDA training, resume parity, forecast and
  worker distribution export pass on the pinned DreamerV3/JAX environment.
  E1/E2 fitting and E0/E1/E2 comparisons also completed on multiple registered
  real Train/Validation folds. No run passes the preregistered R2 promotion
  gate. No Actor, policy imagination, or economic claim is implemented.
- **B-L:** PASS for 0.5.0 contract and OHLC lifecycle cases, native SIM GTC
  LIMIT submit/cancel through `EngineSession`, and one shared non-marketable
  touch-fill trace with equal native/OHLC fill price, quantity, and terminal
  state; native GTD expiry beats an equal-timestamp touch; and native scheduled
  cancel agrees with OHLC for L-04 (cancel before touch) and L-05 (fill before
  later cancel). Native GTC also remains pending through 30 untouched bars
  without fills, positions or equity changes (L-03), and marketable-at-open
  native fill agrees with OHLC on TAKER classification, price and quantity
  (L-07). Unordered same-bar cancel timing fails closed (L-06), and native
  submission rejects invalid tick, quantity step and minimum notional (L-10).
  Native account recorded the maker commission and marked equity in the touch
  case; OHLC still labels its fee `SCENARIO_REQUIRED`. Remaining order-event
  parity across other races/expiry and partial fills, fee-class parity, native-vs-fast
  account/funding parity, and
  historical Gym/planner integration remain NOT STARTED. This is synthetic SIM
  engineering evidence, not tradable execution evidence.
- **Latest validation:** The full Windows suite passed **814 tests** and skipped
  **5 CUDA-only tests** because no CUDA learner is available. After that run,
  six added native L-03/L-06/L-07/L-10 cases passed within the 12-test native
  adapter suite. Ruff passed across the repository;
  Windows MyPy passed for 37 RouteB package source files (the absent Windows JAX
  stubs were excluded as an environment-only import check). The 316-file source
  manifest verifies at identity
  `f65a8409bfe9af2875f95e97154e03d77e1f77b8406b0446f38c9969285adeef`. The CLI
  unit tests inject a frozen code identity because this copied workspace has no
  Git history; the production identity path is separately exercised through
  the explicit base-commit snapshot manifest.

Evidence: [`B0_data_audit.json`](B0_data_audit.json),
[`baselines.json`](baselines.json), [`B1_env_lock.json`](B1_env_lock.json), and
[`B1_cpu_smoke.json`](B1_cpu_smoke.json).

## 2026-10-18 WM world-model implementation

- **Source identity:** The refreshed manifest covers **331 files** (including
  `qa/world_model/`) and verifies at
  `39acedce01819a19cbf54ef4ff46bb7bb8c439ca4a111689beda89b47882ee52`.
- **Verification:** The current full Windows suite passed **847 tests**, with
  **5 CUDA-only tests skipped** because the main workspace has no CUDA learner;
  53 dependency warnings were emitted by existing third-party packages.
  Focused `tests/world_model` passed **50 tests**. The pinned Dreamer worker on
  Python 3.12.10 / JAX 0.11.2 passed both the B1 action-boundary smoke and the
  WM1 synthetic end-to-end smoke: finite training metrics, 79-feature training
  input, incremental/sequence posterior equality, 128 independent 120-minute
  prior paths, valid minute candles, completed-hour and completed-day slow-state
  feedback, one masked hourly auxiliary gradient, and exact forecast parity
  after safe bundle restore. Ruff passed for the
  WorldModel package/tests/worker smoke, MyPy passed for all 15 WorldModel
  modules, both workspace lockfiles passed `uv lock --check`, and
  `git diff --check` passed with only the existing LF-to-CRLF notices.
- **Evidence boundary:** All new training and forecast runs use a deterministic
  synthetic replay. B0 still blocks a fresh eligible 0.4.1 Fold, so no real
  market fitting, WFO, ablation, calibration, or forecast-skill claim exists.
  The worker is CPU-only here; the model has not been trained on CUDA. The
  hourly GRU transition weights are fixed-seed/frozen; the auxiliary next-hour
  objective trains a separate prediction head, not the GRU weights. Recursive
  slow states condition later decoder emissions, while the pinned RSSM latent
  transition remains market-latent-only. CUDA learner performance and multi-seed
  stability are still unverified.

## Route B v2.2 additive source integration

- Merged the supplied v2.2 additive source pack under
  `packages/world_model/src/rles_world_model/routeb/`, with its tests, CPU smoke,
  and B2 demo config. The pack README, code audit, and integration checklist are
  retained under `docs/world_model/`; the repository root README was preserved.
  Existing `rles_world_model` modules, Route A, NautilusTrader source, and the
  pinned DreamerV3 checkout were not replaced or modified.
- The new RouteB package provides a causal 1m/1h/1d clock and dataset tensors,
  Train-only OHLCV scaler and sequence artifacts, B2 single-scale Dreamer RSSM
  training with a constant zero market action, SHA/PyTree-bound optimizer
  checkpoint resume, forecast request validation, and a separate imagined
  account. `jax_model.py` now validates and inserts the pinned upstream source
  before importing its `embodied` modules.
- Continued WB2 with the missing E0/E1 mechanisms: `evaluation.sample_e0_paths`
  generates deterministic multi-horizon S0/S1/S2 return paths from Train-only
  statistics plus explicitly visible anchor context; `jax_gru.py` and
  `jax_gru_train.py` provide a probabilistic GRU next-bar comparator, optimizer
  updates, full checkpoint resume and recursive forecast paths under the same
  Train NPZ/scaler identity as E2. E0 is NumPy-only; E1 runs in the isolated
  Python 3.12 JAX environment. Frozen worker IPC now exposes E1 train, F1
  OHLCV forecast and exact one-step distribution commands alongside E2; both
  model families share request, provenance and artifact geometry checks.
  Neither accepts policy/account state.
- E1 and E2 now expose checkpoint-bound one-step Gaussian feature parameters;
  the E2 worker has a checksummed `next-distribution` artifact path. The exact
  one-minute close-to-close Gaussian NLL is computed from the two diagonal
  log-gap/body coordinates and frozen Train scaler moments. Multi-step
  cumulative-return NLL remains unavailable unless a model provides a valid
  native density for that aggregate target.
- Added the common-anchor predictive scoring contract in
  `routeb/evaluation.py`: cumulative-horizon CRPS, seeded randomized empirical
  PIT, 90/95% coverage and interval width, direction AUC/Brier/log loss/sign
  accuracy, 1/5% VaR breach rates, predictive-distribution CVaR and separately
  observed loss conditional on VaR breach. Native NLL is included only when each
  model supplies per-anchor density NLL; it is never inferred from a finite
  sample ensemble. `compare_path_models` enforces identical model tensor
  geometry and integer-nanosecond labels strictly after each ordered anchor.
  `paired_block_bootstrap_crps` reports paired model-minus-baseline intervals
  while keeping blocks within continuous anchor intervals.
  `build_predictive_comparison_report` fits volatility quantiles on the supplied
  Train volatility series, bins anchor-time volatility, and applies Holm
  family-wise adjustment over the
  preregistered model-pair/horizon family, and emits a canonical SHA-256
  report identity bound to forecast, label, timestamp, stratum and native-NLL
  array hashes; `verify_predictive_report_digest` detects report mutation. It
  requires per-model config/artifact hashes and records train-example count,
  optimizer updates, parameter count, device, training time and forecast seed;
  shared train-example counts and forecast seed are enforced across candidates. It
  sets `authorization_verified=false` because this layer
  cannot verify access-registry or fold manifests. The APIs use caller-provided
  arrays and do not authorize or load any market dataset.
- Added `ohlcv_generation_diagnostics` for generated/realized rolling RV,
  signed and absolute-return lag-one autocorrelation, OHLCV bar/path validity,
  and wick-excursion Brier/reliability bins. Added
  `bucket_aggregation_consistency` to verify exact synthetic minute-to-hour/day
  OHLCV aggregation for caller-specified F2 bucket ranges. These are diagnostic
  functions and do not train the model or certify market performance.
- The comparison report can include those OHLCV diagnostics and F2 bucket
  consistency results when decoded path tensors are supplied; all such arrays
  are included in its input identity.
- **RouteB v2.2 tests:** 39 passed, 3 JAX-dependent tests skipped in the main
  Python 3.14 environment. Ruff passed for all added RouteB package, test, and
  smoke files. The pinned Python 3.12.10 / JAX 0.11.2 CPU B2 and E1 GRU smokes
  passed:
  four total optimizer steps matched uninterrupted training after a 2-step
  checkpoint/resume; the forecast shape was `[3, 3, 5]`; input data was
  synthetic only. E1 GRU likewise matched resumed/uninterrupted optimizer
  state and generated 3 independent 8-step paths. An initial B2 smoke exposed
  the upstream import-order defect, which was fixed before the passing run.
- **Project regression after source merge:** the complete Windows suite passed
  **888 tests** and skipped **8** (5 existing CUDA-only tests plus 3 JAX-only
  tests that require the separate Python 3.12 worker); 53 existing third-party
  dependency warnings were emitted. After that run, only static type narrowing
  and NumPy typing annotations changed; affected RouteB tests and the pinned
  B2 CPU smoke were rerun successfully. MyPy passes all 17 new RouteB modules.
  The refreshed 358-file source manifest verifies at
  `bde1e5ae20014af1ace7c7d7f725eef70bf9089671534a8125b063e6d408f83b`.
- **Predictive-evaluation extension:** all 101 RouteB tests passed, with 3
  JAX-only tests skipped in the available Python 3.12 shell (JAX is absent
  there); Ruff passed for RouteB source and tests, and MyPy passed for the
  changed `evaluation.py`. A broader MyPy invocation cannot resolve optional
  JAX and pinned upstream dependencies in this shell, so it is not counted as
  passing.
- **Native one-step NLL:** E1's Python 3.12.10/JAX CPU train-resume-forecast
  smoke passed with the added `[3,5]` distribution output and finite exact
  one-step return NLL. The pinned DreamerV3 source was restored at the approved
  upstream commit `e01491fad6434b2245a3b8ca201dd7faedcc458c`; the B2 CPU smoke
  then passed through model training, checkpoint resume, forecast, worker
  subprocess and checksummed NPZ distribution export. A withheld synthetic next
  bar produced a finite exact one-step return NLL. This also exposed and fixed
  two integration defects: Ninjax `pure` output unpacking and NumPy's opaque
  serialization of DreamerV3 bfloat16 values (worker artifacts are now float32).
- The end-to-end smoke also verified F1 market-bar forecast IPC. Its test exposed
  an inconsistent client schema check (the worker correctly emits
  `[paths,horizon,5]`, while the client required rank 4); `WorkerClient` now
  validates the emitted rank-3 shape, declared positive path/horizon counts,
  finite values and OHLCV geometry. The pinned worker smoke passes both F1 bar
  paths and the exact one-step distribution endpoint.
- The World Model regression suite now passes **104 tests**, with **3 JAX-only
  skips** in the default Python environment. Ruff passes on RouteB source/tests/
  smoke and MyPy passes `routeb/evaluation.py`.
- This is B2 **engineering/mechanism** evidence, not E2 model-selection
  evidence. No authorized fresh Train/Validation window was read, no real
  market fit or E0/E1/E2 comparison was run, and no forecast skill or economic
  conclusion is available. The source design explicitly gates B3 learned
  hierarchy, policy imagination, and any financial claim on those later
  validations; they remain unstarted.
- **Latest repository regression after E1/E2 worker IPC completion:** the full
  Windows suite passes **903 tests**, with **8 skipped** (5 CUDA-only and 3
  JAX-only in the Python 3.14 environment); 53 existing dependency warnings.
  `tests/world_model` passes **104**, with 3 JAX-only skips. Ruff passes for
  RouteB code/tests/smokes; MyPy passes for evaluation and worker client,
  and the B0 CLI passes with missing optional dependency stubs ignored. The
  pinned Python 3.12.10/JAX 0.11.2 CPU B2 smoke now traverses
  training, resume, direct forecast, checksummed next-distribution export, and
  F1 forecast worker IPC; E1 GRU smoke also passes through worker-side training,
  F1 forecast and next-distribution subprocess endpoints. The 359-file source
  manifest verifies at
  `e8fce2af13da770078cd39bf983dfec33dc3a8e9b159a66b0567fd910e8acdf5`.
- **B0 pre-read identity audit:** `_routeb_code_identity()` now fingerprints the
  complete recursive `rles_world_model` Python source tree, including every
  `routeb/` module. A regression test changes only a nested RouteB source file
  and proves the pre-read SHA changes; the focused CLI suite passes 9 tests.
- **WB2 registered Fold-to-Train bridge:** added `prepare_fold_train_sequences`
  to consume one already-authorized contiguous Train/Validation replay, verify
  exact warmup-to-Validation-end coverage and availability barriers, fit the
  scaler only on Train score coordinates, mask loss targets to the Train score
  half-open interval, and export only Train tensors to the worker. Validation
  rows are used only to satisfy the host-side authorized replay contract and
  are not copied into the Train NPZ. The worker artifact metadata carries the
  frozen Fold hash. The `prepare-fold-train` CLI preflights a fresh output,
  binds the read to the recursive RouteB code identity, invokes the registered
  Fold reader, and writes a provenance summary for the Train-only artifact.
  `train-prepared` then checks the Train NPZ checksum, pinned snapshot/upstream,
  and expected Fold hash before dispatching E1 or E2 to the isolated Python 3.12
  worker. It records shared Train identity, candidate config, optimizer steps,
  parameter count, elapsed training time, device, and checkpoint hash; resume
  is bound to the existing checkpoint identity.
  `build_fold_predictive_report` now creates the fold-bound E0/E1/E2 comparison
  from the registered Train/Validation replay: it fits E0 only on Train score,
  evaluates all candidates on the same ordered Validation anchors, labels and
  forecast seed, and verifies E1/E2 run reports share the same Fold, Train NPZ,
  training budget, example count and device. It rejects replay rows at or past
  Validation end, labels crossing the Validation score boundary, mismatched
  forecast anchor order, and invalid OHLCV paths. The resulting report keeps
  `authorization_verified=false` so callers must retain the registration
  receipt alongside it.
  Regressions cover scaler isolation, target masks, incomplete coverage, late
  availability, CLI reader ordering, E1/E2 worker dispatch and existing-output
  rejection. A pinned Python 3.12 CPU smoke ran both candidates on the same
  synthetic Train artifact with matched seed, batch size, and optimizer steps;
  the 100-step E2 fit lowered its masked next-bar NLL from 5.8945 to 4.5135.
  This is a fixed-synthetic training sanity check, not Validation skill. The
  The previous full Windows regression passed **913 tests**, with **8 skipped**
  (5 CUDA-only and 3 JAX-only). After the subsequent Route B additions,
  `tests/world_model` passes **120**, with 3 JAX-only skips. Ruff
  passes for RouteB source/tests and the new CLI smoke; MyPy passes the changed
  CLI, training, sequence, integration, fold evaluation and smoke modules. The
  53 emitted warnings are existing dependency warnings. After integration into
  the main project, the Route B plus existing project regression set passes
  **207 tests** with **3 JAX-only skips**. Ruff and MyPy pass the merged Route B
  source and authorization adapters. The refreshed 489-file main-project
  source manifest verifies at
  `4349e14fffce781d10b46b2cb7b8a7d9499fcdd5acc8af3aa2f8efca3f916880`.
  No real Train/Validation replay was authorized or read, so this is
  data-boundary and mechanism evidence only; real E0/E1/E2 scoring remains
  pending, and the E3/E4 hierarchy gate remains closed.
- **Validation forecast batching:** the isolated worker now accepts ordered
  anchor batches for both E1 and E2 and returns `[anchors, paths,
  horizon, 5]` OHLCV arrays in one worker process. The host validates the
  checksum, anchor/path/horizon dimensions and every OHLCV path. Added
  `prepare_fold_forecast_batch` to derive each context from the authorized
  Fold replay, apply the frozen Train scaler, enforce per-anchor availability,
  and stop at the Validation horizon boundary; it emits no labels and never
  opens Test. `run-registered-fold` now performs the complete operation under
  one fresh reservation: preflight both candidate configurations, read the
  registered Train/Validation replay once, persist the receipt, prepare Train,
  fit E1/E2 at matched budgets, generate batched Validation forecasts, score
  E0/E1/E2, and write the report with checksums and source identity. This
  one-shot shape follows the registry's rejection of repeated overlapping
  development reads. A synthetic orchestration regression confirms the same
  replay object flows from the single reader call into the experiment.
  The final report also binds each forecast NPZ and the batch request by SHA-256;
  `verify_fold_experiment_report_digest` detects edits to the frozen report.
  Example invocation (the Fold config and E1/E2 JSON files must be frozen and
  registered before this command):

  ```powershell
  rles world-model run-registered-fold `
    --registered-fold-config <fresh-fold.json> `
    --e1-configuration configs/routeb/e1_gru_cpu_demo.json `
    --e2-configuration configs/routeb/b2_cpu_demo.json `
    --worker-python packages/world_model/environment/.venv/Scripts/python.exe `
    --upstream-checkout packages/world_model/environment/.tmp/dreamerv3 `
    --output-dir <new-report-directory>
  ```

  A full synthetic one-shot run also completed with the pinned Python 3.12 CPU
  workers: Train preparation, E1/E2 fit, batch forecasts, E0 baselines and the
  final predictive report. It is mechanism-only evidence. No real Fold was
  available to execute this registered command; real E0/E1/E2 evidence and
  the E3/E4 gate remain pending.
  The pinned Python 3.12 CPU worker also completed E1 and E2 two-anchor batch
  forecasts from synthetic-only contexts, both producing `[2, 2, 3, 5]`; the
  same run repeated the fixed synthetic E2 NLL check. The latest focused set
  passed 36 tests with one environment-specific JAX skip; Ruff passes.

- **Main-project reproduction (2026-10-18):** Installed the locked isolated worker
  environment with Python 3.12.10 / JAX 0.11.2 and the pinned DreamerV3 checkout
  `e01491fad6434b2245a3b8ca201dd7faedcc458c`. From the main Python 3.14
  workspace, `qa/world_model/routeb_cli_worker_smoke.py` passed end to end:
  matched E1/E2 synthetic Train artifact and optimizer budget, both batched
  forecasts shaped `[2, 2, 3, 5]`, synthetic one-shot Train/E0/E1/E2 report,
  and fixed-seed E2 synthetic NLL decreased from 5.8945 to 4.5135. The run
  report is [`B2_main_project_cpu_smoke.json`](B2_main_project_cpu_smoke.json)
  with report SHA-256
  `0ab09a2cf591724d8b742a5c8a55f59661031b365ee38ac84b0c4b9d5bf7b1cc`.
  This confirms host/worker integration in the main project; all bars were
  synthetic, and no authorized market Validation, model selection, WFO, or
  economic conclusion was produced.

- **B2 train/checkpoint CPU recheck (2026-10-18):** The smoke now accepts an
  explicit expected platform and preserves the caller-selected JAX backend;
  CPU remains the default. On the locked Python 3.12.10/JAX 0.11.2 worker,
  training, resumed-vs-uninterrupted model plus Optax arrays, 3-step forecasts,
  exact one-step distribution IPC, and F1 worker forecasts all passed on
  `cpu:0`. Evidence:
  [`B2_train_resume_cpu_smoke.json`](B2_train_resume_cpu_smoke.json). The same
  smoke also passed against the pinned WSL CUDA environment on `cuda:0`, with
  exact model/Optax resume parity and fixed-seed direct and worker forecasts.
  The latest four-step fit took 4.40 seconds and JAX reported a 608 MiB peak live
  allocation (1,026 MiB peak pool) for this small synthetic job; this is not a
  throughput benchmark. Evidence: [`B2_cuda_smoke.json`](B2_cuda_smoke.json).
  The full isolated environment lock contains 78 exact training and CUDA runtime
  pins; its SHA-256 is
  `e15b296ff2575272ad9f8e67c6a03dc74b2af57205321f363bf29b4e01c1243d`.
  Both CPU and CUDA reports are synthetic mechanism evidence only; the fresh
  real-market Fold gate remains blocked.

- **v3.0 R2 risk supervision closure (2026-10-18):** The Train sequence artifact
  is now schema v3 and carries valid-path masks and 1/5/15/30/60-minute return,
  log realized-volatility, high/low excursion, and upper/lower barrier targets.
  Barrier thresholds come from the preregistered 75th percentile of Train-score
  realized volatility at each horizon. The JAX learner optimizes a separate
  risk likelihood/BCE head from the same
  causal prior state as its OHLCV forecast. S0/S1/S2 output metrics continue to
  come from common generated OHLCV path ensembles. The RouteB world-model
  suite passes (128 passed, 4 host-only JAX skips); Ruff and the 184-file
  workspace MyPy check pass. The dedicated PR/nightly/release workflow now
  runs the full RouteB suite and a pinned Python 3.12.10/JAX CPU worker job.
  That worker checks causal prior gradients, independent KL weighting, and
  finite prior rollouts at 1/5/15/60 steps, then runs the complete 2+2 versus
  4-step checkpoint/resume and worker IPC smoke. The current manual WSL CPU
  run passes; workflow YAML and both UV locks validate locally.
  Synthetic CPU and CUDA training each reported 92 valid labels at every
  horizon, finite risk loss, Train-fitted barrier thresholds, exact parameter,
  optimizer, sampler-order and PRNG parity on resume, and passing forecast/
  worker IPC checks. Each alignment artifact contains 93 rows, 72 valid.
  Prior/risk head and transition gradients are finite and nonzero; future
  observation mutation leaves the aligned prior forecast unchanged, and the
  constant-input diagnostic remains finite. Updated evidence:
  [`B2_train_resume_cpu_smoke.json`](B2_train_resume_cpu_smoke.json),
  [`B2_cuda_smoke.json`](B2_cuda_smoke.json),
  [`B2_prior_risk_gradient_smoke.json`](B2_prior_risk_gradient_smoke.json), and
  [`V3_IMPLEMENTATION_STATUS.md`](V3_IMPLEMENTATION_STATUS.md). These are
  engineering checks only. No fresh, authorized market Validation fold was
  available, so baseline superiority and promotion remain unverified.
  The new CI worker recipe was reproduced in a fresh RouteB temporary
  environment created from the locked Python 3.12.10 project: all 61 installed
  packages pass `uv pip check`, prior/risk and 1/5/15/60 rollout smoke passes,
  and the four-update B2 CPU resume/worker smoke passes. The refreshed CPU
  result is [`B2_train_resume_cpu_smoke.json`](B2_train_resume_cpu_smoke.json).


- **CUDA JAX regression:** 47 tests passed, 0 failed, 0 skipped across the
  RouteB additive, E1 and quality-v2 test files using Python 3.12.3/JAX 0.11.2
  on `cuda:0`. This exercises the JAX-only cases skipped by Python 3.14; the
  wider main-project regression remains 207 passed with 3 skips. Pytest was a
  temporary test-only install and was removed by restoring the 78-package lock;
  `uv pip check` and CUDA device enumeration passed afterward. Evidence:
  [`B2_cuda_test_suite.json`](B2_cuda_test_suite.json).

## Latest frozen R2 iteration

The preregistered ETHUSDT 2024-01-10 through 2024-04-12 risk-weight comparison is complete. E2 with risk loss scale 2.0 slightly beats the strongest statistical baseline on 15-minute CRPS, but the paired 95% interval crosses zero and Holm-adjusted p is 1.0. Its free rollout remains under-dispersed; volatility persistence is weaker than E1. Test was not decoded and the scope is consumed. See [`R2_ETHUSDT_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_PRE_REGISTRATION_20261019.md) and [`R2_ETHUSDT_RISK2_60_15_15_RESULTS_20261019.md`](R2_ETHUSDT_RISK2_60_15_15_RESULTS_20261019.md).

## Current v3.0 R2 research gate (2026-10-19)

- Fresh registered LTCUSDT Train/Validation comparison: **COMPLETED** with
  three training seeds, 180 Validation anchors, 96 pooled paths per model and
  1,000 optimizer updates per seed. Test and Holdout payloads were not decoded.
  E2 15m return CRPS is `0.0024538091` versus S0 `0.0024698718`; E2-minus-S0
  95% block-bootstrap CI `[-0.0000825121, +0.0000545594]`, Holm-adjusted
  `p=1.0`. The point estimate favors E2 but does not pass the R2 gate. Free
  rollouts also understate realized volatility clustering and upper-tail spread.
  See [`R2_LTCUSDT_1000STEP_360_15_15_20261019.md`](R2_LTCUSDT_1000STEP_360_15_15_20261019.md)
  and the immutable report under
  `.artifacts/world_model/fresh_ltcusdt_1000step_risk_ensemble_20261019/`.

- Fresh registered LINKUSDT Train/Validation comparison: **COMPLETED** with
  three training seeds, 180 Validation anchors, and 96 pooled paths per model.
  Test and Holdout payloads were not decoded. The 15m return CRPS comparison is
  **INCONCLUSIVE**: E2 `0.0028797185`, strongest baseline S0 `0.0028365066`,
  E2-minus-S0 95% block-bootstrap CI `[-0.0000332263, 0.0001172530]`,
  Holm-adjusted `p=1.0`. See
  [`R2_LINKUSDT_360_15_15_20261019.md`](R2_LINKUSDT_360_15_15_20261019.md)
  and the immutable report under
  `.artifacts/world_model/fresh_linkusdt_ensemble_20261019/`.
- R2 promotion is not granted. Per v3.0, continue conditional volatility and
  extreme-risk work; keep R3/R4/R5/R6 gated.
- The ETHUSDT risk-weight-2 experiment failed its primary gate. It reduced the
  rollout volatility mean and upper tail error relative to E1, but did not
  improve realized volatility clustering. Continue R2 only with a new,
  preregistered hypothesis and an unconsumed ledger scope.
- The E2 prior risk head now exports checkpoint-bound multi-horizon return,
  realized-volatility, high/low excursion distributions and upper/lower
  barrier logits. The fold report scores native risk forecasts with CRPS/Brier
  and paired time-block bootstrap plus Holm adjustment. An initial DOGEUSDT
  run exposed a target-space mismatch for log realized volatility; the scorer
  and regression tests now align this target with training. The immutable DOGE
  log-RV score is invalid and documented in
  [`R2_DOGEUSDT_RISK_SCORE_CORRECTION_20261019.md`](R2_DOGEUSDT_RISK_SCORE_CORRECTION_20261019.md).
  AVAXUSDT's corrected three-seed comparison is complete: E2 15m CRPS is
  `0.0038566243` versus S0 `0.0036998354`, with Holm-adjusted `p=0.232` and a
  positive E2-minus-S0 difference. The native E2 risk head also underperforms
  the selected baseline on 15m return scores. R2 remains NOT PROMOTED; see
  [`R2_AVAXUSDT_CORRECTED_360_15_15_20261019.md`](R2_AVAXUSDT_CORRECTED_360_15_15_20261019.md).
  R3/R4/R5/R6 stay gated.


## Latest frozen R2 run

ETHUSDT R2 long-horizon risk-weight revision 2 completed on the independent 2023-04-13 to 2023-07-15 source interval. At 15 minutes, E2 CRPS was `0.0013115379` against E0 S1 `0.0013427077`; the paired 95% CI favored E2 before correction, but Holm-adjusted p was `0.3418291`, so the registered gate failed. E2 rollouts remained more under-dispersed and less persistent than E1. All three seed reports, the aggregate canonical digest and pooled forecast hashes verify; Test was not decoded. Continue R2 with a new frozen output-distribution hypothesis and unused scope. R3/R4/R5/R6 remain gated. See [`R2_ETHUSDT_LONG_HORIZON_RISK_RESULTS_20261019_REV2.md`](R2_ETHUSDT_LONG_HORIZON_RISK_RESULTS_20261019_REV2.md).


## Latest R2 distribution experiment

ETHUSDT variance-matched Student-t E2 completed on the independent 2022 Q4 scope. The 15m primary score worsened versus strongest E0 S1: E2 `0.0009921716`, S1 `0.0009550564`, paired 95% CI for E2-minus-S1 `[+0.0000051702,+0.0000640337]`, Holm p `0.23188`. The gate failed. Native 30/60m log-volatility risk scores favored E2 across all three seeds, but free-rollout volatility persistence remained weak. Test was not decoded. Continue R2 with one frozen Gaussian-mixture head hypothesis on a new ledger scope; R3/R4/R5/R6 stay gated. See [`R2_ETHUSDT_STUDENT_T_RESULTS_20261019.md`](R2_ETHUSDT_STUDENT_T_RESULTS_20261019.md).

## Latest frozen R2 run: two-component Gaussian mixture

The preregistered ETHUSDT mixture-head experiment is complete. E2 15m return CRPS was `0.0016580219`, worse than E0 S1 `0.0016401095`; E2-minus-S1 was `+0.0000179125` (95% CI `[-0.0000321407,+0.0000774650]`, Holm p `1.0`). The primary gate failed. Native 30/60m log-realized-volatility scores improved for every seed, but free-rollout absolute-return autocorrelation remained far below realized (`0.0041` vs `0.1206`). Test was not decoded; all aggregate, predictive and seed report digests pass. R2 remains open; R3/R4/R5/R6 remain gated. See [`R2_ETHUSDT_MIXTURE_GAUSSIAN_RESULTS_20261019.md`](R2_ETHUSDT_MIXTURE_GAUSSIAN_RESULTS_20261019.md).

## Latest R2 risk-loss scale experiment

ETHUSDT risk scale 0.5 completed on the preregistered February-May 2022 scope. E2 15m CRPS `0.0016427192` was worse than strongest E0 S1 `0.0016327746` (difference `+0.0000099446`, 95% CI `[-0.0000297960,+0.0000494486]`, Holm p `1.0`). E2 beat S0, but Holm p `0.05547` missed the registered threshold. Native 30/60m log-RV scores improved in all seeds, while free-rollout clustering stayed weak. Aggregate, predictive and seed digests verify; Test was not decoded. R2 remains unpromoted. See [`R2_ETHUSDT_RISK05_RESULTS_20261019.md`](R2_ETHUSDT_RISK05_RESULTS_20261019.md).

## Latest R2 risk-loss scale experiment: 0.25

ETHUSDT risk scale 0.25 completed on the May-August 2022 scope. E2 15m return CRPS `0.0034760011` nearly matched E0 S1 `0.0034747115`; difference `+0.0000012896`, paired 95% CI `[-0.0001002373,+0.0000716310]`, Holm p `1.0`. The primary gate failed. Native 30/60m log-RV gains were no longer significant, and free-rollout clustering/tails remained low. Integrity digests pass; Test was not decoded. R2 remains unpromoted. See [`R2_ETHUSDT_RISK025_RESULTS_20261019.md`](R2_ETHUSDT_RISK025_RESULTS_20261019.md).

## Latest R2 risk-scale transfer result

The Gaussian risk-scale-0.5 transfer on the July-October 2023 ETHUSDT scope completed. E2 15m CRPS `0.0007201673` beat the strongest baseline E0 S2 `0.0007362676` by `-0.0000161002`, but the paired 95% CI crossed zero and Holm p was `1.0`. E2 significantly beat S0, which was not the strongest baseline. Native 30-minute risk scores improved in all seeds and 60-minute scores in two of three; free-rollout clustering/tails stayed under-dispersed. Test was not decoded and all digests verify. R2 remains unpromoted. See [`R2_ETHUSDT_RISK05_TRANSFER_RESULTS_20261019.md`](R2_ETHUSDT_RISK05_TRANSFER_RESULTS_20261019.md).

## Latest R2 adjacent-period transfer result

The fixed Gaussian risk-scale-0.5 recipe completed on the adjacent ETHUSDT 2023-10-16 to 2024-01-17 source interval. E2 15m CRPS was `0.0014828516` versus strongest baseline E0 S2 `0.0014909569`; the paired 95% CI `[-0.0000539319,+0.0000161153]` crossed zero and Holm p was `1.0`, so the primary gate failed. Native 30m log-RV scores improved in all seeds and 60m in two of three, while generated volatility persistence/tails remained low. Aggregate, predictive, seed and pooled hashes verify; Test stayed undecoded. R2 remains open and unpromoted; R3/R4/R5/R6 remain gated. See [`R2_ETHUSDT_RISK05_ADJACENT_RESULTS_20261019.md`](R2_ETHUSDT_RISK05_ADJACENT_RESULTS_20261019.md).

## Latest R2 training-objective result

The preregistered Gaussian CRPS training objective completed on the independent ETHUSDT 2021-01-02 to 2021-04-05 scope. E2 15m CRPS `0.0032632485` was slightly worse than strongest E0 S1 `0.0032611095` (difference `+0.0000021390`, 95% CI `[-0.0000520096,+0.0000839570]`, Holm p `1.0`); the primary gate failed. Native 30/60m log-RV scores improved in all seeds after correction. All report/artifact hashes verify and Test remained unread. R2 remains open and unpromoted. See [`R2_ETHUSDT_CRPS_OBJECTIVE_RESULTS_20261019.md`](R2_ETHUSDT_CRPS_OBJECTIVE_RESULTS_20261019.md).

## Latest R2: 15-step open-loop CRPS

The preregistered ETHUSDT 2021-04-05 to 2021-07-08 run completed on three seeds with 1,000 updates per candidate/seed, 180 Validation anchors and 96 pooled paths. E2 15m return CRPS was `0.0033467892` versus strongest statistical baseline E0 S1 `0.0034373483`; the paired 95% interval favored E2, but bootstrap `p=0.1424` and Holm-adjusted `p=1.0`, so the primary gate failed. E2 also failed to match E1's better point estimate (`0.0033056222`). Native 30/60m log-realized-volatility CRPS improved against S2 across all seeds after correction, but rollout volatility clustering remained weak and the p95 tail was too wide. Aggregate/predictive/seed digests and pooled path hashes verify; Test remained undecoded. R2 remains unpromoted; R3/R4/R5/R6 remain gated. See [`R2_ETHUSDT_OPENLOOP_CRPS_RESULTS_20261019.md`](R2_ETHUSDT_OPENLOOP_CRPS_RESULTS_20261019.md) and its [`pre-registration`](R2_ETHUSDT_OPENLOOP_CRPS_PRE_REGISTRATION_20261019.md).

## Latest R2 adjacent confirmation: 15-step open-loop CRPS

The fixed E2 open-loop recipe completed on adjacent ETHUSDT 2021-07-08 to 2021-10-10 data with three seeds. It failed to replicate its prior favorable point estimate: E2 15m return CRPS `0.0031071156` was worse than E0 S1 `0.0030902426` (difference `+0.0000168730`, 95% CI `[-0.0000569696,+0.0001198835]`, Holm `p=1.0`). E1 also had a lower point estimate. Native risk scores were mixed and free rollouts severely understated realized volatility clustering and spread. All aggregate, predictive, seed, and pooled hashes verify; Test stayed undecoded. The open-loop objective is not promoted; continue R2 with a new pre-registered volatility-persistence hypothesis. R3/R4/R5/R6 remain gated. See [`R2_ETHUSDT_OPENLOOP_CRPS_ADJACENT_RESULTS_20261019.md`](R2_ETHUSDT_OPENLOOP_CRPS_ADJACENT_RESULTS_20261019.md).


## Latest R2 native risk target: 15-minute log realized volatility

The frozen Gaussian E2 risk-head recipe was tested on unused ETHUSDT data from 2021-10-10 through 2022-01-12. Native 15m log-RV CRPS improved significantly against Validation-selected E0 S2 for seeds 31 and 20261018, but seed 102 was inconclusive (E2-minus-E0 `-0.02506`, 95% CI `[-0.05482,+0.00989]`, Holm `p=0.26687`), so the preregistered all-seed gate failed. Native 15m return CRPS was significantly worse for E2 across all three seeds. Generated paths stayed under-dispersed despite 100% valid OHLCV. All aggregate, predictive, per-seed and pooled digests verify; Test remained unread. R2 remains unpromoted and R3/R4/R5/R6 remain gated. See [`R2_ETHUSDT_NATIVE_RV15_RESULTS_20261019.md`](R2_ETHUSDT_NATIVE_RV15_RESULTS_20261019.md).

## Latest BTCUSDT path-risk experiment

The BTCUSDT 2022-04-18 to 2022-07-21 risk-conditioned rollout run completed on
three seeds without OOM after forecast chunking. Its new path-derived 15m
log-RV CRPS gate failed in the wrong direction for all seeds: E2-minus-E0
`+0.129895`, `+0.090897`, and `+0.157759`; all 95% intervals were above zero
and Holm p-values were below 0.05. Generated rolling-RV mean/p95 were
`0.001292`/`0.003011` versus realized `0.000919`/`0.001819`. Test stayed
unread and report/artifact hashes verify. A zero-overlap 2022-07-21 to
2022-10-23 geometric-strength-0.5 experiment completed: generated RV mean
matched observed, but p95 stayed low and all three primary intervals crossed
zero (Holm p `1.000`, `0.855`, `0.550`). Test stayed unread; integrity hashes
verify. A variance-matched Student-t prior with tempered risk conditioning is
now running on the next ledger-clear BTC scope. See
[`R2_BTCUSDT_RISK_CONDITIONED_ROLLOUT_RESULTS_20261020_REV3.md`](R2_BTCUSDT_RISK_CONDITIONED_ROLLOUT_RESULTS_20261020_REV3.md),
[`R2_BTCUSDT_RISK_CONDITIONED_TEMPERED_RESULTS_20261020.md`](R2_BTCUSDT_RISK_CONDITIONED_TEMPERED_RESULTS_20261020.md),
and [`R2_BTCUSDT_STUDENT_T_TEMPERED_RISK_PRE_REGISTRATION_20261020.md`](R2_BTCUSDT_STUDENT_T_TEMPERED_RISK_PRE_REGISTRATION_20261020.md).

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

The frozen BTCUSDT strength-0.1 transfer on `[2023-11-03, 2024-02-05)` improved selected-E0 path-derived 15m log-RV CRPS in each seed, but seed 31 missed the registered Holm threshold (`p=0.087956`) so the all-seed gate failed. Pooled E2/E0 CRPS was `0.204828/0.277485`; generated volatility mean/p95 and absolute-return persistence still understated realized values. Independent aggregate, predictive, seed and forecast/risk artifact checks passed; Test/Holdout remained unread. R2 remains unpromoted and R3-R6 gated. Next investigate a training-time path-consistency objective. See [`R2_BTCUSDT_GAUSSIAN_RISK01_ADJACENT_RESULTS_20261020.md`](R2_BTCUSDT_GAUSSIAN_RISK01_ADJACENT_RESULTS_20261020.md).


## Latest matched-objective R4 E2/E3 comparison — DOGEUSDT

The registered three-seed DOGEUSDT comparison completed on a disjoint 360/15-day Train/Validation Fold. E3's multi-horizon risk objective matched the frozen E2 auxiliary settings. Two seeds significantly favored E3 on path log-RV CRPS; seed 102 significantly favored E2, so the all-seed primary gate failed. The return-CRPS no-harm gate passed. Test OHLCV remained sealed; no hierarchy promotion, economic certification, or production claim follows. See [`R4_DOGEUSDT_E2_E3_RESULTS_20261022.md`](R4_DOGEUSDT_E2_E3_RESULTS_20261022.md).


## Matched-objective R4 DOGEUSDT temporal replication — run 02

The second fixed-protocol DOGEUSDT E2/E3 comparison completed on a disjoint earlier interval. Seeds 31 and 20261018 significantly favored E3 on the primary path log-RV CRPS; seed 102 significantly favored E2 again. Return-CRPS no-harm passed, but the all-seed gate failed, with no promotion. Test OHLCV stayed sealed. The same seed-direction disagreement across two periods is a warning about seed sensitivity; its mechanism is unknown. See [`R4_DOGEUSDT_E2_E3_RESULTS_RUN02_20261022.md`](R4_DOGEUSDT_E2_E3_RESULTS_RUN02_20261022.md).
