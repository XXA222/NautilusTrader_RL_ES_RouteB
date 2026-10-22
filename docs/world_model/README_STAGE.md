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
- **B0 data/baseline evidence:** BLOCKED. The registered fresh-fold reader is
  implemented, but no never-consumed eligible 0.4.1 Train/Validation interval
  has been reserved or read; reports contain no real market audit or baseline
  score. The fold must cover an approved 60/15/15 or 360/15/15 protocol.
  The current Route B `data/` inventory contains only completed schema 0.4.0
  snapshots (latest event time 2024-12-01); these cannot satisfy the reader's
  0.4.1 row-group manifest contract. The registry also contains consumed or
  reserved market intervals through 2026-08-11; as of 2026-10-18, a clean
  interval starting after that reservation is at most 67 days, shorter than
  the minimum 90-day 60/15/15 fold. A fresh 0.4.1
  source with a sufficiently long interval outside every registered range is
  required to pass this gate.
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
- **WSL2/CUDA:** A saved device-probe report records a successful JAX 0.11.2
  minimal JIT on the RTX 5070. The current WSL default Python has no JAX, and the
  pinned CUDA environment was not reproduced in this checkout. This proves a
  device probe only; DreamerV3 RSSM training and a CUDA learner remain untested.
- **B2 engineering:** Synthetic-only CPU training, resume parity, forecast and
  worker distribution export have now passed end to end on the pinned
  DreamerV3/JAX environment. **E2 authorized-market model fitting and the E0/E1
  comparison remain blocked** because B0 has no registered Train/Validation
  Fold or dataset identity. No Actor, policy imagination, or economic claim is
  implemented.
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
  53 emitted warnings are existing dependency warnings. The refreshed 363-file
  source manifest verifies at
  `4bafcab4f1d0b995f9aaa5a3b2e17f2700a897503d1435bfdeb0197a68271a2b`.
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
