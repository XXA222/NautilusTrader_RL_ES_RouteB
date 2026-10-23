# RouteB test and profile results — 2026-10-23

### A04 registry circular JSON rejection

`tests/test_research_access.py` plus `tests/test_strict_json_constants.py`: **146 passed** with RouteB `packages/*/src` explicitly first on `PYTHONPATH`. Ruff passed for the seven strict JSON helpers, registry access source, and both test files; MyPy passed for all eight changed source modules. Tests use a temporary registry and synthetic cyclic/deep JSON values only.

Compute worker transport follow-up: the combined compute contract, strict JSON, and registry selector passes **164 tests** with RouteB source imports. Ruff and MyPy pass for the changed transport module. The payload regression is synthetic and in-memory.

Contracts JSON follow-up: `tests/test_layer_invariants.py tests/test_contract_config.py`: **39 passed** with RouteB source imports. Ruff and MyPy pass for the two reader modules.

Direct JSON ingress/readers follow-up: Binance transfer, legacy inventory, and artifact integrity selectors: **169 passed**. Ruff and targeted MyPy pass for all three changed modules. Inputs and files were temporary and synthetic.

This note records checks run against the RouteB working copy backed up on
`backup/routeb-working-copy-20261023`. Detailed machine-readable profile data
is stored under `docs/audit_evidence/20261023/`.

## Automated checks

- Focused data-layer, event-guard and historical-factory tests: **148 passed**.
- After removing the unused historical bar tuple, focused layer, Native
  accumulator, research metric and lazy-frame factory tests: **160 passed**.
- After the Native cashflow detachment/copy changes, Native accumulator parity
  and research metric tests: **130 passed**.
- After adding post-finalization index release, the Native accumulator parity
  and research metric selector passed **131 tests**; Ruff and MyPy passed for
  the changed source and test files.
- After adding disk-backed return spool cleanup, the Native accumulator parity
  and research metric selector passed **132 tests**; Ruff and MyPy passed for
  both changed research modules.
- The follow-up terminal-failure spool cleanup selector passed **134 tests**;
  Ruff and MyPy passed for the changed research modules.
- Native parity, research metrics, application CLI, real application flow and
  orchestrator workflow passed **197 tests** with 13 third-party warnings;
  after making final-snapshot release exception-safe, the focused set passed
  **186 tests**. Ruff and MyPy passed for the changed modules.
- WorldModel additive tests passed **19 with 2 JAX-only skips** in the app
  environment and **21 tests** in RouteB's isolated JAX CPU environment; Ruff
  and MyPy passed for the bundle change.
- ResearchArtifactStore JSON/layer selector passed **7 tests**; Ruff and MyPy
  passed after recursive non-string key rejection was added.
- Ruff on changed source and test files: **passed**.
- MyPy on the changed Native metrics module: **passed**. The changed data,
  Native metrics and historical modules also passed together with package
  source roots configured in `MYPYPATH`.
- Source manifest generation and `--verify`: **passed** locally. The generated
  manifest is intentionally excluded from this public backup because it records
  machine-specific paths and runtime details.
- `git diff --check` and evidence JSON parsing: **passed**.

The test groups above were run at different points in the change sequence;
they are not one full-suite run. Dependency warnings were reported by pytest.

## 90-day synthetic Native replay

- 129,597 decision steps were replayed in both online-summary and lean modes.
- Complete trajectory SHA-256 matched across modes:
  `4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f`.
- Online Native metrics SHA-256:
  `0c9bb2fbdbd241a24d05c4c9b6a3e864bf37db3ea5e78dc71eb9785ea9e50a45`.
- Observed replay RSS peak deltas were 191,860,736 bytes (online) and
  191,827,968 bytes (lean). These measurements show parity for this synthetic
  run; they do not establish a real-history RSS improvement.
- The profile used generated synthetic data only. It did not use a real
  registry, real cashflow history, or Test/Holdout data.

See `f7_current_source_native_growth_20261023.json` and related F7 evidence
files for the complete measurements and source hashes.

The post-finalization release regression and 10,000-cashflow structural
measurement are recorded in
`f7_accumulator_finalize_index_release_20261023.json`.
The return-spool lifecycle regression is recorded in
`f7_accumulator_spool_release_20261023.json`.
The terminal failure-path cleanup regression is recorded in
`f7_accumulator_spool_failure_cleanup_20261023.json`.
Final-snapshot release and its orchestrator integration results are recorded in
`f7_final_snapshot_release_20261023.json`.
WorldModel bundle scaler-manifest validation is recorded in
`a10_bundle_scaler_string_keys_20261023.json`.
ResearchArtifactStore L1 manifest key validation is recorded in
`a10_research_artifact_string_keys_20261023.json`.
The consolidated JSON boundary selectors, including the CLI deep-manifest regression, pass **258 tests**. Ruff and targeted MyPy pass on the changed sources.
