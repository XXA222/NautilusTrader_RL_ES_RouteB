# RouteB test and profile results — 2026-10-23

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
