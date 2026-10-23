# RouteB backup verification — 2026-10-23

This record accompanies a RouteB-only source snapshot in the independent public repository `XXA222/NautilusTrader_RL_ES_RouteB`, branch `backup/routeb-code-test-backup-20261023`.

## Verification

- A04 access-ledger and WFO regression suites: **112 passed** (`tests/test_research_access.py`, `tests/test_research_wfo.py`) after strict historical-inventory import and candidate/fold authorization binding.
- A08 cashflow and native-report regression suites: **117 passed** (`tests/test_research_metrics.py`, `tests/test_binance_cashflows.py`, `tests/test_native_report_accumulator_parity.py`) after exact decimal parsing was enforced for external and Binance cashflows.
- F7 streamed-detail and bounded-tail regression suite: **138 passed** (`tests/test_research_metrics.py`, `tests/test_native_report_accumulator_parity.py`, `tests/test_application_cli.py`). This is a separate run and is not combined with the A04/A08 counts.
- Ruff and mypy passed on the changed RouteB sources and tests as recorded in the development plan. `git diff --check` passed for the A08 changes.
- Test runtime: Python 3.14.7, using RouteB package sources via `PYTHONPATH` and the existing test runner environment. The main project's environment was used read-only.

## Snapshot scope

The snapshot includes changed RouteB implementation files, their regression tests, and this test-results summary. It excludes internal planning/audit documents, temporary test-output directories, local datasets, access inventories, and exploratory scripts.

## Evidence limits

These local regression results do not certify Hosted CI, the isolated RouteB runtime, or authenticated real-account cashflow completeness. No real account credentials or market payloads were used for these checks.



