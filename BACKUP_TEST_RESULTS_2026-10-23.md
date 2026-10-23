# RouteB backup verification — 2026-10-23

RouteB-only source and test snapshot in the independent public repository `XXA222/NautilusTrader_RL_ES_RouteB`, branch `backup/routeb-code-test-backup-20261023`.

## Verification

- A04 access-ledger and WFO regression suites: **112 passed** (`tests/test_research_access.py`, `tests/test_research_wfo.py`) after strict historical-inventory import and candidate/fold authorization binding.
- A08 cashflow and native-report regression suites: **123 passed** (`tests/test_research_metrics.py`, `tests/test_binance_cashflows.py`, `tests/test_native_report_accumulator_parity.py`) including exact cashflow parsing, finite aggregate/report outputs, and failure-state atomicity.
- F7 streamed-detail and bounded-tail regression suite: **138 passed** (`tests/test_research_metrics.py`, `tests/test_native_report_accumulator_parity.py`, `tests/test_application_cli.py`); this is a separate historical run.
- M0.2 source/wheel manifest regressions: **2 passed** (`tests/test_source_manifest.py`, `tests/test_wheel_manifest.py`). The current Windows build produced **12 wheels and 12 sdists**; the wheel manifest matched every built Python file to the source tree (`all_match=true`).
- Ruff and MyPy passed for the changed RouteB sources and tests. The reusable CI workflow parses and runs source/wheel manifest validation before wheel installation.
- Test runtime: Python 3.14.7, RouteB package sources via `PYTHONPATH`; the main-project environment was used read-only.

## Snapshot scope

The snapshot contains changed RouteB implementation and CI files, their regression tests, and this verification summary. It excludes internal planning/audit documents, temporary test outputs, local datasets, access inventories, and exploratory scripts.

## Evidence limits

Local results do not certify Hosted CI until its jobs finish, or certify authenticated real-account cashflow completeness. No real account credentials or market payloads were used for these checks.
