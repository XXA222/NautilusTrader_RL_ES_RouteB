# RouteB GitHub backup — 2026-10-23

This snapshot belongs to the independent RouteB repository:

- Repository: <https://github.com/XXA222/NautilusTrader_RL_ES_RouteB>
- Visibility: public; outside users can read and clone it. GitHub permits writes only to the owner and users explicitly granted write access.
- Snapshot branch: `backup/routeb-20261023`

The snapshot contains the current RouteB source packages, tests, QA scripts, configurations, development and audit documents, and the checked-in test evidence. It excludes local caches, temporary test workspaces, downloaded datasets, and generated model artifacts.

## Verification

On 2026-10-23, the following focused regression suites passed:

```text
tests/test_historical_factory.py
tests/test_native_report_accumulator_parity.py
tests/test_native_decimal_arithmetic.py
tests/test_research_metrics.py
tests/test_native_liquidation_facts.py
175 passed, 18 warnings in 106.36s
```

The test process used Python 3.14.7 from the main project's existing `.venv` and loaded RouteB source through `PYTHONPATH`. No dependencies were installed or modified. This verifies the listed RouteB behavior but does not claim a RouteB-only virtual environment was used.

The refreshed source manifest contains 366 files with source identity SHA-256 `5fc001946596b337cb666de41826924f00460e4d02edec852e17a7506766c08a`.

The locked B2 CPU smoke also passed on Python 3.12.10 with JAX/JAXLIB 0.11.2 and the pinned DreamerV3 checkout. Its synthetic-only result and environment record are in `docs/audit_evidence/20261023/b2_cpu_smoke_20261023.json` and `docs/audit_evidence/20261023/b2_cpu_smoke_result_20261023.json`. This does not claim real-market validation, economic performance, CUDA validation, or Train-sample authorization.
