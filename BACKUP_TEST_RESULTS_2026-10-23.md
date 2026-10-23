# RouteB independent backup and verification — 2026-10-23

This is a RouteB-only snapshot in the independent public repository `XXA222/NautilusTrader_RL_ES_RouteB`, branch `backup/routeb-code-test-backup-20261023`. It does not update the upstream NautilusTrader_RL_ES repository or RouteA.

## Current local verification

- Workspace PR selector on Python 3.14.7: **727 passed, 1 skipped, 253 deselected** (`contracts or unit or micro_golden or algorithm_smoke`). The skip is the CUDA-only provider smoke because CUDA is unavailable.
- RouteB workspace MyPy: **109 source files, no issues** (world-model sources are checked separately in their locked JAX environment).
- RouteB world-model JAX CPU suite on Python 3.12.10: **119 passed**; isolated world-model MyPy: **37 source files, no issues**.
- Full Ruff check: **passed** (`packages ci qa tests`).
- Current-source distribution build: **12 wheels and 12 sdists**; wheel manifest: **12 wheels, all_match=true**, including the world-model `py.typed` marker.
- Source manifest: **370 files**; source identity SHA-256 `4abd7dc89f4545abdd3e0249ef66cb7144df30d66e81be638bb41a2221b3fe24`.
- Git diff whitespace check: **passed**.

The reusable CI workflow now verifies source/wheel payloads before installation and checks the world model in an isolated Python 3.12/JAX CPU job that runs pytest, MyPy, and B2/B3 mechanism smoke tests. The pinned DreamerV3 checkout smoke still requires the Hosted CI runner.

## Snapshot scope and limits

The snapshot contains current changed RouteB implementation, package metadata, CI workflow, regression tests, and this verification summary. The detailed source manifest is intentionally omitted because it contains local environment and fixture metadata; its identity hash is recorded above and CI regenerates the manifest from the checkout. Internal planning/audit documents, temporary test output, local datasets, access inventories, and exploratory scripts are also excluded.

All current results above are local Windows checks. Hosted Linux/Windows/JAX CI has not yet run on this updated branch. No authenticated account statement or market, Validation, Test, or Holdout payload was read; these engineering tests do not certify cashflow completeness or trading performance.
