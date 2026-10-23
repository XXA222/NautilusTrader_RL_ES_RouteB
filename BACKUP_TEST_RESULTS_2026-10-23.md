# RouteB independent backup and verification — 2026-10-23

This is a RouteB-only snapshot in the independent public repository `XXA222/NautilusTrader_RL_ES_RouteB`, branch `backup/routeb-code-test-backup-20261023`. It does not update the upstream NautilusTrader_RL_ES repository or RouteA.

## Current local verification

- Workspace PR selector on Python 3.14.7: **727 passed, 1 skipped, 253 deselected** (`contracts or unit or micro_golden or algorithm_smoke`). The skip is the CUDA-only provider smoke because CUDA is unavailable.
- RouteB workspace MyPy: **109 source files, no issues** (world-model sources are checked separately in their locked JAX environment).
- RouteB world-model JAX CPU suite on Python 3.12.10: **119 passed**; isolated world-model MyPy: **37 source files, no issues**.
- Full Ruff check: **passed** (`packages ci qa tests`).
- Current-source distribution build: **12 wheels and 12 sdists**; wheel manifest: **12 wheels, all_match=true**, including the world-model `py.typed` marker.
- Source manifest v2: **370 files**; source identity SHA-256 `ba0d6bfcbabfeccbe1dc36b184ff62c2eca592d86552192f9d70f56b5dd907f9`. It omits local runtime inventories and untracked upstream paths.
- Git diff whitespace check: **passed**.

The reusable CI workflow now verifies source/wheel payloads before installation and checks the world model in an isolated Python 3.12/JAX CPU job that runs pytest, MyPy, and B2/B3 mechanism smoke tests. Wheel-installed code resolves its workspace from the fixed `RLES_WORKSPACE_ROOT` CI setting, so registry/commit identity does not depend on `site-packages` or test cwd. The pinned DreamerV3 checkout smoke requires the Hosted CI runner.

## Snapshot scope and limits

The snapshot contains current changed RouteB implementation, package metadata, CI workflow, regression tests, and this verification summary. The detailed source manifest is intentionally omitted because it contains local environment and fixture metadata; its identity hash is recorded above and CI regenerates the manifest from the checkout. Internal planning/audit documents, temporary test output, local datasets, access inventories, and exploratory scripts are also excluded.

Hosted verification completed successfully for commit `7fb5f1b56f236cb50fbc0914413df84016c0485f` in [GitHub Actions run 39901004125](https://github.com/XXA222/NautilusTrader_RL_ES_RouteB/actions/runs/39901004125): Linux and Windows workspace jobs passed, and the isolated JAX CPU job passed. Hosted PR selector results were Linux **720 passed, 8 skipped, 253 deselected** and Windows **727 passed, 1 skipped, 253 deselected**; the platform-specific skip counts differ. Hosted JAX world-model suite: **119 passed**; B2/B3 CPU mechanism smokes, both MyPy jobs, and Ruff passed. GPU and release jobs were skipped by lane configuration. No authenticated account statement or market, Validation, Test, or Holdout payload was read; these engineering checks do not certify cashflow completeness or trading performance.

The source-manifest v2 sanitization and regression were verified locally (**3 passed**, focused Ruff passed, manifest self-verification passed). The Hosted run above validates the previous manifest v1 code revision; CI for v2 must run before that change is considered cross-platform verified.

