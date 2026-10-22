# RouteB backup test record — 2026-10-23

This record accompanies the RouteB source snapshot on branch `backup/routeb-current-20261023` in the independent public repository `XXA222/NautilusTrader_RL_ES_RouteB`.

## Recorded verification

- Source selector for native report parity, research metrics, historical factory, and native liquidation facts: **224 passed, 18 dependency warnings**.
- The same selector against the RouteB-built research wheel with `PYTHONPATH` unset: **224 passed, 18 dependency warnings**.
- Research metrics tests reproduced from the public backup checkout: **107 passed**.
- Ruff: full `packages/tests/qa/ci` check passed.
- MyPy: **226 source files passed**.
- RouteB isolated environment dependency check: **72 packages compatible**.
- F7 synthetic 5,000,000-return RSS profile: exact metrics matched; radix tail selection reduced measured process RSS delta by **60.33%** versus the heap reference (10,129,408 B to 4,018,176 B); elapsed-time ratio was **1.109**.

Detailed machine-readable records are in `docs/audit_evidence/20261023/f7_bounded_tail_memory_cap_20261023.json` and `docs/audit_evidence/20261023/f7_tail_process_rss_20261023.json`.

## Scope and limitations

The RSS profile uses deterministic synthetic returns and does not establish strategy or economic validity. The public-checkout reproduction reused the RouteB-local Python environment, so it is not a clean dependency-environment reproduction. The full-project clean-checkout audit item remains open.

The source manifest `qa/reports/source_manifest_current.json` is intentionally not included in this public snapshot: automated publication review flagged that it exposes hashes, sizes, environment dependency metadata, and paths to project data/test fixtures.