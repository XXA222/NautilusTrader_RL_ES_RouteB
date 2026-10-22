# RouteB F7 lazy-frame and FeatureRow memory update �� 2026-10-23

- Lazy per-episode frames use a two-frame cache; scalar episode-window bounds avoid repeated construction of endpoint frames.
- `FeatureRow` now uses slots and retains its immutable fields, canonical hash, and explicit pickle compatibility. Python object-size accounting is 288 B per 5-value row versus 376 B before the change, about 11.4 MB saved at 129,600 rows.
- Source and isolated installed-wheel selectors each pass **118 tests, 18 warnings**; wheel/source module hashes match; all 72 RouteB environment packages are compatible.
- 90-day installed-wheel synthetic replay: 129,597 decisions; online/lean trajectory SHA-256 matches (`4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f`). RSS deltas are 192,876,544 / 194,117,632 B, about 62% lower than baseline; elapsed times 88.74 / 79.13 s, near baseline 86.54 / 76.47 s.
- Ruff, targeted MyPy, 605-file source identity verification, and diff checks pass.

Machine-readable evidence: `docs/audit_evidence/20261023/f7_lazy_barrier_frames_20261023.json`.

F7 remains open for retained CuratedStore event/time indexes, Native session/cache/report facts, and authenticated real-history Train profiling. External provenance, independent Validation, market authorization/depth, trial-coverage and M4.2 gates remain open.
