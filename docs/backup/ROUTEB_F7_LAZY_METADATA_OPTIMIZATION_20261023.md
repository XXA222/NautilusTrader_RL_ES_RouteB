# RouteB F7 lazy-frame metadata optimization �� 2026-10-23

The lazy per-episode frame sequence keeps only a two-frame working set and avoids Fold-wide derived-frame indexes. Profiling found that Gym step info repeatedly queried lazy episode start/end frames. `BarrierEpisode` now stores the validated decision-window timestamps as scalar metadata, so per-step reporting does not rebuild those frames.

## Verification

- Source and isolated installed-wheel selectors: **124 passed each**, 28 dependency warnings each.
- 90-day synthetic Train replay: 129,597 decisions; online-summary and lean trajectory SHA-256 match at `4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f`.
- RSS peak deltas: 203,612,160 B online summary and 197,705,728 B lean, about 60.1% below the pre-change pair.
- Runtime: 88.85 s online summary and 78.77 s lean, near the pre-change 86.54 s and 76.47 s.
- GymBridge wheel and source SHA-256 match for `episode.py` and `historical.py`; 72 RouteB environment packages are compatible.
- Ruff, targeted MyPy and the 605-file source manifest verification pass.

Machine-readable profile: `docs/audit_evidence/20261023/f7_lazy_barrier_frames_20261023.json`.

This is synthetic engineering evidence only. F7 remains open for remaining O(n) source/native/report state and authorized real-history Train profiling. Independent Validation, data provenance, market scope/depth, trial-coverage, and M4.2 gates remain open.
