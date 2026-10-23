# RouteB F7 history-memory update — 2026-10-23

- Fold-wide derived frames are lazy with a bounded two-frame cache; episode-window metadata uses cached scalar bounds.
- `FeatureRow` is slotted while preserving fields, hashes and pickle behavior; object-size accounting falls from 376 B to 288 B per five-value row.
- `CuratedStore` drops three duplicate timestamp tuples and aliases the full visible view when all events are available. The 90-day synthetic store generation peak RSS fell by 10,440,704 B and post-generation RSS by 8,245,248 B.
- Source and isolated installed-wheel selectors each pass **172 tests, 18 warnings**; source/wheel hashes match; all 72 RouteB packages are compatible.
- 90-day installed-wheel replay: 129,597 decisions, exact online/lean trajectory parity (`4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f`), replay RSS deltas 193,503,232 / 192,835,584 B, runtime 90.70 / 79.81 s.

Machine-readable evidence: `docs/audit_evidence/20261023/f7_lazy_barrier_frames_20261023.json`.

F7 remains open for duplicate event-reference indexes and Native long-history state, plus a separately authorized real-history Train profile. External provenance, independent Validation, market authorization/depth, historical-trial and M4.2 gates remain open.

