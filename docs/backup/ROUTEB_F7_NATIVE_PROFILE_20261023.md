# F7 current-source Native profile — 2026-10-23

RouteB isolated profile, schema `rles.f7-synthetic-native-growth.v2`.

- Online-summary sink and lean execution each processed **129,597** synthetic minute steps.
- Full per-step trajectory SHA-256 matched: `4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f`.
- The online summary consumed 129,600 snapshots and produced Native metrics SHA-256 `0c9bb2fbdbd241a24d05c4c9b6a3e864bf37db3ea5e78dc71eb9785ea9e50a45`.
- Online summary and lean RSS peak deltas: **510,554,112 B** and **510,062,592 B**; elapsed time: **86.54 s** and **76.47 s**.
- The summary sink does not retain a per-barrier snapshot sequence. RSS differences from this single synthetic pair do not prove a stable memory gain; the summary run was about 13.2% slower.

Machine-readable evidence: `docs/audit_evidence/20261023/f7_synthetic_native_growth_current_20261023.json`. Reproducer: `qa/profile_f7_synthetic_native_growth.py`.

This does not close F7: O(n) event/report state, a complete authorized real-history Train RSS/trajectory profile, and representative long position/funding/order-history growth remain open. No market, registry, account, Validation, Test, or Holdout data was accessed.