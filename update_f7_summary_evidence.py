from __future__ import annotations

import hashlib
import json
from pathlib import Path

root = Path(r"D:\Program Files\NautilusTrader_RL_ES_RouteB")
evidence = {
    "schema": "rles.routeb.audit-evidence.f7-online-native-summary-stream.v1",
    "date": "2026-10-22",
    "status": "PASS_SYNTHETIC_ENGINEERING_ONLY",
    "finding": "The summary-only online Native accumulator was wired into default rollout, but sourced cashflow on a repeated native timestamp could reach reconciliation before timestamp uniqueness was rejected. The streaming profile also still exercised the disk-spool prototype.",
    "changes": [
        "Reject a second barrier at a sourced cashflow timestamp before equity reconciliation, matching the batch report's explicit duplicate-barrier failure.",
        "Expose the number of consumed native barriers for profiling.",
        "Update the F7 profile to measure online summary reduction rather than disk spooling.",
        "Add parity coverage for duplicate timestamp grouping, sourced deposits and withdrawals, duplicate cashflow barriers, and exact lower-tail metrics.",
    ],
    "verification": {
        "source_selector": "194 passed, 18 existing upstream warnings",
        "installed_wheel_selector": "194 passed, 18 existing upstream warnings; PYTHONPATH unset",
        "ruff": "passed",
        "mypy": "Success: no issues found in native_metrics.py, profile_f7_synthetic_native_growth.py, and test_native_report_accumulator_parity.py",
        "py_compile": "passed",
        "source_manifest": {"files": 597, "source_identity_sha256": "d7771d4ad8683e5a493c9c48545aeeaf9d6108b8d47c938a964823022d12fed2", "verified": True},
        "distribution": {"wheels": 12, "sdists": 12, "wheel_manifest_all_match": True, "installed_modules": 12, "installed_entry_points": 6, "routeb_app_dependencies_compatible": 69},
        "90_day_synthetic_profile": {
            "decision_steps": 129597, "native_barriers": 129600,
            "trajectory_sha256": "4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f",
            "lean_trajectory_sha256": "4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f",
            "native_metrics_sha256": "0c9bb2fbdbd241a24d05c4c9b6a3e864bf37db3ea5e78dc71eb9785ea9e50a45",
            "baseline_native_metrics_sha256": "0c9bb2fbdbd241a24d05c4c9b6a3e864bf37db3ea5e78dc71eb9785ea9e50a45",
            "online_rss_peak_delta_bytes": 509599744, "lean_rss_peak_delta_bytes": 510210048,
            "online_elapsed_seconds": 83.3827784, "lean_elapsed_seconds": 76.9902175,
            "interpretation": "Single observation only. RSS was effectively similar to lean (about 0.6 MB lower); online metric reduction added about 6.4 seconds. No stable RSS reduction claim is made. The same hashes were obtained with substantially lower elapsed time than the earlier disk-spool run.",
        },
        "profile_artifact": "docs/audit_evidence/20261022/f7_online_native_summary_stream_20261022.json",
    },
    "limits": [
        "Only fixed-action synthetic OHLCV, mark, index, funding, and native-engine replay were used.",
        "No curated market archive, account, research registry, project ledger, Validation, Test, or Holdout payload was accessed.",
        "Exact lower-tail statistics retain a worst-return heap sized from the known return-count bound, so memory is not constant in sample count.",
        "Account-row reports intentionally materialize rows; this work validates the summary-only path.",
    ],
    "next": "Continue the local A01-A14 source audit from the current RouteB tree. Preserve external gates for authenticated account cashflows, market rules, independent research windows, and Hosted CI.",
}
summary_path = root / "docs/audit_evidence/20261022/f7_online_native_summary_validation_20261022.json"
summary_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
summary_path.with_suffix(summary_path.suffix + ".sha256").write_text(f"{digest}  {summary_path.name}\n", encoding="ascii")

plan = root / "docs/DEVELOPMENT_PLAN_20261017.md"
plan_section = """

## 2026-10-22 — F7 online Native summary reducer parity and long-window profile

**Finding and fix:** The online `NativeReportAccumulator` was already wired into summary-only rollout. A sourced cashflow landing on a duplicated native timestamp was counted by reconciliation before uniqueness validation, so the stream path could report an equity mismatch instead of rejecting the ambiguous barrier. The stream now rejects the second barrier immediately with the same duplicate-barrier error as the batch report. It also exposes a public snapshot count for profiling. The F7 profile now measures online reduction and lean replay, not the disk-spool prototype. Added parity coverage for duplicate timestamp grouping, sourced deposit/withdrawal return adjustment, ambiguous cashflow barriers, and exact tail metrics.

**Verification:** Focused source and installed-wheel selectors both passed **194 tests** (18 existing upstream warnings); Ruff, py_compile, and MyPy for the accumulator/profile/regression passed. The 90-day synthetic Native profile covered **129,597 decisions / 129,600 barriers**. Stream and lean trajectory SHA both equal `4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f`; online report metrics SHA equals the established baseline `0c9bb2fbdbd241a24d05c4c9b6a3e864bf37db3ea5e78dc71eb9785ea9e50a45`. One observation: online RSS delta **509,599,744 bytes**, lean **510,210,048 bytes**; elapsed **83.38 s** vs **76.99 s**. This does not establish a stable RSS reduction, but avoids retaining the compact per-barrier snapshot series and is about 45 seconds faster than the earlier spool observation with identical hashes. All **12 wheels and 12 sdists** built, all wheel contents match source, and all 12 packages / 6 entry points import from the RouteB `.venv-win-app`; all **69 dependencies** are compatible. Source manifest verifies **597 files**, identity `d7771d4ad8683e5a493c9c48545aeeaf9d6108b8d47c938a964823022d12fed2`.

**Boundary:** Synthetic fixed-action replay only; no curated market archive, account, registry, ledger, Validation, Test, or Holdout payload was read. Exact CVaR state retains a bounded worst-return heap derived from the known sample upper bound, so it is not constant-memory in sample count; `include_account_rows=true` continues to materialize rows. Evidence: `docs/audit_evidence/20261022/f7_online_native_summary_stream_20261022.json` and `f7_online_native_summary_validation_20261022.json`.

**Next cursor:** Continue the dependency-satisfied local A01–A14 source audit. Keep authenticated account cashflow, historical market rules, independent windows, and clean Hosted CI at their external evidence gates.
"""
plan.write_text(plan.read_text(encoding="utf-8").rstrip() + plan_section + "\n", encoding="utf-8")

current = root / "docs/DEVELOPMENT_GOAL_CURRENT.md"
current_section = """

## 2026-10-22 latest cursor — F7 online Native summary reducer

The summary-only online Native reducer consumes each barrier once without retaining the compact per-barrier snapshot series. Sourced cashflow at a repeated timestamp now fails before equity reconciliation with the same explicit duplicate-barrier reason as the batch report. The F7 profile now exercises online reduction.

- The focused source and installed-wheel history/metrics/liquidation/rollout selector passes **194 tests** each; Ruff, py_compile, and targeted MyPy pass.
- The 90-day synthetic replay covered **129,597 decisions / 129,600 barriers**. Online and lean trajectory SHA both equal `4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f`; native metric SHA matches the baseline `0c9bb2fbdbd241a24d05c4c9b6a3e864bf37db3ea5e78dc71eb9785ea9e50a45`.
- One RSS observation: online +509,599,744 bytes / 83.38 s; lean +510,210,048 bytes / 76.99 s. Peak RSS is effectively the same and does not support a stable reduction claim; online avoids retaining O(n) compact snapshots and is substantially faster than the prior disk-spool prototype.
- The 597-file source identity `d7771d4ad8683e5a493c9c48545aeeaf9d6108b8d47c938a964823022d12fed2` verifies. Twelve wheels and twelve sdists built; all wheel modules match source; installed smoke covers 12 packages and 6 entry points; all 69 RouteB app dependencies are compatible.

Evidence: `docs/audit_evidence/20261022/f7_online_native_summary_stream_20261022.json` and `docs/audit_evidence/20261022/f7_online_native_summary_validation_20261022.json`. Continue the local A01–A14 source audit from current RouteB code. Authenticated cashflow provenance, authenticated market rules, independent research windows, and clean Hosted CI remain external gates.
"""
current.write_text(current.read_text(encoding="utf-8").rstrip() + current_section + "\n", encoding="utf-8")
print(f"wrote summary evidence; sha256={digest}")
