from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "ci" / "l6" / "src"))

from l6_ci_release.attestation import GateBlocked, load_gate_reports, verify_gate_reports  # noqa: E402
from l6_ci_release.gates import GATE_CHECKS, EvidenceRef, GateRunner  # noqa: E402


@pytest.mark.parametrize(
    "document",
    [
        b'{"producers":["reviewer"],"producers":["attacker"]}',
        b'{"producers":[NaN]}',
    ],
)
def test_trusted_producer_registry_rejects_ambiguous_json(tmp_path: Path, document: bytes) -> None:
    producer_file = tmp_path / "trusted.json"
    producer_file.write_bytes(document)
    with pytest.raises(GateBlocked, match="invalid reviewer producer registry"):
        _load_from_cli_inputs(tmp_path, producer_file)


def _load_from_cli_inputs(reports: Path, trusted_producers: Path) -> None:
    from l6_ci_release.attestation import main

    old_argv = sys.argv
    try:
        sys.argv = ["l6-attestation", "--reports", str(reports),
                    "--trusted-producers", str(trusted_producers)]
        main()
    finally:
        sys.argv = old_argv


def test_gate_report_loader_rejects_nested_duplicate_keys(tmp_path: Path) -> None:
    trusted = {"reviewer"}
    for index in range(10):
        (tmp_path / f"Q{index}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "Q0.json").write_text(
        '{"schema_version":1,"checks":{"a":true,"a":false}}', encoding="utf-8"
    )
    with pytest.raises(GateBlocked, match="Q0 report unreadable"):
        load_gate_reports(tmp_path, trusted_producers=trusted)


def test_gate_report_loader_rejects_non_finite_json_constant(tmp_path: Path) -> None:
    for index in range(10):
        (tmp_path / f"Q{index}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "Q0.json").write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(GateBlocked, match="Q0 report unreadable"):
        load_gate_reports(tmp_path, trusted_producers={"reviewer"})


def test_valid_gate_attestations_still_load_after_strict_parsing(tmp_path: Path) -> None:
    trusted = {"reviewer"}
    runner = GateRunner(trusted_producers=trusted)
    for gate in (f"Q{index}" for index in range(10)):
        evidence = []
        for check in sorted(GATE_CHECKS[gate]):
            path = tmp_path / f"{gate}-{check}.evidence"
            payload = f"{gate}:{check}".encode("utf-8")
            path.write_bytes(payload)
            evidence.append(EvidenceRef(check, str(path), hashlib.sha256(payload).hexdigest(), "reviewer"))
        runner.evaluate(gate, evidence=evidence,
                        checks={check: True for check in GATE_CHECKS[gate]},
                        reviewer_approved=True)

    reports = runner.reports()
    assert verify_gate_reports(reports, trusted_producers=trusted) == {
        gate: report["report_sha256"] for gate, report in reports.items()
    }
    report_root = tmp_path / "reports"
    report_root.mkdir()
    for gate, report in reports.items():
        (report_root / f"{gate}.json").write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    assert load_gate_reports(report_root, trusted_producers=trusted) == verify_gate_reports(
        reports, trusted_producers=trusted
    )


