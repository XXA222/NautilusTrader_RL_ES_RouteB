from __future__ import annotations

import hashlib
import json

from qa.world_model.run_registered_fold_ensemble import _seal_report_digest


def test_aggregate_report_digest_is_sealed_and_reproducible():
    report = {"status": "COMPLETED", "evidence": {"test_payload_decoded": False}}

    digest = _seal_report_digest(report)

    payload = dict(report)
    recorded = payload.pop("report_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert recorded == digest
    assert recorded == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_aggregate_report_digest_reseals_after_payload_change():
    report = {"status": "COMPLETED", "value": 1}
    first = _seal_report_digest(report)
    report["value"] = 2

    second = _seal_report_digest(report)

    payload = dict(report)
    recorded = payload.pop("report_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert second != first
    assert recorded == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
