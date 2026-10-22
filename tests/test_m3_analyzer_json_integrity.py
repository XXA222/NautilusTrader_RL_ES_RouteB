from pathlib import Path

import pytest

from qa import analyze_m33_action_value, analyze_stage58_train_coverage
from qa import analyze_stage59_train_diagnostics, analyze_stage60_fixed_regimes


READERS = (
    analyze_stage58_train_coverage._load,
    analyze_stage59_train_diagnostics.load,
    analyze_stage60_fixed_regimes.load,
    analyze_m33_action_value.load,
)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("payload", [
    b'{"episode_count":1,"episode_count":4}',
    b'{"report":\xff}',
    b'["not", "an", "object"]',
])
def test_m3_analyzers_reject_ambiguous_or_invalid_evidence_json(
        tmp_path: Path, reader, payload: bytes) -> None:
    source = tmp_path / "evidence.json"
    source.write_bytes(payload)

    with pytest.raises(ValueError, match="JSON object"):
        reader(source)
