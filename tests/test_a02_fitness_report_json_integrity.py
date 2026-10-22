from pathlib import Path

import pytest

from qa import compare_es_fitness_objectives_20261017 as comparison


@pytest.mark.parametrize("reader", [comparison._read_json_object,
                                     lambda path: comparison._parse_json_object(
                                         path.read_bytes(), "embedded evidence")])
@pytest.mark.parametrize("payload", [
    b'{"protocol":"legacy","protocol":"constraint_priority"}',
    b'{"fitness":\xff}',
    b'["not", "an", "object"]',
])
def test_a02_comparison_reader_rejects_ambiguous_evidence_json(
        tmp_path: Path, reader, payload: bytes) -> None:
    source = tmp_path / "evidence.json"
    source.write_bytes(payload)

    with pytest.raises(ValueError, match="A02 JSON"):
        reader(source)
