"""Strict JSON boundary checks for independent M1.3 analysis tools."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(filename: str):
    path = ROOT / "qa" / filename
    spec = importlib.util.spec_from_file_location(f"routeb_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("filename", "reader_name"),
    [
        ("analyze_m13_stage_contribution.py", "_load_json"),
        ("analyze_m13_stage_contribution_v2.py", "_json"),
    ],
)
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_m13_analysis_readers_reject_nonstandard_json_constants(
    tmp_path: Path, filename: str, reader_name: str, constant: str
) -> None:
    module = _load(filename)
    source = tmp_path / "report.json"
    source.write_text(f'{{"untrusted_extension": {constant}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="cannot read JSON object"):
        getattr(module, reader_name)(source)
