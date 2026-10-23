from __future__ import annotations

import pytest

from rles_compute_runtime.json_utils import loads_unique_json as compute_loads
from rles_contracts.json_utils import MAX_JSON_NESTING_DEPTH, ensure_json_nesting_depth
from rles_data.json_utils import loads_unique_json as data_loads
from rles_es_evotorch.json_utils import loads_unique_json as es_loads
from rles_orchestrator.json_utils import loads_unique_json as orchestrator_loads
from rles_research.json_utils import loads_unique_json as research_loads
from rles_rl_sb3.json_utils import loads_unique_json as rl_loads
from rles_world_model.json_utils import loads_unique_json as world_model_loads


@pytest.mark.parametrize("loads", [
    compute_loads, data_loads, es_loads, orchestrator_loads,
    research_loads, rl_loads, world_model_loads,
])
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_package_strict_json_helpers_reject_nonstandard_constants(loads, constant: str) -> None:
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        loads('{"nested":[{"value":' + constant + '}]}')


def test_data_reader_allows_an_explicit_constant_policy_override() -> None:
    assert data_loads('{"value":NaN}', parse_constant=lambda token: token) == {"value": "NaN"}


def test_json_nesting_guard_ignores_delimiters_inside_escaped_strings() -> None:
    payload = '{"text":"escaped quote: \\\" and brackets [] {}"}'
    ensure_json_nesting_depth(payload)
    ensure_json_nesting_depth(payload.encode("utf-8"))


def test_json_nesting_guard_accepts_limit_and_rejects_one_over() -> None:
    ensure_json_nesting_depth("[" * MAX_JSON_NESTING_DEPTH + "0" + "]" * MAX_JSON_NESTING_DEPTH)
    with pytest.raises(ValueError, match="supported nesting depth"):
        ensure_json_nesting_depth("[" * (MAX_JSON_NESTING_DEPTH + 1) + "0" + "]" * (MAX_JSON_NESTING_DEPTH + 1))


@pytest.mark.parametrize("loads", [
    compute_loads, data_loads, es_loads, orchestrator_loads,
    research_loads, rl_loads, world_model_loads,
])
def test_package_strict_json_helpers_normalize_excessive_nesting(loads) -> None:
    payload = "[" * 20_000 + "0" + "]" * 20_000
    with pytest.raises(ValueError, match="supported nesting depth"):
        loads(payload)
