from __future__ import annotations

import pytest

from rles_compute_runtime.json_utils import loads_unique_json as compute_loads
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
