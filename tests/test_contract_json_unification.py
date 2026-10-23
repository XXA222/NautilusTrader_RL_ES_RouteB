from __future__ import annotations

import pytest

from rles_contracts.json_utils import loads_strict_json
from rles_contracts.schema import FieldSpec
from rles_research.json_utils import loads_unique_json


@pytest.mark.parametrize("loader", [loads_strict_json, loads_unique_json])
def test_shared_json_loaders_reject_duplicate_keys(loader) -> None:
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        loader('{"a":1,"a":2}')


@pytest.mark.parametrize("loader", [loads_strict_json, loads_unique_json])
def test_shared_json_loaders_reject_nonstandard_constants(loader) -> None:
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        loader('{"a":NaN}')


@pytest.mark.parametrize("loader", [loads_strict_json, loads_unique_json])
def test_shared_json_loaders_enforce_nesting_limit(loader) -> None:
    payload = "[" * 513 + "0" + "]" * 513
    with pytest.raises(ValueError, match="supported nesting depth"):
        loader(payload)


@pytest.mark.parametrize("loader", [loads_strict_json, loads_unique_json])
def test_shared_json_loaders_ignore_brackets_inside_strings(loader) -> None:
    payload = '{"text":"[[[ ]]]"}'
    assert loader(payload) == {"text": "[[[ ]]]"}


def test_optional_contract_defaults_use_shared_bounded_parser() -> None:
    payload = "[" * 513 + "0" + "]" * 513
    with pytest.raises(ValueError, match="invalid canonical JSON default"):
        FieldSpec("optional", "json", "unit", required=False, default_json=payload)
