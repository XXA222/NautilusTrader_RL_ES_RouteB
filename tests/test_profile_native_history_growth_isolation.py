"""RouteB profile inputs must remain inside the isolated project checkout."""
import importlib.util
from pathlib import Path

import pytest

from rles_data.core import DataError

_PROFILE_PATH = Path(__file__).resolve().parents[1] / "qa" / "profile_native_history_growth_20261016.py"
_PROFILE_SPEC = importlib.util.spec_from_file_location("routeb_profile_native_history_growth", _PROFILE_PATH)
assert _PROFILE_SPEC is not None and _PROFILE_SPEC.loader is not None
_PROFILE = importlib.util.module_from_spec(_PROFILE_SPEC)
_PROFILE_SPEC.loader.exec_module(_PROFILE)
_routeb_path = _PROFILE._routeb_path
run = _PROFILE.run


def test_profile_path_resolves_and_rejects_outside_routeb() -> None:
    routeb_root = Path(__file__).resolve().parents[1]
    inside = routeb_root / "pyproject.toml"
    assert _routeb_path(inside, "test path", must_exist=True) == inside.resolve()

    outside = routeb_root.parent / "outside-routeb" / "does-not-exist.json"
    with pytest.raises(DataError, match="inside the isolated RouteB project"):
        _routeb_path(outside, "test path", must_exist=False)


def test_archived_profile_plan_rejects_cross_project_bundle_before_access() -> None:
    routeb_root = Path(__file__).resolve().parents[1]
    plan = routeb_root / "configs" / "train_signal_baselines_20261016.json"
    output = routeb_root / ".tmp-profile-isolation-must-not-be-created"

    with pytest.raises(DataError, match="frozen Train bundle.*inside the isolated RouteB project"):
        run(plan, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("script", "entrypoint"),
    [
        ("low_capacity_predictor_20261016.py", "run"),
        ("zero_target_exit_probe_20261016.py", "run"),
        ("profile_episode_reset_20261017.py", "run"),
        ("profile_episode_reset_20261017.py", "finalize_existing"),
        ("profile_stage69_episode_reset_20261017.py", "run"),
        ("reconcile_native_equity_long_20261017.py", "run"),
        ("compare_bar_cost_modes_20261017.py", "run"),
        ("compare_episode_horizon_20261017.py", "run"),
    ],
)
def test_legacy_profile_entrypoints_reject_external_plan_before_read(
    script: str, entrypoint: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    routeb_root = Path(__file__).resolve().parents[1]
    qa_root = routeb_root / "qa"
    monkeypatch.syspath_prepend(str(qa_root))
    module_path = qa_root / script
    module_spec = importlib.util.spec_from_file_location(f"routeb_{module_path.stem}", module_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    outside_plan = routeb_root.parent
    output = routeb_root / ".tmp-profile-isolation-entrypoint-output"

    with pytest.raises(DataError, match="inside the isolated RouteB project"):
        if script == "profile_stage69_episode_reset_20261017.py":
            getattr(module, entrypoint)(outside_plan, outside_plan, output)
        elif script == "reconcile_native_equity_long_20261017.py":
            getattr(module, entrypoint)(plan_path=outside_plan, output=output)
        else:
            getattr(module, entrypoint)(outside_plan, output)
