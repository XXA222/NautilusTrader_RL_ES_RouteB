from pathlib import Path

import pytest

from rles_orchestrator.workspace import require_workspace_paths


def test_workspace_local_paths_accept_current_project_files(tmp_path: Path) -> None:
    root = tmp_path / "route-b"
    path = root / "data" / "market.parquet"
    resolved = require_workspace_paths(root, {"market": path, "output": root / "out"})
    assert resolved["market"] == path.resolve()
    assert resolved["output"] == (root / "out").resolve()


def test_workspace_local_paths_reject_cross_checkout_reads_before_use(tmp_path: Path) -> None:
    root = tmp_path / "route-b"
    external = tmp_path / "route-a" / ".artifacts" / "market.parquet"
    with pytest.raises(ValueError, match="inside this workspace"):
        require_workspace_paths(root, {"market": external})


def test_workspace_path_labels_are_validated() -> None:
    with pytest.raises(ValueError, match="labels"):
        require_workspace_paths(Path.cwd(), {"": Path.cwd()})
