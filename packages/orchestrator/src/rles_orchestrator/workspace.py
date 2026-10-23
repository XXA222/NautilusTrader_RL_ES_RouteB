"""Workspace-local path checks for reproducible, repository-isolated runs."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping


def require_workspace_paths(root: Path, paths: Mapping[str, str | Path]) -> dict[str, Path]:
    """Resolve and validate paths before a run reads data or creates output.

    ``Path.resolve`` follows existing symlinks, so a workspace symlink cannot
    silently route a RouteB run into another checkout. Nonexistent paths are
    accepted only when their resolved destination remains under the workspace.
    """
    workspace = root.resolve()
    resolved: dict[str, Path] = {}
    for label, value in paths.items():
        if not isinstance(label, str) or not label:
            raise ValueError("workspace path labels must be nonempty strings")
        candidate = Path(value).resolve()
        if not candidate.is_relative_to(workspace):
            raise ValueError(f"{label} path must remain inside this workspace")
        resolved[label] = candidate
    return resolved
