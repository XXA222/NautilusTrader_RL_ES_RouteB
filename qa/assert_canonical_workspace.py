"""Fail closed when local RouteB development is outside its canonical checkout."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


CANONICAL_PROJECT_ROOT = Path(r"D:\Program Files\NautilusTrader_RL_ES_RouteB")
MARKER_NAME = ".canonical-project-root"


def _normalized(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path)).rstrip("\\/")


def assert_canonical_workspace() -> Path:
    if os.name != "nt":
        raise RuntimeError(
            "RouteB local development is Windows-only at "
            f"{CANONICAL_PROJECT_ROOT}; non-Windows checkouts are read-only/CI-only"
        )

    script_root = Path(__file__).resolve().parents[1]
    expected = _normalized(CANONICAL_PROJECT_ROOT)
    actual = _normalized(script_root)
    if actual != expected:
        raise RuntimeError(
            f"refusing non-canonical RouteB checkout: {script_root}; expected "
            f"{CANONICAL_PROJECT_ROOT}"
        )

    marker = script_root / MARKER_NAME
    marker_value = marker.read_text(encoding="utf-8").strip()
    if _normalized(Path(marker_value)) != expected:
        raise RuntimeError(f"invalid {MARKER_NAME}: {marker_value!r}")

    completed = subprocess.run(
        ["git", "-C", str(script_root), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    git_root = Path(completed.stdout.strip())
    if _normalized(git_root) != expected:
        raise RuntimeError(
            f"Git root mismatch: {git_root}; expected {CANONICAL_PROJECT_ROOT}"
        )
    return script_root


if __name__ == "__main__":
    try:
        root = assert_canonical_workspace()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"canonical RouteB workspace verified: {root}")
