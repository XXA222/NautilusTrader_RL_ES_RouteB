"""Fail closed when local RouteB development is outside its canonical checkout."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


CANONICAL_PROJECT_ROOT = Path(r"D:\Program Files\NautilusTrader_RL_ES_RouteB")
CANONICAL_WSL_PROJECT_ROOT = Path("/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB")
MARKER_NAME = ".canonical-project-root"


def _normalized(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path)).rstrip("\\/")


def _platform_root(*, os_name: str = os.name, platform: str = sys.platform) -> Path:
    if os_name == "nt":
        return CANONICAL_PROJECT_ROOT
    if os_name == "posix" and platform.startswith("linux"):
        return CANONICAL_WSL_PROJECT_ROOT
    raise RuntimeError(
        "RouteB local development is supported only in the canonical Windows or "
        f"WSL checkout: {CANONICAL_PROJECT_ROOT}"
    )


def _marker_matches(marker_value: str, *, expected: Path,
                    os_name: str = os.name) -> bool:
    if os_name == "nt":
        return _normalized(Path(marker_value)) == _normalized(expected)
    # The marker remains a stable Windows project identity in this shared checkout.
    return marker_value.casefold() == str(CANONICAL_PROJECT_ROOT).casefold()


def assert_canonical_workspace() -> Path:
    platform_root = _platform_root()
    script_root = Path(__file__).resolve().parents[1]
    expected = _normalized(platform_root)
    actual = _normalized(script_root)
    if actual != expected:
        raise RuntimeError(
            f"refusing non-canonical RouteB checkout: {script_root}; expected "
            f"{platform_root}"
        )

    marker = script_root / MARKER_NAME
    marker_value = marker.read_text(encoding="utf-8").strip()
    if not _marker_matches(marker_value, expected=platform_root):
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
            f"Git root mismatch: {git_root}; expected {platform_root}"
        )
    return script_root


if __name__ == "__main__":
    try:
        root = assert_canonical_workspace()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"canonical RouteB workspace verified: {root}")
