"""Create and verify a reproducible identity manifest for the RLES workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rles_contracts.json_utils import ensure_json_nesting_depth


_SOURCE_ROOTS = ("packages", "tests", "configs", "ci", "packaging", "requirements", "qa")
_ROOT_FILES = (
    ".gitattributes",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "RLES_v0.4.0_总开发与集成基线.md",
)
_EXCLUDED_PARTS = {"__pycache__", "build", "dist", ".pytest_cache", ".mypy_cache", ".tmp"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


class _DuplicateManifestKey(ValueError):
    """A manifest JSON object repeats a field and is ambiguous."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateManifestKey(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise json.JSONDecodeError(f"non-standard JSON constant: {value}", "", 0)

def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
        ensure_json_nesting_depth(content)
        value = json.loads(content, object_pairs_hook=_unique_json_object,
                            parse_constant=_reject_json_constant)
    except _DuplicateManifestKey as exc:
        raise ValueError(f"duplicate JSON manifest key in {path}: {exc}") from exc
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"cannot read JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text_attributes(root: Path, paths: list[Path]) -> dict[str, str]:
    """Read Git's text attribute for all manifest paths in one invocation."""
    if not paths:
        return {}
    names = [path.relative_to(root).as_posix() for path in paths]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "check-attr", "-z", "--stdin", "text"],
            input=("\0".join(names) + "\0").encode("utf-8"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    fields = result.stdout.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3:
        return {}
    return {
        os.fsdecode(fields[index]): os.fsdecode(fields[index + 2])
        for index in range(0, len(fields), 3)
        if fields[index + 1] == b"text"
    }


def _source_bytes(path: Path, text_attribute: str | None = None) -> bytes:
    data = path.read_bytes()
    if text_attribute == "unset":
        return data
    if text_attribute in {"set", "auto", "input"}:
        return data.replace(b"\r\n", b"\n")
    if text_attribute is None or text_attribute == "unspecified":
        if b"\0" not in data:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                pass
            else:
                return data.replace(b"\r\n", b"\n")
    return data


def _source_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in _ROOT_FILES:
        path = root / relative
        if not path.is_symlink() and path.is_file():
            files.add(path)
    for relative in _SOURCE_ROOTS:
        directory = root / relative
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in directory.rglob("*"):
            relative_parts = path.relative_to(root).parts
            generated_component = any(
                part in _EXCLUDED_PARTS or part.startswith((".venv", "dist"))
                for part in relative_parts
            ) or relative_parts[:2] == ("qa", "reports")
            if (not generated_component and not path.is_symlink() and path.is_file()
                    and path.suffix not in _EXCLUDED_SUFFIXES
                    and ".egg-info" not in path.as_posix()):
                files.add(path)
    workflow_dir = root / ".github" / "workflows"
    if workflow_dir.is_dir() and not workflow_dir.is_symlink():
        files.update(path for path in workflow_dir.rglob("*") if not path.is_symlink() and path.is_file())
    for relative in ("qa/source_manifest.py", "qa/wheel_manifest.py"):
        tool = root / relative
        if tool.is_file() and not tool.is_symlink():
            files.add(tool)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _upstream_identity(root: Path) -> dict[str, Any] | None:
    upstream = root / "nautilus_trader"
    if not upstream.is_dir():
        return None

    def git(*args: str) -> bytes:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={upstream}", "-C", str(upstream), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    status = git("status", "--porcelain", "--untracked-files=all").decode("utf-8").splitlines()
    tracked_diff = git("diff", "HEAD", "--binary")
    return {
        "path": "nautilus_trader",
        "commit": commit,
        "tracked_worktree_clean": not any(not row.startswith("??") for row in status),
        "tracked_diff_sha256": _sha256(tracked_diff),

    }


def build_manifest(root: Path, *, generated_at_utc: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    source_paths = _source_files(root)
    text_attributes = _text_attributes(root, source_paths)
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": len(content),
            "sha256": _sha256(content),
        }
        for path in source_paths
        for content in (_source_bytes(path, text_attributes.get(path.relative_to(root).as_posix())),)
    ]
    upstream = _upstream_identity(root)
    identity = {"files": files, "upstream_nautilus": upstream}
    identity_sha = _sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())

    return {
        "schema": "rles.source-manifest.v3",
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "source_identity_sha256": identity_sha,
        "scope": {
            "included": [*_ROOT_FILES, *_SOURCE_ROOTS, ".github/workflows",
                         "qa/source_manifest.py", "qa/wheel_manifest.py"],
            "excluded_generated": [".tmp", ".venv*", "data", "dist*", "docs/audit reports", "qa/reports"],
        },

        "upstream_nautilus": upstream,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("qa/reports/source_manifest_current.json"))
    parser.add_argument("--verify", action="store_true", help="verify an existing manifest without rewriting it")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    fresh = build_manifest(root)
    if args.verify:
        if not output.is_file():
            print(f"manifest not found: {output}", file=sys.stderr)
            return 2
        try:
            previous = _read_manifest(output)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        matches = (
            previous.get("schema") == fresh["schema"]
            and previous.get("source_identity_sha256") == fresh["source_identity_sha256"]
            and previous.get("files") == fresh["files"]
            and previous.get("upstream_nautilus") == fresh["upstream_nautilus"]
        )
        if not matches:
            print("source manifest is stale", file=sys.stderr)
            return 1
        print(f"verified {len(fresh['files'])} files; source_identity_sha256={fresh['source_identity_sha256']}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fresh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}; files={len(fresh['files'])}; source_identity_sha256={fresh['source_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
