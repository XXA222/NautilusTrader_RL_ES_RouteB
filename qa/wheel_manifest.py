"""Compare a built workspace wheelhouse with its frozen RLES source identity."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import Any

from rles_contracts.json_utils import ensure_json_nesting_depth


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class _DuplicateManifestKey(ValueError):
    """A source manifest JSON object repeats a field and is ambiguous."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateManifestKey(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise json.JSONDecodeError(f"non-standard JSON constant: {value}", "", 0)

def _read_source_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"source manifest must not be a symlink: {path}")
    try:
        content = path.read_text(encoding="utf-8")
        ensure_json_nesting_depth(content)
        value = json.loads(content, object_pairs_hook=_unique_json_object,
                            parse_constant=_reject_json_constant)
    except _DuplicateManifestKey as exc:
        raise ValueError(f"duplicate JSON manifest key in {path}: {exc}") from exc
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"cannot read source manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected source manifest object: {path}")
    return value


def _source_python_files(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for item in source_root.rglob("*.py"):
        if item.is_symlink():
            continue
        if item.is_file():
            files.append(item)
    return files


def build_wheel_manifest(root: Path, wheels_dir: Path, source_identity: str) -> dict[str, Any]:
    root = root.resolve()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if wheels_dir.is_symlink() or not wheels_dir.is_dir():
        raise ValueError(f"wheel directory must be a real directory: {wheels_dir}")
    wheels_dir = wheels_dir.resolve()
    wheel_paths = sorted(wheels_dir.glob("*.whl"))
    wheels = []
    for wheel in wheel_paths:
        if wheel.is_symlink():
            errors.append(f"{wheel.name}: wheel file must not be a symlink")
        elif wheel.is_file():
            wheels.append(wheel)
    packages_root = root / "packages"
    if packages_root.is_symlink() or not packages_root.is_dir():
        raise ValueError(f"workspace packages path must be a real directory: {packages_root}")
    packages = sorted(packages_root.iterdir())
    expected = 0
    for wheel in wheels:
        try:
            with zipfile.ZipFile(wheel) as archive:
                duplicate_paths = sorted(
                    path for path, count in Counter(archive.namelist()).items() if count > 1
                )
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(f"{wheel.name}: invalid wheel archive: {exc}")
            continue
        if duplicate_paths:
            errors.append(f"{wheel.name}: duplicate archive paths={duplicate_paths}")
    for package in packages:
        if package.is_symlink():
            errors.append(f"{package.name}: package path must not be a symlink")
            continue
        if not package.is_dir():
            continue
        project_file = package / "pyproject.toml"
        source_root = package / "src"
        if project_file.is_symlink():
            errors.append(f"{package.name}: pyproject.toml must not be a symlink")
            continue
        if not project_file.is_file():
            continue
        if source_root.is_symlink():
            errors.append(f"{package.name}: source root must not be a symlink")
            continue
        if not source_root.is_dir():
            continue
        with project_file.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        name, version = project.get("name"), project.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError(f"{project_file}: project name/version required")
        expected += 1
        matches: list[tuple[Path, dict[str, str]]] = []
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                metadata_paths = [n for n in archive.namelist()
                                  if n.endswith(".dist-info/METADATA")]
                if len(metadata_paths) != 1:
                    continue
                metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_paths[0]))
                metadata_names = metadata.get_all("Name", [])
                metadata_versions = metadata.get_all("Version", [])
                if len(metadata_names) != 1 or len(metadata_versions) != 1:
                    errors.append(
                        f"{wheel.name}: METADATA must declare exactly one Name and Version"
                    )
                    continue
                if _normalize(str(metadata_names[0])) != _normalize(name):
                    continue
                wheel_version = str(metadata_versions[0])
                if wheel_version != version:
                    errors.append(
                        f"{wheel.name}: {name} version mismatch; expected {version}, got {wheel_version}"
                    )
                    continue
                module_files = {
                    item.filename: _sha256(archive.read(item))
                    for item in archive.infolist()
                    if item.filename.endswith(".py")
                    and ".dist-info/" not in item.filename
                    and ".data/" not in item.filename
                }
                matches.append((wheel, module_files))
        if len(matches) != 1:
            errors.append(f"{name}: expected exactly one wheel, found {len(matches)}")
            continue

        wheel, built_files = matches[0]
        source_files = {
            item.relative_to(source_root).as_posix(): _sha256(item.read_bytes())
            for item in _source_python_files(source_root)
        }
        missing = sorted(set(source_files) - set(built_files))
        extra = sorted(set(built_files) - set(source_files))
        mismatch = sorted(path for path in set(source_files) & set(built_files)
                          if source_files[path] != built_files[path])
        if missing or extra or mismatch:
            errors.append(f"{name}: missing={missing}, extra={extra}, content_mismatch={mismatch}")
        records.append({
            "distribution": name,
            "version": version,
            "wheel": wheel.name,
            "wheel_sha256": _sha256(wheel.read_bytes()),
            "source_file_count": len(source_files),
            "wheel_python_file_count": len(built_files),
            "source_matches_wheel": not (missing or extra or mismatch),
            "missing": missing,
            "extra": extra,
            "content_mismatch": mismatch,
        })
    if len(wheel_paths) != expected:
        errors.append(f"expected {expected} workspace wheels, found {len(wheel_paths)}")
    return {
        "schema": "rles.wheelhouse-manifest.v1",
        "source_identity_sha256": source_identity,
        "wheel_count": len(records),
        "source_matches_all_wheels": not errors and len(records) == expected,
        "wheels": records,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    source_path = args.source_manifest if args.source_manifest.is_absolute() else root / args.source_manifest
    wheels_dir = args.wheels if args.wheels.is_absolute() else root / args.wheels
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        source = _read_source_manifest(source_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    identity = source.get("source_identity_sha256")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("source manifest has no valid source_identity_sha256")
    report = build_wheel_manifest(root, wheels_dir, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}; wheels={report['wheel_count']}; all_match={report['source_matches_all_wheels']}")
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return 0 if report["source_matches_all_wheels"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
