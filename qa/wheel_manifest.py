"""Compare a built workspace wheelhouse with its frozen RLES source identity."""
from __future__ import annotations

import argparse
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def build_wheel_manifest(root: Path, wheels_dir: Path, source_identity: str) -> dict[str, Any]:
    root, wheels_dir = root.resolve(), wheels_dir.resolve()
    wheels = sorted(wheels_dir.glob("*.whl"))
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    packages = sorted((root / "packages").iterdir())
    expected = 0
    for package in packages:
        project_file = package / "pyproject.toml"
        source_root = package / "src"
        if not package.is_dir() or not project_file.is_file() or not source_root.is_dir():
            continue
        with project_file.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        name, version = project.get("name"), project.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError(f"{project_file}: project name/version required")
        expected += 1
        matches: list[tuple[Path, dict[str, str], str, list[str]]] = []
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                metadata_paths = [n for n in archive.namelist()
                                  if n.endswith(".dist-info/METADATA")]
                if len(metadata_paths) != 1:
                    continue
                metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_paths[0]))
                if _normalize(str(metadata.get("Name", ""))) != _normalize(name):
                    continue
                payload_files: dict[str, str] = {}
                duplicate_payload_paths: list[str] = []
                for item in archive.infolist():
                    path = item.filename
                    parts = Path(path).parts
                    if (item.is_dir()
                            or any(part.endswith((".dist-info", ".data", ".egg-info"))
                                   for part in parts)):
                        continue
                    if path in payload_files:
                        duplicate_payload_paths.append(path)
                    payload_files[path] = _sha256(archive.read(item))
                wheel_version = str(metadata.get("Version", ""))
                matches.append((wheel, payload_files, wheel_version,
                                sorted(set(duplicate_payload_paths))))
        if len(matches) != 1:
            errors.append(f"{name}: expected exactly one wheel, found {len(matches)}")
            continue

        wheel, built_files, wheel_version, duplicate_payload_paths = matches[0]
        source_files = {
            item.relative_to(source_root).as_posix(): _sha256(item.read_bytes())
            for item in source_root.rglob("*")
            if (item.is_file() and not item.is_symlink()
                and not any(part == "__pycache__" or part.endswith(".egg-info")
                            for part in item.relative_to(source_root).parts)
                and item.suffix not in {".pyc", ".pyo"})
        }
        missing = sorted(set(source_files) - set(built_files))
        extra = sorted(set(built_files) - set(source_files))
        mismatch = sorted(path for path in set(source_files) & set(built_files)
                          if source_files[path] != built_files[path])
        version_mismatch = wheel_version != version
        if missing or extra or mismatch or version_mismatch or duplicate_payload_paths:
            errors.append(f"{name}: missing={missing}, extra={extra}, content_mismatch={mismatch}")
            if version_mismatch:
                errors.append(f"{name}: project version={version}, wheel metadata version={wheel_version}")
            if duplicate_payload_paths:
                errors.append(f"{name}: duplicate wheel payload paths={duplicate_payload_paths}")
        python_file_count = sum(path.endswith(".py") for path in built_files)
        records.append({
            "distribution": name,
            "version": version,
            "wheel_version": wheel_version,
            "wheel": wheel.name,
            "wheel_sha256": _sha256(wheel.read_bytes()),
            "source_payload_file_count": len(source_files),
            "wheel_payload_file_count": len(built_files),
            "wheel_python_file_count": python_file_count,
            "source_matches_wheel": not (
                missing or extra or mismatch or version_mismatch or duplicate_payload_paths
            ),
            "missing": missing,
            "extra": extra,
            "content_mismatch": mismatch,
        })
    if len(wheels) != expected:
        errors.append(f"expected {expected} workspace wheels, found {len(wheels)}")
    return {
        "schema": "rles.wheelhouse-manifest.v2",
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
    source = json.loads(source_path.read_text(encoding="utf-8"))
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
