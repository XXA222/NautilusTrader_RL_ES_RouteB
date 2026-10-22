"""Require every workspace package and entry point to load from installed wheels."""
from __future__ import annotations

import importlib
from importlib.metadata import distribution
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_wheel_origin(content: str, package_name: str) -> dict[str, Any]:
    try:
        origin = json.loads(content, object_pairs_hook=_unique_json_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(
                                 ValueError(f"non-standard JSON constant: {value}")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{package_name}: invalid direct_url.json") from exc
    if not isinstance(origin, dict):
        raise ValueError(f"{package_name}: direct_url.json root must be an object")
    url = origin.get("url")
    archive_info = origin.get("archive_info")
    dir_info = origin.get("dir_info", {})
    if (not isinstance(url, str) or not url.startswith("file:")
            or not url.lower().endswith(".whl")
            or not isinstance(archive_info, dict)
            or not isinstance(dir_info, dict)
            or ("editable" in dir_info and type(dir_info["editable"]) is not bool)
            or dir_info.get("editable") is True):
        raise ValueError(f"{package_name}: expected an installed wheel archive origin")
    return origin


def _workspace_packages() -> list[tuple[str, tuple[str, ...]]]:
    packages: list[tuple[str, tuple[str, ...]]] = []
    seen_distributions: set[str] = set()
    for project_file in sorted((ROOT / "packages").glob("*/pyproject.toml")):
        with project_file.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        name = project.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit(f"{project_file}: project.name is required")
        normalized = _normalize(name)
        if normalized in seen_distributions:
            raise SystemExit(f"duplicate workspace distribution name: {name}")
        seen_distributions.add(normalized)

        source_root = project_file.parent / "src"
        modules = tuple(sorted(
            path.name for path in source_root.iterdir()
            if path.is_dir() and not path.is_symlink()
            and (path / "__init__.py").is_file()
        )) if source_root.is_dir() else ()
        if not modules:
            raise SystemExit(f"{project_file}: no import packages found under {source_root}")
        packages.append((name, modules))
    if not packages:
        raise SystemExit(f"no workspace packages found under {ROOT / 'packages'}")
    return packages


def _installed_module_path(module_name: str, prefix: Path, label: str) -> Path:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise SystemExit(f"{label}: {module_name} has no installed module file")
    path = Path(module_file).resolve()
    if not path.is_relative_to(prefix):
        raise SystemExit(f"{label}: {module_name} imported outside wheel environment: {path}")
    return path


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    records: dict[str, dict[str, Any]] = {}
    for name, modules in _workspace_packages():
        dist = distribution(name)
        direct_url = dist.read_text("direct_url.json")
        if direct_url is None:
            raise SystemExit(f"{name}: local wheel provenance missing")
        try:
            _parse_wheel_origin(direct_url, name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        imported_modules = {
            module_name: str(_installed_module_path(module_name, prefix, name))
            for module_name in modules
        }
        entry_points = []
        for entry_point in sorted(dist.entry_points, key=lambda item: (item.group, item.name)):
            entry_path = _installed_module_path(
                entry_point.module, prefix,
                f"{name}:{entry_point.group}:{entry_point.name}",
            )
            entry_points.append({
                "group": entry_point.group,
                "name": entry_point.name,
                "module": entry_point.module,
                "module_path": str(entry_path),
            })
        records[name] = {
            "version": dist.version,
            "modules": imported_modules,
            "entry_points": entry_points,
        }
    print(json.dumps({"python": sys.version.split()[0], "prefix": str(prefix),
                      "installed_wheels": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
