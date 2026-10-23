from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

pytestmark = pytest.mark.unit


_GENERATOR = Path(__file__).resolve().parents[1] / "qa" / "wheel_manifest.py"


def _run(root: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    result = subprocess.run(
        [
            sys.executable,
            str(_GENERATOR),
            "--root", str(root),
            "--wheels", str(root / "wheels"),
            "--source-manifest", str(root / "source.json"),
            "--output", str(root / "report.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, json.loads((root / "report.json").read_text(encoding="utf-8"))


def test_wheel_manifest_matches_built_sources_and_detects_drift(tmp_path: Path) -> None:
    package = tmp_path / "packages" / "demo"
    source = package / "src" / "demo" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    typed_marker = source.parent / "py.typed"
    typed_marker.write_text("", encoding="utf-8")
    build_metadata = package / "src" / "demo_pkg.egg-info"
    build_metadata.mkdir()
    (build_metadata / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: demo-pkg\nVersion: 1.0.0\n", encoding="utf-8"
    )
    package.mkdir(exist_ok=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "demo-pkg"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = wheel_dir / "demo_pkg-1.0.0-py3-none-any.whl"

    def write_wheel(*, include_typed_marker: bool = True, version: str = "1.0.0",
                    duplicate_source: bool = False) -> None:
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("demo/__init__.py", source.read_bytes())
            if include_typed_marker:
                archive.writestr("demo/py.typed", typed_marker.read_bytes())
            if duplicate_source:
                archive.writestr("demo/__init__.py", b"VALUE = 999\n")
            archive.writestr(
                "demo_pkg-1.0.0.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: demo-pkg\nVersion: {version}\n",
            )

    write_wheel()
    (tmp_path / "source.json").write_text(
        json.dumps({"source_identity_sha256": "a" * 64}), encoding="utf-8"
    )

    result, report = _run(tmp_path)
    assert result.returncode == 0
    assert report["source_matches_all_wheels"] is True
    assert report["schema"] == "rles.wheelhouse-manifest.v2"
    assert report["wheels"][0]["source_payload_file_count"] == 2
    assert report["wheels"][0]["wheel_payload_file_count"] == 2
    assert report["wheels"][0]["wheel_python_file_count"] == 1

    write_wheel(include_typed_marker=False)
    result, report = _run(tmp_path)
    assert result.returncode == 1
    assert report["wheels"][0]["missing"] == ["demo/py.typed"]

    write_wheel(version="0.9.0")
    result, report = _run(tmp_path)
    assert result.returncode == 1
    assert any("project version=1.0.0, wheel metadata version=0.9.0" in error
               for error in report["errors"])

    with pytest.warns(UserWarning, match="Duplicate name"):
        write_wheel(duplicate_source=True)
    result, report = _run(tmp_path)
    assert result.returncode == 1
    assert any("duplicate wheel payload paths=['demo/__init__.py']" in error
               for error in report["errors"])

    write_wheel()

    extra_wheel = wheel_dir / "unowned_pkg-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(extra_wheel, "w") as archive:
        archive.writestr(
            "unowned_pkg-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: unowned-pkg\nVersion: 1.0.0\n",
        )
    result, report = _run(tmp_path)
    assert result.returncode == 1
    assert report["source_matches_all_wheels"] is False
    assert "expected 1 workspace wheels, found 2" in report["errors"]
    extra_wheel.unlink()

    source.write_text("VALUE = 2\n", encoding="utf-8")
    result, report = _run(tmp_path)
    assert result.returncode == 1
    assert report["source_matches_all_wheels"] is False
    assert report["wheels"][0]["content_mismatch"] == ["demo/__init__.py"]
