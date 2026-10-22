from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from qa.wheel_manifest import _source_python_files
from qa.smoke.wheel_imports import _parse_wheel_origin


_GENERATOR = Path(__file__).resolve().parents[1] / "qa" / "wheel_manifest.py"


@pytest.mark.parametrize("payload", [
    '{"url":"file:///wheels/demo.whl","url":"file:///editable/demo.whl",'
    '"archive_info":{},"dir_info":{"editable":false}}',
    '{"url":"file:///wheels/demo.whl","archive_info":{},'
    '"dir_info":{"editable":false,"editable":true}}',
])
def test_wheel_import_provenance_rejects_duplicate_json_keys(payload: str) -> None:
    with pytest.raises(ValueError, match="invalid direct_url.json"):
        _parse_wheel_origin(payload, "demo-pkg")


@pytest.mark.parametrize("payload", [
    '{"url":"file:///wheels/demo.whl","archive_info":{},"dir_info":{"editable":false}}',
    '{"url":"file:///wheels/demo.whl","archive_info":{}}',
])
def test_wheel_import_provenance_accepts_archive_installation(payload: str) -> None:
    assert _parse_wheel_origin(payload, "demo-pkg")["url"] == "file:///wheels/demo.whl"


@pytest.mark.parametrize("payload", [
    '[]',
    '{"url":7,"archive_info":{}}',
    '{"url":"file:///wheels/demo.whl","archive_info":[],"dir_info":{}}',
    '{"url":"file:///wheels/demo.whl","archive_info":{},"dir_info":{"editable":1}}',
    '{"url":"file:///wheels/demo.whl","archive_info":{},"dir_info":{"editable":true}}',
])
def test_wheel_import_provenance_rejects_invalid_source_shapes(payload: str) -> None:
    with pytest.raises(ValueError, match="expected an installed wheel archive origin|root must be an object"):
        _parse_wheel_origin(payload, "demo-pkg")


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
    package.mkdir(exist_ok=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "demo-pkg"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = wheel_dir / "demo_pkg-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo/__init__.py", source.read_bytes())
        archive.writestr(
            "demo_pkg-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: demo-pkg\nVersion: 1.0.0\n",
        )
    (tmp_path / "source.json").write_text(
        json.dumps({"source_identity_sha256": "a" * 64}), encoding="utf-8"
    )

    result, report = _run(tmp_path)
    assert result.returncode == 0
    assert report["source_matches_all_wheels"] is True
    assert report["wheels"][0]["source_file_count"] == 1

    source.write_text("VALUE = 2\n", encoding="utf-8")
    result, report = _run(tmp_path)
    assert result.returncode == 1
    assert report["source_matches_all_wheels"] is False
    assert report["wheels"][0]["content_mismatch"] == ["demo/__init__.py"]


def test_wheel_manifest_rejects_duplicate_source_manifest_keys(tmp_path: Path) -> None:
    (tmp_path / "packages").mkdir()
    (tmp_path / "wheels").mkdir()
    (tmp_path / "source.json").write_text(
        '{"source_identity_sha256":"' + "a" * 64 + '","source_identity_sha256":"' + "a" * 64 + '"}',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--wheels", str(tmp_path / "wheels"),
         "--source-manifest", str(tmp_path / "source.json"),
         "--output", str(tmp_path / "report.json")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "duplicate JSON manifest key" in result.stderr


def test_wheel_manifest_normalizes_invalid_source_manifest_utf8(tmp_path: Path) -> None:
    (tmp_path / "packages").mkdir()
    (tmp_path / "wheels").mkdir()
    (tmp_path / "source.json").write_bytes(b"{\"source_identity_sha256\":\"\xff\"}")
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--wheels", str(tmp_path / "wheels"),
         "--source-manifest", str(tmp_path / "source.json"),
         "--output", str(tmp_path / "report.json")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "cannot read source manifest" in result.stderr


def test_wheel_manifest_rejects_duplicate_archive_paths(tmp_path: Path) -> None:
    package = tmp_path / "packages" / "demo"
    source = package / "src" / "demo" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    package.mkdir(exist_ok=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "demo-pkg"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = wheel_dir / "demo_pkg-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo/__init__.py", source.read_bytes())
        archive.writestr("demo/__init__.py", b"VALUE = 2\n")
        archive.writestr(
            "demo_pkg-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: demo-pkg\nVersion: 1.0.0\n",
        )
    (tmp_path / "source.json").write_text(
        json.dumps({"source_identity_sha256": "a" * 64}), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--wheels", str(wheel_dir), "--source-manifest", str(tmp_path / "source.json"),
         "--output", str(tmp_path / "report.json")],
        capture_output=True, text=True, check=False,
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert report["source_matches_all_wheels"] is False
    assert any("duplicate archive paths" in error for error in report["errors"])


def test_wheel_manifest_rejects_stale_distribution_version(tmp_path: Path) -> None:
    package = tmp_path / "packages" / "demo"
    source = package / "src" / "demo" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    package.mkdir(exist_ok=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "demo-pkg"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = wheel_dir / "demo_pkg-0.9.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo/__init__.py", source.read_bytes())
        archive.writestr(
            "demo_pkg-0.9.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: demo-pkg\nVersion: 0.9.0\n",
        )
    (tmp_path / "source.json").write_text(
        json.dumps({"source_identity_sha256": "a" * 64}), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--wheels", str(wheel_dir), "--source-manifest", str(tmp_path / "source.json"),
         "--output", str(tmp_path / "report.json")],
        capture_output=True, text=True, check=False,
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert report["source_matches_all_wheels"] is False
    assert any("version mismatch" in error for error in report["errors"])


def test_wheel_manifest_rejects_duplicate_metadata_identity_headers(tmp_path: Path) -> None:
    package = tmp_path / "packages" / "demo"
    source = package / "src" / "demo" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    package.mkdir(exist_ok=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "demo-pkg"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = wheel_dir / "demo_pkg-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo/__init__.py", source.read_bytes())
        archive.writestr(
            "demo_pkg-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: demo-pkg\nVersion: 1.0.0\nVersion: 2.0.0\n",
        )
    (tmp_path / "source.json").write_text(
        json.dumps({"source_identity_sha256": "a" * 64}), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--wheels", str(wheel_dir), "--source-manifest", str(tmp_path / "source.json"),
         "--output", str(tmp_path / "report.json")],
        capture_output=True, text=True, check=False,
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert report["source_matches_all_wheels"] is False
    assert any("exactly one Name and Version" in error for error in report["errors"])


def test_wheel_manifest_checks_source_symlink_before_file_stat(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    linked = source_root / "linked.py"
    linked.write_text("VALUE = 1\n", encoding="utf-8")
    original_is_symlink = Path.is_symlink
    original_is_file = Path.is_file

    def is_symlink(path: Path) -> bool:
        return True if path == linked else original_is_symlink(path)

    def is_file(path: Path) -> bool:
        if path == linked:
            raise AssertionError("is_file must not stat a symlink target")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    monkeypatch.setattr(Path, "is_file", is_file)
    assert _source_python_files(source_root) == []


def test_wheel_manifest_rejects_symlink_wheel_directory(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "packages").mkdir(parents=True)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return True if path == wheels else original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    from qa.wheel_manifest import build_wheel_manifest

    with pytest.raises(ValueError, match="wheel directory must be a real directory"):
        build_wheel_manifest(root, wheels, "a" * 64)


def test_wheel_manifest_rejects_nonstandard_source_identity_constant(tmp_path: Path) -> None:
    (tmp_path / "packages").mkdir()
    (tmp_path / "wheels").mkdir()
    (tmp_path / "source.json").write_text(
        '{"source_identity_sha256":"' + "a" * 64 + '","extra":NaN}',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--wheels", str(tmp_path / "wheels"),
         "--source-manifest", str(tmp_path / "source.json"),
         "--output", str(tmp_path / "report.json")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "cannot read source manifest" in result.stderr


def test_wheel_import_provenance_rejects_nonstandard_direct_url_constant() -> None:
    payload = ('{"url":"file:///wheels/demo.whl","archive_info":{},'
               '"dir_info":{"editable":false},"extra":NaN}')
    with pytest.raises(ValueError, match="invalid direct_url.json"):
        _parse_wheel_origin(payload, "demo-pkg")
