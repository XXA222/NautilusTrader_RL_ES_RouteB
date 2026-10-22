from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from qa.source_manifest import _source_files


_GENERATOR = Path(__file__).resolve().parents[1] / "qa" / "source_manifest.py"


def _generate(root: Path, output: Path) -> dict:
    subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(root), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_source_identity_tracks_workspace_inputs_and_ignores_generated_reports(tmp_path: Path) -> None:
    (tmp_path / "packages" / "demo").mkdir(parents=True)
    (tmp_path / "packages" / "demo" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "packages" / "demo" / ".tmp").mkdir()
    (tmp_path / "packages" / "demo" / ".tmp" / "generated.py").write_text(
        "VALUE = 'temporary build output'\n", encoding="utf-8",
    )
    (tmp_path / "packages" / "demo" / "dist-build" / "generated.py").parent.mkdir()
    (tmp_path / "packages" / "demo" / "dist-build" / "generated.py").write_text(
        "VALUE = 'temporary distribution output'\n", encoding="utf-8",
    )
    (tmp_path / "packages" / "demo" / ".venv-runtime" / "Scripts").mkdir(parents=True)
    (tmp_path / "packages" / "demo" / ".venv-runtime" / "Scripts" / "python.exe").write_bytes(
        b"generated interpreter placeholder",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text("def test_demo(): pass\n", encoding="utf-8")
    (tmp_path / "qa" / "reports").mkdir(parents=True)
    (tmp_path / "qa" / "reports" / "generated.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "qa" / "tools").mkdir()
    (tmp_path / "qa" / "tools" / "delivered.py").write_text(
        "VALUE = 'delivered QA tool'\n", encoding="utf-8",
    )
    manifest_path = tmp_path / "qa" / "reports" / "source_manifest_current.json"

    first = _generate(tmp_path, manifest_path)
    second = _generate(tmp_path, manifest_path)
    assert first["source_identity_sha256"] == second["source_identity_sha256"]
    (tmp_path / "packages" / "demo" / "module.py").write_bytes(b"VALUE = 1\r\n")
    crlf = _generate(tmp_path, manifest_path)
    assert crlf["source_identity_sha256"] == first["source_identity_sha256"]
    assert [item["path"] for item in first["files"]] == [
        "packages/demo/module.py", "qa/tools/delivered.py", "tests/test_demo.py",
    ]

    (tmp_path / "qa" / "tools" / "delivered.py").write_text(
        "VALUE = 'changed delivered QA tool'\n", encoding="utf-8",
    )
    qa_changed = _generate(tmp_path, manifest_path)
    assert qa_changed["source_identity_sha256"] != first["source_identity_sha256"]

    (tmp_path / "packages" / "demo" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = _generate(tmp_path, manifest_path)
    assert changed["source_identity_sha256"] != qa_changed["source_identity_sha256"]


def test_source_manifest_verify_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"source_identity_sha256":"' + "a" * 64 + '","source_identity_sha256":"' + "a" * 64 + '"}',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--output", str(manifest), "--verify"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "duplicate JSON manifest key" in result.stderr


def test_source_manifest_verify_normalizes_invalid_utf8(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{\"source_identity_sha256\":\"\xff\"}")
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--output", str(manifest), "--verify"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "cannot read JSON manifest" in result.stderr


def test_source_manifest_checks_symlink_before_file_stat(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "packages" / "demo"
    source_root.mkdir(parents=True)
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
    assert linked not in _source_files(tmp_path)


def test_source_manifest_verify_rejects_nonstandard_json_constant(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"source_identity_sha256":"' + "a" * 64 + '","extra":NaN}',
                        encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--output", str(manifest), "--verify"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "cannot read JSON manifest" in result.stderr
