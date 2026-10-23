from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from qa.source_manifest import build_manifest


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
    for generated_dir in (
        ".uv-cache", ".artifacts", ".ruff_cache", ".rles-python",
        ".codex-remote-attachments", "__pycache__", "outputs", "runs",
        "checkpoints", "models",
    ):
        generated_file = tmp_path / "packages" / "demo" / generated_dir / "generated.py"
        generated_file.parent.mkdir()
        generated_file.write_text("VALUE = 'generated output'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text("def test_demo(): pass\n", encoding="utf-8")
    (tmp_path / "qa" / "reports").mkdir(parents=True)
    (tmp_path / "qa" / "research_trial_inventory.py").write_text(
        "VALUE = 'trial inventory'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "train_signal_baselines.py").write_text(
        "VALUE = 'signal baseline runner'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "world_model").mkdir()
    (tmp_path / "qa" / "world_model" / "wm1_cpu_smoke.py").write_text(
        "VALUE = 'world model cpu smoke'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_stage58_train_coverage.py").write_text(
        "VALUE = 'train coverage'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_stage59_train_diagnostics.py").write_text(
        "VALUE = 'stage59 analysis'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "preregister_stage59_train_diagnostics.py").write_text(
        "VALUE = 'stage59 preregistration'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_stage60_fixed_regimes.py").write_text(
        "VALUE = 'stage60 fixed regime analysis'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "preregister_stage60_fixed_regimes.py").write_text(
        "VALUE = 'stage60 fixed regime preregistration'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_m33_action_value.py").write_text(
        "VALUE = 'M3.3 action value analysis'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "preregister_m33_action_value.py").write_text(
        "VALUE = 'M3.3 action value preregistration'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "m13_stage_contribution_20261017.py").write_text(
        "VALUE = 'M1.3 stage contribution runner'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_m13_stage_contribution.py").write_text(
        "VALUE = 'M1.3 stage contribution independent analysis'\n", encoding="utf-8",
    )
    manifest_path = tmp_path / "qa" / "reports" / "source_manifest_current.json"

    first = _generate(tmp_path, manifest_path)
    second = _generate(tmp_path, manifest_path)
    assert first["source_identity_sha256"] == second["source_identity_sha256"]
    assert [item["path"] for item in first["files"]] == [
        "packages/demo/module.py", "qa/analyze_m13_stage_contribution.py",
        "qa/analyze_m33_action_value.py",
        "qa/analyze_stage58_train_coverage.py",
        "qa/analyze_stage59_train_diagnostics.py",
        "qa/analyze_stage60_fixed_regimes.py",
        "qa/m13_stage_contribution_20261017.py",
        "qa/preregister_m33_action_value.py",
        "qa/preregister_stage59_train_diagnostics.py",
        "qa/preregister_stage60_fixed_regimes.py",
        "qa/research_trial_inventory.py", "qa/train_signal_baselines.py",
        "qa/world_model/wm1_cpu_smoke.py",
        "tests/test_demo.py",
    ]

    (tmp_path / "packages" / "demo" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = _generate(tmp_path, manifest_path)
    assert changed["source_identity_sha256"] != first["source_identity_sha256"]


def test_source_identity_ignores_untracked_upstream_paths_and_runtime_details(tmp_path: Path) -> None:
    upstream = tmp_path / "nautilus_trader"
    upstream.mkdir()
    subprocess.run(["git", "init", str(upstream)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(upstream), "config", "user.email", "routeb-test@example.invalid"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(upstream), "config", "user.name", "RouteB test"],
                   check=True, capture_output=True)
    tracked = upstream / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(upstream), "add", "tracked.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(upstream), "commit", "-m", "fixture"],
                   check=True, capture_output=True)

    first = build_manifest(tmp_path, generated_at_utc="2026-01-01T00:00:00+00:00")
    private = upstream / "private-local-fixture-path.txt"
    private.write_text("machine-specific artifact\n", encoding="utf-8")
    second = build_manifest(tmp_path, generated_at_utc="2026-01-02T00:00:00+00:00")

    assert first["schema"] == "rles.source-manifest.v2"
    assert "runtime" not in first
    assert "status_entries" not in first["upstream_nautilus"]
    assert first["source_identity_sha256"] == second["source_identity_sha256"]
    assert "private-local-fixture-path.txt" not in json.dumps(second)

    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    changed = build_manifest(tmp_path, generated_at_utc="2026-01-03T00:00:00+00:00")
    assert changed["source_identity_sha256"] != second["source_identity_sha256"]
    assert changed["upstream_nautilus"]["tracked_worktree_clean"] is False

