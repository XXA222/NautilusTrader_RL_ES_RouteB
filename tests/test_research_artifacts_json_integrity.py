from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rles_research.artifacts import ArtifactError, ArtifactStore


def _completed_attempt(tmp_path: Path):
    manifest = {
        name: f"{index:x}" * 64
        for index, name in enumerate(
            ("data_hash", "code_hash", "config_hash", "seed_hash", "deps_hash",
             "adapter_hash", "compute_hash", "execution_hash", "risk_hash"),
            start=1,
        )
    }
    store = ArtifactStore(tmp_path / "artifacts", trusted_sources={"qa-local"})
    writer = store.begin("run-strict-json", "attempt-1")
    writer.add("policy.txt", b"policy")
    directory = writer.complete(l1_manifest=manifest, required_files=["policy.txt"],
                                trusted_source_id="qa-local")
    return store, directory


def _update_index_and_marker(directory: Path) -> None:
    index_path = directory / "integrity.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest_path = directory / "manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    index["files"]["manifest.json"] = manifest_hash
    index["manifest_sha256"] = manifest_hash
    index_bytes = json.dumps(index, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    index_path.write_bytes(index_bytes)
    marker = {"integrity_sha256": hashlib.sha256(index_bytes).hexdigest(), "state": "COMPLETED"}
    (directory / "COMPLETED").write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def test_artifact_store_rejects_nonfinite_integrity_index_even_when_marker_matches(tmp_path: Path) -> None:
    store, directory = _completed_attempt(tmp_path)
    index_path = directory / "integrity.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["ambiguous"] = float("nan")
    index_bytes = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    index_path.write_bytes(index_bytes)
    marker = {"integrity_sha256": hashlib.sha256(index_bytes).hexdigest(), "state": "COMPLETED"}
    (directory / "COMPLETED").write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="invalid marker/index"):
        store.verify("run-strict-json", "attempt-1")


def test_artifact_store_rejects_nonfinite_completion_marker(tmp_path: Path) -> None:
    store, directory = _completed_attempt(tmp_path)
    marker_path = directory / "COMPLETED"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["ambiguous"] = float("inf")
    marker_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    with pytest.raises(ArtifactError, match="invalid marker/index"):
        store.verify("run-strict-json", "attempt-1")


def test_artifact_store_rejects_nonfinite_manifest_after_all_hashes_are_recomputed(tmp_path: Path) -> None:
    store, directory = _completed_attempt(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ambiguous"] = float("-inf")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    _update_index_and_marker(directory)
    with pytest.raises(ArtifactError, match="unreadable L1 experiment manifest"):
        store.verify("run-strict-json", "attempt-1")


@pytest.mark.parametrize("extra", [
    {1: "root-key-coercion"},
    {"nested": {1: "nested-key-coercion"}},
])
def test_artifact_store_rejects_non_string_manifest_keys_before_completion(tmp_path: Path, extra):
    manifest = {
        name: f"{index:x}" * 64
        for index, name in enumerate(
            ("data_hash", "code_hash", "config_hash", "seed_hash", "deps_hash",
             "adapter_hash", "compute_hash", "execution_hash", "risk_hash"),
            start=1,
        )
    }
    manifest.update(extra)
    store = ArtifactStore(tmp_path / "artifacts", trusted_sources={"qa-local"})
    writer = store.begin("run-non-string-manifest-key", "attempt-1")
    writer.add("policy.txt", b"policy")

    with pytest.raises(ArtifactError, match="string object keys"):
        writer.complete(
            l1_manifest=manifest, required_files=["policy.txt"],
            trusted_source_id="qa-local",
        )
