"""Authenticated Train-only B3 hierarchy artifacts for the isolated JAX worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from rles_data.io import strict_json_loads

from .codec import TrainOnlyStandardizer
from .hierarchy_dataset import HierarchyScalers, HierarchyTensors
from .jax_model import PINNED_UPSTREAM
from .jax_train import SNAPSHOT_SHA


_TENSOR_FIELDS = (
    "minute_x", "hour_x", "day_x", "hour_index", "day_index", "minute_mask",
    "hour_mask", "day_mask", "minute_available_at", "hour_available_at",
    "day_available_at", "hour_close_at", "day_close_at",
)
_SCALE_FIELDS = ("minute", "hour", "day")
_ARTIFACT_SCHEMA = "routeb.hierarchy-train.v1"


def hierarchy_scaler_sha256(scalers: HierarchyScalers) -> str:
    if not isinstance(scalers, HierarchyScalers):
        raise ValueError("B3 Train scalers are required")
    fingerprints = tuple(getattr(scalers, name).fingerprint for name in _SCALE_FIELDS)
    return hashlib.sha256("".join(fingerprints).encode("ascii")).hexdigest()


def _hierarchy_tensor_sha256(tensors: HierarchyTensors) -> str:
    from .jax_hierarchy_train import hierarchy_tensor_sha256

    return hierarchy_tensor_sha256(tensors)


def export_hierarchical_train_artifact(path: str | Path, *,
        tensors: HierarchyTensors, scalers: HierarchyScalers,
        train_scope_id: str, fold_sha256: str,
        snapshot_sha: str = SNAPSHOT_SHA,
        upstream_sha: str = PINNED_UPSTREAM) -> tuple[Path, dict[str, Any]]:
    """Persist one verified Train hierarchy and its frozen scale-specific scalers.

    The artifact contains prepared features only; no Validation/Test labels or
    raw market-file path is passed to the Python 3.12 worker.
    """
    if (not isinstance(tensors, HierarchyTensors)
            or not isinstance(scalers, HierarchyScalers)
            or not isinstance(train_scope_id, str) or not train_scope_id.strip()
            or not isinstance(fold_sha256, str) or len(fold_sha256) != 64
            or any(char not in "0123456789abcdef" for char in fold_sha256)
            or snapshot_sha != SNAPSHOT_SHA or upstream_sha != PINNED_UPSTREAM
            or any(scaler.dataset_id != scalers.minute.dataset_id
                   or scaler.train_scope != "train"
                   for scaler in (scalers.minute, scalers.hour, scalers.day))):
        raise ValueError("invalid B3 Train hierarchy or frozen identity")
    target = Path(path).resolve()
    meta_path = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or meta_path.exists():
        raise FileExistsError("refusing to overwrite B3 Train hierarchy artifact")
    arrays = {name: np.asarray(getattr(tensors, name)) for name in _TENSOR_FIELDS}
    for name in _SCALE_FIELDS:
        scaler = getattr(scalers, name)
        arrays[f"{name}_scaler_mean"] = scaler.mean
        arrays[f"{name}_scaler_std"] = scaler.std
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as stream:
        temporary_npz = Path(stream.name)
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    temporary_meta = meta_path.with_suffix(".json.partial")
    try:
        artifact_sha = hashlib.sha256(temporary_npz.read_bytes()).hexdigest()
        tensor_sha = _hierarchy_tensor_sha256(tensors)
        scaler_sha = hierarchy_scaler_sha256(scalers)
        metadata: dict[str, Any] = {
            "schema": _ARTIFACT_SCHEMA,
            "purpose": "train",
            "train_scope_id": train_scope_id,
            "dataset_id": scalers.minute.dataset_id,
            "fold_sha256": fold_sha256,
            "snapshot_sha": snapshot_sha,
            "upstream_sha": upstream_sha,
            "artifact_sha256": artifact_sha,
            "hierarchy_tensor_sha256": tensor_sha,
            "scaler_sha256": scaler_sha,
            "tensor_shapes": {name: list(arrays[name].shape) for name in _TENSOR_FIELDS},
            "validation_payload_in_worker": False,
            "test_payload_in_worker": False,
        }
        temporary_meta.write_text(json.dumps(
            metadata, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary_npz, target)
        os.replace(temporary_meta, meta_path)
    finally:
        temporary_npz.unlink(missing_ok=True)
        temporary_meta.unlink(missing_ok=True)
    return target, metadata


def load_hierarchical_train_artifact(path: str | Path, *,
        expected_scope: str, expected_fold_sha256: str,
        expected_snapshot_sha: str = SNAPSHOT_SHA,
        expected_upstream_sha: str = PINNED_UPSTREAM
        ) -> tuple[HierarchyTensors, HierarchyScalers, dict[str, Any]]:
    """Load a checksum- and identity-bound B3 Train artifact in the worker."""
    target = Path(path)
    metadata = strict_json_loads(
        target.with_suffix(".json").read_bytes(), context="B3 Train hierarchy manifest")
    if (not isinstance(metadata, dict)
            or metadata.get("schema") != _ARTIFACT_SCHEMA
            or metadata.get("purpose") != "train"
            or metadata.get("train_scope_id") != expected_scope
            or metadata.get("fold_sha256") != expected_fold_sha256
            or metadata.get("snapshot_sha") != expected_snapshot_sha
            or metadata.get("upstream_sha") != expected_upstream_sha
            or metadata.get("validation_payload_in_worker") is not False
            or metadata.get("test_payload_in_worker") is not False):
        raise ValueError("B3 Train hierarchy provenance mismatch")
    if hashlib.sha256(target.read_bytes()).hexdigest() != metadata.get("artifact_sha256"):
        raise ValueError("B3 Train hierarchy checksum mismatch")
    dataset_id = metadata.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("B3 Train hierarchy dataset identity is invalid")
    expected_members = set(_TENSOR_FIELDS)
    expected_members.update(f"{name}_scaler_{moment}"
                            for name in _SCALE_FIELDS for moment in ("mean", "std"))
    with np.load(target, allow_pickle=False) as archive:
        if set(archive.files) != expected_members:
            raise ValueError("B3 Train hierarchy NPZ member schema mismatch")
        tensor_kwargs = {name: np.asarray(archive[name]).copy() for name in _TENSOR_FIELDS}
        scalers = HierarchyScalers(*(
            TrainOnlyStandardizer(
                np.asarray(archive[f"{name}_scaler_mean"], dtype=np.float64),
                np.asarray(archive[f"{name}_scaler_std"], dtype=np.float64),
                dataset_id, "train")
            for name in _SCALE_FIELDS
        ))
    tensors = HierarchyTensors(**tensor_kwargs)
    if (metadata.get("tensor_shapes") != {
            name: list(getattr(tensors, name).shape) for name in _TENSOR_FIELDS}
            or _hierarchy_tensor_sha256(tensors) != metadata.get("hierarchy_tensor_sha256")
            or hierarchy_scaler_sha256(scalers) != metadata.get("scaler_sha256")
            or any(scaler.dataset_id != dataset_id for scaler in
                   (scalers.minute, scalers.hour, scalers.day))):
        raise ValueError("B3 Train hierarchy tensors or scaler identity mismatch")
    return tensors, scalers, metadata
