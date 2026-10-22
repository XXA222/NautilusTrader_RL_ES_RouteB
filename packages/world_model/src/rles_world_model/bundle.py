"""Safe, content-addressed Route B model bundles (NPZ, no object deserialization)."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from rles_data.core import DataError

from .upstream import UPSTREAM_COMMIT
from .learner import FORECAST_SUPPORT
from .json_utils import loads_unique_json

BUNDLE_SCHEMA = "rles.world_model.bundle.v1"


def save_bundle(path: str | Path, learner: Any, *, feature_spec_hash: str,
                scaler_manifest: dict[str, Any], instrument: str,
                source_code_sha256: str) -> Path:
    """Write a new model bundle atomically; existing bundle directories are immutable."""
    target = Path(path).resolve()
    if target.exists():
        raise DataError("model bundle target already exists")
    if (not _sha256(feature_spec_hash) or not _sha256(source_code_sha256)
            or not isinstance(instrument, str) or not instrument.strip()
            or not isinstance(scaler_manifest, dict)):
        raise DataError("bundle requires feature/source hashes, instrument, and scaler manifest")
    try:
        scaler_payload = json.dumps(scaler_manifest, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DataError("scaler manifest must contain JSON-safe values") from exc
    checkpoint = learner.checkpoint()
    if not isinstance(learner.input_feature_count, int) or learner.input_feature_count < 1:
        raise DataError("model bundle requires an initialized feature input shape")
    params = _arrays(checkpoint["params"], learner.jax)
    optimizer = _arrays(checkpoint["optimizer_state"], learner.jax)
    if not params:
        raise DataError("model bundle cannot contain empty parameters")
    payload = {f"parameter_{index:05d}": value for index, value in enumerate(params)}
    payload.update({f"optimizer_{index:05d}": value for index, value in enumerate(optimizer)})
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise DataError("model bundle temporary path already exists")
    try:
        temporary.mkdir(parents=False)
        weights_path = temporary / "state.npz"
        with weights_path.open("xb") as stream:
            np.savez_compressed(stream, **cast(Any, payload))
        weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        config = learner.config
        config_data = _config_dict(config)
        manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "upstream_repository": "https://github.com/danijar/dreamerv3.git",
            "upstream_commit": UPSTREAM_COMMIT,
            "adapter_version": "rles-market-rssm-v0",
            "feature_spec_hash": feature_spec_hash,
            "scaler_manifest": json.loads(scaler_payload),
            "instrument": instrument,
            "feature_count": learner.input_feature_count,
            "forecast_support": FORECAST_SUPPORT,
            "source_code_sha256": source_code_sha256,
            "rssm_config": config_data,
            "training_step": checkpoint["step"],
            "seed": checkpoint["seed"],
            "parameter_count": len(params),
            "optimizer_state_count": len(optimizer),
            "parameter_tree": str(learner.jax.tree_util.tree_structure(checkpoint["params"])),
            "optimizer_tree": str(learner.jax.tree_util.tree_structure(checkpoint["optimizer_state"])),
            "state_sha256": weights_hash,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2,
                                            ensure_ascii=True, allow_nan=False) + "\n",
                                 encoding="utf-8", newline="\n")
        _fsync_file(manifest_path)
        _fsync_file(weights_path)
        temporary.rename(target)
    except Exception:
        if temporary.exists():
            for child in temporary.iterdir():
                if child.is_file():
                    child.unlink()
            temporary.rmdir()
        raise
    return target


def load_bundle(path: str | Path, learner: Any, *, expected_feature_spec_hash: str,
                expected_source_code_sha256: str) -> dict[str, Any]:
    """Restore parameters and optimizer into an initialized compatible worker."""
    root = Path(path).resolve()
    try:
        manifest = loads_unique_json((root / "manifest.json").read_text(encoding="utf-8"))
        weights_path = root / "state.npz"
        raw = weights_path.read_bytes()
    except (OSError, UnicodeError, ValueError) as exc:
        raise DataError("bundle manifest or state archive is unreadable") from exc
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != BUNDLE_SCHEMA
            or manifest.get("upstream_commit") != UPSTREAM_COMMIT
            or manifest.get("feature_spec_hash") != expected_feature_spec_hash
            or manifest.get("source_code_sha256") != expected_source_code_sha256
            or manifest.get("rssm_config") != _config_dict(learner.config)
            or manifest.get("feature_count") != learner.input_feature_count
            or manifest.get("forecast_support") != FORECAST_SUPPORT
            or hashlib.sha256(raw).hexdigest() != manifest.get("state_sha256")):
        raise DataError("bundle identity, compatibility, or state checksum validation failed")
    template = learner.checkpoint()
    p_leaves, p_tree = learner.jax.tree_util.tree_flatten(template["params"])
    o_leaves, o_tree = learner.jax.tree_util.tree_flatten(template["optimizer_state"])
    if (str(p_tree) != manifest.get("parameter_tree")
            or str(o_tree) != manifest.get("optimizer_tree")
            or len(p_leaves) != manifest.get("parameter_count")
            or len(o_leaves) != manifest.get("optimizer_state_count")):
        raise DataError("bundle tree shape does not match the initialized learner")
    with np.load(weights_path, allow_pickle=False) as archive:
        expected_names = ({f"parameter_{index:05d}" for index in range(len(p_leaves))}
                          | {f"optimizer_{index:05d}" for index in range(len(o_leaves))})
        if set(archive.files) != expected_names:
            raise DataError("bundle archive contains missing or unexpected arrays")
        new_params: list[Any] = []
        new_optimizer: list[Any] = []
        for prefix, templates, output in (("parameter", p_leaves, new_params),
                                          ("optimizer", o_leaves, new_optimizer)):
            for index, expected in enumerate(templates):
                array = np.array(archive[f"{prefix}_{index:05d}"], copy=True)
                if array.shape != expected.shape or array.dtype != np.asarray(expected).dtype:
                    raise DataError("bundle array shape or dtype differs from learner template")
                output.append(learner.jnp.asarray(array))
    learner.params = learner.jax.tree_util.tree_unflatten(p_tree, new_params)
    learner.opt_state = learner.jax.tree_util.tree_unflatten(o_tree, new_optimizer)
    learner.step = manifest.get("training_step")
    if type(learner.step) is not int or learner.step < 0:
        raise DataError("bundle training step is invalid")
    return manifest


def _arrays(tree: Any, jax: Any) -> list[np.ndarray]:
    return [np.asarray(value) for value in jax.tree_util.tree_leaves(tree)]


def _config_dict(config: Any) -> dict[str, Any]:
    if not is_dataclass(config):
        raise DataError("learner configuration must be a dataclass")
    instance = cast(Any, config)
    return {field.name: getattr(instance, field.name) for field in fields(instance)}


def _sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())
