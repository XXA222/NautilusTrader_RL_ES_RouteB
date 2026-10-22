from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from rles_rl_sb3.actor import InferenceActor
from rles_rl_sb3.checkpoint import checkpoint_verify
from rles_rl_sb3.config import digest
from rles_rl_sb3.errors import ArtifactError


def test_sb3_checkpoint_manifest_rejects_duplicate_schema_key(tmp_path):
    payload = b'{"schema":"rles.l4.stage.v3","schema":"rles.l4.stage.v4"}'
    (tmp_path / "manifest.json").write_bytes(payload)
    (tmp_path / "COMPLETED").write_text(hashlib.sha256(payload).hexdigest(), encoding="ascii")

    with pytest.raises(ArtifactError, match="manifest JSON is invalid or ambiguous"):
        checkpoint_verify(tmp_path, None, trusted_source=True)


def test_l0_actor_rejects_duplicate_schema_key_after_integrity_verification(tmp_path):
    mapping = {
        "schema": "rles.l4.actor.mapping.v1",
        "layers": [],
        "input_dim": 1,
        "observation_schema": "obs-v1",
        "action_schema": "action-v1",
    }
    mapping["actor_signature"] = digest(mapping)
    actor = {
        "schema": "rles.l4.inference.v1",
        "mapping": mapping,
        "fingerprint": {"feature_scaler": "a" * 64},
        "policy": {"observation_schema_hash": "obs-v1",
                   "action_schema_hash": "action-v1", "observation_dim": 1},
        "normalization": None,
        "external_feature_scaler_sha256": "a" * 64,
    }
    actor_payload = (
        '{"schema":"ambiguous","schema":"rles.l4.inference.v1",'
        + json.dumps({key: value for key, value in actor.items() if key != "schema"})[1:]
    ).encode("utf-8")
    weights_path = tmp_path / "weights.npz"
    with weights_path.open("wb") as stream:
        np.savez_compressed(stream)
    actor_path = tmp_path / "actor.json"
    actor_path.write_bytes(actor_payload)
    ledger = {
        "weights.npz": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "actor.json": hashlib.sha256(actor_payload).hexdigest(),
    }
    ledger_payload = json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (tmp_path / "integrity.json").write_bytes(ledger_payload)
    (tmp_path / "COMPLETED").write_text(
        hashlib.sha256(ledger_payload).hexdigest(), encoding="ascii")

    with pytest.raises(ArtifactError, match="actor JSON is invalid or ambiguous"):
        InferenceActor(tmp_path, verify=True)
