from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from io import BytesIO

import numpy as np
import pytest

from rles_es_evotorch.checkpoint import ESCheckpointEnvelope, ESCheckpointStore
from rles_es_evotorch.provider import _decode_state


def _checkpoint(root: Path) -> Path:
    components = {name: (name + "-state").encode() for name in (
        "distribution", "optimizer", "population", "rng_torch", "rng_numpy",
        "rng_python", "rng_evotorch", "batch_evaluator", "provider",
    )}
    envelope = ESCheckpointEnvelope(
        algorithm="pgpe", attempt_id="duplicate-json-attempt",
        actor_signature="a" * 64, dataset_hash="b" * 64, config_hash="c" * 64,
        generation=1, evaluations=2, data_cursor=3, native_codec_id="codec-v1",
        state_components=components,
    )
    return ESCheckpointStore(root).save(envelope)


def test_es_checkpoint_manifest_rejects_duplicate_identity_even_with_matching_marker(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path / "checkpoints")
    manifest = directory / "manifest.json"
    payload = manifest.read_bytes()
    duplicate = payload.replace(b'"generation":1', b'"generation":1,"generation":2')
    assert duplicate != payload
    manifest.write_bytes(duplicate)
    (directory / "COMPLETED").write_text(sha256(duplicate).hexdigest(), encoding="ascii")
    with pytest.raises(RuntimeError, match="invalid or ambiguous JSON"):
        ESCheckpointStore(directory.parent).load(directory.name)


def test_native_state_npz_metadata_rejects_duplicate_keys() -> None:
    metadata = b'{"state":1,"state":2}'
    buffer = BytesIO()
    np.savez_compressed(buffer, __metadata__=np.asarray(metadata.decode("ascii")))
    with pytest.raises(ValueError, match="invalid native ES state component"):
        _decode_state(buffer.getvalue())

