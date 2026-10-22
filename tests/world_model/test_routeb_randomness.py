from __future__ import annotations

import json

import pytest

from rles_world_model.routeb.jax_hierarchy_train import HierarchyTrainingConfig
from rles_world_model.routeb.jax_train import TrainingConfig
from rles_world_model.routeb.randomness import (
    MAX_JAX_SEED_EXCLUSIVE,
    SeedStreams,
    resolve_training_seed_streams,
    seed_stream_identity,
    validate_jax_seed,
    validate_jax_seed_offset,
    validate_training_seed,
)


def test_jax_seed_domain_includes_uint32_max_and_rejects_alias_values() -> None:
    validate_jax_seed(0)
    validate_jax_seed(MAX_JAX_SEED_EXCLUSIVE - 1)
    for invalid in (-1, MAX_JAX_SEED_EXCLUSIVE, True, 1.0):
        with pytest.raises(ValueError, match=r"\[0, 2\*\*32\)"):
            validate_jax_seed(invalid)  # type: ignore[arg-type]


def test_training_seed_reserves_room_for_every_per_update_key() -> None:
    validate_training_seed(MAX_JAX_SEED_EXCLUSIVE - 2, 1)
    with pytest.raises(ValueError, match="exceeds the JAX seed domain"):
        validate_training_seed(MAX_JAX_SEED_EXCLUSIVE - 1, 1)
    with pytest.raises(ValueError, match="exceeds the JAX seed domain"):
        validate_training_seed(1, 0)


def test_seed_offset_rejects_batch_chunk_wraparound() -> None:
    validate_jax_seed_offset(MAX_JAX_SEED_EXCLUSIVE - 5, 4)
    with pytest.raises(ValueError, match="derived offset"):
        validate_jax_seed_offset(MAX_JAX_SEED_EXCLUSIVE - 4, 4)


def test_independent_training_stream_plan_preserves_defaults_and_records_overrides() -> None:
    default = resolve_training_seed_streams(31, 5)
    assert seed_stream_identity(default) == {
        "initialization_seed": 31,
        "sampler_seed": 31,
        "model_seed": 32,
    }
    split = resolve_training_seed_streams(
        31, 5, SeedStreams(initialization_seed=102, sampler_seed=31, model_seed=32))
    assert seed_stream_identity(split) == {
        "initialization_seed": 102,
        "sampler_seed": 31,
        "model_seed": 32,
    }


def test_independent_training_stream_plan_rejects_step_key_overflow() -> None:
    with pytest.raises(ValueError, match="sampler seed"):
        resolve_training_seed_streams(
            31, 2,
            SeedStreams(initialization_seed=31,
                        sampler_seed=MAX_JAX_SEED_EXCLUSIVE - 1,
                        model_seed=32))
    with pytest.raises(ValueError, match="model seed"):
        resolve_training_seed_streams(
            31, 2,
            SeedStreams(initialization_seed=31, sampler_seed=31,
                        model_seed=MAX_JAX_SEED_EXCLUSIVE - 1))


def test_worker_rejects_derived_chunk_seed_wraparound_before_context_read(
        tmp_path, capsys) -> None:
    from rles_world_model.routeb.worker_cli import main

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": {}, "training": {}}), encoding="utf-8")
    request = tmp_path / "request.json"
    payload = {
        "schema": "routeb.forecast-batch-request.v1",
        "train_scope_id": "synthetic-train",
        "dataset_id": "synthetic-dataset",
        "history_npz": str(tmp_path / "must-not-be-read.npz"),
        "context_sha256": "a" * 64,
        "scaler_sha256": "b" * 64,
        "horizon": 1,
        "n_paths": 2,
        "seed": MAX_JAX_SEED_EXCLUSIVE - 4,
        "train_npz_sha256": "c" * 64,
        "instrument": "SYNTH.SIM",
        "anchor_timestamps_ns": [1, 2, 3, 4, 5],
        "decision_timestamps_ns": [1, 2, 3, 4, 5],
    }
    request.write_text(json.dumps(payload), encoding="utf-8")
    result = main([
        "forecast-batch", "--request", str(request), "--checkout", "unused",
        "--checkpoint", "unused", "--output", str(tmp_path / "out.npz"),
        "--configuration", str(config),
    ])
    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "BLOCKED"
    assert "derived offset" in error["detail"]


def test_jax_accepts_distinct_high_uint32_keys() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    seeds = jnp.asarray((0, 1 << 31, MAX_JAX_SEED_EXCLUSIVE - 1), dtype=jnp.uint32)
    keys = jax.vmap(jax.random.PRNGKey)(seeds)
    assert keys.shape == (3, 2)
    assert len({tuple(int(word) for word in key) for key in keys.tolist()}) == 3


def test_f2_full_filter_rejects_derived_seed_overflow_before_context_use() -> None:
    pytest.importorskip("jax")
    from rles_world_model.routeb.jax_hierarchy_f2 import rollout_hierarchical_f2_jax

    with pytest.raises(ValueError, match="step-derived random seed"):
        rollout_hierarchical_f2_jax(
            prefix=None, context_tensors=None, scalers=None, checkout="unused",
            params={}, steps=1, mode="F2_FULL_FILTER",
            seed=MAX_JAX_SEED_EXCLUSIVE - 1)


@pytest.mark.parametrize("config_type", [TrainingConfig, HierarchyTrainingConfig])
def test_jax_training_configs_reject_seed_wraparound(config_type) -> None:
    assert config_type(seed=MAX_JAX_SEED_EXCLUSIVE - 2, steps=1).seed == (
        MAX_JAX_SEED_EXCLUSIVE - 2)
    with pytest.raises(ValueError, match="seed/update range"):
        config_type(seed=MAX_JAX_SEED_EXCLUSIVE - 1, steps=1)
    with pytest.raises(ValueError, match="seed/update range"):
        config_type(seed=MAX_JAX_SEED_EXCLUSIVE, steps=1)
