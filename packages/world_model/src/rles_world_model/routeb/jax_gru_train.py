"""Train and resume the E1 GRU baseline on registered Train-only sequences."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from .checkpoint import load_checkpoint, optimizer_recipe, save_checkpoint
from .jax_gru import (GRUConfig, forecast_standardized, init_parameters,
                      loss_and_metrics, next_distribution_standardized)
from .jax_train import PINNED_UPSTREAM, SNAPSHOT_SHA, TrainingConfig
from .sequences import load_worker_batch


def train_gru(*, dataset_npz: str | Path, train_scope_id: str, output_checkpoint: str | Path,
              model_config: GRUConfig | None = None,
              training_config: TrainingConfig | None = None,
              resume: bool = False) -> list[dict]:
    """Fit E1 with deterministic batches and a resumable full optimizer PyTree."""
    import jax
    import jax.numpy as jnp
    import optax

    model = model_config or GRUConfig()
    training = training_config or TrainingConfig()
    batch, metadata = load_worker_batch(
        dataset_npz,
        expected_scope=train_scope_id,
        snapshot_sha=SNAPSHOT_SHA,
        upstream_sha=PINNED_UPSTREAM,
    )
    count = len(batch.features)
    if count < training.batch_size:
        raise ValueError("not enough Train sequences for E1 batch size")

    params = init_parameters(training.seed, model)
    optimizer = optax.chain(
        optax.clip_by_global_norm(training.grad_clip), optax.adam(training.lr)
    )
    optimizer_state = optimizer.init(params)
    identity = {
        "model_family": "E1_PROBABILISTIC_GRU",
        "train_scope_id": train_scope_id,
        "dataset_id": metadata["dataset_id"],
        "snapshot_sha": SNAPSHOT_SHA,
        "upstream_sha": PINNED_UPSTREAM,
        "train_npz_sha256": metadata["artifact_sha256"],
        "scaler_sha256": metadata["scaler_sha256"],
    }
    recipe = optimizer_recipe(model, training)
    start = 0
    checkpoint = Path(output_checkpoint)
    if checkpoint.exists() and not resume:
        raise FileExistsError("E1 checkpoint exists; pass resume=True to continue it")
    if resume:
        params, optimizer_state, start, seed = load_checkpoint(
            checkpoint,
            params_template=params,
            opt_template=optimizer_state,
            expected_identity=identity,
            expected_config=recipe,
        )
        if seed != training.seed:
            raise ValueError("E1 checkpoint seed differs from requested seed")
    if start > training.steps:
        raise ValueError("resume step exceeds requested E1 training steps")

    @jax.jit
    def update(current_params, current_optimizer, features, reset, masks):
        def objective(parameters):
            return loss_and_metrics(parameters, features, reset, masks, model)

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current_params)
        changes, next_optimizer = optimizer.update(gradients, current_optimizer, current_params)
        next_params = optax.apply_updates(current_params, changes)
        return next_params, next_optimizer, {**metrics, "loss": loss,
                                              "grad_norm": optax.global_norm(gradients)}

    reports: list[dict[str, object]] = []
    training_started = time.perf_counter()
    for step in range(start, training.steps):
        selection = np.random.default_rng(training.seed + step).choice(
            count, size=training.batch_size, replace=False
        )
        params, optimizer_state, metrics = update(
            params,
            optimizer_state,
            jnp.asarray(batch.features[selection]),
            jnp.asarray(batch.reset[selection]),
            jnp.asarray(batch.masks[selection]),
        )
        scalars = {key: float(np.asarray(value)) for key, value in metrics.items()}
        if not all(np.isfinite(value) for value in scalars.values()):
            raise RuntimeError("nonfinite E1 training metric; failed closed")
        if (step + 1) % training.report_interval == 0 or step + 1 == training.steps:
            reports.append({"step": step + 1, **scalars})
            save_checkpoint(
                checkpoint,
                params=params,
                opt_state=optimizer_state,
                step=step + 1,
                seed=training.seed,
                identity=identity,
                config=recipe,
            )
    if not reports:
        raise ValueError("resume checkpoint already reached the requested E1 max steps")
    parameter_count = sum(
        int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params)
    )
    reports[-1].update({
        "parameter_count": parameter_count,
        "train_examples": count,
        "optimizer_updates": training.steps,
        "training_seconds": time.perf_counter() - training_started,
        "device": str(jax.devices()[0]),
    })
    return reports


def forecast_gru(*, checkpoint: str | Path, model_config: GRUConfig,
                 training_config: TrainingConfig, train_scope_id: str,
                 dataset_id: str, train_npz_sha256: str, scaler_sha256: str,
                 history: np.ndarray, steps: int, forecast_seed: int) -> np.ndarray:
    """Load a provenance-bound E1 checkpoint and sample recursive paths."""
    import optax

    values = np.asarray(history, dtype=np.float32)
    if (values.ndim != 3 or values.shape[-1] != model_config.feature_dim
            or values.shape[1] < 2 or not np.isfinite(values).all()):
        raise ValueError("E1 history must be finite [batch,time,5]")
    if type(steps) is not int or not 1 <= steps <= 60:
        raise ValueError("E1 forecast horizon outside supported range")
    if type(forecast_seed) is not int or forecast_seed < 0:
        raise ValueError("E1 forecast seed must be a nonnegative integer")
    params = init_parameters(training_config.seed, model_config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.grad_clip),
        optax.adam(training_config.lr),
    )
    identity = {
        "model_family": "E1_PROBABILISTIC_GRU",
        "train_scope_id": train_scope_id,
        "dataset_id": dataset_id,
        "snapshot_sha": SNAPSHOT_SHA,
        "upstream_sha": PINNED_UPSTREAM,
        "train_npz_sha256": train_npz_sha256,
        "scaler_sha256": scaler_sha256,
    }
    params, _, _, seed = load_checkpoint(
        checkpoint,
        params_template=params,
        opt_template=optimizer.init(params),
        expected_identity=identity,
        expected_config=optimizer_recipe(model_config, training_config),
    )
    if seed != training_config.seed:
        raise ValueError("E1 checkpoint seed differs from forecast configuration")
    paths = forecast_standardized(
        params, values, steps=steps, seed=forecast_seed, config=model_config
    )
    result = np.asarray(paths)
    if result.shape != (values.shape[0], steps, model_config.feature_dim) or not np.isfinite(result).all():
        raise RuntimeError("E1 forecast output has invalid shape or nonfinite values")
    return result


def forecast_gru_distribution(*, checkpoint: str | Path, model_config: GRUConfig,
                              training_config: TrainingConfig, train_scope_id: str,
                              dataset_id: str, train_npz_sha256: str, scaler_sha256: str,
                              history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return provenance-bound E1 one-step Gaussian feature means/scales."""
    import optax

    values = np.asarray(history, dtype=np.float32)
    if (values.ndim != 3 or values.shape[-1] != model_config.feature_dim
            or values.shape[1] < 2 or not np.isfinite(values).all()):
        raise ValueError("E1 history must be finite [batch,time,5]")
    params = init_parameters(training_config.seed, model_config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.grad_clip), optax.adam(training_config.lr))
    identity = {
        "model_family": "E1_PROBABILISTIC_GRU",
        "train_scope_id": train_scope_id,
        "dataset_id": dataset_id,
        "snapshot_sha": SNAPSHOT_SHA,
        "upstream_sha": PINNED_UPSTREAM,
        "train_npz_sha256": train_npz_sha256,
        "scaler_sha256": scaler_sha256,
    }
    params, _, _, seed = load_checkpoint(
        checkpoint, params_template=params, opt_template=optimizer.init(params),
        expected_identity=identity, expected_config=optimizer_recipe(model_config, training_config))
    if seed != training_config.seed:
        raise ValueError("E1 checkpoint seed differs from forecast configuration")
    mean, scale = next_distribution_standardized(params, values, config=model_config)
    mean_array, scale_array = np.asarray(mean), np.asarray(scale)
    if (mean_array.shape != (len(values), model_config.feature_dim)
            or scale_array.shape != mean_array.shape or not np.isfinite(mean_array).all()
            or not np.isfinite(scale_array).all() or np.any(scale_array <= 0)):
        raise RuntimeError("E1 one-step forecast distribution is invalid")
    return mean_array, scale_array
