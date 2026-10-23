"""Train-only optimizer for the experimental three-scale B3 RSSM."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import math
import time

import numpy as np

from .checkpoint import load_checkpoint, optimizer_recipe, save_checkpoint
from .hierarchy_dataset import HierarchyTensors, make_hierarchy_windows
from .jax_model import PINNED_UPSTREAM, RSSMConfig
from .jax_train import SNAPSHOT_SHA
from .jax_hierarchy import make_hierarchical_model
from .hierarchy import derive_future_scale_events
from .clock import MultiScaleClock


@dataclass(frozen=True, slots=True)
class HierarchicalTrainingConfig:
    seed: int = 47
    steps: int = 100
    batch_size: int = 2
    window_length: int = 64
    burn_in: int = 8
    stride: int = 16
    lr: float = 3e-4
    grad_clip: float = 100.0
    report_interval: int = 25

    def __post_init__(self) -> None:
        ints = (self.seed, self.steps, self.batch_size, self.window_length,
                self.burn_in, self.stride, self.report_interval)
        if (any(type(x) is not int for x in ints) or self.seed < 0
                or self.steps < 1 or self.batch_size < 1
                or self.window_length < 3 or not 1 <= self.burn_in < self.window_length - 1
                or self.stride < 1 or self.report_interval < 1
                or not math.isfinite(self.lr) or self.lr <= 0
                or not math.isfinite(self.grad_clip) or self.grad_clip <= 0):
            raise ValueError("invalid hierarchical training config")


@dataclass(frozen=True, slots=True)
class HierarchicalModelConfig:
    minute: RSSMConfig
    hour: RSSMConfig
    day: RSSMConfig


def hierarchy_tensor_sha256(tensors: HierarchyTensors) -> str:
    digest = sha256()
    for name in ("minute_x", "hour_x", "day_x", "hour_index", "day_index",
                 "minute_mask", "hour_mask", "day_mask", "minute_available_at",
                 "hour_available_at", "day_available_at", "hour_close_at", "day_close_at"):
        array = np.ascontiguousarray(getattr(tensors, name))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def train_hierarchical(*, tensors: HierarchyTensors, train_scope_id: str,
        dataset_id: str, scaler_sha256: str, checkout: str | Path,
        output_checkpoint: str | Path,
        model_config: HierarchicalModelConfig | None = None,
        training_config: HierarchicalTrainingConfig | None = None,
        resume: bool = False) -> list[dict[str, float | int | str]]:
    """Optimize all three RSSMs using only the supplied Train tensors.

    The caller must obtain ``tensors`` through ``build_train_hierarchy`` on an
    approved Train replay. Checkpoints bind exact tensor bytes, dataset/scope,
    frozen scaler, source snapshot and upstream revision.
    """
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax

    if not train_scope_id or not dataset_id or len(scaler_sha256) != 64:
        raise ValueError("Train scope, dataset identity and scaler SHA-256 are required")
    try:
        int(scaler_sha256, 16)
    except ValueError as exc:
        raise ValueError("invalid scaler SHA-256") from exc
    model_cfg = model_config or HierarchicalModelConfig(
        RSSMConfig(), RSSMConfig(), RSSMConfig())
    cfg = training_config or HierarchicalTrainingConfig()
    windows = make_hierarchy_windows(tensors, length=cfg.window_length,
                                     burn_in=cfg.burn_in, stride=cfg.stride)
    if len(windows.starts) < cfg.batch_size:
        raise ValueError("not enough causal Train windows for batch size")
    model = make_hierarchical_model(checkout, minute=model_cfg.minute,
                                    hour=model_cfg.hour, day=model_cfg.day)
    slow_mask_h = tensors.hour_mask
    slow_mask_d = tensors.day_mask
    sample = np.arange(cfg.batch_size)
    args = (jnp.asarray(windows.minute_x[sample]), jnp.asarray(tensors.hour_x),
            jnp.asarray(tensors.day_x), jnp.asarray(windows.hour_index[sample]),
            jnp.asarray(windows.day_index[sample]), jnp.asarray(windows.minute_mask[sample]),
            jnp.asarray(slow_mask_h), jnp.asarray(slow_mask_d),
            jnp.asarray(windows.reset[sample]))
    params = nj.init(model)({}, *args, seed=cfg.seed)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                            optax.adam(cfg.lr))
    opt_state = optimizer.init(params)
    identity = {"train_scope_id": train_scope_id, "dataset_id": dataset_id,
                "snapshot_sha": SNAPSHOT_SHA, "upstream_sha": PINNED_UPSTREAM,
                "hierarchy_tensor_sha256": hierarchy_tensor_sha256(tensors),
                "scaler_sha256": scaler_sha256}
    configuration = optimizer_recipe(model_cfg, cfg)
    start = 0
    if resume:
        params, opt_state, start, seed = load_checkpoint(
            output_checkpoint, params_template=params, opt_template=opt_state,
            expected_identity=identity, expected_config=configuration)
        if seed != cfg.seed:
            raise ValueError("hierarchical checkpoint seed mismatch")
    elif Path(output_checkpoint).exists():
        raise FileExistsError("B3 checkpoint exists; pass resume=True to continue")
    if start >= cfg.steps:
        raise ValueError("resume checkpoint already reached requested maximum steps")

    @jax.jit
    def update(parameters, state, batch_x, hour_x, day_x, hour_index, day_index,
               minute_mask, hour_mask, day_mask, reset, rng_seed):
        def loss_fn(p):
            _, diagnostics = nj.pure(model)(
                p, batch_x, hour_x, day_x, hour_index, day_index,
                minute_mask, hour_mask, day_mask, reset, seed=rng_seed)
            return diagnostics["loss"], diagnostics
        (value, diagnostics), grads = jax.value_and_grad(loss_fn, has_aux=True)(parameters)
        changes, next_state = optimizer.update(grads, state, parameters)
        next_params = optax.apply_updates(parameters, changes)
        named_leaves = jax.tree_util.tree_flatten_with_path(grads)[0]
        group_norms = {}
        for scale in ("minute_dyn", "hour_dyn", "day_dyn"):
            selected = [leaf for path, leaf in named_leaves
                        if scale in jax.tree_util.keystr(path)]
            if not selected:
                raise ValueError(f"B3 optimizer tree has no {scale} parameters")
            group_norms[f"grad_{scale}_norm"] = optax.global_norm(selected)
        diagnostics = {**diagnostics, **group_norms, "loss": value,
                       "grad_norm": optax.global_norm(grads)}
        return next_params, next_state, diagnostics

    reports: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    for step in range(start, cfg.steps):
        selected = np.random.default_rng(cfg.seed + step).choice(
            len(windows.starts), size=cfg.batch_size, replace=False)
        params, opt_state, metrics = update(
            params, opt_state, jnp.asarray(windows.minute_x[selected]),
            jnp.asarray(tensors.hour_x), jnp.asarray(tensors.day_x),
            jnp.asarray(windows.hour_index[selected]), jnp.asarray(windows.day_index[selected]),
            jnp.asarray(windows.minute_mask[selected]), jnp.asarray(slow_mask_h),
            jnp.asarray(slow_mask_d), jnp.asarray(windows.reset[selected]), cfg.seed + step + 1)
        scalar = {key: float(np.asarray(value)) for key, value in metrics.items()}
        if not all(math.isfinite(value) for value in scalar.values()):
            raise RuntimeError("nonfinite B3 training metric; failed closed")
        if (step + 1) % cfg.report_interval == 0 or step + 1 == cfg.steps:
            reports.append({"step": step + 1, **scalar})
            save_checkpoint(output_checkpoint, params=params, opt_state=opt_state,
                            step=step + 1, seed=cfg.seed, identity=identity,
                            config=configuration)
    reports[-1].update({"parameter_count": sum(
        int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params)),
        "train_windows": len(windows.starts), "optimizer_updates": cfg.steps,
        "training_seconds": time.perf_counter() - started,
        "device": str(jax.devices()[0]),
        "train_hierarchy_tensor_sha256": identity["hierarchy_tensor_sha256"]})
    return reports


def forecast_hierarchical_f2(*, tensors: HierarchyTensors, train_scope_id: str,
        dataset_id: str, scaler_sha256: str, checkout: str | Path,
        checkpoint: str | Path, model_config: HierarchicalModelConfig,
        training_config: HierarchicalTrainingConfig,
        training_tensor_sha256: str, inference_context_sha256: str,
        future_minute_decisions_ns: np.ndarray,
        hour_availability_delay_ns: int,
        day_availability_delay_ns: int,
        forecast_seed: int = 811
        ) -> dict[str, np.ndarray]:
    """Load a provenance-bound B3 checkpoint and recursively imagine F2 priors.

    Event flags are derived from the UTC calendar in ``MultiScaleClock`` plus
    explicit, frozen per-scale availability delays. Future market observations
    are deliberately not accepted by this API.
    """
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax

    if type(forecast_seed) is not int or forecast_seed < 0:
        raise ValueError("invalid F2 forecast seed")
    actual_context_sha256 = hierarchy_tensor_sha256(tensors)
    if (not isinstance(training_tensor_sha256, str)
            or len(training_tensor_sha256) != 64
            or any(char not in "0123456789abcdef" for char in training_tensor_sha256)
            or not isinstance(inference_context_sha256, str)
            or inference_context_sha256 != actual_context_sha256):
        raise ValueError("F2 training identity or authorized inference context identity is invalid")
    decision_times = np.asarray(future_minute_decisions_ns)
    if (type(hour_availability_delay_ns) is not int or hour_availability_delay_ns < 0
            or type(day_availability_delay_ns) is not int or day_availability_delay_ns < 0):
        raise ValueError("F2 availability delays must be nonnegative integer nanoseconds")
    if len(decision_times) and decision_times[0] <= tensors.minute_available_at[-1]:
        raise ValueError("F2 future decisions must follow the final observed Train availability barrier")
    hour_times = np.asarray(MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decision_times, timeframe_minutes=60,
        availability_delay_ns=hour_availability_delay_ns,
        observed_close_through_ns=int(tensors.hour_close_at[-1])), dtype=np.int64)
    day_times = np.asarray(MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decision_times, timeframe_minutes=1440,
        availability_delay_ns=day_availability_delay_ns,
        observed_close_through_ns=int(tensors.day_close_at[-1])), dtype=np.int64)
    future_hour_events, future_day_events = derive_future_scale_events(
        minute_decisions_ns=decision_times,
        hour_available_at_ns=hour_times,
        day_available_at_ns=day_times)
    if not future_hour_events:
        raise ValueError("F2 needs at least one future minute decision")
    model = make_hierarchical_model(checkout, minute=model_config.minute,
                                    hour=model_config.hour, day=model_config.day)
    batch_size = tensors.minute_x.shape[0]
    minute_reset = np.zeros(tensors.minute_x.shape[:2], dtype=np.bool_)
    minute_reset[:, 0] = True
    template_args = (jnp.asarray(tensors.minute_x), jnp.asarray(tensors.hour_x),
        jnp.asarray(tensors.day_x), jnp.asarray(tensors.hour_index),
        jnp.asarray(tensors.day_index), jnp.asarray(tensors.minute_mask),
        jnp.asarray(tensors.hour_mask), jnp.asarray(tensors.day_mask),
        jnp.asarray(minute_reset))
    params = nj.init(model)({}, *template_args, seed=training_config.seed)
    optimizer = optax.chain(optax.clip_by_global_norm(training_config.grad_clip),
                            optax.adam(training_config.lr))
    identity = {"train_scope_id": train_scope_id, "dataset_id": dataset_id,
        "snapshot_sha": SNAPSHOT_SHA, "upstream_sha": PINNED_UPSTREAM,
        "hierarchy_tensor_sha256": training_tensor_sha256,
        "scaler_sha256": scaler_sha256}
    params, _, _, checkpoint_seed = load_checkpoint(checkpoint,
        params_template=params, opt_template=optimizer.init(params),
        expected_identity=identity,
        expected_config=optimizer_recipe(model_config, training_config))
    if checkpoint_seed != training_config.seed:
        raise ValueError("F2 checkpoint seed differs from training configuration")
    _, result = nj.pure(model.rollout_f2)(params,
        jnp.asarray(tensors.minute_x), jnp.asarray(tensors.hour_x),
        jnp.asarray(tensors.day_x), jnp.asarray(tensors.hour_index),
        jnp.asarray(tensors.day_index), future_hour_events, future_day_events,
        seed=forecast_seed)
    outputs = {key: np.asarray(jax.device_get(value)) for key, value in result.items()}
    horizon = len(future_hour_events)
    if (outputs["minute_mean"].shape != (batch_size, horizon, 5)
            or outputs["minute_scale"].shape != (batch_size, horizon, 5)
            or outputs["anchor_minute_mean"].shape != (batch_size, 5)
            or outputs["anchor_minute_scale"].shape != (batch_size, 5)
            or outputs["first_prior_minute_mean"].shape != (batch_size, 5)
            or outputs["first_prior_minute_scale"].shape != (batch_size, 5)
            or outputs["hour_latent_diagnostic"].shape != (batch_size, horizon, 5)
            or outputs["day_latent_diagnostic"].shape != (batch_size, horizon, 5)
            or any(not np.isfinite(value).all() for value in outputs.values())
            or np.any(outputs["minute_scale"] <= 0)
            or np.any(outputs["anchor_minute_scale"] <= 0)
            or np.any(outputs["first_prior_minute_scale"] <= 0)
            or not np.array_equal(outputs["minute_mean"][:, 0],
                                  outputs["anchor_minute_mean"])
            or not np.array_equal(outputs["minute_scale"][:, 0],
                                  outputs["anchor_minute_scale"])):
        raise RuntimeError("F2 produced malformed or nonfinite cross-scale outputs")
    return outputs
