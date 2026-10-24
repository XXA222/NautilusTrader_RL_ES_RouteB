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
from .codec import TrainOnlyStandardizer
from .hierarchical_paths import (
    RecursiveHierarchicalOHLCVPaths,
    generate_recursive_hierarchical_ohlcv_paths,
)
from .market import Candle


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
        forecast_seed: int = 811,
        advance_slow_priors: bool = True,
        minute_feedback_features: np.ndarray | None = None,
        hour_feedback_features: np.ndarray | None = None,
        day_feedback_features: np.ndarray | None = None,
        ) -> dict[str, np.ndarray]:
    """Load a provenance-bound B3 checkpoint for a matched E3/E4 rollout.

    Event flags are derived from the UTC calendar in ``MultiScaleClock`` plus
    explicit, frozen per-scale availability delays. Future market observations
    are deliberately not accepted by this API.

    ``advance_slow_priors=True`` selects E4 (F2 slow-prior transitions).
    ``False`` selects E3 (same three-scale checkpoint and minute-prior rollout,
    with slow latent carries frozen at their observed posterior states).
    Both modes return latent-space features, not independently valid OHLCV.
    When all three feedback tensors are supplied, the forecast is a causal
    replay conditioned on already sampled path features and path-derived slow
    aggregates. The first forecast must be sampled separately; this API does
    not claim to be a single-pass market-path generator.
    """
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax

    if (type(forecast_seed) is not int or forecast_seed < 0
            or type(advance_slow_priors) is not bool):
        raise ValueError("invalid matched E3/E4 forecast mode or seed")
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
    supplied_feedback = (minute_feedback_features is not None,
                         hour_feedback_features is not None,
                         day_feedback_features is not None)
    if any(supplied_feedback) and not all(supplied_feedback):
        raise ValueError("minute, hour and day feedback features must be supplied together")
    feedback: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if all(supplied_feedback):
        minute_feedback = np.asarray(minute_feedback_features, dtype=np.float32)
        hour_feedback = np.asarray(hour_feedback_features, dtype=np.float32)
        day_feedback = np.asarray(day_feedback_features, dtype=np.float32)
        if (minute_feedback.ndim != 3 or minute_feedback.shape[0] < 1
                or minute_feedback.shape[1:] != (len(future_hour_events), 5)):
            raise ValueError("minute feedback must be [paths,horizon,5]")
        batch_size = minute_feedback.shape[0]
        expected_feedback_shape = (batch_size, len(future_hour_events), 5)
        if (hour_feedback.shape != expected_feedback_shape
                or day_feedback.shape != expected_feedback_shape
                or any(not np.isfinite(value).all() for value in
                       (minute_feedback, hour_feedback, day_feedback))):
            raise ValueError("all feedback features must be finite and path/horizon aligned")
        hour_event_mask = np.asarray(future_hour_events, dtype=np.bool_)
        day_event_mask = np.asarray(future_day_events, dtype=np.bool_)
        if (np.any(hour_feedback[:, ~hour_event_mask] != 0)
                or np.any(day_feedback[:, ~day_event_mask] != 0)):
            raise ValueError("slow feedback features must be zero before their availability event")
        feedback = minute_feedback, hour_feedback, day_feedback
    else:
        batch_size = tensors.minute_x.shape[0]
    history_batch_size = tensors.minute_x.shape[0]
    if history_batch_size not in (1, batch_size):
        raise ValueError("observed history batch cannot align with feedback paths")

    def batch_history(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if array.shape[0] == batch_size:
            return array
        if array.shape[0] == 1:
            return np.broadcast_to(array, (batch_size, *array.shape[1:]))
        raise ValueError("observed history batch cannot align with forecast batch")

    minute_x = batch_history(tensors.minute_x)
    hour_x = batch_history(tensors.hour_x)
    day_x = batch_history(tensors.day_x)
    hour_index = batch_history(tensors.hour_index)
    day_index = batch_history(tensors.day_index)
    minute_mask = batch_history(tensors.minute_mask)
    hour_mask = batch_history(tensors.hour_mask)
    day_mask = batch_history(tensors.day_mask)
    model = make_hierarchical_model(checkout, minute=model_config.minute,
                                    hour=model_config.hour, day=model_config.day)
    minute_reset = np.zeros(minute_x.shape[:2], dtype=np.bool_)
    minute_reset[:, 0] = True
    template_args = (jnp.asarray(minute_x), jnp.asarray(hour_x),
        jnp.asarray(day_x), jnp.asarray(hour_index),
        jnp.asarray(day_index), jnp.asarray(minute_mask),
        jnp.asarray(hour_mask), jnp.asarray(day_mask),
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
    rollout_kwargs: dict[str, object] = {
        "advance_slow_priors": advance_slow_priors, "seed": forecast_seed,
    }
    if feedback is not None:
        rollout_kwargs.update({"minute_feedback": jnp.asarray(feedback[0]),
                               "hour_feedback": jnp.asarray(feedback[1]),
                               "day_feedback": jnp.asarray(feedback[2])})
    _, result = nj.pure(model.rollout_f2)(params,
        jnp.asarray(minute_x), jnp.asarray(hour_x), jnp.asarray(day_x),
        jnp.asarray(hour_index), jnp.asarray(day_index),
        future_hour_events, future_day_events, **rollout_kwargs)
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


class StatefulHierarchicalF2Predictor:
    """Checkpoint-loaded, cached B3 predictor for recursive OHLCV generation.

    The observed hierarchy is encoded once. Each call consumes the previously
    sampled minute feature and any slow aggregate available at that decision,
    then emits the next conditional head without replaying the context prefix.
    """

    def __init__(self, *, model, parameters, state, jax, nj,
                 advance_slow_priors: bool, paths: int, forecast_seed: int) -> None:
        self._model = model
        self._parameters = parameters
        self._state = state
        self._jax = jax
        self._nj = nj
        self._advance_slow_priors = advance_slow_priors
        self._paths = paths
        self._forecast_seed = forecast_seed
        self._next_step = 0
        self._compiled: dict[tuple[bool, bool, bool], object] = {}

    def _step_function(self, *, has_feedback: bool, hour_event: bool,
                       day_event: bool):
        key = has_feedback, hour_event, day_event
        function = self._compiled.get(key)
        if function is None:
            def apply(parameters, state, minute_feedback, hour_feedback,
                      day_feedback, seed):
                return self._nj.pure(self._model.rollout_f2_step)(
                    parameters, state, minute_feedback, hour_feedback,
                    day_feedback, has_feedback=has_feedback,
                    hour_event=hour_event, day_event=day_event,
                    advance_slow_priors=self._advance_slow_priors,
                    seed=seed)
            function = self._jax.jit(apply)
            self._compiled[key] = function
        return function

    def predict_step(self, *, step: int, minute_feedback: np.ndarray,
                     hour_feedback: np.ndarray, day_feedback: np.ndarray,
                     hour_event: bool, day_event: bool
                     ) -> tuple[np.ndarray, np.ndarray]:
        if (type(step) is not int or step != self._next_step
                or type(hour_event) is not bool or type(day_event) is not bool):
            raise ValueError("stateful B3 predictions must follow the ordered event plan")
        minute = np.asarray(minute_feedback, dtype=np.float32)
        hour = np.asarray(hour_feedback, dtype=np.float32)
        day = np.asarray(day_feedback, dtype=np.float32)
        if (minute.shape != (self._paths, 5) or hour.shape != minute.shape
                or day.shape != minute.shape
                or any(not np.isfinite(value).all() for value in (minute, hour, day))):
            raise ValueError("stateful B3 feedback must be finite [paths,5]")
        if step == 0:
            if (hour_event or day_event or np.any(minute) or np.any(hour) or np.any(day)):
                raise ValueError("the initial B3 forecast cannot contain generated feedback")
        elif ((not hour_event and np.any(hour)) or (not day_event and np.any(day))):
            raise ValueError("slow feedback appeared before its availability event")
        function = self._step_function(
            has_feedback=step > 0, hour_event=hour_event, day_event=day_event)
        seed = np.random.SeedSequence(
            [self._forecast_seed, step, 0xB3]).generate_state(
                1, dtype=np.uint32)[0]
        self._parameters, result = function(
            self._parameters, self._state,
            self._jax.numpy.asarray(minute), self._jax.numpy.asarray(hour),
            self._jax.numpy.asarray(day), self._jax.numpy.asarray(seed))
        self._state, output = result
        mean = np.asarray(self._jax.device_get(output["minute_mean"]), dtype=np.float64)
        scale = np.asarray(self._jax.device_get(output["minute_scale"]), dtype=np.float64)
        if (mean.shape != (self._paths, 5) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0)):
            raise RuntimeError("stateful B3 worker produced an invalid one-step head")
        self._next_step += 1
        return mean, scale


def create_stateful_hierarchical_f2_predictor(*, tensors: HierarchyTensors,
        train_scope_id: str, dataset_id: str, scaler_sha256: str,
        checkout: str | Path, checkpoint: str | Path,
        model_config: HierarchicalModelConfig,
        training_config: HierarchicalTrainingConfig,
        training_tensor_sha256: str, inference_context_sha256: str,
        paths: int, forecast_seed: int = 811, advance_slow_priors: bool = True
        ) -> StatefulHierarchicalF2Predictor:
    """Load a B3 checkpoint and encode one observable context a single time."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax

    if (not isinstance(tensors, HierarchyTensors) or type(paths) is not int
            or not 1 <= paths <= 1024 or type(forecast_seed) is not int
            or forecast_seed < 0 or type(advance_slow_priors) is not bool
            or not isinstance(train_scope_id, str) or not train_scope_id.strip()
            or not isinstance(dataset_id, str) or not dataset_id.strip()
            or not isinstance(scaler_sha256, str) or len(scaler_sha256) != 64
            or not isinstance(training_config, HierarchicalTrainingConfig)
            or not isinstance(model_config, HierarchicalModelConfig)):
        raise ValueError("invalid stateful B3 rollout identity or configuration")
    actual_context_sha256 = hierarchy_tensor_sha256(tensors)
    for name, value in (("Train tensor", training_tensor_sha256),
                        ("inference context", inference_context_sha256)):
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"invalid B3 {name} SHA-256")
    if inference_context_sha256 != actual_context_sha256:
        raise ValueError("stateful B3 context does not match its declared identity")
    if (tensors.minute_x.shape[0] != 1 or tensors.hour_x.shape[0] != 1
            or tensors.day_x.shape[0] != 1):
        raise ValueError("stateful B3 expects one observable anchor context")

    def repeated(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(value, (paths, *value.shape[1:]))

    minute_x = repeated(tensors.minute_x)
    hour_x = repeated(tensors.hour_x)
    day_x = repeated(tensors.day_x)
    hour_index = repeated(tensors.hour_index)
    day_index = repeated(tensors.day_index)
    minute_mask = repeated(tensors.minute_mask)
    hour_mask = repeated(tensors.hour_mask)
    day_mask = repeated(tensors.day_mask)
    resets = np.zeros(minute_x.shape[:2], dtype=np.bool_)
    resets[:, 0] = True
    model = make_hierarchical_model(checkout, minute=model_config.minute,
        hour=model_config.hour, day=model_config.day)
    template_args = tuple(jnp.asarray(value) for value in (
        minute_x, hour_x, day_x, hour_index, day_index,
        minute_mask, hour_mask, day_mask, resets))
    parameters = nj.init(model)({}, *template_args, seed=training_config.seed)
    optimizer = optax.chain(optax.clip_by_global_norm(training_config.grad_clip),
                            optax.adam(training_config.lr))
    identity = {"train_scope_id": train_scope_id, "dataset_id": dataset_id,
        "snapshot_sha": SNAPSHOT_SHA, "upstream_sha": PINNED_UPSTREAM,
        "hierarchy_tensor_sha256": training_tensor_sha256,
        "scaler_sha256": scaler_sha256}
    parameters, _, _, checkpoint_seed = load_checkpoint(
        checkpoint, params_template=parameters,
        opt_template=optimizer.init(parameters), expected_identity=identity,
        expected_config=optimizer_recipe(model_config, training_config))
    if checkpoint_seed != training_config.seed:
        raise ValueError("stateful B3 checkpoint seed differs from training configuration")
    parameters, state = nj.pure(model.initialize_rollout_f2)(
        parameters, jnp.asarray(minute_x), jnp.asarray(hour_x), jnp.asarray(day_x),
        jnp.asarray(hour_index), jnp.asarray(day_index), seed=forecast_seed)
    return StatefulHierarchicalF2Predictor(
        model=model, parameters=parameters, state=state, jax=jax, nj=nj,
        advance_slow_priors=advance_slow_priors, paths=paths,
        forecast_seed=forecast_seed)


def generate_recursive_hierarchical_f2_paths(*, tensors: HierarchyTensors,
        train_scope_id: str, dataset_id: str, scaler_sha256: str,
        checkout: str | Path, checkpoint: str | Path,
        model_config: HierarchicalModelConfig,
        training_config: HierarchicalTrainingConfig,
        training_tensor_sha256: str, inference_context_sha256: str,
        future_minute_decisions_ns: np.ndarray,
        hour_availability_delay_ns: int, day_availability_delay_ns: int,
        minute_scaler: TrainOnlyStandardizer,
        hour_scaler: TrainOnlyStandardizer,
        day_scaler: TrainOnlyStandardizer,
        observed_prefix: tuple[Candle, ...], paths: int, forecast_seed: int = 811,
        advance_slow_priors: bool = True,
        max_attempts_per_bar: int = 20) -> RecursiveHierarchicalOHLCVPaths:
    """Generate legal F2 OHLCV paths autoregressively with path-derived feedback.

    Each minute is forecast, sampled and decoded before the next conditional
    forecast. Completed UTC hour/day candles from each path are encoded with
    the frozen Train scalers and become visible only at their scheduled event.
    The checkpoint and observable context are loaded once. The context is
    encoded into reusable RSSM carries, then each sampled minute and any
    newly-available aggregate advances those carries exactly once.
    """
    decisions = np.asarray(future_minute_decisions_ns)
    if (type(forecast_seed) is not int or forecast_seed < 0
            or type(advance_slow_priors) is not bool):
        raise ValueError("invalid recursive F2 seed or E3/E4 mode")
    if (type(hour_availability_delay_ns) is not int or hour_availability_delay_ns < 0
            or type(day_availability_delay_ns) is not int or day_availability_delay_ns < 0):
        raise ValueError("recursive F2 availability delays must be nonnegative integers")
    hour_times = np.asarray(MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decisions, timeframe_minutes=60,
        availability_delay_ns=hour_availability_delay_ns,
        observed_close_through_ns=int(tensors.hour_close_at[-1])), dtype=np.int64)
    day_times = np.asarray(MultiScaleClock.future_availability_schedule(
        minute_decisions_ns=decisions, timeframe_minutes=1440,
        availability_delay_ns=day_availability_delay_ns,
        observed_close_through_ns=int(tensors.day_close_at[-1])), dtype=np.int64)
    hour_events, day_events = derive_future_scale_events(
        minute_decisions_ns=decisions,
        hour_available_at_ns=hour_times,
        day_available_at_ns=day_times)

    predictor = create_stateful_hierarchical_f2_predictor(
        tensors=tensors, train_scope_id=train_scope_id, dataset_id=dataset_id,
        scaler_sha256=scaler_sha256, checkout=checkout, checkpoint=checkpoint,
        model_config=model_config, training_config=training_config,
        training_tensor_sha256=training_tensor_sha256,
        inference_context_sha256=inference_context_sha256, paths=paths,
        forecast_seed=forecast_seed,
        advance_slow_priors=advance_slow_priors)

    return generate_recursive_hierarchical_ohlcv_paths(
        predictor=predictor, future_minute_decisions_ns=decisions,
        future_hour_events=hour_events, future_day_events=day_events,
        hour_availability_delay_ns=hour_availability_delay_ns,
        day_availability_delay_ns=day_availability_delay_ns,
        minute_scaler=minute_scaler, hour_scaler=hour_scaler,
        day_scaler=day_scaler, observed_prefix=observed_prefix,
        paths=paths, seed=forecast_seed,
        max_attempts_per_bar=max_attempts_per_bar)
