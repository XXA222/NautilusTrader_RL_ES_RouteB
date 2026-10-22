"""Train-only JAX optimizer for the experimental RouteB three-scale RSSM.

This first trainer deliberately consumes one bounded, prebuilt Train tensor
sequence. It is an engineering path for deterministic optimization and resume;
it does not authorize Validation/Test access or claim R3 predictive value.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from .jax_hierarchy import make_hierarchical_model
from .jax_model import PINNED_UPSTREAM, RSSMConfig
from .checkpoint import load_checkpoint, save_checkpoint
from .hierarchy_dataset import HierarchyTensors
from .market import MINUTE_NS
from .randomness import (SeedStreams, resolve_training_seed_streams,
                         seed_stream_identity, validate_training_seed,
                         validate_jax_seed)


@dataclass(frozen=True)
class HierarchyTrainingConfig:
    seed: int = 31
    steps: int = 100
    hierarchy_variant: str = "e3b_fusion"
    window_length: int = 1024
    burn_in: int = 32
    hour_context_steps: int = 48
    day_context_steps: int = 32
    learning_rate: float = 3e-4
    grad_clip: float = 100.0
    report_interval: int = 25
    risk_auxiliary_scale: float = 0.0
    risk_min_scale: float | None = None
    risk_horizons: tuple[int, ...] = (1, 5, 15, 30, 60)
    risk_horizon_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    risk_barrier_quantile: float = 0.75

    def __post_init__(self) -> None:
        import math
        try:
            validate_training_seed(self.seed, self.steps)
        except ValueError as exc:
            raise ValueError("invalid hierarchical training seed/update range") from exc
        if (type(self.steps) is not int or self.steps < 1
                or type(self.report_interval) is not int or self.report_interval < 1
                or not isinstance(self.hierarchy_variant, str)
                or self.hierarchy_variant not in {"e3_features", "e3b_fusion", "e4_conditional"}
                or type(self.window_length) is not int or self.window_length < 16
                or type(self.burn_in) is not int or self.burn_in < 0
                or self.burn_in >= self.window_length
                or type(self.hour_context_steps) is not int
                or self.hour_context_steps < (self.window_length + 59) // 60 + 2
                or type(self.day_context_steps) is not int
                or self.day_context_steps < (self.window_length + 1439) // 1440 + 2
                or isinstance(self.learning_rate, bool)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or isinstance(self.grad_clip, bool)
                or not math.isfinite(self.grad_clip) or self.grad_clip <= 0
                or isinstance(self.risk_auxiliary_scale, bool)
                or not math.isfinite(self.risk_auxiliary_scale)
                or self.risk_auxiliary_scale < 0
                or (self.risk_min_scale is not None and (
                    isinstance(self.risk_min_scale, bool)
                    or not math.isfinite(self.risk_min_scale)
                    or self.risk_min_scale <= 0))
                or isinstance(self.risk_barrier_quantile, bool)
                or not math.isfinite(self.risk_barrier_quantile)
                or not 0.5 < self.risk_barrier_quantile < 1.0):
            raise ValueError("invalid hierarchical training configuration")
        horizons = tuple(self.risk_horizons)
        weights = tuple(self.risk_horizon_weights)
        object.__setattr__(self, "risk_horizons", horizons)
        object.__setattr__(self, "risk_horizon_weights", weights)
        if (not horizons or tuple(sorted(set(horizons))) != horizons
                or any(type(value) is not int or value < 1 for value in horizons)
                or len(weights) != len(horizons)
                or any(isinstance(value, bool) or not math.isfinite(value) or value <= 0
                       for value in weights)):
            raise ValueError("invalid hierarchical risk horizon configuration")


def _validated_arrays(tensors: HierarchyTensors) -> dict[str, np.ndarray]:
    if tensors.scope not in {"train", "validation"}:
        raise ValueError("hierarchy tensors must declare Train or Validation scope")
    hour_available = np.asarray(tensors.hour_available_at, dtype=np.int64)
    day_available = np.asarray(tensors.day_available_at, dtype=np.int64)
    hour_day_index = (np.searchsorted(day_available, hour_available, side="right") - 1
                      if tensors.hour_day_index is None else
                      np.asarray(tensors.hour_day_index, dtype=np.int32).reshape(-1))
    arrays = {
        "minute_x": np.asarray(tensors.minute_x, dtype=np.float32),
        "hour_x": np.asarray(tensors.hour_x, dtype=np.float32),
        "day_x": np.asarray(tensors.day_x, dtype=np.float32),
        "hour_index": np.asarray(tensors.hour_index, dtype=np.int32),
        "day_index": np.asarray(tensors.day_index, dtype=np.int32),
        "hour_day_index": np.asarray(hour_day_index, dtype=np.int32)[None],
        "minute_mask": np.asarray(tensors.minute_mask, dtype=np.float32),
        "hour_mask": np.asarray(tensors.hour_mask, dtype=np.float32),
        "day_mask": np.asarray(tensors.day_mask, dtype=np.float32),
        "minute_available_at": np.asarray(tensors.minute_available_at, dtype=np.int64),
        "hour_available_at": hour_available,
        "day_available_at": day_available,
        "risk_targets": (np.zeros((1, np.asarray(tensors.minute_x).shape[1],
                                   len(tensors.risk_horizons), 6),
                                   dtype=np.float32) if tensors.risk_targets is None else
                         np.asarray(tensors.risk_targets, dtype=np.float32)),
        "risk_mask": (np.zeros((1, np.asarray(tensors.minute_x).shape[1],
                                len(tensors.risk_horizons)),
                                dtype=np.float32) if tensors.risk_masks is None else
                      np.asarray(tensors.risk_masks, dtype=np.float32)),
    }
    minute, hour, day = (arrays[k] for k in ("minute_x", "hour_x", "day_x"))
    B, T = minute.shape[:2] if minute.ndim == 3 else (0, 0)
    if (B != 1 or minute.shape[2:] != (5,) or hour.ndim != 3 or day.ndim != 3
            or hour.shape[0] != B or day.shape[0] != B
            or hour.shape[1] < 2 or day.shape[1] < 2
            or hour.shape[2:] != (5,) or day.shape[2:] != (5,)):
        raise ValueError("hierarchy training requires [1,T,5], [1,H,5], [1,D,5] tensors")
    for key, shape in (("hour_index", (B, T)), ("day_index", (B, T)),
                       ("minute_mask", (B, T)),
                       ("hour_mask", hour.shape[:2]), ("day_mask", day.shape[:2]),
                       ("minute_available_at", (T,)),
                       ("hour_day_index", (B, hour.shape[1])),
                       ("hour_available_at", (hour.shape[1],)),
                       ("day_available_at", (day.shape[1],))):
        if arrays[key].shape != shape:
            raise ValueError(f"hierarchy tensor {key} must have shape {shape}")
    for key in ("minute_x", "hour_x", "day_x", "minute_mask", "hour_mask", "day_mask"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"hierarchy tensor {key} contains nonfinite values")
    for key in ("minute_mask", "hour_mask", "day_mask"):
        if np.any((arrays[key] < 0) | (arrays[key] > 1)):
            raise ValueError(f"hierarchy mask {key} must be in [0,1]")
    if (np.any(np.diff(arrays["minute_available_at"]) <= 0)
            or np.any(np.diff(arrays["hour_available_at"]) <= 0)
            or np.any(np.diff(arrays["day_available_at"]) <= 0)):
        raise ValueError("hierarchy availability clocks must be strictly increasing")
    for key, clock in (("hour_index", "hour_available_at"),
                       ("day_index", "day_available_at")):
        indices = arrays[key][0]
        if (np.any(indices < -1) or np.any(indices >= len(arrays[clock]))
                or np.any(np.diff(indices) < 0)):
            raise ValueError(f"hierarchy tensor {key} violates bounded monotone indexing")
        selected = indices >= 0
        if selected.any() and np.any(arrays[clock][indices[selected]] >
                                     arrays["minute_available_at"][selected]):
            raise ValueError(f"hierarchy tensor {key} leaks a future slow state")
    hour_day = arrays["hour_day_index"][0]
    if (np.any(hour_day < -1) or np.any(hour_day >= len(arrays["day_available_at"]))
            or np.any(np.diff(hour_day) < 0)):
        raise ValueError("hierarchy tensor hour_day_index violates bounded monotone indexing")
    visible_day = hour_day >= 0
    if visible_day.any() and np.any(
            arrays["day_available_at"][hour_day[visible_day]] >
            arrays["hour_available_at"][visible_day]):
        raise ValueError("hour-scale transition leaks a future daily state")
    if not np.any(arrays["minute_mask"] > 0):
        raise ValueError("hierarchy Train sequence has no scored minute targets")
    if (arrays["risk_targets"].shape != (B, T, len(tensors.risk_horizons), 6)
            or arrays["risk_mask"].shape != (B, T, len(tensors.risk_horizons))
            or not np.isfinite(arrays["risk_targets"]).all()
            or not np.isfinite(arrays["risk_mask"]).all()
            or np.any((arrays["risk_mask"] < 0) | (arrays["risk_mask"] > 1))):
        raise ValueError("hierarchy multi-horizon risk tensors have invalid shape or values")
    if tensors.risk_barrier_fractions is not None and (
            len(tensors.risk_barrier_fractions) != len(tensors.risk_horizons)
            or any(not np.isfinite(value) or not 0 < value < 1
                   for value in tensors.risk_barrier_fractions)):
        raise ValueError("hierarchy Train risk barriers are invalid")
    return arrays


def _tensor_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _hierarchy_source_identity(tensors: HierarchyTensors,
                              dataset_id: str) -> dict[str, object]:
    if tensors.scope != "train":
        raise ValueError("checkpoint identity must be derived from Train tensors only")
    if tensors.dataset_id is not None and tensors.dataset_id != dataset_id:
        raise ValueError("hierarchy tensor dataset identity differs from training metadata")
    fingerprints = tensors.scaler_fingerprints
    if fingerprints is not None:
        if (not isinstance(fingerprints, tuple) or len(fingerprints) != 3
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in fingerprints)):
            raise ValueError("hierarchy tensor scaler fingerprints are invalid")
    return {
        "hierarchy_dataset_id": tensors.dataset_id,
        "scaler_fingerprints": list(fingerprints) if fingerprints is not None else None,
        "risk_horizons": list(tensors.risk_horizons),
        "risk_barrier_quantile": tensors.risk_barrier_quantile,
        "risk_barrier_fractions": (
            list(tensors.risk_barrier_fractions)
            if tensors.risk_barrier_fractions is not None else None),
    }


def _window_arrays(arrays: dict[str, np.ndarray], *, start: int,
                   config: HierarchyTrainingConfig) -> dict[str, np.ndarray]:
    """Crop one fixed-shape minute window and its causally visible slow context."""
    end = start + config.window_length
    if start < 0 or end > arrays["minute_x"].shape[1]:
        raise ValueError("hierarchy training window is outside the Train sequence")
    sampled = dict(arrays)
    sampled["minute_x"] = arrays["minute_x"][:, start:end]
    sampled["minute_mask"] = arrays["minute_mask"][:, start:end].copy()
    sampled["minute_mask"][:, :config.burn_in] = 0.0
    sampled["risk_targets"] = arrays["risk_targets"][:, start:end]
    sampled["risk_mask"] = arrays["risk_mask"][:, start:end].copy()
    sampled["risk_mask"][:, :config.burn_in] = 0.0
    sampled["minute_available_at"] = arrays["minute_available_at"][start:end]
    context_bounds: dict[str, tuple[int, int]] = {}
    for scale in ("hour", "day"):
        index_key = f"{scale}_index"
        clock_key = f"{scale}_available_at"
        mask_key = f"{scale}_mask"
        context_size = getattr(config, f"{scale}_context_steps")
        visible = arrays[index_key][0, start:end]
        referenced = visible[visible >= 0]
        if referenced.size:
            last_visible = int(referenced[-1])
            first_context = max(0, last_visible - context_size + 1)
            stop_context = last_visible + 1
        else:
            # Keep a valid two-token native-scale input while no slow candle
            # is available. Both entries remain unreferenced and loss-masked.
            first_context = 0
            stop_context = min(2, arrays[f"{scale}_x"].shape[1])
        context_bounds[scale] = (first_context, stop_context)
        source = arrays[f"{scale}_x"][:, first_context:stop_context]
        source_mask = arrays[mask_key][:, first_context:stop_context]
        valid_count = source.shape[1]
        if valid_count < 2:
            source = np.pad(source, ((0, 0), (0, 2 - valid_count), (0, 0)))
            source_mask = np.pad(source_mask, ((0, 0), (0, 2 - valid_count)))
            valid_count = 2
        if valid_count > context_size:
            raise ValueError(f"{scale} context does not cover the minute window span")
        if valid_count < context_size:
            pad = context_size - valid_count
            source = np.pad(source, ((0, 0), (0, pad), (0, 0)))
            source_mask = np.pad(source_mask, ((0, 0), (0, pad)))
        index = visible.copy()
        selected = index >= 0
        index[selected] -= first_context
        valid_count = stop_context - first_context
        if selected.any() and np.any(index[selected] >= valid_count):
            raise ValueError(f"{scale} context crop omitted a referenced slow state")
        if selected.any() and np.any(
                arrays[clock_key][first_context:stop_context][index[selected]] >
                sampled["minute_available_at"][selected]):
            raise ValueError(f"{scale} cropped window exposes a future state")
        sampled[f"{scale}_x"] = source.astype(np.float32, copy=False)
        sampled[mask_key] = source_mask.astype(np.float32, copy=False)
        sampled[index_key] = index[None].astype(np.int32, copy=False)
        clock = arrays[clock_key][first_context:stop_context]
        if len(clock) < context_size:
            clock = np.pad(clock, (0, context_size - len(clock)))
        sampled[clock_key] = clock.astype(np.int64, copy=False)
    hour_start, hour_stop = context_bounds["hour"]
    day_start, day_stop = context_bounds["day"]
    hour_day = arrays["hour_day_index"][0, hour_start:hour_stop].copy()
    selected_hour_day = hour_day >= 0
    hour_day[selected_hour_day] -= day_start
    if selected_hour_day.any() and np.any(hour_day[selected_hour_day] >= day_stop - day_start):
        raise ValueError("daily context crop omitted a referenced hour-level condition")
    if len(hour_day) < config.hour_context_steps:
        hour_day = np.pad(hour_day,
                          (0, config.hour_context_steps - len(hour_day)),
                          constant_values=-1)
    sampled["hour_day_index"] = hour_day[None].astype(np.int32, copy=False)
    return sampled


def _as_model_inputs(arrays: dict[str, np.ndarray]):
    import jax.numpy as jnp
    return [jnp.asarray(arrays[name]) for name in (
        "minute_x", "hour_x", "day_x", "hour_index", "day_index",
        "hour_day_index", "minute_mask", "hour_mask", "day_mask",
        "risk_targets", "risk_mask")]


def _validate_risk_objective(tensors: HierarchyTensors,
                             arrays: dict[str, np.ndarray],
                             config: HierarchyTrainingConfig) -> None:
    if tuple(tensors.risk_horizons) != config.risk_horizons:
        raise ValueError("hierarchy risk horizons differ from the frozen training objective")
    if (config.risk_auxiliary_scale > 0
            and tensors.risk_barrier_quantile != config.risk_barrier_quantile):
        raise ValueError("hierarchy risk barrier quantile differs from the frozen training objective")
    if config.risk_auxiliary_scale > 0 and (
            tensors.scope != "train" or tensors.risk_targets is None
            or tensors.risk_masks is None or tensors.risk_barrier_fractions is None
            or not np.any(arrays["risk_mask"] > 0)):
        raise ValueError("risk auxiliary training requires Train-only targets and fitted barriers")


def _require_static_context_horizon(anchor_available_ns: int, steps: int) -> None:
    """Fail closed if a static slow-state rollout reaches a UTC slow close."""
    anchor_close = (anchor_available_ns // MINUTE_NS) * MINUTE_NS
    final_close = anchor_close + steps * MINUTE_NS
    next_hour_close = ((anchor_close // (60 * MINUTE_NS)) + 1) * 60 * MINUTE_NS
    next_day_close = ((anchor_close // (1440 * MINUTE_NS)) + 1) * 1440 * MINUTE_NS
    if next_hour_close <= final_close or next_day_close <= final_close:
        raise ValueError("static-context forecast crosses a future hour/day event")


def train_hierarchical(*, tensors: HierarchyTensors, train_scope_id: str,
        dataset_id: str, checkout: str | Path, output_checkpoint: str | Path,
        minute_config: RSSMConfig | None = None,
        hour_config: RSSMConfig | None = None, day_config: RSSMConfig | None = None,
        training_config: HierarchyTrainingConfig | None = None,
        resume: bool = False,
        seed_streams: SeedStreams | None = None) -> list[dict]:
    """Optimize causal Train windows with deterministic checkpoint resume.

    The selected E3/E3b/E4 objective is trained on fixed causal windows.
    E3 uses observed slow features; E3b/E4 train independent slow RSSMs.
    Per-step window seeds make resumed and uninterrupted optimizer
    trajectories identical. Registered research runners remain separate work.
    """
    import jax
    import ninjax as nj
    import optax  # type: ignore[import-not-found]

    if not isinstance(train_scope_id, str) or not train_scope_id.strip():
        raise ValueError("an explicit frozen Train scope identity is required")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("an explicit source dataset identity is required")
    minute_config = minute_config or RSSMConfig()
    hour_config = hour_config or RSSMConfig(deter=128, hidden=96, stoch=8, classes=8, blocks=8)
    day_config = day_config or RSSMConfig(deter=96, hidden=64, stoch=8, classes=8, blocks=8)
    cfg = training_config or HierarchyTrainingConfig()
    streams = resolve_training_seed_streams(cfg.seed, cfg.steps, seed_streams)
    if tensors.scope != "train":
        raise ValueError("hierarchical optimization accepts Train tensors only")
    arrays = _validated_arrays(tensors)
    _validate_risk_objective(tensors, arrays, cfg)
    if arrays["minute_x"].shape[1] < cfg.window_length:
        raise ValueError("Train sequence is shorter than the configured minute window")
    day_visible = arrays["day_index"][0]
    min_start = next((i for i in range(len(day_visible) - cfg.window_length + 1)
                      if day_visible[i + cfg.window_length - 1] >= 1), None)
    if min_start is None:
        raise ValueError("Train sequence has no window with two observable daily states")
    score_starts = np.flatnonzero(arrays["minute_mask"][0] > 0)
    if not score_starts.size:
        raise ValueError("Train sequence has no scored minute targets")
    min_start = max(min_start, int(score_starts[0]))
    max_start = arrays["minute_x"].shape[1] - cfg.window_length
    if min_start > max_start:
        raise ValueError("no eligible three-scale training windows")
    eligible_start_count = max_start - min_start + 1
    digest = _tensor_digest(arrays)
    model = make_hierarchical_model(checkout, minute=minute_config,
                                    hour=hour_config, day=day_config,
                                    variant=cfg.hierarchy_variant,
                                    risk_horizons=cfg.risk_horizons,
                                    risk_horizon_weights=cfg.risk_horizon_weights,
                                    risk_auxiliary_scale=cfg.risk_auxiliary_scale,
                                    risk_min_scale=cfg.risk_min_scale)
    first_window = _window_arrays(arrays, start=min_start, config=cfg)
    inputs = _as_model_inputs(first_window)
    params = nj.init(model)({}, *inputs, seed=streams.initialization_seed)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                            optax.adam(cfg.learning_rate))
    opt_state = optimizer.init(params)
    identity = {"train_scope_id": train_scope_id, "dataset_id": dataset_id,
                "tensor_sha256": digest, "upstream_sha": PINNED_UPSTREAM,
                **_hierarchy_source_identity(tensors, dataset_id)}
    model_recipe = {"minute": asdict(minute_config), "hour": asdict(hour_config),
                    "day": asdict(day_config)}
    train_recipe = {"learning_rate": cfg.learning_rate, "grad_clip": cfg.grad_clip,
                    "seed": cfg.seed, "window_length": cfg.window_length,
                    "burn_in": cfg.burn_in,
                    "hour_context_steps": cfg.hour_context_steps,
                    "day_context_steps": cfg.day_context_steps,
                    "hierarchy_variant": cfg.hierarchy_variant,
                    "risk_auxiliary_scale": cfg.risk_auxiliary_scale,
                    "risk_min_scale": cfg.risk_min_scale,
                    "risk_horizons": list(cfg.risk_horizons),
                    "risk_horizon_weights": list(cfg.risk_horizon_weights),
                    "risk_barrier_quantile": cfg.risk_barrier_quantile}
    configuration = json.loads(json.dumps(
        {"model": model_recipe, "train": train_recipe,
         "trainer_schema": "routeb.hierarchical-trainer.v2"},
        sort_keys=True, allow_nan=False))
    if seed_streams is not None:
        configuration["seed_streams"] = seed_stream_identity(streams)
    start = 0
    if resume:
        params, opt_state, start, saved_seed = load_checkpoint(
            output_checkpoint, params_template=params, opt_template=opt_state,
            expected_identity=identity, expected_config=configuration)
        if saved_seed != cfg.seed:
            raise ValueError("hierarchy checkpoint seed mismatch")
    if start >= cfg.steps:
        raise ValueError("resume checkpoint already reached the requested update count")

    @jax.jit
    def update(parameters, state, batch_inputs, rng_seed):
        def loss_fn(candidate):
            _, diagnostics = nj.pure(model)(candidate, *batch_inputs, seed=rng_seed)
            return diagnostics["loss"], diagnostics
        (loss, diagnostics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(parameters)
        updates, new_state = optimizer.update(gradients, state, parameters)
        new_parameters = optax.apply_updates(parameters, updates)
        return new_parameters, new_state, {
            **diagnostics, "loss": loss, "grad_norm": optax.global_norm(gradients)}

    reports: list[dict] = []
    sampled_starts: list[int] = []
    started = time.perf_counter()
    for step in range(start, cfg.steps):
        sampler_seed = streams.sampler_seed + step
        selected_start = int(np.random.default_rng(sampler_seed).integers(
            min_start, max_start + 1))
        sampled_starts.append(selected_start)
        window = _window_arrays(arrays, start=selected_start, config=cfg)
        window_inputs = _as_model_inputs(window)
        params, opt_state, metrics = update(
            params, opt_state, window_inputs, streams.model_seed + step)
        scalars = {key: float(np.asarray(value)) for key, value in metrics.items()}
        if not all(np.isfinite(value) for value in scalars.values()):
            raise RuntimeError("nonfinite hierarchical training metric; failed closed")
        if (step + 1) % cfg.report_interval == 0 or step + 1 == cfg.steps:
            report = {"step": step + 1, **scalars,
                      "model_rng_seed": streams.model_seed + step,
                      "sampler_seed": sampler_seed,
                      "window_start": selected_start,
                      "eligible_window_count": eligible_start_count,
                      "window_start_sha256": hashlib.sha256(
                          np.asarray([selected_start], dtype="<i8").tobytes()).hexdigest()}
            reports.append(report)
            save_checkpoint(output_checkpoint, params=params, opt_state=opt_state,
                            step=step + 1, seed=cfg.seed, identity=identity,
                            config=configuration)
    report = reports[-1]
    report.update({"optimizer_updates": cfg.steps,
                   "initialization_seed": streams.initialization_seed,
                   "sampler_seed_base": streams.sampler_seed,
                   "model_seed_base": streams.model_seed,
                   "parameters": sum(int(np.prod(leaf.shape)) for leaf in
                                     jax.tree_util.tree_leaves(params)),
                   "train_tensor_sha256": digest,
                   "sampled_unique_window_starts": len(set(sampled_starts)),
                   "sampled_window_starts_sha256": hashlib.sha256(
                       np.asarray(sampled_starts, dtype="<i8").tobytes()).hexdigest(),
                   "nonoverlapping_score_window_capacity": int(
                       np.count_nonzero(arrays["minute_mask"]) // cfg.window_length),
                   "training_seconds": time.perf_counter() - started,
                   "device": str(jax.devices()[0])})
    return reports


def forecast_hierarchical_static_context_standardized(*, tensors: HierarchyTensors,
        start: int, steps: int, train_scope_id: str, dataset_id: str,
        checkout: str | Path, checkpoint: str | Path,
        minute_config: RSSMConfig | None = None,
        hour_config: RSSMConfig | None = None, day_config: RSSMConfig | None = None,
        training_config: HierarchyTrainingConfig | None = None,
        forecast_seed: int = 701) -> dict[str, np.ndarray]:
    """Sample prior OHLCV-coordinate paths while holding current H/D states fixed.

    This is a bounded F1 diagnostic, not F2. The caller must use it only when
    no H/D event becomes available over ``steps``; event-driven rollouts must
    use a future implementation that synthesizes and filters slow bars.
    """
    import jax
    import ninjax as nj
    import optax  # type: ignore[import-not-found]

    validate_jax_seed(forecast_seed, name="hierarchical forecast seed")
    if type(steps) is not int or steps < 1 or steps > 60:
        raise ValueError("hierarchical forecast steps must be in [1,60]")
    if not isinstance(train_scope_id, str) or not train_scope_id.strip():
        raise ValueError("an explicit frozen Train scope identity is required")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("an explicit source dataset identity is required")
    if tensors.scope != "train":
        raise ValueError("static-context Train forecast requires Train tensors")
    minute_config = minute_config or RSSMConfig()
    hour_config = hour_config or RSSMConfig(deter=128, hidden=96, stoch=8, classes=8, blocks=8)
    day_config = day_config or RSSMConfig(deter=96, hidden=64, stoch=8, classes=8, blocks=8)
    cfg = training_config or HierarchyTrainingConfig()
    arrays = _validated_arrays(tensors)
    _validate_risk_objective(tensors, arrays, cfg)
    digest = _tensor_digest(arrays)
    if start < 0 or start + cfg.window_length > arrays["minute_x"].shape[1]:
        raise ValueError("static-context forecast window is outside the frozen Train tensors")
    anchor_available = int(arrays["minute_available_at"][start + cfg.window_length - 1])
    _require_static_context_horizon(anchor_available, steps)
    window = _window_arrays(arrays, start=start, config=cfg)
    inputs = _as_model_inputs(window)
    model = make_hierarchical_model(checkout, minute=minute_config,
                                    hour=hour_config, day=day_config,
                                    variant=cfg.hierarchy_variant,
                                    risk_horizons=cfg.risk_horizons,
                                    risk_horizon_weights=cfg.risk_horizon_weights,
                                    risk_auxiliary_scale=cfg.risk_auxiliary_scale,
                                    risk_min_scale=cfg.risk_min_scale)
    params_template = nj.init(model)({}, *inputs, seed=cfg.seed)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                            optax.adam(cfg.learning_rate))
    opt_template = optimizer.init(params_template)
    identity = {"train_scope_id": train_scope_id, "dataset_id": dataset_id,
                "tensor_sha256": digest, "upstream_sha": PINNED_UPSTREAM,
                **_hierarchy_source_identity(tensors, dataset_id)}
    configuration = json.loads(json.dumps({
        "model": {"minute": asdict(minute_config), "hour": asdict(hour_config),
                  "day": asdict(day_config)},
        "train": {"learning_rate": cfg.learning_rate, "grad_clip": cfg.grad_clip,
                  "seed": cfg.seed, "window_length": cfg.window_length,
                  "burn_in": cfg.burn_in,
                  "hour_context_steps": cfg.hour_context_steps,
                  "day_context_steps": cfg.day_context_steps,
                  "hierarchy_variant": cfg.hierarchy_variant,
                  "risk_auxiliary_scale": cfg.risk_auxiliary_scale,
                  "risk_min_scale": cfg.risk_min_scale,
                  "risk_horizons": list(cfg.risk_horizons),
                  "risk_horizon_weights": list(cfg.risk_horizon_weights),
                  "risk_barrier_quantile": cfg.risk_barrier_quantile},
        "trainer_schema": "routeb.hierarchical-trainer.v2"},
        sort_keys=True, allow_nan=False))
    params, _, _, saved_seed = load_checkpoint(
        checkpoint, params_template=params_template, opt_template=opt_template,
        expected_identity=identity, expected_config=configuration,
        allow_seed_stream_metadata=True)
    if saved_seed != cfg.seed:
        raise ValueError("hierarchy forecast seed differs from checkpoint seed")
    _, result = nj.pure(model.rollout_static_context)(
        params, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
        inputs[5], steps, seed=forecast_seed)
    host = {key: np.asarray(jax.device_get(value), dtype=np.float32)
            for key, value in result.items()}
    expected = (1, steps, 5)
    if (set(host) != {"samples", "means", "scales"}
            or any(value.shape != expected or not np.isfinite(value).all()
                   for value in host.values())
            or np.any(host["scales"] <= 0)):
        raise RuntimeError("invalid hierarchical prior forecast output")
    return host


def forecast_hierarchical_validation_static_context_standardized(*,
        train_tensors: HierarchyTensors, validation_tensors: HierarchyTensors,
        start: int, steps: int, train_scope_id: str, dataset_id: str,
        checkout: str | Path, checkpoint: str | Path,
        minute_config: RSSMConfig | None = None,
        hour_config: RSSMConfig | None = None, day_config: RSSMConfig | None = None,
        training_config: HierarchyTrainingConfig | None = None,
        forecast_seed: int = 701,
        forecast_seeds: tuple[int, ...] | None = None) -> dict[str, np.ndarray]:
    """Forecast registered Validation anchors using Train-only checkpoint parameters.

    Validation tensors supply causal context and scale indexes only. The
    checkpoint identity and parameters are loaded exclusively from the frozen
    Train tensor set, and the scored rollout must remain inside its interval.
    This remains a bounded static-context diagnostic, not F2.
    """
    if train_tensors.scope != "train" or validation_tensors.scope != "validation":
        raise ValueError("Validation forecast requires separate Train and Validation tensors")
    if (train_tensors.dataset_id != dataset_id
            or validation_tensors.dataset_id != dataset_id
            or train_tensors.scaler_fingerprints is None
            or validation_tensors.scaler_fingerprints != train_tensors.scaler_fingerprints):
        raise ValueError("Validation tensors must share the frozen Train dataset and scalers")
    try:
        validate_jax_seed(forecast_seed, name="hierarchical forecast seed")
        if forecast_seeds is not None and (
                not isinstance(forecast_seeds, tuple) or not forecast_seeds
                or any(type(seed) is not int or not 0 <= seed < (1 << 32)
                       for seed in forecast_seeds)
                or len(set(forecast_seeds)) != len(forecast_seeds)):
            raise ValueError("invalid hierarchical forecast seed batch")
    except ValueError as exc:
        raise ValueError("invalid hierarchical Validation forecast seed or horizon") from exc
    if (type(steps) is not int or steps < 1 or steps > 60):
        raise ValueError("invalid hierarchical Validation forecast seed or horizon")
    score_start, score_end = validation_tensors.score_start_ns, validation_tensors.score_end_ns
    if (type(score_start) is not int or type(score_end) is not int
            or score_start <= 0 or score_start >= score_end
            or score_start % MINUTE_NS or score_end % MINUTE_NS):
        raise ValueError("Validation tensors are missing a bounded score interval")
    cfg = training_config or HierarchyTrainingConfig()
    if (type(start) is not int or start < 0
            or start + cfg.window_length > validation_tensors.minute_x.shape[1]):
        raise ValueError("Validation forecast window is outside the approved sequence")
    arrays = _validated_arrays(validation_tensors)
    anchor_index = start + cfg.window_length - 1
    if arrays["minute_mask"][0, anchor_index] != 1.0:
        raise ValueError("forecast anchor is outside the registered Validation score window")
    anchor_available = int(arrays["minute_available_at"][anchor_index])
    anchor_close = (anchor_available // MINUTE_NS) * MINUTE_NS
    if anchor_close < score_start or anchor_close + steps * MINUTE_NS >= score_end:
        raise ValueError("Validation forecast targets extend outside the score interval")
    _require_static_context_horizon(anchor_available, steps)
    import jax
    import ninjax as nj

    params = load_hierarchical_checkpoint_parameters(
        tensors=train_tensors, train_scope_id=train_scope_id, dataset_id=dataset_id,
        checkout=checkout, checkpoint=checkpoint,
        minute_config=minute_config, hour_config=hour_config, day_config=day_config,
        training_config=cfg)
    minute_config = minute_config or RSSMConfig()
    hour_config = hour_config or RSSMConfig(deter=128, hidden=96, stoch=8, classes=8, blocks=8)
    day_config = day_config or RSSMConfig(deter=96, hidden=64, stoch=8, classes=8, blocks=8)
    window = _window_arrays(arrays, start=start, config=cfg)
    inputs = _as_model_inputs(window)
    model = make_hierarchical_model(checkout, minute=minute_config,
                                    hour=hour_config, day=day_config,
                                    variant=cfg.hierarchy_variant,
                                    risk_horizons=cfg.risk_horizons,
                                    risk_horizon_weights=cfg.risk_horizon_weights,
                                    risk_auxiliary_scale=cfg.risk_auxiliary_scale,
                                    risk_min_scale=cfg.risk_min_scale)
    rollout = nj.pure(model.rollout_static_context)
    if forecast_seeds is None:
        _, result = rollout(
            params, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            inputs[5], steps, seed=forecast_seed)
        expected: tuple[int, ...] = (1, steps, 5)
    else:
        import jax.numpy as jnp

        def sample(seed_value):
            _, sampled = rollout(
                params, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                inputs[5], steps, seed=seed_value)
            return sampled

        result = jax.vmap(sample)(jnp.asarray(forecast_seeds, dtype=jnp.uint32))
        expected = (len(forecast_seeds), 1, steps, 5)
    host = {key: np.asarray(jax.device_get(value), dtype=np.float32)
            for key, value in result.items()}
    if (set(host) != {"samples", "means", "scales"}
            or any(value.shape != expected or not np.isfinite(value).all()
                   for value in host.values())
            or np.any(host["scales"] <= 0)):
        raise RuntimeError("invalid hierarchical Validation forecast output")
    return host


def forecast_hierarchical_validation_static_context_batch_standardized(*,
        train_tensors: HierarchyTensors, validation_tensors: HierarchyTensors,
        starts: tuple[int, ...], steps: int, forecast_seeds: tuple[int, ...],
        train_scope_id: str, dataset_id: str, checkout: str | Path,
        checkpoint: str | Path, minute_config: RSSMConfig | None = None,
        hour_config: RSSMConfig | None = None, day_config: RSSMConfig | None = None,
        training_config: HierarchyTrainingConfig | None = None,
        forecast_batch_size: int = 8) -> dict[str, np.ndarray]:
    """Score many Validation anchors and paths with one Train checkpoint load.

    Output arrays have shape ``[anchor, path, step, coordinate]``. Anchor
    windows share model, parameters, configuration and fixed shapes; all
    checkpoint provenance comes from Train tensors.
    """
    import jax
    import jax.numpy as jnp
    import ninjax as nj

    if train_tensors.scope != "train" or validation_tensors.scope != "validation":
        raise ValueError("Validation forecast requires separate Train and Validation tensors")
    if (train_tensors.dataset_id != dataset_id
            or validation_tensors.dataset_id != dataset_id
            or train_tensors.scaler_fingerprints is None
            or validation_tensors.scaler_fingerprints != train_tensors.scaler_fingerprints):
        raise ValueError("Validation tensors must share the frozen Train dataset and scalers")
    if (not isinstance(starts, tuple) or not starts
            or any(type(start) is not int or start < 0 for start in starts)
            or tuple(sorted(set(starts))) != starts
            or not isinstance(forecast_seeds, tuple) or not forecast_seeds
            or any(type(seed) is not int or not 0 <= seed < (1 << 32)
                   for seed in forecast_seeds)
            or len(set(forecast_seeds)) != len(forecast_seeds)
            or type(forecast_batch_size) is not int or forecast_batch_size < 1
            or type(steps) is not int or steps < 1 or steps > 60):
        raise ValueError("invalid Validation anchor, path seed, or horizon batch")
    score_start, score_end = validation_tensors.score_start_ns, validation_tensors.score_end_ns
    if (type(score_start) is not int or type(score_end) is not int
            or score_start <= 0 or score_start >= score_end
            or score_start % MINUTE_NS or score_end % MINUTE_NS):
        raise ValueError("Validation tensors are missing a bounded score interval")
    cfg = training_config or HierarchyTrainingConfig()
    arrays = _validated_arrays(validation_tensors)
    windows = []
    for start in starts:
        if start + cfg.window_length > arrays["minute_x"].shape[1]:
            raise ValueError("Validation forecast window is outside the approved sequence")
        anchor_index = start + cfg.window_length - 1
        if arrays["minute_mask"][0, anchor_index] != 1.0:
            raise ValueError("forecast anchor is outside the registered Validation score window")
        anchor_available = int(arrays["minute_available_at"][anchor_index])
        anchor_close = (anchor_available // MINUTE_NS) * MINUTE_NS
        if anchor_close < score_start or anchor_close + steps * MINUTE_NS >= score_end:
            raise ValueError("Validation forecast targets extend outside the score interval")
        _require_static_context_horizon(anchor_available, steps)
        windows.append(_window_arrays(arrays, start=start, config=cfg))
    batched = {name: np.concatenate([window[name] for window in windows], axis=0)
               for name in ("minute_x", "hour_x", "day_x", "hour_index", "day_index",
                            "hour_day_index", "minute_mask", "hour_mask", "day_mask",
                            "risk_targets", "risk_mask")}
    inputs = _as_model_inputs(batched)
    params = load_hierarchical_checkpoint_parameters(
        tensors=train_tensors, train_scope_id=train_scope_id, dataset_id=dataset_id,
        checkout=checkout, checkpoint=checkpoint, minute_config=minute_config,
        hour_config=hour_config, day_config=day_config, training_config=cfg)
    minute_config = minute_config or RSSMConfig()
    hour_config = hour_config or RSSMConfig(deter=128, hidden=96, stoch=8, classes=8, blocks=8)
    day_config = day_config or RSSMConfig(deter=96, hidden=64, stoch=8, classes=8, blocks=8)
    model = make_hierarchical_model(checkout, minute=minute_config,
                                    hour=hour_config, day=day_config,
                                    variant=cfg.hierarchy_variant,
                                    risk_horizons=cfg.risk_horizons,
                                    risk_horizon_weights=cfg.risk_horizon_weights,
                                    risk_auxiliary_scale=cfg.risk_auxiliary_scale,
                                    risk_min_scale=cfg.risk_min_scale)
    # The model validates the horizon as a Python integer; keep that argument
    # static while forecast seeds remain vectorized JAX values.
    rollout = jax.jit(nj.pure(model.rollout_static_context), static_argnums=(7,))

    def sample(seed_value):
        _, result = rollout(
            params, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            inputs[5], steps, seed=seed_value)
        return result

    chunks: dict[str, list[np.ndarray]] = {"samples": [], "means": [], "scales": []}
    for offset in range(0, len(forecast_seeds), forecast_batch_size):
        seed_chunk = forecast_seeds[offset:offset + forecast_batch_size]
        result = jax.vmap(sample)(jnp.asarray(seed_chunk, dtype=jnp.uint32))
        for key, value in result.items():
            chunks[key].append(np.asarray(jax.device_get(value), dtype=np.float32))
    host: dict[str, np.ndarray] = {}
    for key, values in chunks.items():
        combined = np.concatenate(values, axis=0)
        host[key] = np.transpose(combined, (1, 0, 2, 3))
    expected = (len(starts), len(forecast_seeds), steps, 5)
    if (set(host) != {"samples", "means", "scales"}
            or any(value.shape != expected or not np.isfinite(value).all()
                   for value in host.values())
            or np.any(host["scales"] <= 0)):
        raise RuntimeError("invalid batched hierarchical Validation forecast output")
    return host


def load_hierarchical_checkpoint_parameters(*, tensors: HierarchyTensors,
        train_scope_id: str, dataset_id: str, checkout: str | Path,
        checkpoint: str | Path, minute_config: RSSMConfig | None = None,
        hour_config: RSSMConfig | None = None, day_config: RSSMConfig | None = None,
        training_config: HierarchyTrainingConfig | None = None):
    """Load a hierarchy parameter tree only when its Train identity matches."""
    if tensors.scope != "train":
        raise ValueError("checkpoint parameters can only be loaded from Train tensors")
    import ninjax as nj
    import optax  # type: ignore[import-not-found]
    if not isinstance(train_scope_id, str) or not train_scope_id.strip():
        raise ValueError("an explicit frozen Train scope identity is required")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("an explicit source dataset identity is required")
    minute_config = minute_config or RSSMConfig()
    hour_config = hour_config or RSSMConfig(
        deter=128, hidden=96, stoch=8, classes=8, blocks=8)
    day_config = day_config or RSSMConfig(
        deter=96, hidden=64, stoch=8, classes=8, blocks=8)
    cfg = training_config or HierarchyTrainingConfig()
    arrays = _validated_arrays(tensors)
    _validate_risk_objective(tensors, arrays, cfg)
    if arrays["minute_x"].shape[1] < cfg.window_length:
        raise ValueError("Train sequence is shorter than the configured minute window")
    day_visible = arrays["day_index"][0]
    min_start = next((i for i in range(len(day_visible) - cfg.window_length + 1)
                      if day_visible[i + cfg.window_length - 1] >= 1), None)
    if min_start is None:
        raise ValueError("Train sequence has no window with two observable daily states")
    score_starts = np.flatnonzero(arrays["minute_mask"][0] > 0)
    if not score_starts.size:
        raise ValueError("Train sequence has no scored minute targets")
    min_start = max(min_start, int(score_starts[0]))
    model = make_hierarchical_model(
        checkout, minute=minute_config, hour=hour_config, day=day_config,
        variant=cfg.hierarchy_variant, risk_horizons=cfg.risk_horizons,
        risk_horizon_weights=cfg.risk_horizon_weights,
        risk_auxiliary_scale=cfg.risk_auxiliary_scale,
        risk_min_scale=cfg.risk_min_scale)
    inputs = _as_model_inputs(_window_arrays(arrays, start=min_start, config=cfg))
    params_template = nj.init(model)({}, *inputs, seed=cfg.seed)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                            optax.adam(cfg.learning_rate))
    opt_template = optimizer.init(params_template)
    identity = {"train_scope_id": train_scope_id, "dataset_id": dataset_id,
                "tensor_sha256": _tensor_digest(arrays), "upstream_sha": PINNED_UPSTREAM,
                **_hierarchy_source_identity(tensors, dataset_id)}
    configuration = json.loads(json.dumps({
        "model": {"minute": asdict(minute_config), "hour": asdict(hour_config),
                  "day": asdict(day_config)},
        "train": {"learning_rate": cfg.learning_rate, "grad_clip": cfg.grad_clip,
                  "seed": cfg.seed, "window_length": cfg.window_length,
                  "burn_in": cfg.burn_in,
                  "hour_context_steps": cfg.hour_context_steps,
                  "day_context_steps": cfg.day_context_steps,
                  "hierarchy_variant": cfg.hierarchy_variant,
                  "risk_auxiliary_scale": cfg.risk_auxiliary_scale,
                  "risk_min_scale": cfg.risk_min_scale,
                  "risk_horizons": list(cfg.risk_horizons),
                  "risk_horizon_weights": list(cfg.risk_horizon_weights),
                  "risk_barrier_quantile": cfg.risk_barrier_quantile},
        "trainer_schema": "routeb.hierarchical-trainer.v2"},
        sort_keys=True, allow_nan=False))
    params, _, step, saved_seed = load_checkpoint(
        checkpoint, params_template=params_template, opt_template=opt_template,
        expected_identity=identity, expected_config=configuration,
        allow_seed_stream_metadata=True)
    if saved_seed != cfg.seed:
        raise ValueError("hierarchy checkpoint seed mismatch")
    if step < 1:
        raise ValueError("hierarchy checkpoint has no completed optimizer update")
    return params
