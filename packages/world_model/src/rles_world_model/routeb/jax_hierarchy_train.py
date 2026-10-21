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


@dataclass(frozen=True)
class HierarchyTrainingConfig:
    seed: int = 31
    steps: int = 100
    window_length: int = 1024
    hour_context_steps: int = 48
    day_context_steps: int = 32
    learning_rate: float = 3e-4
    grad_clip: float = 100.0
    report_interval: int = 25

    def __post_init__(self) -> None:
        import math
        if (type(self.seed) is not int or self.seed < 0
                or type(self.steps) is not int or self.steps < 1
                or type(self.report_interval) is not int or self.report_interval < 1
                or type(self.window_length) is not int or self.window_length < 16
                or type(self.hour_context_steps) is not int
                or self.hour_context_steps < (self.window_length + 59) // 60 + 2
                or type(self.day_context_steps) is not int
                or self.day_context_steps < (self.window_length + 1439) // 1440 + 2
                or isinstance(self.learning_rate, bool)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or isinstance(self.grad_clip, bool)
                or not math.isfinite(self.grad_clip) or self.grad_clip <= 0):
            raise ValueError("invalid hierarchical training configuration")


def _validated_arrays(tensors: HierarchyTensors) -> dict[str, np.ndarray]:
    arrays = {
        "minute_x": np.asarray(tensors.minute_x, dtype=np.float32),
        "hour_x": np.asarray(tensors.hour_x, dtype=np.float32),
        "day_x": np.asarray(tensors.day_x, dtype=np.float32),
        "hour_index": np.asarray(tensors.hour_index, dtype=np.int32),
        "day_index": np.asarray(tensors.day_index, dtype=np.int32),
        "minute_mask": np.asarray(tensors.minute_mask, dtype=np.float32),
        "hour_mask": np.asarray(tensors.hour_mask, dtype=np.float32),
        "day_mask": np.asarray(tensors.day_mask, dtype=np.float32),
        "minute_available_at": np.asarray(tensors.minute_available_at, dtype=np.int64),
        "hour_available_at": np.asarray(tensors.hour_available_at, dtype=np.int64),
        "day_available_at": np.asarray(tensors.day_available_at, dtype=np.int64),
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
    if not np.any(arrays["minute_mask"] > 0):
        raise ValueError("hierarchy Train sequence has no scored minute targets")
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


def _window_arrays(arrays: dict[str, np.ndarray], *, start: int,
                   config: HierarchyTrainingConfig) -> dict[str, np.ndarray]:
    """Crop one fixed-shape minute window and its causally visible slow context."""
    end = start + config.window_length
    if start < 0 or end > arrays["minute_x"].shape[1]:
        raise ValueError("hierarchy training window is outside the Train sequence")
    sampled = dict(arrays)
    sampled["minute_x"] = arrays["minute_x"][:, start:end]
    sampled["minute_mask"] = arrays["minute_mask"][:, start:end]
    sampled["minute_available_at"] = arrays["minute_available_at"][start:end]
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
    return sampled


def _as_model_inputs(arrays: dict[str, np.ndarray]):
    import jax.numpy as jnp
    return [jnp.asarray(arrays[name]) for name in (
        "minute_x", "hour_x", "day_x", "hour_index", "day_index",
        "minute_mask", "hour_mask", "day_mask")]


def train_hierarchical(*, tensors: HierarchyTensors, train_scope_id: str,
        dataset_id: str, checkout: str | Path, output_checkpoint: str | Path,
        minute_config: RSSMConfig | None = None,
        hour_config: RSSMConfig | None = None, day_config: RSSMConfig | None = None,
        training_config: HierarchyTrainingConfig | None = None,
        resume: bool = False) -> list[dict]:
    """Optimize causal Train windows with deterministic checkpoint resume.

    All three RSSMs and both slow-scale reconstruction objectives are trained
    jointly. Per-step window seeds make resumed and uninterrupted optimizer
    trajectories identical. Registered E3/E4 runners remain separate work.
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
    arrays = _validated_arrays(tensors)
    if arrays["minute_x"].shape[1] < cfg.window_length:
        raise ValueError("Train sequence is shorter than the configured minute window")
    day_visible = arrays["day_index"][0]
    min_start = next((i for i in range(len(day_visible) - cfg.window_length + 1)
                      if day_visible[i + cfg.window_length - 1] >= 1), None)
    if min_start is None:
        raise ValueError("Train sequence has no window with two observable daily states")
    max_start = arrays["minute_x"].shape[1] - cfg.window_length
    if min_start > max_start:
        raise ValueError("no eligible three-scale training windows")
    eligible_start_count = max_start - min_start + 1
    digest = _tensor_digest(arrays)
    model = make_hierarchical_model(checkout, minute=minute_config,
                                    hour=hour_config, day=day_config)
    first_window = _window_arrays(arrays, start=min_start, config=cfg)
    inputs = _as_model_inputs(first_window)
    seed = cfg.seed
    params = nj.init(model)({}, *inputs, seed=seed)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.grad_clip),
                            optax.adam(cfg.learning_rate))
    opt_state = optimizer.init(params)
    identity = {"train_scope_id": train_scope_id, "dataset_id": dataset_id,
                "tensor_sha256": digest, "upstream_sha": PINNED_UPSTREAM}
    model_recipe = {"minute": asdict(minute_config), "hour": asdict(hour_config),
                    "day": asdict(day_config)}
    train_recipe = {"learning_rate": cfg.learning_rate, "grad_clip": cfg.grad_clip,
                    "seed": cfg.seed, "window_length": cfg.window_length,
                    "hour_context_steps": cfg.hour_context_steps,
                    "day_context_steps": cfg.day_context_steps}
    configuration = json.loads(json.dumps(
        {"model": model_recipe, "train": train_recipe,
         "trainer_schema": "routeb.hierarchical-trainer.v1"},
        sort_keys=True, allow_nan=False))
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
    started = time.perf_counter()
    for step in range(start, cfg.steps):
        sampler_seed = cfg.seed + step
        selected_start = int(np.random.default_rng(sampler_seed).integers(
            min_start, max_start + 1))
        window = _window_arrays(arrays, start=selected_start, config=cfg)
        window_inputs = _as_model_inputs(window)
        params, opt_state, metrics = update(
            params, opt_state, window_inputs, cfg.seed + step + 1)
        scalars = {key: float(np.asarray(value)) for key, value in metrics.items()}
        if not all(np.isfinite(value) for value in scalars.values()):
            raise RuntimeError("nonfinite hierarchical training metric; failed closed")
        if (step + 1) % cfg.report_interval == 0 or step + 1 == cfg.steps:
            report = {"step": step + 1, **scalars,
                      "model_rng_seed": cfg.seed + step + 1,
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
                   "parameters": sum(int(np.prod(leaf.shape)) for leaf in
                                     jax.tree_util.tree_leaves(params)),
                   "train_tensor_sha256": digest,
                   "training_seconds": time.perf_counter() - started,
                   "device": str(jax.devices()[0])})
    return reports
