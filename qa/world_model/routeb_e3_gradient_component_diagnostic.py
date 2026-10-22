"""Decompose final E3 objective-term gradients on a saved synthetic factorial.

Only synthetic Train tensors and checkpoints from the seed-factorial probe are
read. The result is an engineering diagnostic, not market or economic evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from rles_world_model.cli import _read_json
from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
from rles_world_model.routeb.jax_hierarchy import make_hierarchical_model
from rles_world_model.routeb.jax_hierarchy_train import (
    HierarchyTrainingConfig,
    _as_model_inputs,
    _validated_arrays,
    _window_arrays,
    load_hierarchical_checkpoint_parameters,
)
from rles_world_model.routeb.jax_model import RSSMConfig
from rles_world_model.routeb.market import MarketSequence
from routeb_seed_stream_factorial import _synthetic_bars


FACTOR_CASES = (
    "tied_31",
    "initialization_only_102",
    "sampling_only_102",
    "model_noise_only_102",
    "initialization_sampling_102",
    "initialization_model_noise_102",
    "sampling_model_noise_102",
    "tied_102",
)
COMPONENT_NAMES = (
    "minute_prediction",
    "posterior_auxiliary",
    "slow_scale_auxiliary",
    "risk_log_return_auxiliary",
    "risk_log_realized_volatility_auxiliary",
    "risk_max_high_return_auxiliary",
    "risk_min_low_return_auxiliary",
    "risk_barrier_auxiliary",
    "shared_kl_residual",
)


def _gradient_summary(component_gradients: Any) -> dict[str, Any]:
    """Summarize a [component, ...parameter-shape] pytree without flatten copies."""
    import jax

    leaves = [np.asarray(value, dtype=np.float64)
              for value in jax.tree_util.tree_leaves(component_gradients)]
    if not leaves or any(value.ndim < 1 or value.shape[0] != len(COMPONENT_NAMES)
                         for value in leaves):
        raise RuntimeError("component gradient pytree has an unexpected shape")
    by_component = [
        [leaf[index] for leaf in leaves] for index in range(len(COMPONENT_NAMES))
    ]
    total_gradient = [
        sum(by_component[index][leaf_index] for index in range(len(COMPONENT_NAMES)))
        for leaf_index in range(len(leaves))
    ]
    norms = [
        math.sqrt(sum(float(np.square(leaf).sum()) for leaf in component))
        for component in by_component
    ]
    total_norm = math.sqrt(sum(float(np.square(leaf).sum())
                               for leaf in total_gradient))
    cosines: dict[str, float] = {}
    for index, name in enumerate(COMPONENT_NAMES):
        dot = sum(float(np.sum(component_leaf * total_leaf))
                  for component_leaf, total_leaf in
                  zip(by_component[index], total_gradient, strict=True))
        cosines[name] = dot / max(norms[index] * total_norm, 1e-30)
    return {
        "gradient_norms": {name: float(norms[index])
                           for index, name in enumerate(COMPONENT_NAMES)},
        "gradient_cosine_with_total": cosines,
        "total_gradient_norm": float(total_norm),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    import jax
    import jax.numpy as jnp
    import ninjax as nj

    devices = jax.devices()
    if not any(device.platform == "cpu" for device in devices):
        raise RuntimeError(f"gradient decomposition requires a CPU JAX device: {devices}")
    run_dir = args.run_dir.resolve()
    run_report = _read_json(str(run_dir / "report.json"))
    if (run_report.get("status") != "PASS_SYNTHETIC_ENGINEERING_ONLY"
            or run_report.get("validation_test_holdout_used") is not False
            or run_report.get("market_data_used") is not False):
        raise ValueError("source factorial report is not an eligible synthetic-only run")

    dataset_id = "routeb-synthetic-rng-factorial-v1"
    tensors, _ = build_train_hierarchy(
        MarketSequence(_synthetic_bars(), dataset_id, "train"), burn_in=32)
    arrays = _validated_arrays(tensors)
    train_config = HierarchyTrainingConfig(
        seed=31, steps=12, hierarchy_variant="e3_features", window_length=256,
        burn_in=32, hour_context_steps=16, day_context_steps=8,
        learning_rate=3e-4, report_interval=12, risk_auxiliary_scale=1.0)
    model_config = RSSMConfig(deter=16, hidden=16, stoch=2, classes=4, blocks=2)
    model = make_hierarchical_model(
        args.checkout, minute=model_config, hour=model_config, day=model_config,
        variant=train_config.hierarchy_variant,
        risk_horizons=train_config.risk_horizons,
        risk_horizon_weights=train_config.risk_horizon_weights,
        risk_auxiliary_scale=train_config.risk_auxiliary_scale,
        risk_min_scale=train_config.risk_min_scale)
    pure_model = nj.pure(model)
    cases: dict[str, dict[str, Any]] = {}

    for label in FACTOR_CASES:
        checkpoint = run_dir / f"e3-{label}.npz"
        metrics = run_report["e3_features"][label]["metrics"]
        start = int(metrics["window_start"])
        model_seed = int(metrics["model_rng_seed"])
        checkpoint_meta = _read_json(str(checkpoint.with_suffix(".json")))
        training_seed = checkpoint_meta.get("config", {}).get("train", {}).get("seed")
        if (type(training_seed) is not int
                or checkpoint_meta.get("seed") != training_seed):
            raise ValueError(f"checkpoint training seed metadata is invalid for {label}")
        case_training_config = replace(train_config, seed=training_seed)
        inputs = _as_model_inputs(
            _window_arrays(arrays, start=start, config=case_training_config))
        parameters = load_hierarchical_checkpoint_parameters(
            tensors=tensors, train_scope_id="synthetic-rng-factorial-only",
            dataset_id=dataset_id, checkout=args.checkout, checkpoint=checkpoint,
            minute_config=model_config, hour_config=model_config,
            day_config=model_config, training_config=case_training_config)

        def component_losses(candidate):
            _, diagnostics = pure_model(
                candidate, *inputs, seed=jnp.asarray(model_seed, dtype=jnp.uint32))
            terms = jnp.stack((
                model_config.prediction_scale * diagnostics["minute_nll"],
                model_config.posterior_auxiliary_scale
                * diagnostics["posterior_aux_nll"],
                diagnostics["slow_aux"],
                train_config.risk_auxiliary_scale * diagnostics["risk_log_return_nll"],
                train_config.risk_auxiliary_scale
                * diagnostics["risk_log_realized_volatility_nll"],
                train_config.risk_auxiliary_scale
                * diagnostics["risk_max_high_return_nll"],
                train_config.risk_auxiliary_scale
                * diagnostics["risk_min_low_return_nll"],
                train_config.risk_auxiliary_scale * diagnostics["risk_barrier_bce"],
            ))
            residual = diagnostics["loss"] - terms.sum()
            return jnp.concatenate((terms, residual[None]))

        losses = component_losses(parameters)
        component_gradients = jax.jacrev(component_losses)(parameters)
        values = np.asarray(jax.device_get(losses), dtype=np.float64)
        summary = _gradient_summary(component_gradients)
        if (values.shape != (len(COMPONENT_NAMES),) or not np.isfinite(values).all()
                or any(not math.isfinite(value)
                       for value in summary["gradient_norms"].values())):
            raise RuntimeError(f"nonfinite terminal E3 objective decomposition for {label}")
        cases[label] = {
            "window_start": start,
            "model_rng_seed": model_seed,
            "training_seed": training_seed,
            "component_losses": {name: float(values[index])
                                 for index, name in enumerate(COMPONENT_NAMES)},
            "total_loss": float(values.sum()),
            **summary,
        }

    report = {
        "schema": "routeb.synthetic-e3-gradient-components.v1",
        "status": "PASS_SYNTHETIC_ENGINEERING_ONLY",
        "source_factorial_report": str(run_dir / "report.json"),
        "source_factorial_steps": run_report["steps_per_arm"],
        "component_names": list(COMPONENT_NAMES),
        "cases": cases,
        "market_data_used": False,
        "research_ledger_read": False,
        "validation_test_holdout_used": False,
        "predictive_skill_claim": False,
        "economic_claim": False,
        "gradient_point": (
            "terminal checkpoint evaluated on the final reported synthetic Train window"
        ),
    }
    output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "case_count": len(cases),
        "median_component_gradient_norms": {
            name: float(np.median([
                case["gradient_norms"][name] for case in cases.values()]))
            for name in COMPONENT_NAMES
        },
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
