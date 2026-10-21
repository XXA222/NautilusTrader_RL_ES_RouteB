"""Synthetic-only proof of the R1 causal prior forecast path and gradients.

Run in the pinned Python 3.12 JAX environment. This reads no market files.
"""
from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from rles_world_model.routeb.jax_model import RSSMConfig, make_model
from rles_world_model.routeb.jax_hierarchy import masked_weighted_kl


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    rng = np.random.default_rng(20261018)
    features = rng.normal(size=(2, 9, 5)).astype(np.float32)
    reset = np.zeros((2, 9), dtype=bool)
    reset[:, 0] = True
    mask = np.ones((2, 9), dtype=np.float32)
    model = make_model(os.environ["ROUTEB_DREAMERV3_CHECKOUT"], RSSMConfig(
        deter=32, hidden=32, stoch=2, classes=4, blocks=4))
    risk_targets=rng.normal(0.0,0.02,size=(2,9,5,6)).astype(np.float32)
    risk_targets[...,4:6]=(risk_targets[...,4:6]>0).astype(np.float32)
    risk_mask=np.ones((2,9,5),dtype=bool)
    model_args = (jnp.asarray(features), jnp.asarray(reset), jnp.asarray(mask), True,
            jnp.asarray(risk_targets),jnp.asarray(risk_mask))
    params = nj.init(model)({}, *model_args, seed=31)

    def loss_fn(parameters):
        _, diagnostics = nj.pure(model)(parameters, *model_args, seed=71)
        return diagnostics["prior_forecast_nll"]

    loss, gradients = jax.value_and_grad(loss_fn)(params)
    leaves = jax.tree_util.tree_flatten_with_path(gradients)[0]
    norms = {jax.tree_util.keystr(path): float(jnp.linalg.norm(value))
             for path, value in leaves}
    head_norm = sum(value for key, value in norms.items() if "prior_forecast" in key)
    transition_norm = sum(value for key, value in norms.items()
                          if "dyngru" in key or "priorlogit" in key)
    if not np.isfinite(float(loss)) or head_norm <= 0 or transition_norm <= 0:
        raise AssertionError(f"prior path lacks finite nonzero gradients: {norms}")

    crps_model = make_model(os.environ["ROUTEB_DREAMERV3_CHECKOUT"], RSSMConfig(
        deter=32, hidden=32, stoch=2, classes=4, blocks=4,
        forecast_objective="open_loop_crps", open_loop_crps_horizon=3))
    crps_params = nj.init(crps_model)({}, *model_args, seed=31)

    def crps_loss_fn(parameters):
        _, diagnostics = nj.pure(crps_model)(parameters, *model_args, seed=74)
        return diagnostics["prior_forecast_objective"], diagnostics

    (crps_loss, crps_diagnostics), crps_gradients = jax.value_and_grad(
        crps_loss_fn, has_aux=True)(crps_params)
    crps_leaves = jax.tree_util.tree_flatten_with_path(crps_gradients)[0]
    crps_norms = {jax.tree_util.keystr(path): float(jnp.linalg.norm(value))
                  for path, value in crps_leaves}
    crps_head_norm = sum(value for key, value in crps_norms.items()
                         if "prior_forecast" in key)
    crps_transition_norm = sum(value for key, value in crps_norms.items()
                               if "dyngru" in key or "priorlogit" in key)
    if (not np.isfinite(float(crps_loss)) or float(crps_loss) <= 0
            or crps_head_norm <= 0 or crps_transition_norm <= 0
            or not np.isclose(float(crps_diagnostics["prior_forecast_objective"]),
                              float(crps_diagnostics["open_loop_prior_crps"]), rtol=1e-6)
            or np.asarray(crps_diagnostics["open_loop_prior_crps_by_horizon"]).shape!=(3,)
            or not np.isfinite(np.asarray(
                crps_diagnostics["open_loop_prior_crps_by_horizon"])).all()):
        raise AssertionError(f"CRPS objective lacks finite nonzero gradients: {crps_norms}")

    path_targets=risk_targets[:,:,:3].copy()
    path_mask=risk_mask[:,:,:3].copy()
    # Only the first anchor per sequence is scored. Mask all tested horizons
    # at the later anchor so future-observation mutation cannot change its context.
    path_mask[:,3,1:3]=False
    path_model=make_model(os.environ["ROUTEB_DREAMERV3_CHECKOUT"],RSSMConfig(
        deter=32,hidden=32,stoch=2,classes=4,blocks=4,
        risk_horizons=(1,3,5),risk_horizon_weights=(1.,1.,1.),
        forecast_objective="path_log_rv_crps",path_log_rv_crps_horizon=3,
        path_log_rv_crps_horizons=(3,5),path_log_rv_crps_paths=4,
        rollout_risk_conditioning=True,rollout_risk_sampling=True,
        rollout_risk_conditioning_strength=0.5,
        rollout_risk_noise_persistence=0.65))
    path_args=(jnp.asarray(features),jnp.asarray(reset),jnp.asarray(mask),True,
        jnp.asarray(path_targets),jnp.asarray(path_mask),
        jnp.asarray([0.,0.,0.,0.,0.],dtype=jnp.float32),
        jnp.asarray([.001,.001,1.,1.,1.],dtype=jnp.float32))
    path_params=nj.init(path_model)({},*path_args,seed=31)

    def path_loss_fn(parameters,values):
        changed_args=(jnp.asarray(values),*path_args[1:])
        _,diagnostics=nj.pure(path_model)(parameters,*changed_args,seed=117)
        return diagnostics["path_log_rv_crps"],diagnostics

    (path_loss,path_diagnostics),path_gradients=jax.value_and_grad(
        lambda parameters:path_loss_fn(parameters,features),has_aux=True)(path_params)
    path_norms={jax.tree_util.keystr(path):float(jnp.linalg.norm(value))
                for path,value in jax.tree_util.tree_flatten_with_path(path_gradients)[0]}
    path_head_norm=sum(value for key,value in path_norms.items()
                       if "prior_forecast" in key)
    path_transition_norm=sum(value for key,value in path_norms.items()
                             if "dyngru" in key or "priorlogit" in key)
    path_risk_head_norm=sum(value for key,value in path_norms.items()
                            if "prior_risk" in key)
    path_changed=features.copy()
    path_changed[:,1:,:]+=rng.normal(0.0,7.0,size=path_changed[:,1:,:].shape).astype(
        np.float32)
    _,mutated_path_diagnostics=path_loss_fn(path_params,path_changed)
    if (not np.isfinite(float(path_loss)) or float(path_loss)<=0
            or float(path_diagnostics["valid_path_log_rv_crps"])!=2.0
            or path_head_norm<=0 or path_transition_norm<=0 or path_risk_head_norm<=0
            or not np.isclose(float(path_diagnostics["path_log_rv_crps"]),
                              float(mutated_path_diagnostics["path_log_rv_crps"]),
                              rtol=0.0,atol=1e-6)):
        raise AssertionError(
            "path log-RV CRPS failed gradient/mask/causality: "
            f"loss={float(path_loss)}, valid={float(path_diagnostics['valid_path_log_rv_crps'])}, "
            f"head={path_head_norm}, transition={path_transition_norm}, "
            f"risk={path_risk_head_norm}, "
            f"mutated={float(mutated_path_diagnostics['path_log_rv_crps'])}, "
            f"norms={path_norms}")

    def risk_loss_fn(parameters):
        _,diagnostics=nj.pure(model)(parameters,*model_args,seed=72)
        return diagnostics["prior_risk_nll"]

    risk_loss,risk_gradients=jax.value_and_grad(risk_loss_fn)(params)
    risk_leaves=jax.tree_util.tree_flatten_with_path(risk_gradients)[0]
    risk_norms={jax.tree_util.keystr(path):float(jnp.linalg.norm(value))
                for path,value in risk_leaves}
    risk_head_norm=sum(value for key,value in risk_norms.items() if "prior_risk" in key)
    risk_transition_norm=sum(value for key,value in risk_norms.items()
                             if "dyngru" in key or "priorlogit" in key)
    if (not np.isfinite(float(risk_loss)) or risk_head_norm<=0
            or risk_transition_norm<=0):
        raise AssertionError(f"risk objective lacks finite nonzero gradients: {risk_norms}")

    constant=np.zeros_like(features)
    constant_targets=np.zeros_like(risk_targets)
    _,constant_metrics=nj.pure(model)(params,jnp.asarray(constant),jnp.asarray(reset),
        jnp.asarray(mask),True,jnp.asarray(constant_targets),jnp.asarray(risk_mask),seed=73)
    constant_loss=float(constant_metrics["loss"])
    constant_variance=float(constant_metrics["latent_feature_variance"])
    if (not np.isfinite(constant_loss) or not np.isfinite(constant_variance)
            or constant_variance<0):
        raise AssertionError("constant-input negative case produced invalid loss/variance")

    dyn_kl=jnp.asarray([[1.0,3.0,100.0]])
    rep_kl=jnp.asarray([[2.0,5.0,200.0]])
    kl_mask=jnp.asarray([[1.0,1.0,0.0]])
    def weighted_kl(dyn_weight,rep_weight):
        return masked_weighted_kl(dyn_kl,rep_kl,kl_mask,
            dynamics_scale=dyn_weight,representation_scale=rep_weight)
    dyn_weight_gradient,rep_weight_gradient=jax.grad(
        weighted_kl,argnums=(0,1))(0.25,0.1)
    if (not np.isclose(float(dyn_weight_gradient),2.0)
            or not np.isclose(float(rep_weight_gradient),3.5)):
        raise AssertionError("hierarchical dynamics/representation KL weights are entangled")

    def distributions(values):
        _, result = nj.pure(model.prior_forecast_sequence)(
            params, jnp.asarray(values), seed=991)
        return tuple(np.asarray(jax.device_get(item)) for item in result)

    first_mean, first_scale = distributions(features)
    changed = features.copy()
    changed[:, 1:, :] += rng.normal(0.0, 7.0, size=changed[:, 1:, :].shape).astype(np.float32)
    mutated_mean, mutated_scale = distributions(changed)
    # q(x_2) and all later observations must not feed the forecast for x_2.
    if (not np.array_equal(first_mean[:, 0], mutated_mean[:, 0])
            or not np.array_equal(first_scale[:, 0], mutated_scale[:, 0])):
        raise AssertionError("future observation changed its own prior forecast")
    if (first_mean.shape != (2, 8, 5) or first_scale.shape != first_mean.shape
            or not np.isfinite(first_mean).all() or not np.isfinite(first_scale).all()
            or np.any(first_scale <= 0)):
        raise AssertionError("prior forecast sequence shape or values invalid")

    # The CI worker exercises the specified free-prior forecast horizons. The
    # model receives only the fixed observable prefix; no future fixture rows
    # are passed to rollout.
    rollout_shapes = {}
    for horizon in (1, 5, 15, 60):
        _, generated = nj.pure(model.rollout)(
            params, jnp.asarray(features[:, :3]), horizon, seed=1700 + horizon)
        generated = np.asarray(jax.device_get(generated))
        if generated.shape != (2, horizon, 5) or not np.isfinite(generated).all():
            raise AssertionError(f"invalid F1 prior rollout for horizon {horizon}")
        rollout_shapes[str(horizon)] = list(generated.shape)

    conditioned_model=make_model(
        os.environ["ROUTEB_DREAMERV3_CHECKOUT"],RSSMConfig(
            deter=32,hidden=32,stoch=2,classes=4,blocks=4,
            rollout_risk_conditioning=True,rollout_risk_sampling=True,
            rollout_risk_noise_persistence=0.65))
    conditioned_params=nj.init(conditioned_model)(
        {},*model_args,seed=31)
    train_feature_std=jnp.asarray([0.001,0.001,0.1,0.1,1.0],dtype=jnp.float32)
    _,conditioned_generated=nj.pure(conditioned_model.rollout)(
        conditioned_params,jnp.asarray(features[:,:3]),5,train_feature_std,
        seed=1915)
    _,conditioned_generated_repeat=nj.pure(conditioned_model.rollout)(
        conditioned_params,jnp.asarray(features[:,:3]),5,train_feature_std,
        seed=1915)
    conditioned_generated=np.asarray(jax.device_get(conditioned_generated))
    conditioned_generated_repeat=np.asarray(jax.device_get(conditioned_generated_repeat))
    if (conditioned_generated.shape!=(2,5,5)
            or not np.isfinite(conditioned_generated).all()
            or not np.array_equal(conditioned_generated,conditioned_generated_repeat)):
        raise AssertionError("native-risk-conditioned prior rollout is invalid")
    report={
        "status": "PASS", "scope": "synthetic_mechanism_only",
        "loss": float(loss), "prior_head_gradient_norm": head_norm,
        "prior_transition_gradient_norm": transition_norm,
        "crps_loss": float(crps_loss), "crps_head_gradient_norm": crps_head_norm,
        "crps_transition_gradient_norm": crps_transition_norm,
        "open_loop_crps_horizon": 3,
        "open_loop_crps_by_horizon": np.asarray(
            crps_diagnostics["open_loop_prior_crps_by_horizon"]).astype(float).tolist(),
        "path_log_rv_crps":float(path_loss),
        "path_log_rv_crps_valid_anchors":float(path_diagnostics[
            "valid_path_log_rv_crps"]),
        "path_log_rv_crps_prior_head_gradient_norm":path_head_norm,
        "path_log_rv_crps_prior_transition_gradient_norm":path_transition_norm,
        "path_log_rv_crps_risk_head_gradient_norm":path_risk_head_norm,
        "path_log_rv_crps_future_mutation_invariant":True,
        "risk_loss":float(risk_loss),"risk_head_gradient_norm":risk_head_norm,
        "risk_transition_gradient_norm":risk_transition_norm,
        "constant_input_finite":True,"constant_input_latent_variance":constant_variance,
        "hierarchy_kl_weight_gradients":[float(dyn_weight_gradient),float(rep_weight_gradient)],
        "prior_sequence_shape": list(first_mean.shape),
        "f1_prior_rollout_shapes": rollout_shapes,
        "risk_conditioned_rollout_shape":list(conditioned_generated.shape),
        "risk_conditioned_rollout_finite":True,
        "future_observation_mutation_invariant": True,
        "real_market_validation": False, "economic_claim": False,
    }
    rendered=json.dumps(report,sort_keys=True,indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
