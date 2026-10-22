"""Minimal real gradient-based B2 training loop for pinned DreamerV3 RSSM.

This is explicitly Train-only and has no Validation reader. It is intentionally
run out-of-process in the project's separate Python 3.12 JAX environment.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time
import numpy as np

from .sequences import load_worker_batch
from .jax_model import PINNED_UPSTREAM,RSSMConfig,make_model
from .checkpoint import load_checkpoint,save_checkpoint,optimizer_recipe
from .randomness import (SeedStreams, resolve_training_seed_streams,
                         seed_stream_identity, validate_training_seed,
                         validate_jax_seed)

SNAPSHOT_SHA="80194a587444bf4a11f06f6e273723e5754f2483"

def _update_failure_context(*, step: int, sampler_seed: int,
                            selection: np.ndarray, features: np.ndarray,
                            risk_targets: np.ndarray,
                            risk_masks: np.ndarray, cause: Exception | None = None) -> str:
    """Describe a failed Train-only update without exposing input payloads."""
    selected_features = features[selection]
    selected_risks = risk_targets[selection]
    selected_risk_masks = risk_masks[selection]
    valid_risks = selected_risks[np.broadcast_to(
        selected_risk_masks[..., None], selected_risks.shape)]
    selection_sha = hashlib.sha256(
        np.asarray(selection, dtype="<i8").tobytes()).hexdigest()
    risk_abs_max = float(np.max(np.abs(valid_risks))) if valid_risks.size else 0.0
    cause_summary = ""
    if cause is not None:
        cause_text = str(cause)
        cause_message = cause_text.splitlines()[0] if cause_text else "<empty>"
        cause_summary = f"; cause_type={type(cause).__name__}; cause={cause_message[:240]}"
    return (
        f"JAX optimizer update failed at step={step} sampler_seed={sampler_seed}; "
        f"selection_sha256={selection_sha}; "
        f"feature_abs_max={float(np.max(np.abs(selected_features))):.8g}; "
        f"valid_risk_target_abs_max={risk_abs_max:.8g}{cause_summary}"
    )


@dataclass(frozen=True)
class TrainingConfig:
    seed:int=31
    steps:int=100
    batch_size:int=8
    lr:float=3e-4
    grad_clip:float=100.0
    report_interval:int=25

    def __post_init__(self):
        import math
        try:
            validate_training_seed(self.seed, self.steps)
        except ValueError as exc:
            raise ValueError("invalid training seed/update range") from exc
        if (type(self.steps) is not int
            or self.steps<1 or type(self.batch_size) is not int or self.batch_size<1
            or type(self.report_interval) is not int or self.report_interval<1
            or not math.isfinite(self.lr) or self.lr<=0
            or not math.isfinite(self.grad_clip) or self.grad_clip<=0):
            raise ValueError("invalid training config")


def train(*, dataset_npz:str|Path,train_scope_id:str,checkout:str|Path,
          output_checkpoint:str|Path,model_config:RSSMConfig|None=None,
          training_config:TrainingConfig|None=None,resume:bool=False,
          seed_streams:SeedStreams|None=None)->list[dict]:
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax  # type: ignore[import-not-found]
    model_config=model_config or RSSMConfig()
    cfg=training_config or TrainingConfig()
    streams=resolve_training_seed_streams(cfg.seed,cfg.steps,seed_streams)
    batch,meta=load_worker_batch(dataset_npz,expected_scope=train_scope_id,
                                 snapshot_sha=SNAPSHOT_SHA,upstream_sha=PINNED_UPSTREAM)
    with np.load(dataset_npz,allow_pickle=False) as arrays:
        feature_mean=np.asarray(arrays["scaler_mean"],dtype=np.float32)
        feature_std=np.asarray(arrays["scaler_std"],dtype=np.float32)
    if (feature_mean.shape!=(5,) or feature_std.shape!=(5,)
            or not np.isfinite(feature_mean).all()
            or not np.isfinite(feature_std).all() or np.any(feature_std<=0)):
        raise ValueError("Train NPZ contains invalid frozen scaler moments")
    if tuple(batch.risk_horizons)!=tuple(model_config.risk_horizons):
        raise ValueError("Train risk horizons differ from the frozen model configuration")
    model=make_model(checkout,model_config)
    n,T,D=batch.features.shape
    if n<cfg.batch_size:
        raise ValueError("not enough Train sequences for batch size")
    x0=jnp.asarray(batch.features[:cfg.batch_size]);r0=jnp.asarray(batch.reset[:cfg.batch_size]);
    m0=jnp.asarray(batch.masks[:cfg.batch_size])
    rt0=jnp.asarray(batch.risk_targets[:cfg.batch_size])
    rm0=jnp.asarray(batch.risk_masks[:cfg.batch_size])
    fm=jnp.asarray(feature_mean);fs=jnp.asarray(feature_std)
    params=nj.init(model)({},x0,r0,m0,True,rt0,rm0,fm,fs,
                         seed=streams.initialization_seed)
    opt=optax.chain(optax.clip_by_global_norm(cfg.grad_clip),optax.adam(cfg.lr))
    opt_state=opt.init(params)
    identity={"train_scope_id":train_scope_id,"dataset_id":meta['dataset_id'],
              "snapshot_sha":SNAPSHOT_SHA,"upstream_sha":PINNED_UPSTREAM,
              "train_npz_sha256":meta["artifact_sha256"],
              "scaler_sha256":meta["scaler_sha256"]}
    configuration=optimizer_recipe(model_config,cfg)
    if seed_streams is not None:
        configuration["seed_streams"] = seed_stream_identity(streams)
    start=0
    if resume:
        params,opt_state,start,seed=load_checkpoint(output_checkpoint,
            params_template=params,opt_template=opt_state,
            expected_identity=identity,expected_config=configuration)
        if seed!=cfg.seed:
            raise ValueError("seed mismatch")
    if start>cfg.steps:
        raise ValueError("resume step exceeds requested max steps")

    @jax.jit
    def update(parameters,optstate,features,reset,mask,risk_targets,risk_mask,
                scaler_mean,scaler_std,rngseed):
        def loss_fn(p):
            updated,diag=nj.pure(model)(p,features,reset,mask,True,
                risk_targets,risk_mask,scaler_mean,scaler_std,seed=rngseed)
            return diag['loss'],diag
        (value,diagnostics),grads=jax.value_and_grad(loss_fn,has_aux=True)(parameters)
        changes,newstate=opt.update(grads,optstate,parameters)
        newparams=optax.apply_updates(parameters,changes)
        grad_norm=optax.global_norm(grads)
        diagnostics={**diagnostics,'grad_norm':grad_norm,'loss':value}
        return newparams,newstate,diagnostics

    reports: list[dict[str, object]]=[]
    r2_batch_losses: list[float]=[]
    training_started=time.perf_counter()
    for step in range(start,cfg.steps):
        # Reproducible batch sampling from train data, identical after resume.
        sampler_seed=streams.sampler_seed+step
        selection=np.random.default_rng(sampler_seed).choice(n,size=cfg.batch_size,replace=False)
        model_rng_seed=streams.model_seed+step
        try:
            params,opt_state,metrics=update(params,opt_state,
                jnp.asarray(batch.features[selection]),jnp.asarray(batch.reset[selection]),
                jnp.asarray(batch.masks[selection]),jnp.asarray(batch.risk_targets[selection]),
                jnp.asarray(batch.risk_masks[selection]),fm,fs,model_rng_seed)
            scalars={}
            for key,value in metrics.items():
                array=np.asarray(value)
                if array.ndim==0:
                    scalars[key]=float(array)
                else:
                    scalars[key]=array.astype(float).tolist()
            if not all(np.isfinite(value).all() if isinstance(value,list)
                       else np.isfinite(value) for value in scalars.values()):
                raise RuntimeError("nonfinite JAX training metric; failed closed")
        except Exception as exc:
            context = _update_failure_context(
                step=step+1, sampler_seed=sampler_seed, selection=selection,
                features=batch.features, risk_targets=batch.risk_targets,
                risk_masks=batch.risk_masks, cause=exc)
            raise RuntimeError(context) from exc
        r2_batch_losses.append(scalars["r2_regularizer_loss"])
        if (step+1)%cfg.report_interval==0 or step+1==cfg.steps:
            report={"step":step+1,**scalars,
                "sampler_seed":sampler_seed,
                "sampler_order_sha256":hashlib.sha256(
                    np.asarray(selection,dtype="<i8").tobytes()).hexdigest(),
                "model_rng_seed":model_rng_seed}
            if r2_batch_losses:
                values=np.asarray(r2_batch_losses,dtype=np.float64)
                mean=float(values.mean())
                report.update({
                    "r2_batch_loss_mean":mean,
                    "r2_batch_loss_std":float(values.std()),
                    "r2_batch_loss_cv":float(values.std()/max(abs(mean),1e-12)),
                    "r2_batch_count":len(r2_batch_losses),
                })
                r2_batch_losses.clear()
            reports.append(report)
            save_checkpoint(output_checkpoint,params=params,opt_state=opt_state,
                            step=step+1,seed=cfg.seed,identity=identity,config=configuration)
    if not reports:
        raise ValueError("resume checkpoint already reached the requested max steps")
    parameter_count=sum(int(np.prod(leaf.shape))
                        for leaf in jax.tree_util.tree_leaves(params))
    reports[-1].update({
        "initialization_seed":streams.initialization_seed,
        "sampler_seed_base":streams.sampler_seed,
        "model_seed_base":streams.model_seed,
        "parameter_count":parameter_count,
        "train_examples":n,
        "optimizer_updates":cfg.steps,
        "training_seconds":time.perf_counter()-training_started,
        "device":str(jax.devices()[0]),
    })
    return reports


def forecast_standardized(*,checkout:str|Path,checkpoint:str|Path,
                          model_config:RSSMConfig,training_config:TrainingConfig,
                          train_scope_id:str,dataset_id:str,train_npz_sha256:str,scaler_sha256:str,history:np.ndarray,
                          steps:int,forecast_seed:int,
                          feature_std:np.ndarray|None=None)->np.ndarray:
    """Requires checkpoint provenance and returns generated standardized OHLCV coordinates."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax  # type: ignore[import-not-found]
    try:
        validate_jax_seed(forecast_seed, name="forecast_seed")
    except ValueError as exc:
        raise ValueError("invalid forecast horizon/seed") from exc
    if type(steps) is not int or steps<1 or steps>60:
        raise ValueError("invalid forecast horizon/seed")
    x=np.asarray(history,dtype=np.float32)
    if x.ndim!=3 or x.shape[2]!=5 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError("history [B,T,5] required")
    if model_config.rollout_risk_conditioning:
        if (feature_std is None or np.asarray(feature_std).shape!=(5,)
                or not np.isfinite(feature_std).all() or np.any(np.asarray(feature_std)<=0)):
            raise ValueError("risk-conditioned forecast requires positive frozen Train feature_std")
    model=make_model(checkout,model_config)
    mask=jnp.ones(x.shape[:2],dtype=jnp.bool_)
    reset=jnp.zeros(x.shape[:2],dtype=jnp.bool_).at[:,0].set(True)
    templ=nj.init(model)({},jnp.asarray(x),reset,mask,False,seed=training_config.seed)
    optim=optax.chain(optax.clip_by_global_norm(training_config.grad_clip),
                      optax.adam(training_config.lr))
    identity={"train_scope_id":train_scope_id,"dataset_id":dataset_id,
              "snapshot_sha":SNAPSHOT_SHA,"upstream_sha":PINNED_UPSTREAM,
              "train_npz_sha256":train_npz_sha256,
              "scaler_sha256":scaler_sha256}
    config=optimizer_recipe(model_config,training_config)
    params,_,_,_=load_checkpoint(checkpoint,params_template=templ,
        opt_template=optim.init(templ),expected_identity=identity,expected_config=config,
        allow_seed_stream_metadata=True)
    if model_config.rollout_risk_conditioning:
        _,out=nj.pure(model.rollout)(
            params,jnp.asarray(x),steps,jnp.asarray(feature_std,dtype=jnp.float32),
            seed=forecast_seed)
    else:
        _,out=nj.pure(model.rollout)(params,jnp.asarray(x),steps,seed=forecast_seed)
    result=np.asarray(jax.device_get(out))
    if result.shape!=(x.shape[0],steps,5) or not np.isfinite(result).all():
        raise RuntimeError("invalid forecast output")
    return result


def forecast_distribution_standardized(*, checkout:str|Path, checkpoint:str|Path,
        model_config:RSSMConfig, training_config:TrainingConfig,
        train_scope_id:str, dataset_id:str, train_npz_sha256:str, scaler_sha256:str,
        history:np.ndarray) -> tuple[np.ndarray,np.ndarray] | dict[str,np.ndarray]:
    """Return the checkpoint-bound E2 location and marginal standard deviation [B,5]."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax  # type: ignore[import-not-found]
    x=np.asarray(history,dtype=np.float32)
    if x.ndim!=3 or x.shape[2]!=5 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError("history [B,T>=2,5] required")
    model=make_model(checkout,model_config)
    mask=jnp.ones(x.shape[:2],dtype=jnp.bool_)
    reset=jnp.zeros(x.shape[:2],dtype=jnp.bool_).at[:,0].set(True)
    templ=nj.init(model)({},jnp.asarray(x),reset,mask,False,seed=training_config.seed)
    optim=optax.chain(optax.clip_by_global_norm(training_config.grad_clip),
                      optax.adam(training_config.lr))
    identity={"train_scope_id":train_scope_id,"dataset_id":dataset_id,
              "snapshot_sha":SNAPSHOT_SHA,"upstream_sha":PINNED_UPSTREAM,
              "train_npz_sha256":train_npz_sha256,"scaler_sha256":scaler_sha256}
    params,_,_,seed=load_checkpoint(checkpoint,params_template=templ,
        opt_template=optim.init(templ),expected_identity=identity,
        expected_config=optimizer_recipe(model_config,training_config),
        allow_seed_stream_metadata=True)
    if seed!=training_config.seed:
        raise ValueError("forecast seed differs from E2 checkpoint seed")
    _,outputs=nj.pure(model.next_distribution)(
        params,jnp.asarray(x),seed=training_config.seed)
    if isinstance(outputs,dict):
        result={key:np.asarray(jax.device_get(value)) for key,value in outputs.items()}
        if (set(result)!={"mean","scale","mixture_means","mixture_scales","mixture_weights"}
                or result["mean"].shape!=(x.shape[0],5)
                or result["scale"].shape!=result["mean"].shape
                or result["mixture_means"].shape!=(x.shape[0],2,5)
                or result["mixture_scales"].shape!=(x.shape[0],2,5)
                or result["mixture_weights"].shape!=(x.shape[0],2)
                or any(not np.isfinite(value).all() for value in result.values())
                or np.any(result["scale"]<=0) or np.any(result["mixture_scales"]<=0)
                or np.any(result["mixture_weights"]<0)
                or not np.allclose(result["mixture_weights"].sum(-1),1.0,atol=1e-5)):
            raise RuntimeError("E2 one-step mixture forecast is invalid: "+str({"shapes":{key:value.shape for key,value in result.items()},"finite":{key:bool(np.isfinite(value).all()) for key,value in result.items()},"min_scale":float(np.nanmin(result["mixture_scales"])),"weight_sums":result["mixture_weights"].sum(-1).tolist()}))
        return result
    mean,scale=outputs
    result_mean=np.asarray(jax.device_get(mean))
    result_scale=np.asarray(jax.device_get(scale))
    if (result_mean.shape!=(x.shape[0],5) or result_scale.shape!=result_mean.shape
            or not np.isfinite(result_mean).all() or not np.isfinite(result_scale).all()
            or np.any(result_scale<=0)):
        raise RuntimeError("E2 one-step forecast distribution is invalid")
    return result_mean,result_scale


def forecast_risk_distribution_standardized(*, checkout:str|Path, checkpoint:str|Path,
        model_config:RSSMConfig, training_config:TrainingConfig,
        train_scope_id:str, dataset_id:str, train_npz_sha256:str, scaler_sha256:str,
        history:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return checkpoint-bound causal prior risk heads for each configured horizon.

    The returned arrays have shapes ``[B,H,4]`` for continuous means/scales and
    ``[B,H,2]`` for upper/lower barrier logits. Continuous outputs are in the
    target units used by ``build_risk_targets``; this function does not infer
    future OHLCV paths from these diagnostics.
    """
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax  # type: ignore[import-not-found]
    x=np.asarray(history,dtype=np.float32)
    if x.ndim!=3 or x.shape[2]!=5 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError("history [B,T>=2,5] required")
    model=make_model(checkout,model_config)
    mask=jnp.ones(x.shape[:2],dtype=jnp.bool_)
    reset=jnp.zeros(x.shape[:2],dtype=jnp.bool_).at[:,0].set(True)
    templ=nj.init(model)({},jnp.asarray(x),reset,mask,False,seed=training_config.seed)
    optim=optax.chain(optax.clip_by_global_norm(training_config.grad_clip),
                      optax.adam(training_config.lr))
    identity={"train_scope_id":train_scope_id,"dataset_id":dataset_id,
              "snapshot_sha":SNAPSHOT_SHA,"upstream_sha":PINNED_UPSTREAM,
              "train_npz_sha256":train_npz_sha256,"scaler_sha256":scaler_sha256}
    params,_,_,seed=load_checkpoint(checkpoint,params_template=templ,
        opt_template=optim.init(templ),expected_identity=identity,
        expected_config=optimizer_recipe(model_config,training_config),
        allow_seed_stream_metadata=True)
    if seed!=training_config.seed:
        raise ValueError("risk forecast seed differs from E2 checkpoint seed")
    _,outputs=nj.pure(model.next_risk_distribution)(
        params,jnp.asarray(x),seed=training_config.seed)
    mean,scale,barrier_logits=(np.asarray(jax.device_get(value)) for value in outputs)
    expected_horizons=len(model_config.risk_horizons)
    if (mean.shape!=(x.shape[0],expected_horizons,4)
            or scale.shape!=mean.shape
            or barrier_logits.shape!=(x.shape[0],expected_horizons,2)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not np.isfinite(barrier_logits).all() or np.any(scale<=0)):
        raise RuntimeError("E2 prior risk distribution is invalid")
    return mean,scale,barrier_logits


def alignment_distributions_standardized(*, checkout:str|Path, checkpoint:str|Path,
        model_config:RSSMConfig, training_config:TrainingConfig,
        train_scope_id:str, dataset_id:str, train_npz_sha256:str, scaler_sha256:str,
        history:np.ndarray) -> dict[str,np.ndarray]:
    """Return aligned causal prior/posterior distributions and categorical logits."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax  # type: ignore[import-not-found]
    x=np.asarray(history,dtype=np.float32)
    if x.ndim!=3 or x.shape[2]!=5 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError("history [B,T>=2,5] required")
    model=make_model(checkout,model_config)
    mask=jnp.ones(x.shape[:2],dtype=jnp.bool_)
    reset=jnp.zeros(x.shape[:2],dtype=jnp.bool_).at[:,0].set(True)
    templ=nj.init(model)({},jnp.asarray(x),reset,mask,False,seed=training_config.seed)
    optim=optax.chain(optax.clip_by_global_norm(training_config.grad_clip),
                      optax.adam(training_config.lr))
    identity={"train_scope_id":train_scope_id,"dataset_id":dataset_id,
              "snapshot_sha":SNAPSHOT_SHA,"upstream_sha":PINNED_UPSTREAM,
              "train_npz_sha256":train_npz_sha256,"scaler_sha256":scaler_sha256}
    params,_,_,seed=load_checkpoint(checkpoint,params_template=templ,
        opt_template=optim.init(templ),expected_identity=identity,
        expected_config=optimizer_recipe(model_config,training_config),
        allow_seed_stream_metadata=True)
    if seed!=training_config.seed:
        raise ValueError("alignment seed differs from E2 checkpoint seed")
    _,outputs=nj.pure(model.alignment_distributions)(
        params,jnp.asarray(x),seed=training_config.seed)
    result={key:np.asarray(jax.device_get(value)) for key,value in outputs.items()}
    expected=(x.shape[0],x.shape[1]-1)
    if (result["prior_mean"].shape!=(*expected,5)
            or result["prior_scale"].shape!=(*expected,5)
            or result["posterior_mean"].shape!=(*expected,5)
            or result["posterior_scale"].shape!=(*expected,5)
            or result["prior_logits"].shape[:2]!=expected
            or result["posterior_logits"].shape!=result["prior_logits"].shape
            or any(not np.isfinite(value).all() for value in result.values())
            or np.any(result["prior_scale"]<=0)
            or np.any(result["posterior_scale"]<=0)):
        raise RuntimeError("checkpoint alignment distributions are invalid")
    return result
