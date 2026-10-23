"""Minimal real gradient-based B2 training loop for pinned DreamerV3 RSSM.

This is explicitly Train-only and has no Validation reader. It is intentionally
run out-of-process in the project's separate Python 3.12 JAX environment.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

from .sequences import load_worker_batch
from .jax_model import PINNED_UPSTREAM,RSSMConfig,make_model
from .checkpoint import load_checkpoint,save_checkpoint,optimizer_recipe

SNAPSHOT_SHA="80194a587444bf4a11f06f6e273723e5754f2483"


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
        if (type(self.seed) is not int or self.seed<0 or type(self.steps) is not int
            or self.steps<1 or type(self.batch_size) is not int or self.batch_size<1
            or type(self.report_interval) is not int or self.report_interval<1
            or not math.isfinite(self.lr) or self.lr<=0
            or not math.isfinite(self.grad_clip) or self.grad_clip<=0):
            raise ValueError("invalid training config")


def train(*, dataset_npz:str|Path,train_scope_id:str,checkout:str|Path,
          output_checkpoint:str|Path,model_config:RSSMConfig|None=None,
          training_config:TrainingConfig|None=None,resume:bool=False)->list[dict]:
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax
    model_config=model_config or RSSMConfig()
    cfg=training_config or TrainingConfig()
    batch,meta=load_worker_batch(dataset_npz,expected_scope=train_scope_id,
                                 snapshot_sha=SNAPSHOT_SHA,upstream_sha=PINNED_UPSTREAM)
    model=make_model(checkout,model_config)
    n,T,D=batch.features.shape
    if n<cfg.batch_size:
        raise ValueError("not enough Train sequences for batch size")
    x0=jnp.asarray(batch.features[:cfg.batch_size]);r0=jnp.asarray(batch.reset[:cfg.batch_size]);
    m0=jnp.asarray(batch.masks[:cfg.batch_size])
    params=nj.init(model)({},x0,r0,m0,True,seed=cfg.seed)
    opt=optax.chain(optax.clip_by_global_norm(cfg.grad_clip),optax.adam(cfg.lr))
    opt_state=opt.init(params)
    identity={"train_scope_id":train_scope_id,"dataset_id":meta['dataset_id'],
              "snapshot_sha":SNAPSHOT_SHA,"upstream_sha":PINNED_UPSTREAM,
              "train_npz_sha256":meta["artifact_sha256"],
              "scaler_sha256":meta["scaler_sha256"]}
    configuration=optimizer_recipe(model_config,cfg)
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
    def update(parameters,optstate,features,reset,mask,rngseed):
        def loss_fn(p):
            updated,diag=nj.pure(model)(p,features,reset,mask,True,seed=rngseed)
            return diag['loss'],diag
        (value,diagnostics),grads=jax.value_and_grad(loss_fn,has_aux=True)(parameters)
        changes,newstate=opt.update(grads,optstate,parameters)
        newparams=optax.apply_updates(parameters,changes)
        grad_norm=optax.global_norm(grads)
        diagnostics={**diagnostics,'grad_norm':grad_norm,'loss':value}
        return newparams,newstate,diagnostics

    reports: list[dict[str, object]]=[]
    training_started=time.perf_counter()
    for step in range(start,cfg.steps):
        # Reproducible batch sampling from train data, identical after resume.
        selection=np.random.default_rng(cfg.seed+step).choice(n,size=cfg.batch_size,replace=False)
        params,opt_state,metrics=update(params,opt_state,
            jnp.asarray(batch.features[selection]),jnp.asarray(batch.reset[selection]),
            jnp.asarray(batch.masks[selection]),cfg.seed+step+1)
        scalars={k:float(np.asarray(v)) for k,v in metrics.items()}
        if not all(np.isfinite(v) for v in scalars.values()):
            raise RuntimeError("nonfinite JAX training metric; failed closed")
        if (step+1)%cfg.report_interval==0 or step+1==cfg.steps:
            reports.append({"step":step+1,**scalars})
            save_checkpoint(output_checkpoint,params=params,opt_state=opt_state,
                            step=step+1,seed=cfg.seed,identity=identity,config=configuration)
    if not reports:
        raise ValueError("resume checkpoint already reached the requested max steps")
    parameter_count=sum(int(np.prod(leaf.shape))
                        for leaf in jax.tree_util.tree_leaves(params))
    reports[-1].update({
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
                          steps:int,forecast_seed:int)->np.ndarray:
    """Requires checkpoint provenance and returns generated standardized OHLCV coordinates."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax
    if steps<1 or steps>60 or forecast_seed<0:
        raise ValueError("invalid forecast horizon/seed")
    x=np.asarray(history,dtype=np.float32)
    if x.ndim!=3 or x.shape[2]!=5 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError("history [B,T,5] required")
    model=make_model(checkout,model_config)
    mask=jnp.ones(x.shape[:2],dtype=jnp.bool_)
    reset=jnp.zeros(x.shape[:2],dtype=jnp.bool_).at[:,0].set(True)
    templ=nj.init(model)({},jnp.asarray(x),reset,mask,True,seed=training_config.seed)
    optim=optax.chain(optax.clip_by_global_norm(training_config.grad_clip),
                      optax.adam(training_config.lr))
    identity={"train_scope_id":train_scope_id,"dataset_id":dataset_id,
              "snapshot_sha":SNAPSHOT_SHA,"upstream_sha":PINNED_UPSTREAM,
              "train_npz_sha256":train_npz_sha256,
              "scaler_sha256":scaler_sha256}
    config=optimizer_recipe(model_config,training_config)
    params,_,_,_=load_checkpoint(checkpoint,params_template=templ,
        opt_template=optim.init(templ),expected_identity=identity,expected_config=config)
    _,out=nj.pure(model.rollout)(params,jnp.asarray(x),steps,seed=forecast_seed)
    result=np.asarray(jax.device_get(out))
    if result.shape!=(x.shape[0],steps,5) or not np.isfinite(result).all():
        raise RuntimeError("invalid forecast output")
    return result


def forecast_distribution_standardized(*, checkout:str|Path, checkpoint:str|Path,
        model_config:RSSMConfig, training_config:TrainingConfig,
        train_scope_id:str, dataset_id:str, train_npz_sha256:str, scaler_sha256:str,
        history:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Return the checkpoint-bound E2 one-step Gaussian feature head [B,5]."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import optax
    x=np.asarray(history,dtype=np.float32)
    if x.ndim!=3 or x.shape[2]!=5 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError("history [B,T>=2,5] required")
    model=make_model(checkout,model_config)
    mask=jnp.ones(x.shape[:2],dtype=jnp.bool_)
    reset=jnp.zeros(x.shape[:2],dtype=jnp.bool_).at[:,0].set(True)
    templ=nj.init(model)({},jnp.asarray(x),reset,mask,True,seed=training_config.seed)
    optim=optax.chain(optax.clip_by_global_norm(training_config.grad_clip),
                      optax.adam(training_config.lr))
    identity={"train_scope_id":train_scope_id,"dataset_id":dataset_id,
              "snapshot_sha":SNAPSHOT_SHA,"upstream_sha":PINNED_UPSTREAM,
              "train_npz_sha256":train_npz_sha256,"scaler_sha256":scaler_sha256}
    params,_,_,seed=load_checkpoint(checkpoint,params_template=templ,
        opt_template=optim.init(templ),expected_identity=identity,
        expected_config=optimizer_recipe(model_config,training_config))
    if seed!=training_config.seed:
        raise ValueError("forecast seed differs from E2 checkpoint seed")
    _,(mean,scale)=nj.pure(model.next_distribution)(
        params,jnp.asarray(x),seed=training_config.seed)
    result_mean=np.asarray(jax.device_get(mean))
    result_scale=np.asarray(jax.device_get(scale))
    if (result_mean.shape!=(x.shape[0],5) or result_scale.shape!=result_mean.shape
            or not np.isfinite(result_mean).all() or not np.isfinite(result_scale).all()
            or np.any(result_scale<=0)):
        raise RuntimeError("E2 one-step forecast distribution is invalid")
    return result_mean,result_scale
