"""Additive JAX world-model module using the pinned *unmodified* DreamerV3 RSSM.

Run ONLY inside packages/world_model/environment (Python 3.12). Model learns
current reconstruction, next-bar conditional OHLCV and multi-horizon risk
distributions, plus Dreamer RSSM KL. Market actions are constant zero even
when trading actions change.

Training is single-scale B2. The B3 hierarchy plan is a separate, currently
untrained component and MUST NOT be advertised as a learned 3-scale model.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import sys

PINNED_UPSTREAM="e01491fad6434b2245a3b8ca201dd7faedcc458c"


@dataclass(frozen=True)
class RSSMConfig:
    deter: int=256
    hidden: int=128
    stoch: int=16
    classes: int=16
    blocks: int=8
    feature_dim: int=5
    dynamics_scale: float=1.0
    representation_scale: float=0.1
    reconstruction_scale: float=0.2
    prediction_scale: float=1.0
    posterior_auxiliary_scale: float=0.1
    risk_prediction_scale: float=1.0
    risk_horizons: tuple[int,...]=(1,5,15,30,60)
    risk_horizon_weights: tuple[float,...]=(1.0,1.0,1.0,1.0,1.0)
    forecast_distribution: str="gaussian"
    forecast_objective: str="nll"
    open_loop_crps_horizon: int=15
    path_log_rv_crps_horizon: int=15
    path_log_rv_crps_horizons: tuple[int,...]=()
    path_log_rv_crps_paths: int=8
    path_log_rv_crps_scale: float=1.0
    forecast_student_t_df: float=5.0
    rollout_risk_conditioning: bool=False
    rollout_risk_conditioning_strength: float=1.0
    min_scale: float=0.02

    def __post_init__(self)->None:
        horizons=tuple(self.risk_horizons)
        horizon_weights=tuple(self.risk_horizon_weights)
        path_horizons=tuple(self.path_log_rv_crps_horizons)
        if not path_horizons:
            path_horizons=(self.path_log_rv_crps_horizon,)
        object.__setattr__(self,"risk_horizons",horizons)
        object.__setattr__(self,"risk_horizon_weights",horizon_weights)
        object.__setattr__(self,"path_log_rv_crps_horizons",path_horizons)
        if (any(type(x) is not int for x in (self.deter,self.hidden,self.stoch,
                                                  self.classes,self.blocks,self.feature_dim))
            or self.deter<8 or self.hidden<8 or self.stoch<1 or self.classes<2
            or self.blocks<1 or self.deter%self.blocks!=0 or self.feature_dim!=5
            or any(not math.isfinite(x) or x<=0 for x in (
                self.dynamics_scale,self.representation_scale,
                self.reconstruction_scale,self.prediction_scale,
                self.posterior_auxiliary_scale,self.risk_prediction_scale,
                self.forecast_student_t_df,self.min_scale,
                self.path_log_rv_crps_scale))
            or (isinstance(self.rollout_risk_conditioning_strength,bool)
                or not isinstance(self.rollout_risk_conditioning_strength,(int,float))
                or not math.isfinite(self.rollout_risk_conditioning_strength)
                or not 0.0<=self.rollout_risk_conditioning_strength<=1.0)
            or self.forecast_distribution not in {"gaussian","student_t","mixture_gaussian"}
            or self.forecast_objective not in {
                "nll","crps","open_loop_crps","path_log_rv_crps"}
            or (self.forecast_objective in {
                    "crps","open_loop_crps","path_log_rv_crps"}
                and self.forecast_distribution!="gaussian")
            or type(self.open_loop_crps_horizon) is not int
            or not 1<=self.open_loop_crps_horizon<=60
            or type(self.path_log_rv_crps_horizon) is not int
            or not 1<=self.path_log_rv_crps_horizon<=60
            or not path_horizons
            or tuple(sorted(set(path_horizons)))!=path_horizons
            or any(type(h) is not int or not 1<=h<=60 or h not in horizons
                   for h in path_horizons)
            or type(self.path_log_rv_crps_paths) is not int
            or not 2<=self.path_log_rv_crps_paths<=32
            or (self.forecast_objective=="path_log_rv_crps"
                and (self.path_log_rv_crps_horizon not in horizons
                     or self.path_log_rv_crps_horizon not in path_horizons))
            or self.forecast_student_t_df<=2.0
            or type(self.rollout_risk_conditioning) is not bool
            or (self.rollout_risk_conditioning
                and self.forecast_distribution=="mixture_gaussian")
            or not self.risk_horizons
            or tuple(sorted(set(horizons)))!=horizons
            or any(type(h) is not int or h<1 for h in horizons)):
            raise ValueError("invalid RSSM configuration")
        if (len(horizon_weights)!=len(horizons)
            or any(isinstance(weight,bool) or not isinstance(weight,(int,float))
                   or not math.isfinite(weight) or weight<=0 for weight in horizon_weights)):
            raise ValueError("risk horizon weights must be finite, positive and match the horizons")


def _weighted_risk_loss(continuous,barrier,weights):
    import jax.numpy as jnp
    per_horizon=continuous+barrier
    horizon_weights=jnp.asarray(weights,dtype=per_horizon.dtype)
    return (per_horizon*horizon_weights).sum()/horizon_weights.sum()


def _risk_conditioned_return_scale(prior_scale,risk_mean,risk_scale,feature_std,
                                   strength:float=1.0,student_t_df:float|None=None):
    """Match one-step raw log-return spread to the prior log-RV forecast.

    Channels 0/1 are the opening-gap and close-body log returns. Their frozen
    Train standard deviations convert their Gaussian scales back to raw return
    units. The log-normal mean of the native one-minute RV head gives expected
    absolute return; the distribution-specific E|X| / std ratio maps it to the
    marginal return standard deviation.
    """
    import jax.numpy as jnp
    if student_t_df is not None and (isinstance(student_t_df,bool)
            or not isinstance(student_t_df,(int,float))
            or not math.isfinite(student_t_df) or student_t_df<=2.0):
        raise ValueError("Student-t degrees of freedom must be finite and above two")
    output_dtype=prior_scale.dtype
    prior_scale=prior_scale.astype(jnp.float32)
    risk_mean=risk_mean.astype(jnp.float32)
    risk_scale=risk_scale.astype(jnp.float32)
    feature_std=jnp.asarray(feature_std,dtype=jnp.float32)
    log_rv_mean=risk_mean[...,0,1]
    log_rv_scale=risk_scale[...,0,1]
    expected_rv=jnp.maximum(
        jnp.exp(log_rv_mean+0.5*jnp.square(log_rv_scale))-1e-8,1e-12)
    if student_t_df is None:
        target_return_std=expected_rv*math.sqrt(math.pi/2.0)
    else:
        expected_abs_unit_std=(math.sqrt(student_t_df-2.0)
            *math.gamma((student_t_df-1.0)/2.0)
            /(math.sqrt(math.pi)*math.gamma(student_t_df/2.0)))
        target_return_std=expected_rv/expected_abs_unit_std
    raw_return_std=jnp.sqrt(jnp.maximum(
        jnp.square(prior_scale[...,0]*feature_std[0])
        +jnp.square(prior_scale[...,1]*feature_std[1]),1e-24))
    multiplier=target_return_std/raw_return_std
    tempered_multiplier=jnp.exp(jnp.asarray(strength,dtype=jnp.float32)
                                *jnp.log(jnp.maximum(multiplier,1e-12)))
    return prior_scale.at[...,:2].multiply(tempered_multiplier[...,None]).astype(output_dtype)


def import_pinned_upstream(path:str|Path):
    """Does not edit upstream files or let action values affect market RSSM."""
    path=Path(path).resolve()
    if not path.is_dir():
        raise RuntimeError("DreamerV3 checkout missing")
    proc=subprocess.run(["git","-C",str(path),"rev-parse","HEAD"],
                        capture_output=True,text=True,check=True)
    if proc.stdout.strip()!=PINNED_UPSTREAM:
        raise RuntimeError("upstream DreamerV3 checkout SHA mismatch")
    if str(path) not in sys.path:
        sys.path.insert(0,str(path))
    from dreamerv3.rssm import RSSM  # noqa: PLC0415
    return RSSM


def make_model(checkout:str|Path,config:RSSMConfig):
    """Return a Ninjax module with the pinned RSSM and causal OHLCV/risk heads."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    import elements
    # The pinned checkout provides ``embodied`` as a source package. Validate
    # and add that checkout to sys.path before importing any of its modules.
    RSSM=import_pinned_upstream(checkout)
    import embodied.jax.nets as nn  # type: ignore[import-not-found]

    class MarketWorldModel(nj.Module):
        def _rssm(self):
            return self.sub("market_dyn",RSSM,
                act_space={"market_dummy": elements.Space(np.float32, (1,))},
                deter=config.deter,hidden=config.hidden,stoch=config.stoch,
                classes=config.classes,blocks=config.blocks,
                obslayers=1,imglayers=1,dynlayers=1,free_nats=1.0)

        @staticmethod
        def _state_vector(feat):
            return jnp.concatenate([feat["deter"],
                                    feat["stoch"].reshape((*feat["stoch"].shape[:-2], -1))],axis=-1)

        def _head(self,z,prefix:str):
            mu=self.sub(prefix+"_mean",nn.Linear,5)(z)
            raw=self.sub(prefix+"_scale",nn.Linear,5)(z)
            scale=jax.nn.softplus(jnp.clip(raw,-12.,8.))+config.min_scale
            return mu,scale

        def _prior_head(self,z):
            if config.forecast_distribution!="mixture_gaussian":
                return self._head(z,"prior_forecast")
            components=2
            shape=(*z.shape[:-1],components,5)
            means=self.sub("prior_forecast_component_mean",nn.Linear,components*5)(z).reshape(shape)
            raw=self.sub("prior_forecast_component_scale",nn.Linear,components*5)(z).reshape(shape)
            scales=jax.nn.softplus(jnp.clip(raw,-12.,8.))+config.min_scale
            logits=self.sub("prior_forecast_mixture_logits",nn.Linear,components)(z)
            return {"means":means,"scales":scales,"logits":logits}

        def _prior_moments(self,distribution):
            if isinstance(distribution,dict):
                weights=jax.nn.softmax(
                    distribution["logits"].astype(jnp.float32),axis=-1)
                means=distribution["means"].astype(jnp.float32)
                scales=distribution["scales"].astype(jnp.float32)
                mean=(weights[...,None]*means).sum(axis=-2)
                variance=(weights[...,None]*(jnp.square(scales)
                          +jnp.square(means-mean[...,None,:]))).sum(axis=-2)
                return mean,jnp.sqrt(jnp.maximum(variance,1e-12))
            return distribution

        def _risk_head(self,z):
            horizons=len(config.risk_horizons)
            shape=(*z.shape[:-1],horizons,4)
            mu=self.sub("prior_risk_mean",nn.Linear,horizons*4)(z).reshape(shape)
            raw=self.sub("prior_risk_scale",nn.Linear,horizons*4)(z).reshape(shape)
            scale=jax.nn.softplus(jnp.clip(raw,-12.,8.))+config.min_scale
            barrier=self.sub("prior_risk_barrier",nn.Linear,horizons*2)(z).reshape(
                (*z.shape[:-1],horizons,2))
            return mu,scale,barrier

        def _prior_next_state(self,dyn,feat):
            """Sample p(z[t+1] | posterior[t]) without observing x[t+1]."""
            deter=feat["deter"][:,:-1]
            stoch=feat["stoch"][:,:-1]
            batch,time=deter.shape[:2]
            action={"market_dummy":jnp.zeros((batch,time,1),jnp.float32)}
            actemb=nn.DictConcat(dyn.act_space,1)(action)
            prior_deter=jax.vmap(
                lambda d,s,a:dyn._core(d,s,a),in_axes=(1,1,1),out_axes=1
            )(deter,stoch,actemb)
            # DreamerV3's imagined step explicitly casts transition values to
            # its bfloat16 compute dtype; the vectorized one-step equivalent
            # must preserve that contract before entering Ninjax Linear heads.
            prior_deter=nn.cast(prior_deter)
            prior_logits=jax.vmap(dyn._prior,in_axes=1,out_axes=1)(prior_deter)
            prior_logits=nn.cast(prior_logits)
            prior_stoch=nn.cast(dyn._dist(prior_logits).sample(seed=nj.seed()))
            z=jnp.concatenate([prior_deter,prior_stoch.reshape(
                (*prior_stoch.shape[:-2],-1))],axis=-1)
            return z,prior_logits

        def _prior_next_features(self,dyn,feat):
            return self._prior_next_state(dyn,feat)[0]

        @staticmethod
        def _nll(x,mu,scale):
            return (0.5*jnp.square((x-mu)/scale)+jnp.log(scale)+
                    0.5*math.log(2*math.pi)).sum(-1)

        def _prior_forecast_nll(self,x,distribution):
            if isinstance(distribution,dict):
                means=distribution["means"]
                scales=distribution["scales"]
                logits=distribution["logits"]
                component_nll=(0.5*jnp.square((x[...,None,:]-means)/scales)
                               +jnp.log(scales)+0.5*math.log(2*math.pi)).sum(-1)
                log_weights=jax.nn.log_softmax(logits.astype(jnp.float32),axis=-1)
                return -jax.scipy.special.logsumexp(log_weights-component_nll,axis=-1)
            mu,std=distribution
            if config.forecast_distribution=="gaussian":
                return self._nll(x,mu,std)
            df=config.forecast_student_t_df
            t_scale=std*math.sqrt((df-2.0)/df)
            constant=(math.lgamma(df/2.0)-math.lgamma((df+1.0)/2.0)
                      +0.5*math.log(df*math.pi))
            standardized=jnp.square((x-mu)/t_scale)
            return (constant+jnp.log(t_scale)
                    +0.5*(df+1.0)*jnp.log1p(standardized/df)).sum(-1)

        @staticmethod
        def _normal_crps(x,mu,scale):
            """Elementwise CRPS for a univariate Normal distribution."""
            z=(x-mu)/scale
            cdf=jax.scipy.special.ndtr(z)
            density=jnp.exp(-0.5*jnp.square(z))/math.sqrt(2.0*math.pi)
            return scale*(z*(2.0*cdf-1.0)+2.0*density-1.0/math.sqrt(math.pi))

        def _open_loop_prior_crps(self,dyn,feat,features,mask,reset):
            """Average per-step CRPS on free prior states from posterior anchors."""
            steps=config.open_loop_crps_horizon
            deter=feat["deter"][:,:-1]
            stoch=feat["stoch"][:,:-1]
            losses=[]
            for step in range(1,steps+1):
                batch,length=deter.shape[:2]
                action={"market_dummy":jnp.zeros((batch,length,1),jnp.float32)}
                action_embedding=nn.DictConcat(dyn.act_space,1)(action)
                prior_deter=jax.vmap(
                    lambda d,s,a:dyn._core(d,s,a),in_axes=(1,1,1),out_axes=1
                )(deter,stoch,action_embedding)
                prior_deter=nn.cast(prior_deter)
                prior_logits=jax.vmap(dyn._prior,in_axes=1,out_axes=1)(prior_deter)
                prior_logits=nn.cast(prior_logits)
                prior_stoch=nn.cast(dyn._dist(prior_logits).sample(seed=nj.seed()))
                z=jnp.concatenate([prior_deter,prior_stoch.reshape(
                    (*prior_stoch.shape[:-2],-1))],axis=-1)
                mean,scale=self._prior_head(z)
                target=features[:,step:,:]
                valid=mask[:,step:]*jnp.ones_like(mask[:,step:])
                for offset in range(1,step+1):
                    valid=valid*(~reset[:,offset:offset+length].astype(jnp.bool_)).astype(mask.dtype)
                crps=self._normal_crps(target,mean,scale).sum(-1)
                losses.append((crps*valid).sum()/jnp.maximum(valid.sum(),1.0))
                deter=prior_deter[:,:-1]
                stoch=prior_stoch[:,:-1]
            values=jnp.stack(losses)
            return values.mean(),values

        def _path_log_rv_crps(self,dyn,feat,features,mask,reset,
                              risk_targets,risk_mask,feature_mean,feature_std):
            """Sample CRPS over several free-prior path log-RV horizons."""
            horizons=config.path_log_rv_crps_horizons
            max_horizon=max(horizons)
            anchor_step=min(horizons)
            paths=config.path_log_rv_crps_paths
            if feature_mean is None or feature_std is None:
                raise ValueError("path log-RV CRPS requires frozen Train scaler moments")
            feature_mean=jnp.asarray(feature_mean,dtype=jnp.float32)
            feature_std=jnp.asarray(feature_std,dtype=jnp.float32)
            if (feature_mean.shape!=(config.feature_dim,)
                    or feature_std.shape!=(config.feature_dim,)):
                raise ValueError("path log-RV CRPS requires valid frozen Train scaler moments")
            batch,time=features.shape[:2]
            anchor_count=time-max_horizon
            if anchor_count<1:
                raise ValueError("path log-RV CRPS horizon must be shorter than sequence")
            anchor_indices=jnp.arange(0,anchor_count,anchor_step,dtype=jnp.int32)
            deter=feat["deter"][:,:-1][:,anchor_indices]
            stoch=feat["stoch"][:,:-1][:,anchor_indices]
            anchor_count=anchor_indices.shape[0]
            deter=jnp.tile(deter,(paths,1,1)).reshape(
                (paths*batch*anchor_count,-1))
            stoch=jnp.tile(stoch,(paths,1,1,1)).reshape(
                (paths*batch*anchor_count,*stoch.shape[2:]))
            returns=[]
            action={"market_dummy":jnp.zeros((paths*batch*anchor_count,1),jnp.float32)}
            action_embedding=nn.DictConcat(dyn.act_space,1)(action)
            for _ in range(max_horizon):
                deter=dyn._core(deter,stoch,action_embedding)
                deter=nn.cast(deter)
                logits=nn.cast(dyn._prior(deter))
                stoch=nn.cast(dyn._dist(logits).sample(seed=nj.seed()))
                z=jnp.concatenate([deter,stoch.reshape((stoch.shape[0],-1))],axis=-1)
                mean,scale=self._prior_head(z)
                observation=mean+scale*jax.random.normal(nj.seed(),mean.shape)
                raw=observation*feature_std+feature_mean
                returns.append(raw[:,0]+raw[:,1])
            sample_returns=jnp.stack(returns,axis=0).reshape(
                (max_horizon,paths,batch,anchor_count)).transpose(2,3,1,0)
            horizon_losses=[]
            valid_counts=[]
            off_diagonal=None
            for horizon in horizons:
                horizon_index=config.risk_horizons.index(horizon)
                sample_log_rv=jnp.log(
                    jnp.sqrt(jnp.square(sample_returns[...,:horizon]).sum(-1))+1e-8)
                truth=risk_targets[:,anchor_indices,horizon_index,1]
                valid=(mask[:,anchor_indices].astype(jnp.float32)
                       *risk_mask[:,anchor_indices,horizon_index].astype(jnp.float32))
                for offset in range(1,horizon+1):
                    valid=valid*(~reset[:,anchor_indices+offset]).astype(jnp.float32)
                first=jnp.abs(sample_log_rv-truth[...,None]).mean(-1)
                pairwise=jnp.abs(sample_log_rv[..., :,None]-sample_log_rv[...,None,:])
                off_diagonal=(1.0-jnp.eye(paths,dtype=pairwise.dtype))
                second=(pairwise*off_diagonal).sum((-1,-2))/(2.0*paths*(paths-1))
                per_anchor=first-second
                count=valid.sum()
                horizon_losses.append(
                    (per_anchor*valid).sum()/jnp.maximum(count,1.0))
                valid_counts.append(count)
            return jnp.stack(horizon_losses).mean(),jnp.stack(valid_counts).mean()

        def _sample_prior_forecast(self,distribution):
            if isinstance(distribution,dict):
                selector=jax.nn.one_hot(
                    jax.random.categorical(nj.seed(),distribution["logits"],axis=-1),
                    2,dtype=distribution["means"].dtype)[...,None]
                mu=(distribution["means"]*selector).sum(axis=-2)
                scale=(distribution["scales"]*selector).sum(axis=-2)
                return mu+scale*jax.random.normal(nj.seed(),mu.shape)
            mu,std=distribution
            if config.forecast_distribution=="gaussian":
                return mu+std*jax.random.normal(nj.seed(),mu.shape)
            df=config.forecast_student_t_df
            t_scale=std*math.sqrt((df-2.0)/df)
            return mu+t_scale*jax.random.t(nj.seed(),df,mu.shape)

        def __call__(self,features,reset,mask,training=True,
                     risk_targets=None,risk_mask=None,
                     feature_mean=None,feature_std=None):
            """Batch teacher-forced representation + strictly shifted next-bar NLL.

            features[t+1] is a supervised label, NEVER fed to head(pred_t).
            RSSM observes features[t] to produce posterior[t].
            """
            features=jnp.asarray(features,dtype=jnp.float32)
            reset=jnp.asarray(reset,dtype=jnp.bool_)
            mask=jnp.asarray(mask,dtype=jnp.float32)
            B,T,D=features.shape
            if D!=config.feature_dim:
                raise ValueError("feature dimension mismatch")
            if (config.forecast_objective=="open_loop_crps"
                    and T<=config.open_loop_crps_horizon):
                raise ValueError("open-loop CRPS horizon must be shorter than the Train sequence")
            if (config.forecast_objective=="path_log_rv_crps"
                    and T<=max(config.path_log_rv_crps_horizons)):
                raise ValueError("path log-RV CRPS horizons must be shorter than the Train sequence")
            dyn=self._rssm()
            action={"market_dummy":jnp.zeros((B,T,1),dtype=jnp.float32)}
            carry=dyn.initial(B)
            carry,entries,kl,feat,metrics=dyn.loss(carry,features,action,reset,training)
            z=self._state_vector(feat)
            prior_z=self._prior_next_features(dyn,feat)
            prior_distribution=self._prior_head(prior_z)
            prior_mean,prior_scale=self._prior_moments(prior_distribution)
            risk_mean,risk_scale,risk_logits=self._risk_head(prior_z)
            posterior_mean,posterior_scale=self._head(z[:,:-1],"posterior_aux")
            rec_mean,rec_scale=self._head(z,"reconstruct")
            nll_prior=self._prior_forecast_nll(features[:,1:,:],prior_distribution)
            if config.forecast_distribution=="gaussian":
                prior_mean,prior_scale=self._prior_moments(prior_distribution)
                crps_prior=self._normal_crps(features[:,1:,:],prior_mean,prior_scale).sum(-1)
            else:
                crps_prior=jnp.zeros_like(nll_prior)
            nll_posterior=self._nll(features[:,1:,:],posterior_mean,posterior_scale)
            nll_rec=self._nll(features,rec_mean,rec_scale)
            # Mask describes the TARGET at t+1. The observation at t may be
            # burn-in (visible and state-forming) without contributing loss.
            # A prediction must not cross an episode reset boundary.
            valid_pair=mask[:,1:]*(~reset[:,1:]).astype(mask.dtype)
            loss_prior=(nll_prior*valid_pair).sum()/jnp.maximum(valid_pair.sum(),1.0)
            loss_crps=(crps_prior*valid_pair).sum()/jnp.maximum(valid_pair.sum(),1.0)
            open_loop_crps=loss_crps
            open_loop_crps_by_horizon=jnp.zeros(
                (config.open_loop_crps_horizon,),dtype=loss_crps.dtype)
            path_log_rv_crps=jnp.asarray(0.0,dtype=loss_crps.dtype)
            path_log_rv_count=jnp.asarray(0.0,dtype=loss_crps.dtype)
            if config.forecast_objective=="open_loop_crps":
                open_loop_crps,open_loop_crps_by_horizon=self._open_loop_prior_crps(
                    dyn,feat,features,mask,reset)
            elif config.forecast_objective=="path_log_rv_crps":
                # `training` is dynamic under Ninjax/JIT. The inactive branch
                # keeps inference initialization free of Train-only path labels.
                default_mean=jnp.zeros((config.feature_dim,),dtype=jnp.float32)
                default_std=jnp.ones((config.feature_dim,),dtype=jnp.float32)
                scaler_mean=default_mean if feature_mean is None else feature_mean
                scaler_std=default_std if feature_std is None else feature_std
                path_targets=(jnp.zeros((B,T,len(config.risk_horizons),6),
                                         dtype=jnp.float32)
                              if risk_targets is None else risk_targets)
                path_mask=(jnp.zeros((B,T,len(config.risk_horizons)),
                                      dtype=jnp.bool_)
                           if risk_mask is None else risk_mask)
                path_log_rv_crps,path_log_rv_count=jax.lax.cond(
                    jnp.asarray(training,dtype=jnp.bool_),
                    lambda _:self._path_log_rv_crps(
                        dyn,feat,features,mask,reset,path_targets,path_mask,
                        scaler_mean,scaler_std),
                    lambda _:(jnp.zeros_like(loss_crps),jnp.zeros_like(loss_crps)),
                    operand=None)
            if config.forecast_objective=="nll":
                objective_prior=loss_prior
            elif config.forecast_objective=="crps":
                objective_prior=loss_crps
            elif config.forecast_objective=="path_log_rv_crps":
                objective_prior=(loss_prior+config.path_log_rv_crps_scale
                                 *jnp.where(jnp.asarray(training,dtype=jnp.bool_),
                                            path_log_rv_crps,0.0))
            else:
                objective_prior=open_loop_crps
            loss_posterior=(nll_posterior*valid_pair).sum()/jnp.maximum(valid_pair.sum(),1.0)
            loss_rec=(nll_rec*mask).sum()/jnp.maximum(mask.sum(),1.0)
            loss_dyn=(kl['dyn']*mask).sum()/jnp.maximum(mask.sum(),1.0)
            loss_rep=(kl['rep']*mask).sum()/jnp.maximum(mask.sum(),1.0)
            H=len(config.risk_horizons)
            if risk_targets is None:
                risk_targets=jnp.zeros((B,T,H,6),dtype=jnp.float32)
            else:
                risk_targets=jnp.asarray(risk_targets,dtype=jnp.float32)
            if risk_mask is None:
                risk_mask=jnp.zeros((B,T,H),dtype=jnp.float32)
            else:
                risk_mask=jnp.asarray(risk_mask,dtype=jnp.float32)
            if risk_targets.shape!=(B,T,H,6) or risk_mask.shape!=(B,T,H):
                raise ValueError("multi-horizon risk target shape mismatch")
            valid_risk=risk_mask[:,:-1]*mask[:,:-1,None]
            continuous=self._nll(risk_targets[:,:-1,:,:4],risk_mean,risk_scale)
            labels=risk_targets[:,:-1,:,4:6]
            barrier_bce=(jax.nn.softplus(risk_logits)-labels*risk_logits).sum(-1)
            horizon_counts=valid_risk.sum((0,1))
            risk_cont_by_h=(continuous*valid_risk).sum((0,1))/jnp.maximum(horizon_counts,1.0)
            risk_barrier_by_h=(barrier_bce*valid_risk).sum((0,1))/jnp.maximum(horizon_counts,1.0)
            loss_risk=_weighted_risk_loss(
                risk_cont_by_h,risk_barrier_by_h,config.risk_horizon_weights)
            total=(config.prediction_scale*objective_prior +
                   config.posterior_auxiliary_scale*loss_posterior +
                   config.reconstruction_scale*loss_rec +
                   config.dynamics_scale*loss_dyn+
                   config.representation_scale*loss_rep+
                   config.risk_prediction_scale*loss_risk)
            return {"loss":total,"prior_forecast_nll":loss_prior,
                    "prior_forecast_crps":loss_crps,
                    "open_loop_prior_crps":open_loop_crps,
                    "open_loop_prior_crps_by_horizon":open_loop_crps_by_horizon,
                    "path_log_rv_crps":path_log_rv_crps,
                    "valid_path_log_rv_crps":path_log_rv_count,
                    "prior_forecast_objective":objective_prior,
                    "prior_risk_nll":loss_risk,
                    "risk_continuous_nll_by_horizon":risk_cont_by_h,
                    "risk_barrier_bce_by_horizon":risk_barrier_by_h,
                    "valid_risk_by_horizon":horizon_counts,
                    "latent_feature_variance":jnp.var(z,axis=(0,1)).mean(),
                    "posterior_aux_nll":loss_posterior,"pred_nll":loss_prior,"rec_nll":loss_rec,
                    "dyn_kl":loss_dyn,"rep_kl":loss_rep,
                    "valid_predictions":valid_pair.sum(),
                    "mean_prior_scale":prior_scale.mean()}

        def prior_forecast_sequence(self,history):
            """Return each causal one-step prior distribution [B,T-1,5]."""
            x=jnp.asarray(history,dtype=jnp.float32)
            B,T,D=x.shape
            if D!=config.feature_dim or T<2:
                raise ValueError("history must have shape [batch,time>=2,5]")
            dyn=self._rssm()
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            reset=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            carry=dyn.initial(B)
            carry,entries,losses,feat,metrics=dyn.loss(
                carry,x,actions,reset,False)
            del carry,entries,losses,metrics
            prior_z=self._prior_next_features(dyn,feat)
            return self._prior_moments(self._prior_head(prior_z))

        def alignment_distributions(self,history):
            """Return aligned prior/posterior forecast distributions and RSSM logits."""
            x=jnp.asarray(history,dtype=jnp.float32)
            B,T,D=x.shape
            if D!=config.feature_dim or T<2:
                raise ValueError("history must have shape [batch,time>=2,5]")
            dyn=self._rssm()
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            reset=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            carry=dyn.initial(B)
            carry,entries,losses,feat,metrics=dyn.loss(
                carry,x,actions,reset,False)
            del carry,entries,losses,metrics
            prior_z,prior_logits=self._prior_next_state(dyn,feat)
            prior_mean,prior_scale=self._prior_moments(self._prior_head(prior_z))
            posterior_mean,posterior_scale=self._head(self._state_vector(feat)[:,:-1],
                                                      "posterior_aux")
            posterior_logits=feat["logit"][:,1:]
            return {"prior_mean":prior_mean,"prior_scale":prior_scale,
                    "posterior_mean":posterior_mean,"posterior_scale":posterior_scale,
                    "prior_logits":prior_logits,"posterior_logits":posterior_logits}

        def rollout(self,history,steps:int,feature_std=None):
            """Generate strict F1 open-loop prior standardized feature samples.

            Output [B,H,5]. Input [B,T,5] is observable historical context.
            At each step, predict next-bar distribution from current state,
            sample a conditional observation, then advance *prior* state. When
            ``rollout_risk_conditioning`` is enabled, the forecast marginal
            return scale on the opening-gap/body coordinates is conditioned on
            the prior head's one-minute log-RV forecast. Gaussian and
            variance-normalized Student-t priors are supported. Frozen Train
            scaler moments convert those coordinates to raw return units.
            Samples are NOT fed back into the state, by design for F1.
            No true future observations are used.
            """
            x=jnp.asarray(history,dtype=jnp.float32)
            B,T,D=x.shape
            if config.rollout_risk_conditioning:
                if feature_std is None:
                    raise ValueError("risk-conditioned rollout requires frozen Train feature_std")
                feature_std=jnp.asarray(feature_std,dtype=jnp.float32)
                if feature_std.shape!=(config.feature_dim,):
                    raise ValueError("risk-conditioned rollout feature_std must have shape [5]")
            dyn=self._rssm()
            carry=dyn.initial(B)
            zero={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            reset=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            carry,entries,feat=dyn.observe(carry,x,zero,reset,False)
            f=jax.tree.map(lambda arr:arr[:,-1],feat)
            outcomes=[]
            for _ in range(steps):
                carry,(f,_)=dyn.imagine(
                    carry,{"market_dummy":jnp.zeros((B,1),jnp.float32)},
                    length=1,training=False,single=True)
                distribution=self._prior_head(self._state_vector(f))
                if config.rollout_risk_conditioning:
                    if isinstance(distribution,dict):
                        raise ValueError("risk-conditioned rollout requires a Gaussian prior head")
                    _,prior_scale=distribution
                    risk_mean,risk_scale,_=self._risk_head(self._state_vector(f))
                    distribution=(distribution[0],_risk_conditioned_return_scale(
                        prior_scale,risk_mean,risk_scale,feature_std,
                        config.rollout_risk_conditioning_strength,
                        config.forecast_student_t_df
                        if config.forecast_distribution=="student_t" else None))
                observation=self._sample_prior_forecast(distribution)
                outcomes.append(observation)
            return jnp.stack(outcomes,axis=1)

        def next_distribution(self,history):
            """Expose the configured prior location and marginal standard deviation."""
            x=jnp.asarray(history,dtype=jnp.float32)
            B,T,D=x.shape
            if D!=config.feature_dim or T<2:
                raise ValueError("history must have shape [batch,time>=2,5]")
            dyn=self._rssm()
            action={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            reset=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            carry=dyn.initial(B)
            carry,entries,feat=dyn.observe(carry,x,action,reset,False)
            del entries
            # The forecast head is evaluated on a sampled p(z[t+1] | q(z[t]))
            # state, matching both its direct training target and F1 rollout.
            carry,(prior_feat,_)=dyn.imagine(
                carry,{"market_dummy":jnp.zeros((B,1),jnp.float32)},
                length=1,training=False,single=True)
            del carry
            distribution=self._prior_head(self._state_vector(prior_feat))
            if isinstance(distribution,dict):
                mean,scale=self._prior_moments(distribution)
                return {"mean":mean,"scale":scale,
                        "mixture_means":distribution["means"].astype(jnp.float32),
                        "mixture_scales":distribution["scales"].astype(jnp.float32),
                        "mixture_weights":jax.nn.softmax(
                            distribution["logits"].astype(jnp.float32),axis=-1)}
            return distribution

        def next_risk_distribution(self,history):
            """Expose checkpoint-trained multi-horizon risk forecasts from the causal prior.

            Continuous targets are ordered as return, realized volatility, maximum
            high excursion, and minimum low excursion for each configured horizon.
            Barrier logits are ordered upper then lower. The input contains history
            only; no future observations enter the prior state.
            """
            x=jnp.asarray(history,dtype=jnp.float32)
            B,T,D=x.shape
            if D!=config.feature_dim or T<2:
                raise ValueError("history must have shape [batch,time>=2,5]")
            dyn=self._rssm()
            action={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            reset=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            carry=dyn.initial(B)
            carry,entries,feat=dyn.observe(carry,x,action,reset,False)
            del entries
            carry,(prior_feat,_)=dyn.imagine(
                carry,{"market_dummy":jnp.zeros((B,1),jnp.float32)},
                length=1,training=False,single=True)
            del carry
            return self._risk_head(self._state_vector(prior_feat))

    return MarketWorldModel(name="route_b_market_world_model")
