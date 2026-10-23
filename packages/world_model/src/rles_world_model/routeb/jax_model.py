"""Additive JAX world-model module using the pinned *unmodified* DreamerV3 RSSM.

Run ONLY inside packages/world_model/environment (Python 3.12). Model learns
current reconstruction, next-bar conditional distribution and Dreamer RSSM KL.
Market actions are constant zero even when trading actions change.

The B3 trainer lives in ``jax_hierarchy_train`` and has its own provenance-
bound checkpoints. Do not infer B3 learning quality from synthetic smoke runs.
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
    min_scale: float=0.02

    def __post_init__(self)->None:
        if (any(type(x) is not int for x in (self.deter,self.hidden,self.stoch,
                                                  self.classes,self.blocks,self.feature_dim))
            or self.deter<8 or self.hidden<8 or self.stoch<1 or self.classes<2
            or self.blocks<1 or self.deter%self.blocks!=0 or self.feature_dim!=5
            or any(not math.isfinite(x) or x<=0 for x in (
                self.dynamics_scale,self.representation_scale,
                self.reconstruction_scale,self.prediction_scale,self.min_scale))):
            raise ValueError("invalid RSSM configuration")


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
    """Return a Ninjax module holding pinned RSSM and two normal forecast heads."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    import elements
    # The pinned checkout provides ``embodied`` as a source package. Validate
    # and add that checkout to sys.path before importing any of its modules.
    RSSM=import_pinned_upstream(checkout)
    import embodied.jax.nets as nn

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

        @staticmethod
        def _nll(x,mu,scale):
            return (0.5*jnp.square((x-mu)/scale)+jnp.log(scale)+
                    0.5*math.log(2*math.pi)).sum(-1)

        def __call__(self,features,reset,mask,training=True):
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
            dyn=self._rssm()
            action={"market_dummy":jnp.zeros((B,T,1),dtype=jnp.float32)}
            carry=dyn.initial(B)
            carry,entries,kl,feat,metrics=dyn.loss(carry,features,action,reset,training)
            z=self._state_vector(feat)
            pred_mean,pred_scale=self._head(z,"next")
            rec_mean,rec_scale=self._head(z,"reconstruct")
            nll_pred=self._nll(features[:,1:,:],pred_mean[:,:-1,:],pred_scale[:,:-1,:])
            nll_rec=self._nll(features,rec_mean,rec_scale)
            # Mask describes the TARGET at t+1. The observation at t may be
            # burn-in (visible and state-forming) without contributing loss.
            # A prediction must not cross an episode reset boundary.
            valid_pair=mask[:,1:]*(~reset[:,1:]).astype(mask.dtype)
            loss_pred=(nll_pred*valid_pair).sum()/jnp.maximum(valid_pair.sum(),1.0)
            loss_rec=(nll_rec*mask).sum()/jnp.maximum(mask.sum(),1.0)
            loss_dyn=(kl['dyn']*mask).sum()/jnp.maximum(mask.sum(),1.0)
            loss_rep=(kl['rep']*mask).sum()/jnp.maximum(mask.sum(),1.0)
            total=(config.prediction_scale*loss_pred +
                   config.reconstruction_scale*loss_rec +
                   config.dynamics_scale*loss_dyn+
                   config.representation_scale*loss_rep)
            return {"loss":total,"pred_nll":loss_pred,"rec_nll":loss_rec,
                    "dyn_kl":loss_dyn,"rep_kl":loss_rep,
                    "valid_predictions":valid_pair.sum(),
                    "mean_pred_scale":pred_scale.mean()}

        def rollout(self,history,steps:int):
            """Generate strict F1 open-loop prior standardized feature samples.

            Output [B,H,5]. Input [B,T,5] is observable historical context.
            At each step, predict next-bar distribution from current state,
            sample a conditional observation, then advance *prior* state.
            Samples are NOT fed back into the state, by design for F1.
            No true future observations are used.
            """
            x=jnp.asarray(history,dtype=jnp.float32)
            B,T,D=x.shape
            dyn=self._rssm()
            carry=dyn.initial(B)
            zero={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            reset=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            carry,entries,feat=dyn.observe(carry,x,zero,reset,False)
            f=jax.tree.map(lambda arr:arr[:,-1],feat)
            outcomes=[]
            for _ in range(steps):
                loc,scale=self._head(self._state_vector(f),"next")
                observation=loc+scale*jax.random.normal(nj.seed(),loc.shape)
                outcomes.append(observation)
                carry,(f,_) =dyn.imagine(
                    carry,{"market_dummy":jnp.zeros((B,1),jnp.float32)},
                    length=1,training=False,single=True)
            return jnp.stack(outcomes,axis=1)

        def next_distribution(self,history):
            """Expose the exact one-step diagonal Gaussian head from anchor state."""
            x=jnp.asarray(history,dtype=jnp.float32)
            B,T,D=x.shape
            if D!=config.feature_dim or T<2:
                raise ValueError("history must have shape [batch,time>=2,5]")
            dyn=self._rssm()
            action={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            reset=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            carry=dyn.initial(B)
            carry,entries,feat=dyn.observe(carry,x,action,reset,False)
            del carry,entries
            last=jax.tree.map(lambda arr:arr[:,-1],feat)
            return self._head(self._state_vector(last),"next")

    return MarketWorldModel(name="route_b_market_world_model")
