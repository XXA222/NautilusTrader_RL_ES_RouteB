"""EXPERIMENTAL B3: three trainable, causally-indexed upstream RSSM levels.

This module provides a JAX loss network, NOT a validated B3 optimizer, fully
recurrent F2 sampler, or economic result. Future high-timeframe states are
rejected by host-side build_plan and gathered using only visible indices.

Keep B3 disabled in production until an end-to-end B3 smoke and E3/E4 ablation.
"""
from __future__ import annotations
from .jax_model import RSSMConfig, import_pinned_upstream


def masked_weighted_kl(dyn_kl, rep_kl, mask, *, dynamics_scale: float,
                       representation_scale: float):
    """Weight each per-update KL term exactly once over its valid updates."""
    import jax.numpy as jnp

    weights = jnp.asarray(mask, dtype=jnp.float32)
    denominator = jnp.maximum(weights.sum(), 1.0)
    dyn = (jnp.asarray(dyn_kl) * weights).sum() / denominator
    rep = (jnp.asarray(rep_kl) * weights).sum() / denominator
    return dynamics_scale * dyn + representation_scale * rep


def make_hierarchical_model(checkout,*,minute:RSSMConfig,hour:RSSMConfig,day:RSSMConfig):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    import elements
    RSSM=import_pinned_upstream(checkout)
    # The pinned checkout adds its vendored ``embodied`` package to sys.path.
    import embodied.jax.nets as nn

    class ThreeScale(nj.Module):
        def _dyn(self,name,cfg):
            return self.sub(name,RSSM,act_space={"market_dummy":elements.Space(np.float32,(1,))},
                            deter=cfg.deter,hidden=cfg.hidden,stoch=cfg.stoch,
                            classes=cfg.classes,blocks=cfg.blocks,
                            obslayers=1,imglayers=1,dynlayers=1,free_nats=1.0)

        def _features(self,features,name,cfg):
            B,T,_=features.shape
            r=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            dyn=self._dyn(name,cfg)
            state,entries,kls,feat,metrics=dyn.loss(dyn.initial(B),features,actions,r,True)
            latent=jnp.concatenate([feat['deter'],
                  feat['stoch'].reshape((B,T,-1))],axis=-1)
            return latent,kls

        @staticmethod
        def _causal_gather(latent,indices):
            B,T,D=latent.shape
            indices=jnp.asarray(indices,dtype=jnp.int32)
            safe=jnp.maximum(indices,0)
            pulled=jnp.take_along_axis(latent,safe[:,:,None],axis=1)
            return jnp.where((indices>=0)[:,:,None],pulled,0.0)

        def __call__(self,minute_x,hour_x,day_x,hour_index,day_index,
                     minute_mask,hour_mask,day_mask):
            """Tensors: x scale [B,T_s,5], index [B,T_min] (-1 or observed state).

            Caller MUST build indexes from UTC event-close and actual availability.
            """
            minute_x=jnp.asarray(minute_x,dtype=jnp.float32)
            hour_x=jnp.asarray(hour_x,dtype=jnp.float32)
            day_x=jnp.asarray(day_x,dtype=jnp.float32)
            hour_index=jnp.asarray(hour_index,dtype=jnp.int32)
            day_index=jnp.asarray(day_index,dtype=jnp.int32)
            if (hour_x.shape[1]<2 or day_x.shape[1]<2 or
                minute_x.shape[2]!=5 or hour_x.shape[2]!=5 or day_x.shape[2]!=5):
                raise ValueError("B3 needs nonempty correctly shaped native-scale batches")
            h_latent,h_kl=self._features(hour_x,'hour_dyn',hour)
            d_latent,d_kl=self._features(day_x,'day_dyn',day)
            h_visible=self._causal_gather(h_latent,hour_index)
            d_visible=self._causal_gather(d_latent,day_index)
            joined=jnp.concatenate([minute_x,h_visible,d_visible,
                                     (hour_index>=0)[:,:,None].astype(jnp.float32),
                                     (day_index>=0)[:,:,None].astype(jnp.float32)],axis=-1)
            tokens=self.sub('minute_conditioned',nn.Linear,minute.hidden)(nn.cast(joined))
            B,T,_=minute_x.shape
            resets=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            mdyn=self._dyn('minute_dyn',minute)
            _,_,m_kl,m_feat,_=mdyn.loss(mdyn.initial(B),tokens,actions,resets,True)
            mz=jnp.concatenate([m_feat['deter'],
                                  m_feat['stoch'].reshape((B,T,-1))],axis=-1)
            pred=self.sub('next_minute_mean',nn.Linear,5)(mz)
            raw_scale=self.sub('next_minute_scale',nn.Linear,5)(mz)
            scale=jax.nn.softplus(jnp.clip(raw_scale,-12.,8.))+minute.min_scale
            loss_per_dim=(.5*jnp.square((minute_x[:,1:]-pred[:,:-1])/scale[:,:-1])+
                          jnp.log(scale[:,:-1])+.5*1.8378770664093453)
            # Target t+1 may be scored even when t was burn-in.
            pair=jnp.asarray(minute_mask[:,1:],dtype=jnp.float32)
            nll=(loss_per_dim.sum(-1)*pair).sum()/jnp.maximum(1.0,pair.sum())
            def scale_aux(latent,source,kl,mask,cfg,prefix):
                reconstruction=self.sub(prefix+'_reconstruction',nn.Linear,5)(latent)
                mse=(jnp.square(source-reconstruction).sum(-1)*mask).sum()/jnp.maximum(mask.sum(),1)
                dyn=(kl['dyn']*mask).sum()/jnp.maximum(mask.sum(),1)
                rep=(kl['rep']*mask).sum()/jnp.maximum(mask.sum(),1)
                return cfg.reconstruction_scale*mse + cfg.dynamics_scale*dyn + cfg.representation_scale*rep
            m_kl_mask=jnp.asarray(minute_mask,dtype=jnp.float32)
            h_mask=jnp.asarray(hour_mask,dtype=jnp.float32)
            d_mask=jnp.asarray(day_mask,dtype=jnp.float32)
            mkl=masked_weighted_kl(
                m_kl['dyn'],m_kl['rep'],m_kl_mask,
                dynamics_scale=minute.dynamics_scale,
                representation_scale=minute.representation_scale)
            aux=scale_aux(h_latent,hour_x,h_kl,h_mask,hour,'hour')+scale_aux(d_latent,day_x,d_kl,d_mask,day,'day')
            return {'loss':minute.prediction_scale*nll + mkl + aux,
                    'minute_nll':nll, 'slow_aux':aux}

    return ThreeScale(name='route_b_hierarchical_world_model')
