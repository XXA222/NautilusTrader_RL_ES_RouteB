"""EXPERIMENTAL B3: three trainable, causally-indexed upstream RSSM levels.

This module provides the trainable B3 loss network and an F2 latent-prior
rollout. It is NOT an economically validated model. Training and rollout use
host-built causal indexes/event plans; future high-timeframe observations are
never accepted by the F2 sampling API.

Keep B3 disabled in production until F2 event-plan perturbation tests and
independent E3/E4 ablations complete.
"""
from __future__ import annotations
from .jax_model import RSSMConfig, import_pinned_upstream


def make_hierarchical_model(checkout,*,minute:RSSMConfig,hour:RSSMConfig,day:RSSMConfig):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    import elements
    RSSM=import_pinned_upstream(checkout)
    import embodied.jax.nets as nn

    class ThreeScale(nj.Module):
        def _dyn(self,name,cfg,slow_context_dim=0):
            action_space={"market_dummy":elements.Space(np.float32,(1,))}
            if slow_context_dim:
                action_space["slow_context"]=elements.Space(np.float32,(slow_context_dim,))
            return self.sub(name,RSSM,act_space=action_space,
                            deter=cfg.deter,hidden=cfg.hidden,stoch=cfg.stoch,
                            classes=cfg.classes,blocks=cfg.blocks,
                            obslayers=1,imglayers=1,dynlayers=1,free_nats=1.0)

        def _features(self,features,name,cfg,slow_context=None,reset=None):
            B,T,_=features.shape
            r=(jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
               if reset is None else jnp.asarray(reset,dtype=jnp.bool_))
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            if slow_context is not None:
                actions["slow_context"]=jnp.asarray(slow_context,dtype=jnp.float32)
            dyn=self._dyn(name,cfg,0 if slow_context is None else slow_context.shape[-1])
            state,entries,kls,feat,metrics=dyn.loss(dyn.initial(B),features,actions,r,True)
            latent=jnp.concatenate([feat['deter'],
                  feat['stoch'].reshape((B,T,-1))],axis=-1)
            return latent,kls,state

        @staticmethod
        def _causal_gather(latent,indices):
            B,T,D=latent.shape
            indices=jnp.asarray(indices,dtype=jnp.int32)
            safe=jnp.maximum(indices,0)
            pulled=jnp.take_along_axis(latent,safe[:,:,None],axis=1)
            return jnp.where((indices>=0)[:,:,None],pulled,0.0)

        def __call__(self,minute_x,hour_x,day_x,hour_index,day_index,
                     minute_mask,hour_mask,day_mask,minute_reset=None):
            """Tensors: x scale [B,T_s,5], index [B,T_min] (-1 or observed state).

            Caller MUST build indexes from UTC event-close and actual availability.
            """
            minute_x=jnp.asarray(minute_x,dtype=jnp.float32)
            hour_x=jnp.asarray(hour_x,dtype=jnp.float32)
            day_x=jnp.asarray(day_x,dtype=jnp.float32)
            hour_index=jnp.asarray(hour_index,dtype=jnp.int32)
            day_index=jnp.asarray(day_index,dtype=jnp.int32)
            B,T,_=minute_x.shape
            # Slow streams are shared across sampled minute windows. Broadcast
            # them per window so each batch carries its own causal index map.
            if hour_x.shape[0]==1 and B!=1:
                hour_x=jnp.broadcast_to(hour_x,(B,*hour_x.shape[1:]))
                hour_mask=jnp.broadcast_to(hour_mask,(B,*hour_mask.shape[1:]))
            if day_x.shape[0]==1 and B!=1:
                day_x=jnp.broadcast_to(day_x,(B,*day_x.shape[1:]))
                day_mask=jnp.broadcast_to(day_mask,(B,*day_mask.shape[1:]))
            if (hour_x.shape[1]<2 or day_x.shape[1]<2 or
                minute_x.shape[2]!=5 or hour_x.shape[2]!=5 or day_x.shape[2]!=5
                or hour_index.shape!=(B,T) or day_index.shape!=(B,T)
                or hour_x.shape[0]!=B or day_x.shape[0]!=B):
                raise ValueError("B3 needs nonempty correctly shaped native-scale batches")
            h_latent,h_kl,_=self._features(hour_x,'hour_dyn',hour)
            d_latent,d_kl,_=self._features(day_x,'day_dyn',day)
            h_visible=self._causal_gather(h_latent,hour_index)
            d_visible=self._causal_gather(d_latent,day_index)
            joined=jnp.concatenate([minute_x,h_visible,d_visible,
                                     (hour_index>=0)[:,:,None].astype(jnp.float32),
                                     (day_index>=0)[:,:,None].astype(jnp.float32)],axis=-1)
            tokens=self.sub('minute_conditioned',nn.Linear,minute.hidden)(
                joined.astype(nn.COMPUTE_DTYPE))
            if minute_reset is None:
                resets=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            else:
                resets=jnp.asarray(minute_reset,dtype=jnp.bool_)
                if resets.shape!=(B,T):
                    raise ValueError("minute reset shape mismatch")
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            slow_context=jnp.concatenate([h_visible,d_visible],axis=-1)
            actions['slow_context']=slow_context
            mdyn=self._dyn('minute_dyn',minute,slow_context.shape[-1])
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
            mkl=((m_kl['dyn']+minute.representation_scale*m_kl['rep'])*m_kl_mask).sum()/jnp.maximum(1,m_kl_mask.sum())
            aux=scale_aux(h_latent,hour_x,h_kl,h_mask,hour,'hour')+scale_aux(d_latent,day_x,d_kl,d_mask,day,'day')
            return {'loss':minute.prediction_scale*nll + minute.dynamics_scale*mkl + aux,
                    'minute_nll':nll, 'slow_aux':aux}

        def rollout_f2(self,minute_x,hour_x,day_x,hour_index,day_index,
                       future_hour_events,future_day_events):
            """Generate a causal minute path while advancing slow RSSM priors.

            Event tuples state when a newly closed hour/day becomes available
            at each future decision. The caller derives them from UTC close plus
            exchange availability delay; future observations are never inputs.
            """
            minute_x=jnp.asarray(minute_x,dtype=jnp.float32)
            hour_x=jnp.asarray(hour_x,dtype=jnp.float32)
            day_x=jnp.asarray(day_x,dtype=jnp.float32)
            hour_index=jnp.asarray(hour_index,dtype=jnp.int32)
            day_index=jnp.asarray(day_index,dtype=jnp.int32)
            B,T,_=minute_x.shape
            if (T<2 or hour_x.shape[1]<2 or day_x.shape[1]<2
                    or hour_index.shape!=(B,T) or day_index.shape!=(B,T)
                    or len(future_hour_events)!=len(future_day_events)
                    or not future_hour_events):
                raise ValueError("F2 requires aligned histories and a nonempty future event plan")
            h_dyn=self._dyn('hour_dyn',hour)
            d_dyn=self._dyn('day_dyn',day)
            zero_h={"market_dummy":jnp.zeros((*hour_x.shape[:2],1),jnp.float32)}
            zero_d={"market_dummy":jnp.zeros((*day_x.shape[:2],1),jnp.float32)}
            hr=jnp.zeros(hour_x.shape[:2],bool).at[:,0].set(True)
            dr=jnp.zeros(day_x.shape[:2],bool).at[:,0].set(True)
            hcarry,_,hfeat=h_dyn.observe(h_dyn.initial(B),hour_x,zero_h,hr,False)
            dcarry,_,dfeat=d_dyn.observe(d_dyn.initial(B),day_x,zero_d,dr,False)
            h_history=jnp.concatenate([hfeat['deter'],hfeat['stoch'].reshape((B,hour_x.shape[1],-1))],-1)
            d_history=jnp.concatenate([dfeat['deter'],dfeat['stoch'].reshape((B,day_x.shape[1],-1))],-1)
            h_visible=self._causal_gather(h_history,hour_index)
            d_visible=self._causal_gather(d_history,day_index)
            joined=jnp.concatenate([minute_x,h_visible,d_visible,
                    (hour_index>=0)[:,:,None].astype(jnp.float32),
                    (day_index>=0)[:,:,None].astype(jnp.float32)],-1)
            tokens=self.sub('minute_conditioned',nn.Linear,minute.hidden)(joined.astype(nn.COMPUTE_DTYPE))
            slow_context=jnp.concatenate([h_visible,d_visible],-1)
            m_dyn=self._dyn('minute_dyn',minute,slow_context.shape[-1])
            mr=jnp.zeros((B,T),bool).at[:,0].set(True)
            m_actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32),
                       "slow_context":slow_context}
            mcarry,_,_=m_dyn.observe(m_dyn.initial(B),tokens,m_actions,mr,False)
            means=[];scales=[];hour_generated=[];day_generated=[]
            for hour_event,day_event in zip(future_hour_events,future_day_events):
                if day_event:
                    dcarry,(dprior,_)=d_dyn.imagine(dcarry,
                        {"market_dummy":jnp.zeros((B,1),jnp.float32)},1,False,single=True)
                    dz=jnp.concatenate([dprior['deter'],dprior['stoch'].reshape((B,-1))],-1)
                    day_generated.append(self.sub('day_reconstruction',nn.Linear,5)(dz))
                else:
                    day_generated.append(jnp.zeros((B,5),jnp.float32))
                if hour_event:
                    hcarry,(hprior,_)=h_dyn.imagine(hcarry,
                        {"market_dummy":jnp.zeros((B,1),jnp.float32)},1,False,single=True)
                    hz=jnp.concatenate([hprior['deter'],hprior['stoch'].reshape((B,-1))],-1)
                    hour_generated.append(self.sub('hour_reconstruction',nn.Linear,5)(hz))
                else:
                    hour_generated.append(jnp.zeros((B,5),jnp.float32))
                hstate=jnp.concatenate([hcarry['deter'],hcarry['stoch'].reshape((B,-1))],-1)
                dstate=jnp.concatenate([dcarry['deter'],dcarry['stoch'].reshape((B,-1))],-1)
                context=jnp.concatenate([hstate,dstate],-1)
                mcarry,(mprior,_)=m_dyn.imagine(mcarry,
                    {"market_dummy":jnp.zeros((B,1),jnp.float32),
                     "slow_context":context},1,False,single=True)
                mz=jnp.concatenate([mprior['deter'],mprior['stoch'].reshape((B,-1))],-1)
                mean=self.sub('next_minute_mean',nn.Linear,5)(mz)
                raw=self.sub('next_minute_scale',nn.Linear,5)(mz)
                scale=jax.nn.softplus(jnp.clip(raw,-12.,8.))+minute.min_scale
                means.append(mean);scales.append(scale)
            return {"minute_mean":jnp.stack(means,axis=1),
                    "minute_scale":jnp.stack(scales,axis=1),
                    "hour_generated":jnp.stack(hour_generated,axis=1),
                    "day_generated":jnp.stack(day_generated,axis=1)}

    return ThreeScale(name='route_b_hierarchical_world_model')
