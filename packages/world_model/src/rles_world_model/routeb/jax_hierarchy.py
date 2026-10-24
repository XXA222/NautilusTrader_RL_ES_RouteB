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

        def initialize_rollout_f2(self,minute_x,hour_x,day_x,hour_index,day_index):
            """Encode an observed context once and return reusable causal carries."""
            import jax
            import jax.numpy as jnp
            minute_x=jnp.asarray(minute_x,dtype=jnp.float32)
            hour_x=jnp.asarray(hour_x,dtype=jnp.float32)
            day_x=jnp.asarray(day_x,dtype=jnp.float32)
            hour_index=jnp.asarray(hour_index,dtype=jnp.int32)
            day_index=jnp.asarray(day_index,dtype=jnp.int32)
            B,T,_=minute_x.shape
            if (T<2 or hour_x.ndim!=3 or day_x.ndim!=3
                    or hour_x.shape[0]!=B or day_x.shape[0]!=B
                    or hour_x.shape[1]<2 or day_x.shape[1]<2
                    or hour_index.shape!=(B,T) or day_index.shape!=(B,T)):
                raise ValueError("stateful F2 requires aligned nonempty observed histories")
            h_dyn=self._dyn('hour_dyn',hour)
            d_dyn=self._dyn('day_dyn',day)
            zero_h={"market_dummy":jnp.zeros((*hour_x.shape[:2],1),jnp.float32)}
            zero_d={"market_dummy":jnp.zeros((*day_x.shape[:2],1),jnp.float32)}
            hr=jnp.zeros(hour_x.shape[:2],bool).at[:,0].set(True)
            dr=jnp.zeros(day_x.shape[:2],bool).at[:,0].set(True)
            hcarry,_,hfeat=h_dyn.observe(h_dyn.initial(B),hour_x,zero_h,hr,False)
            dcarry,_,dfeat=d_dyn.observe(d_dyn.initial(B),day_x,zero_d,dr,False)
            hhistory=jnp.concatenate([hfeat['deter'],
                hfeat['stoch'].reshape((B,hour_x.shape[1],-1))],-1)
            dhistory=jnp.concatenate([dfeat['deter'],
                dfeat['stoch'].reshape((B,day_x.shape[1],-1))],-1)
            hvisible=self._causal_gather(hhistory,hour_index)
            dvisible=self._causal_gather(dhistory,day_index)
            joined=jnp.concatenate([minute_x,hvisible,dvisible,
                (hour_index>=0)[:,:,None].astype(jnp.float32),
                (day_index>=0)[:,:,None].astype(jnp.float32)],-1)
            tokens=self.sub('minute_conditioned',nn.Linear,minute.hidden)(
                joined.astype(nn.COMPUTE_DTYPE))
            slow_context=jnp.concatenate([hvisible,dvisible],-1)
            m_dyn=self._dyn('minute_dyn',minute,slow_context.shape[-1])
            mr=jnp.zeros((B,T),bool).at[:,0].set(True)
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32),
                     "slow_context":slow_context}
            mcarry,_,mfeat=m_dyn.observe(m_dyn.initial(B),tokens,actions,mr,False)
            mstate=jax.tree.map(lambda value:value[:,-1],mfeat)
            hstate=jnp.concatenate([hcarry['deter'],
                hcarry['stoch'].reshape((B,-1))],-1)
            dstate=jnp.concatenate([dcarry['deter'],
                dcarry['stoch'].reshape((B,-1))],-1)
            return {"minute_carry":mcarry,"minute_state":mstate,
                    "hour_carry":hcarry,"hour_state":hstate,
                    "day_carry":dcarry,"day_state":dstate,
                    "hour_visible":hour_index[:,-1]>=0,
                    "day_visible":day_index[:,-1]>=0}

        def rollout_f2_step(self,state,minute_feedback,hour_feedback,day_feedback,
                            *,has_feedback,hour_event,day_event,
                            advance_slow_priors=True):
            """Advance one cached path state and emit its next-minute head.

            Feedback describes the previously generated minute and any slow
            aggregate that has become available by this decision. This makes
            the public step boundary match ``rollout_f2``'s emit-then-observe
            ordering while avoiding history replay between steps.
            """
            import jax
            import jax.numpy as jnp
            if type(has_feedback) is not bool or type(hour_event) is not bool or type(day_event) is not bool:
                raise ValueError("stateful F2 event controls must be booleans")
            mcarry=state['minute_carry']
            mstate=state['minute_state']
            hcarry=state['hour_carry']
            dcarry=state['day_carry']
            hvisible=state['hour_visible']
            dvisible=state['day_visible']
            B=mstate['deter'].shape[0]
            if has_feedback:
                minute_feedback=jnp.asarray(minute_feedback,dtype=jnp.float32)
                hour_feedback=jnp.asarray(hour_feedback,dtype=jnp.float32)
                day_feedback=jnp.asarray(day_feedback,dtype=jnp.float32)
                if (minute_feedback.shape!=(B,5) or hour_feedback.shape!=(B,5)
                        or day_feedback.shape!=(B,5)):
                    raise ValueError("stateful F2 feedback must have shape [paths,5]")
                h_dyn=self._dyn('hour_dyn',hour)
                d_dyn=self._dyn('day_dyn',day)
                m_dyn=self._dyn('minute_dyn',minute,
                    state['hour_state'].shape[-1]+state['day_state'].shape[-1])
                if advance_slow_priors and day_event:
                    dcarry,_,dfeat=d_dyn.observe(dcarry,day_feedback,
                        {"market_dummy":jnp.zeros((B,1),jnp.float32)},
                        jnp.zeros((B,),dtype=jnp.bool_),False,single=True)
                    dstate=jnp.concatenate([dfeat['deter'],
                        dfeat['stoch'].reshape((B,-1))],-1)
                    dvisible=jnp.ones((B,),dtype=jnp.bool_)
                else:
                    dstate=state['day_state']
                if advance_slow_priors and hour_event:
                    hcarry,_,hfeat=h_dyn.observe(hcarry,hour_feedback,
                        {"market_dummy":jnp.zeros((B,1),jnp.float32)},
                        jnp.zeros((B,),dtype=jnp.bool_),False,single=True)
                    hstate=jnp.concatenate([hfeat['deter'],
                        hfeat['stoch'].reshape((B,-1))],-1)
                    hvisible=jnp.ones((B,),dtype=jnp.bool_)
                else:
                    hstate=state['hour_state']
                context=jnp.concatenate([hstate,dstate],-1)
                joined=jnp.concatenate([minute_feedback,hstate,dstate,
                    hvisible[:,None].astype(jnp.float32),
                    dvisible[:,None].astype(jnp.float32)],axis=-1)
                token=self.sub('minute_conditioned',nn.Linear,minute.hidden)(
                    joined.astype(nn.COMPUTE_DTYPE))
                actions={"market_dummy":jnp.zeros((B,1),jnp.float32),
                         "slow_context":context}
                mcarry,(mprior,_)=m_dyn.imagine(mcarry,
                    {"market_dummy":jnp.zeros((B,1),jnp.float32),
                     "slow_context":context},1,False,single=True)
                mcarry,_,mstate=m_dyn.observe(mcarry,token,actions,
                    jnp.zeros((B,),dtype=jnp.bool_),False,single=True)
                state={"minute_carry":mcarry,"minute_state":mstate,
                       "hour_carry":hcarry,"hour_state":hstate,
                       "day_carry":dcarry,"day_state":dstate,
                       "hour_visible":hvisible,"day_visible":dvisible}
                prior_state=mprior
            else:
                prior_state=mstate
            mean_state=state['minute_state']
            mean_z=jnp.concatenate([mean_state['deter'],
                mean_state['stoch'].reshape((B,-1))],-1)
            mean=self.sub('next_minute_mean',nn.Linear,5)(mean_z)
            raw=self.sub('next_minute_scale',nn.Linear,5)(mean_z)
            scale=jax.nn.softplus(jnp.clip(raw,-12.,8.))+minute.min_scale
            prior_z=jnp.concatenate([prior_state['deter'],
                prior_state['stoch'].reshape((B,-1))],-1)
            prior_mean=self.sub('next_minute_mean',nn.Linear,5)(prior_z)
            prior_raw=self.sub('next_minute_scale',nn.Linear,5)(prior_z)
            prior_scale=jax.nn.softplus(jnp.clip(prior_raw,-12.,8.))+minute.min_scale
            return state,{"minute_mean":mean,"minute_scale":scale,
                          "first_prior_minute_mean":prior_mean,
                          "first_prior_minute_scale":prior_scale}

        def rollout_f2(self,minute_x,hour_x,day_x,hour_index,day_index,
                       future_hour_events,future_day_events,
                       advance_slow_priors=True,minute_feedback=None,
                       hour_feedback=None,day_feedback=None):
            """Generate a causal minute path while advancing slow RSSM priors.

            Event tuples state when a newly closed hour/day becomes available
            at each future decision. The caller derives them from UTC close plus
            exchange availability delay; future observations are never inputs.

            ``advance_slow_priors=False`` is the matched E3 ablation: use the
            same trained three-scale model and minute prior rollout, but freeze
            both slow latent carries at their final observed posterior. E4 uses
            the default and advances the slow prior carries at scheduled events.
            When feedback features are supplied, slow carries observe aggregate-
            derived generated bars and minute carries observe sampled generated
            minute features; the next forecast is then conditioned on that
            simulated past. Without feedback features, the legacy prior-only
            rollout is retained for compatibility/mechanism comparisons.
            This switch does not make the latent decoder outputs market OHLCV.
            """
            minute_x=jnp.asarray(minute_x,dtype=jnp.float32)
            hour_x=jnp.asarray(hour_x,dtype=jnp.float32)
            day_x=jnp.asarray(day_x,dtype=jnp.float32)
            hour_index=jnp.asarray(hour_index,dtype=jnp.int32)
            day_index=jnp.asarray(day_index,dtype=jnp.int32)
            B,T,_=minute_x.shape
            feedback_active=minute_feedback is not None
            if feedback_active:
                minute_feedback=jnp.asarray(minute_feedback,dtype=jnp.float32)
                if (minute_feedback.ndim!=3 or minute_feedback.shape[0]!=B
                        or minute_feedback.shape[2]!=5
                        or minute_feedback.shape[1]!=len(future_hour_events)
                        or hour_feedback is None or day_feedback is None):
                    raise ValueError("feedback replay requires aligned minute/hour/day feature tensors")
                hour_feedback=jnp.asarray(hour_feedback,dtype=jnp.float32)
                day_feedback=jnp.asarray(day_feedback,dtype=jnp.float32)
                expected_feedback_shape=(B,len(future_hour_events),5)
                if (hour_feedback.shape!=expected_feedback_shape
                        or day_feedback.shape!=expected_feedback_shape):
                    raise ValueError("feedback feature tensors must align to batch and forecast horizon")
            elif hour_feedback is not None or day_feedback is not None:
                raise ValueError("slow feedback cannot be provided without minute feedback")
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
            mcarry,_,mfeat=m_dyn.observe(m_dyn.initial(B),tokens,m_actions,mr,False)
            mstate=jax.tree.map(lambda value:value[:,-1],mfeat)
            anchor_z=jnp.concatenate([
                mstate['deter'],mstate['stoch'].reshape((B,-1))],-1)
            anchor_mean=self.sub('next_minute_mean',nn.Linear,5)(anchor_z)
            anchor_raw=self.sub('next_minute_scale',nn.Linear,5)(anchor_z)
            anchor_scale=jax.nn.softplus(jnp.clip(anchor_raw,-12.,8.))+minute.min_scale
            means=[];scales=[];hour_diagnostics=[];day_diagnostics=[]
            first_prior_mean=None;first_prior_scale=None
            hour_visible=hour_index[:,-1]>=0
            day_visible=day_index[:,-1]>=0
            for hour_event,day_event in zip(future_hour_events,future_day_events):
                step_index=len(means)
                # The next-bar head is trained on posterior[t] -> x[t+1].
                # Emit from the current posterior/prior state first. Slow events
                # at the target close may condition only the following target.
                mz=jnp.concatenate([
                    mstate['deter'],mstate['stoch'].reshape((B,-1))],-1)
                mean=self.sub('next_minute_mean',nn.Linear,5)(mz)
                raw=self.sub('next_minute_scale',nn.Linear,5)(mz)
                scale=jax.nn.softplus(jnp.clip(raw,-12.,8.))+minute.min_scale
                means.append(mean);scales.append(scale)
                if advance_slow_priors and day_event:
                    if feedback_active:
                        dcarry,_,dfeat=d_dyn.observe(dcarry,day_feedback[:,step_index],
                            {"market_dummy":jnp.zeros((B,1),jnp.float32)},
                            jnp.zeros((B,),dtype=jnp.bool_),False,single=True)
                        dstate=jnp.concatenate([dfeat['deter'],
                            dfeat['stoch'].reshape((B,-1))],-1)
                        day_visible=jnp.ones((B,),dtype=jnp.bool_)
                        day_diagnostics.append(self.sub('day_reconstruction',nn.Linear,5)(dstate))
                    else:
                        dcarry,(dprior,_)=d_dyn.imagine(dcarry,
                            {"market_dummy":jnp.zeros((B,1),jnp.float32)},1,False,single=True)
                        dz=jnp.concatenate([dprior['deter'],dprior['stoch'].reshape((B,-1))],-1)
                        day_diagnostics.append(self.sub('day_reconstruction',nn.Linear,5)(dz))
                else:
                    day_diagnostics.append(jnp.zeros((B,5),jnp.float32))
                if advance_slow_priors and hour_event:
                    if feedback_active:
                        hcarry,_,hfeat=h_dyn.observe(hcarry,hour_feedback[:,step_index],
                            {"market_dummy":jnp.zeros((B,1),jnp.float32)},
                            jnp.zeros((B,),dtype=jnp.bool_),False,single=True)
                        hstate=jnp.concatenate([hfeat['deter'],
                            hfeat['stoch'].reshape((B,-1))],-1)
                        hour_visible=jnp.ones((B,),dtype=jnp.bool_)
                        hour_diagnostics.append(self.sub('hour_reconstruction',nn.Linear,5)(hstate))
                    else:
                        hcarry,(hprior,_)=h_dyn.imagine(hcarry,
                            {"market_dummy":jnp.zeros((B,1),jnp.float32)},1,False,single=True)
                        hz=jnp.concatenate([hprior['deter'],hprior['stoch'].reshape((B,-1))],-1)
                        hour_diagnostics.append(self.sub('hour_reconstruction',nn.Linear,5)(hz))
                else:
                    hour_diagnostics.append(jnp.zeros((B,5),jnp.float32))
                hstate=jnp.concatenate([hcarry['deter'],hcarry['stoch'].reshape((B,-1))],-1)
                dstate=jnp.concatenate([dcarry['deter'],dcarry['stoch'].reshape((B,-1))],-1)
                context=jnp.concatenate([hstate,dstate],-1)
                minute_action={"market_dummy":jnp.zeros((B,1),jnp.float32),
                               "slow_context":context}
                if feedback_active:
                    joined_feedback=jnp.concatenate([
                        minute_feedback[:,step_index],hstate,dstate,
                        hour_visible[:,None].astype(jnp.float32),
                        day_visible[:,None].astype(jnp.float32)],axis=-1)
                    feedback_token=self.sub('minute_conditioned',nn.Linear,minute.hidden)(
                        joined_feedback.astype(nn.COMPUTE_DTYPE))
                    _,(mprior,_)=m_dyn.imagine(
                        mcarry,{"market_dummy":jnp.zeros((B,1),jnp.float32),
                                "slow_context":context},1,False,single=True)
                    mcarry,_,mfeat=m_dyn.observe(mcarry,feedback_token,minute_action,
                        jnp.zeros((B,),dtype=jnp.bool_),False,single=True)
                    mstate=mfeat
                else:
                    mcarry,(mprior,_)=m_dyn.imagine(mcarry,
                        {"market_dummy":jnp.zeros((B,1),jnp.float32),
                         "slow_context":context},1,False,single=True)
                    mstate=mprior
                if first_prior_mean is None:
                    prior_z=jnp.concatenate([
                        mprior['deter'],mprior['stoch'].reshape((B,-1))],-1)
                    first_prior_mean=self.sub('next_minute_mean',nn.Linear,5)(prior_z)
                    prior_raw=self.sub('next_minute_scale',nn.Linear,5)(prior_z)
                    first_prior_scale=(jax.nn.softplus(jnp.clip(prior_raw,-12.,8.))+
                                       minute.min_scale)
            return {"minute_mean":jnp.stack(means,axis=1),
                    "minute_scale":jnp.stack(scales,axis=1),
                    "anchor_minute_mean":anchor_mean,
                    "anchor_minute_scale":anchor_scale,
                    "first_prior_minute_mean":first_prior_mean,
                    "first_prior_minute_scale":first_prior_scale,
                    "hour_latent_diagnostic":jnp.stack(hour_diagnostics,axis=1),
                    "day_latent_diagnostic":jnp.stack(day_diagnostics,axis=1)}

    return ThreeScale(name='route_b_hierarchical_world_model')
