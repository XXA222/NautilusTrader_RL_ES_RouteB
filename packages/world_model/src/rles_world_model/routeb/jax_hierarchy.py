"""Causal E3/E3b/E4 market-world-model candidates over pinned DreamerV3 RSSMs.

E3 uses observed closed-bar H/D features; E3b uses independent slow RSSMs and
token fusion; E4 injects latest-available day state into the hour transition and
latest-available hour/day states into both the minute posterior and prior
transition. These are training networks, not a validated F2 rollout yet.
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


def _weighted_risk_terms(continuous, barrier, weights):
    """Return diagnostic subterms and the unchanged combined risk objective."""
    import jax.numpy as jnp

    weights = jnp.asarray(weights, dtype=jnp.float32)
    denominator = jnp.maximum(weights.sum(), 1.0)
    continuous = jnp.asarray(continuous, dtype=jnp.float32)
    barrier = jnp.asarray(barrier, dtype=jnp.float32)
    continuous_nll = (continuous * weights).sum() / denominator
    barrier_bce = (barrier * weights).sum() / denominator
    combined = ((continuous + barrier) * weights).sum() / denominator
    return continuous_nll, barrier_bce, combined


def _weighted_risk_channels(channel_losses, weights):
    """Aggregate per-horizon continuous risk NLLs by their frozen weights."""
    import jax.numpy as jnp

    weights = jnp.asarray(weights, dtype=jnp.float32)
    channel_losses = jnp.asarray(channel_losses, dtype=jnp.float32)
    denominator = jnp.maximum(weights.sum(), 1.0)
    return (channel_losses * weights[:, None]).sum(axis=0) / denominator


def make_hierarchical_model(checkout, *, minute: RSSMConfig, hour: RSSMConfig,
                            day: RSSMConfig, variant: str = "e3b_fusion",
                            risk_horizons: tuple[int, ...] = (1, 5, 15, 30, 60),
                            risk_horizon_weights: tuple[float, ...] =
                            (1.0, 1.0, 1.0, 1.0, 1.0),
                            risk_auxiliary_scale: float = 0.0,
                            risk_min_scale: float | None = None):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    import elements
    RSSM=import_pinned_upstream(checkout)
    # The pinned checkout adds its vendored ``embodied`` package to sys.path.
    import embodied.jax.nets as nn
    if variant not in {"e3_features", "e3b_fusion", "e4_conditional"}:
        raise ValueError("hierarchy variant must be E3, E3b or E4")
    conditional = variant == "e4_conditional"
    risk_horizons = tuple(risk_horizons)
    risk_horizon_weights = tuple(risk_horizon_weights)
    risk_min_scale = minute.min_scale if risk_min_scale is None else risk_min_scale
    if (isinstance(risk_auxiliary_scale, bool)
            or not isinstance(risk_auxiliary_scale, (int, float))
            or not np.isfinite(risk_auxiliary_scale) or risk_auxiliary_scale < 0
            or isinstance(risk_min_scale, bool)
            or not isinstance(risk_min_scale, (int, float))
            or not np.isfinite(risk_min_scale) or risk_min_scale <= 0
            or not risk_horizons
            or tuple(sorted(set(risk_horizons))) != risk_horizons
            or any(type(value) is not int or value < 1 for value in risk_horizons)
            or len(risk_horizon_weights) != len(risk_horizons)
            or any(isinstance(value, bool) or not np.isfinite(value) or value <= 0
                   for value in risk_horizon_weights)):
        raise ValueError("invalid hierarchical risk-head configuration")
    risk_auxiliary = risk_auxiliary_scale > 0

    class ThreeScale(nj.Module):
        def _dyn(self,name,cfg,slow_dim:int=0):
            act_space={"market_dummy":elements.Space(np.float32,(1,))}
            if slow_dim:
                act_space["slow_context"] = elements.Space(np.float32,(slow_dim,))
            return self.sub(name,RSSM,act_space=act_space,
                            deter=cfg.deter,hidden=cfg.hidden,stoch=cfg.stoch,
                            classes=cfg.classes,blocks=cfg.blocks,
                            obslayers=1,imglayers=1,dynlayers=1,free_nats=1.0)

        def _features(self,features,name,cfg,slow_context=None):
            B,T,_=features.shape
            r=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            slow_dim=0
            if slow_context is not None:
                slow_context=jnp.asarray(slow_context,dtype=jnp.float32)
                if slow_context.shape[:2]!=(B,T):
                    raise ValueError("slow transition context must align with native-scale updates")
                # RSSM action[t] advances state t-1 -> t. It must use the
                # context visible at t-1; the current slow state is separately
                # available to the posterior observation encoder at time t.
                actions["slow_context"]=jnp.concatenate(
                    [jnp.zeros_like(slow_context[:,:1]),slow_context[:,:-1]],axis=1)
                features=jnp.concatenate([features,slow_context],axis=-1)
                slow_dim=slow_context.shape[-1]
            dyn=self._dyn(name,cfg,slow_dim)
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

        @staticmethod
        def _f2_latent(carry):
            return jnp.concatenate([carry['deter'],
                carry['stoch'].reshape((carry['stoch'].shape[0],-1))],axis=-1)

        def f2_initialize(self,minute_x,hour_x,day_x,hour_index,day_index,
                          hour_day_index,hour_mask,day_mask):
            """Filter only the supplied observed prefix into live RSSM carries."""
            minute_x=jnp.asarray(minute_x,dtype=jnp.float32)
            hour_x=jnp.asarray(hour_x,dtype=jnp.float32)
            day_x=jnp.asarray(day_x,dtype=jnp.float32)
            hour_index=jnp.asarray(hour_index,dtype=jnp.int32)
            day_index=jnp.asarray(day_index,dtype=jnp.int32)
            hour_day_index=jnp.asarray(hour_day_index,dtype=jnp.int32)
            hour_mask=jnp.asarray(hour_mask,dtype=jnp.float32)
            day_mask=jnp.asarray(day_mask,dtype=jnp.float32)
            B,T,_=minute_x.shape
            if (B!=1 or T<2 or hour_x.shape[1]<2 or day_x.shape[1]<2
                    or hour_index.shape!=(B,T) or day_index.shape!=(B,T)
                    or hour_day_index.shape!=hour_x.shape[:2]
                    or hour_mask.shape!=hour_x.shape[:2]
                    or day_mask.shape!=day_x.shape[:2]):
                raise ValueError("F2 initialization needs aligned one-market prefix tensors")
            last_hour=jnp.maximum(jnp.sum(hour_mask[0]>0)-1,0)
            last_day=jnp.maximum(jnp.sum(day_mask[0]>0)-1,0)
            if variant=="e3_features":
                hour_state=hour_x[:,last_hour]
                day_state=day_x[:,last_day]
                hour_day_context=day_state
            else:
                day_dyn=self._dyn('day_dyn',day)
                day_reset=jnp.zeros(day_x.shape[:2],dtype=jnp.bool_).at[:,0].set(True)
                day_actions={"market_dummy":jnp.zeros((*day_x.shape[:2],1),jnp.float32)}
                _,day_entries,day_feat=day_dyn.observe(
                    day_dyn.initial(B),day_x,day_actions,day_reset,False)
                day_latent=jnp.concatenate([day_feat['deter'],
                    day_feat['stoch'].reshape((B,day_x.shape[1],-1))],axis=-1)
                hour_day_context=self._causal_gather(
                    day_latent,hour_day_index)
                if conditional:
                    hour_tokens=jnp.concatenate([hour_x,hour_day_context],axis=-1)
                    hour_actions={
                        "market_dummy":jnp.zeros((*hour_x.shape[:2],1),jnp.float32),
                        "slow_context":jnp.concatenate([
                            jnp.zeros_like(hour_day_context[:,:1]),
                            hour_day_context[:,:-1]],axis=1)}
                else:
                    hour_tokens=hour_x
                    hour_actions={"market_dummy":jnp.zeros(
                        (*hour_x.shape[:2],1),jnp.float32)}
                hour_dyn=self._dyn('hour_dyn',hour,
                                   hour_day_context.shape[-1] if conditional else 0)
                hour_reset=jnp.zeros(hour_x.shape[:2],dtype=jnp.bool_).at[:,0].set(True)
                _,hour_entries,hour_feat=hour_dyn.observe(
                    hour_dyn.initial(B),hour_tokens,hour_actions,hour_reset,False)
                hour_latent=jnp.concatenate([hour_feat['deter'],
                    hour_feat['stoch'].reshape((B,hour_x.shape[1],-1))],axis=-1)
                hour_state=jax.tree.map(lambda x:x[:,last_hour],hour_entries)
                day_state=jax.tree.map(lambda x:x[:,last_day],day_entries)
                hour_day_context=hour_day_context[:,last_hour]
            if variant=="e3_features":
                h_visible=self._causal_gather(hour_x,hour_index)
                d_visible=self._causal_gather(day_x,day_index)
                minute_context=None
            else:
                h_visible=self._causal_gather(hour_latent,hour_index)
                d_visible=self._causal_gather(day_latent,day_index)
                minute_context=jnp.concatenate([h_visible,d_visible],axis=-1)
            flags=jnp.concatenate([(hour_index>=0)[:,:,None].astype(jnp.float32),
                                   (day_index>=0)[:,:,None].astype(jnp.float32)],axis=-1)
            tokens=self.sub('minute_conditioned',nn.Linear,minute.hidden)(
                nn.cast(jnp.concatenate([minute_x,h_visible,d_visible,flags],axis=-1)))
            minute_actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            minute_slow_dim=0
            if conditional:
                minute_actions["slow_context"]=jnp.concatenate([
                    jnp.zeros_like(minute_context[:,:1]),minute_context[:,:-1]],axis=1)
                minute_slow_dim=minute_context.shape[-1]
            minute_dyn=self._dyn('minute_dyn',minute,minute_slow_dim)
            minute_reset=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            minute_carry,_,_=minute_dyn.observe(
                minute_dyn.initial(B),tokens,minute_actions,minute_reset,False)
            return minute_carry,hour_state,day_state,hour_day_context

        def _f2_context(self,state,scale="hour"):
            if variant=="e3_features":
                return jnp.asarray(state,dtype=jnp.float32)
            return self._f2_latent(state)

        def f2_prior_step(self,minute_carry,hour_state,day_state):
            """Advance the minute prior once and sample one standardized bar."""
            h_context=self._f2_context(hour_state)
            d_context=self._f2_context(day_state,scale="day")
            action={"market_dummy":jnp.zeros((1,1),jnp.float32)}
            if conditional:
                action["slow_context"]=jnp.concatenate([h_context,d_context],axis=-1)
            minute_dyn=self._dyn('minute_dyn',minute,
                h_context.shape[-1]+d_context.shape[-1] if conditional else 0)
            prior_carry,(feat,_)=minute_dyn.imagine(
                minute_carry,action,length=1,training=False,single=True)
            latent=jnp.concatenate([feat['deter'],
                feat['stoch'].reshape((1,-1))],axis=-1)
            mean=self.sub('next_minute_mean',nn.Linear,5)(latent).astype(jnp.float32)
            raw=self.sub('next_minute_scale',nn.Linear,5)(latent).astype(jnp.float32)
            scale=jax.nn.softplus(jnp.clip(raw,-12.,8.))+minute.min_scale
            sample=mean+scale*jax.random.normal(nj.seed(),mean.shape)
            return prior_carry,sample,mean,scale,feat['logit']

        def f2_update_day(self,day_carry,day_feature):
            if variant=="e3_features":
                return jnp.asarray(day_feature,dtype=jnp.float32)
            feature=jnp.asarray(day_feature,dtype=jnp.float32).reshape((1,1,5))
            dyn=self._dyn('day_dyn',day)
            actions={"market_dummy":jnp.zeros((1,1),jnp.float32)}
            reset=jnp.zeros((1,),dtype=jnp.bool_)
            carry,_,_=dyn.observe(day_carry,feature,actions,reset,False,single=True)
            return carry

        def f2_update_hour(self,hour_carry,hour_feature,previous_day_context,
                           current_day_state):
            if variant=="e3_features":
                current=self._f2_context(current_day_state)
                return jnp.asarray(hour_feature,dtype=jnp.float32)[None],current
            feature=jnp.asarray(hour_feature,dtype=jnp.float32).reshape((1,1,5))
            actions={"market_dummy":jnp.zeros((1,1),jnp.float32)}
            slow_dim=0
            current=self._f2_context(current_day_state,scale="day")
            if conditional:
                previous=jnp.asarray(previous_day_context,dtype=jnp.float32).reshape(
                    (1,-1))
                feature=jnp.concatenate([feature,current[:,None,:]],axis=-1)
                actions["slow_context"]=previous
                slow_dim=current.shape[-1]
            dyn=self._dyn('hour_dyn',hour,slow_dim)
            reset=jnp.zeros((1,),dtype=jnp.bool_)
            carry,_,_=dyn.observe(hour_carry,feature,actions,reset,False,single=True)
            return carry,current

        def f2_filter_minute(self,previous_minute_carry,minute_feature,
                             previous_hour_state,previous_day_state,
                             current_hour_state,current_day_state):
            """Filter a generated minute at its close using then-visible slow states."""
            feature=jnp.asarray(minute_feature,dtype=jnp.float32).reshape((1,1,5))
            previous_h=self._f2_context(previous_hour_state)
            previous_d=self._f2_context(previous_day_state,scale="day")
            current_h=self._f2_context(current_hour_state)
            current_d=self._f2_context(current_day_state,scale="day")
            flags=jnp.ones((1,1,2),dtype=jnp.float32)
            tokens=self.sub('minute_conditioned',nn.Linear,minute.hidden)(nn.cast(
                jnp.concatenate([feature,current_h[:,None,:],
                                 current_d[:,None,:],flags],axis=-1)))
            actions={"market_dummy":jnp.zeros((1,1),jnp.float32)}
            slow_dim=0
            if conditional:
                actions["slow_context"]=jnp.concatenate(
                    [previous_h,previous_d],axis=-1)
                slow_dim=actions["slow_context"].shape[-1]
            dyn=self._dyn('minute_dyn',minute,slow_dim)
            reset=jnp.zeros((1,),dtype=jnp.bool_)
            carry,_,feat=dyn.observe(previous_minute_carry,tokens,actions,
                                     reset,False,single=True)
            return carry,feat['logit']

        def __call__(self,minute_x,hour_x,day_x,hour_index,day_index,
                     hour_day_index,minute_mask,hour_mask,day_mask,
                     risk_targets,risk_mask):
            """Tensors: x scale [B,T_s,5], index [B,T_min] (-1 or observed state).

            Caller MUST build indexes from UTC event-close and actual availability.
            """
            minute_x=jnp.asarray(minute_x,dtype=jnp.float32)
            hour_x=jnp.asarray(hour_x,dtype=jnp.float32)
            day_x=jnp.asarray(day_x,dtype=jnp.float32)
            hour_index=jnp.asarray(hour_index,dtype=jnp.int32)
            day_index=jnp.asarray(day_index,dtype=jnp.int32)
            hour_day_index=jnp.asarray(hour_day_index,dtype=jnp.int32)
            if (hour_x.shape[1]<2 or day_x.shape[1]<2 or
                minute_x.shape[2]!=5 or hour_x.shape[2]!=5 or day_x.shape[2]!=5):
                raise ValueError("B3 needs nonempty correctly shaped native-scale batches")
            if variant=="e3_features":
                h_visible=self._causal_gather(hour_x,hour_index)
                d_visible=self._causal_gather(day_x,day_index)
                aux=jnp.asarray(0.0,dtype=jnp.float32)
            else:
                d_latent,d_kl=self._features(day_x,'day_dyn',day)
                h_context=self._causal_gather(d_latent,hour_day_index)
                h_latent,h_kl=self._features(
                    hour_x,'hour_dyn',hour,h_context if conditional else None)
                h_visible=self._causal_gather(h_latent,hour_index)
                d_visible=self._causal_gather(d_latent,day_index)
            if conditional:
                h_visible_flag=(hour_index>=0)[:,:,None].astype(jnp.float32)
                d_visible_flag=(day_index>=0)[:,:,None].astype(jnp.float32)
                minute_context=jnp.concatenate([h_visible,d_visible],axis=-1)
                joined=jnp.concatenate([minute_x,h_visible,d_visible,
                                         h_visible_flag,d_visible_flag],axis=-1)
            else:
                minute_context=None
                joined=jnp.concatenate([minute_x,h_visible,d_visible,
                    (hour_index>=0)[:,:,None].astype(jnp.float32),
                    (day_index>=0)[:,:,None].astype(jnp.float32)],axis=-1)
            tokens=self.sub('minute_conditioned',nn.Linear,minute.hidden)(nn.cast(joined))
            B,T,_=minute_x.shape
            resets=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            slow_dim=0
            if conditional:
                actions["slow_context"]=jnp.concatenate(
                    [jnp.zeros_like(minute_context[:,:1]),minute_context[:,:-1]],axis=1)
                slow_dim=minute_context.shape[-1]
            mdyn=self._dyn('minute_dyn',minute,slow_dim)
            _,_,m_kl,m_feat,_=mdyn.loss(mdyn.initial(B),tokens,actions,resets,True)
            mz=jnp.concatenate([m_feat['deter'],
                                  m_feat['stoch'].reshape((B,T,-1))],axis=-1)
            posterior_mean=self.sub('posterior_aux_mean',nn.Linear,5)(mz[:,:-1])
            posterior_raw=self.sub('posterior_aux_scale',nn.Linear,5)(mz[:,:-1])
            posterior_scale=jax.nn.softplus(jnp.clip(posterior_raw,-12.,8.))+minute.min_scale
            risk_loss=jnp.asarray(0.0,dtype=jnp.float32)
            risk_continuous_nll=jnp.asarray(0.0,dtype=jnp.float32)
            risk_barrier_bce=jnp.asarray(0.0,dtype=jnp.float32)
            risk_channel_nll=jnp.zeros((4,),dtype=jnp.float32)
            # Score the observation from p(z[t+1]|posterior[t], c[t]); never
            # use posterior[t+1] in the prior prediction path.
            deter=m_feat['deter'][:,:-1]
            stoch=m_feat['stoch'][:,:-1]
            transition_action={"market_dummy":jnp.zeros((B,T-1,1),jnp.float32)}
            if conditional:
                transition_action["slow_context"]=minute_context[:,:-1]
            actemb=nn.DictConcat(mdyn.act_space,1)(transition_action)
            prior_deter=jax.vmap(
                lambda d,s,a:mdyn._core(d,s,a),in_axes=(1,1,1),out_axes=1
            )(deter,stoch,actemb)
            prior_deter=nn.cast(prior_deter)
            prior_logits=jax.vmap(mdyn._prior,in_axes=1,out_axes=1)(prior_deter)
            prior_logits=nn.cast(prior_logits)
            prior_stoch=nn.cast(mdyn._dist(prior_logits).sample(seed=nj.seed()))
            prior_z=jnp.concatenate([prior_deter,prior_stoch.reshape(
                (*prior_stoch.shape[:-2],-1))],axis=-1)
            if risk_auxiliary:
                risk_targets=jnp.asarray(risk_targets,dtype=jnp.float32)
                risk_mask=jnp.asarray(risk_mask,dtype=jnp.float32)
                expected=(B,T,len(risk_horizons),6)
                if risk_targets.shape!=expected or risk_mask.shape!=expected[:-1]:
                    raise ValueError("hierarchical risk target shape mismatch")
                count=len(risk_horizons)
                risk_mean=self.sub("prior_risk_mean",nn.Linear,count*4)(prior_z).reshape(
                    (B,T-1,count,4))
                risk_raw=self.sub("prior_risk_scale",nn.Linear,count*4)(prior_z).reshape(
                    (B,T-1,count,4))
                risk_scale=jax.nn.softplus(jnp.clip(risk_raw,-12.,8.))+risk_min_scale
                risk_logits=self.sub("prior_risk_barrier",nn.Linear,count*2)(prior_z).reshape(
                    (B,T-1,count,2))
                continuous_channels=(.5*jnp.square(
                    (risk_targets[:,:-1,:,:4]-risk_mean)/risk_scale)
                    +jnp.log(risk_scale)
                    +.5*1.8378770664093453)
                continuous=continuous_channels.sum(-1)
                labels=risk_targets[:,:-1,:,4:6]
                barrier_bce=(jax.nn.softplus(risk_logits)
                             -labels*risk_logits).sum(-1)
                valid=risk_mask[:,:-1]*jnp.asarray(minute_mask[:,:-1])[:,:,None]
                horizon_count=valid.sum((0,1))
                cont=(continuous*valid).sum((0,1))/jnp.maximum(horizon_count,1.0)
                bce=(barrier_bce*valid).sum((0,1))/jnp.maximum(horizon_count,1.0)
                weights=jnp.asarray(risk_horizon_weights,dtype=jnp.float32)
                risk_continuous_nll,risk_barrier_bce,risk_loss=_weighted_risk_terms(
                    cont,bce,weights)
                channel_count=jnp.maximum(horizon_count[:,None],1.0)
                channel_by_horizon=(continuous_channels*valid[:,:,:,None]).sum((0,1))/channel_count
                risk_channel_nll=_weighted_risk_channels(channel_by_horizon,weights)
            pred=self.sub('next_minute_mean',nn.Linear,5)(prior_z)
            raw_scale=self.sub('next_minute_scale',nn.Linear,5)(prior_z)
            scale=jax.nn.softplus(jnp.clip(raw_scale,-12.,8.))+minute.min_scale
            target=minute_x[:,1:]
            loss_per_dim=(.5*jnp.square((target-pred)/scale)+
                          jnp.log(scale)+.5*1.8378770664093453)
            posterior_per_dim=(.5*jnp.square((target-posterior_mean)/posterior_scale)+
                               jnp.log(posterior_scale)+.5*1.8378770664093453)
            # Target t+1 may be scored even when t is burn-in.
            pair=jnp.asarray(minute_mask[:,1:],dtype=jnp.float32)
            nll=(loss_per_dim.sum(-1)*pair).sum()/jnp.maximum(1.0,pair.sum())
            posterior_nll=(posterior_per_dim.sum(-1)*pair).sum()/jnp.maximum(1.0,pair.sum())
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
            if variant!="e3_features":
                aux=scale_aux(h_latent,hour_x,h_kl,h_mask,hour,'hour')+scale_aux(d_latent,day_x,d_kl,d_mask,day,'day')
            return {'loss':minute.prediction_scale*nll +
                    minute.posterior_auxiliary_scale*posterior_nll + mkl + aux +
                    risk_auxiliary_scale*risk_loss,
                    'minute_nll':nll, 'posterior_aux_nll':posterior_nll,
                    'slow_aux':aux, 'risk_aux_nll':risk_loss,
                    'risk_continuous_nll':risk_continuous_nll,
                    'risk_barrier_bce':risk_barrier_bce,
                    'risk_log_return_nll':risk_channel_nll[0],
                    'risk_log_realized_volatility_nll':risk_channel_nll[1],
                    'risk_max_high_return_nll':risk_channel_nll[2],
                    'risk_min_low_return_nll':risk_channel_nll[3]}

        def rollout_static_context(self,minute_history,hour_x,day_x,hour_index,
                                   day_index,hour_day_index,steps:int):
            """F1 prior rollout conditioned on the latest slow state held fixed.

            This diagnostic mode is causal, but it is valid only until a new
            slow-scale event becomes available. It intentionally does not
            claim F2 feedback or a full event-driven 1m/1h/1d rollout.
            Returns prior-conditional sample, location and scale per step.
            """
            if type(steps) is not int or steps < 1 or steps > 60:
                raise ValueError("static-context rollout horizon must be in [1,60]")
            minute_history=jnp.asarray(minute_history,dtype=jnp.float32)
            hour_x=jnp.asarray(hour_x,dtype=jnp.float32)
            day_x=jnp.asarray(day_x,dtype=jnp.float32)
            hour_index=jnp.asarray(hour_index,dtype=jnp.int32)
            day_index=jnp.asarray(day_index,dtype=jnp.int32)
            hour_day_index=jnp.asarray(hour_day_index,dtype=jnp.int32)
            B,T,_=minute_history.shape
            if (T < 2 or hour_x.shape[1] < 2 or day_x.shape[1] < 2
                    or hour_index.shape != (B,T) or day_index.shape != (B,T)
                    or hour_day_index.shape != hour_x.shape[:2]):
                raise ValueError("static-context rollout inputs have incompatible shapes")
            if variant=="e3_features":
                h_visible=self._causal_gather(hour_x,hour_index)
                d_visible=self._causal_gather(day_x,day_index)
            else:
                d_latent,_=self._features(day_x,'day_dyn',day)
                h_context=self._causal_gather(d_latent,hour_day_index)
                h_latent,_=self._features(
                    hour_x,'hour_dyn',hour,h_context if conditional else None)
                h_visible=self._causal_gather(h_latent,hour_index)
                d_visible=self._causal_gather(d_latent,day_index)
            minute_context=jnp.concatenate([h_visible,d_visible],axis=-1)
            flags=jnp.concatenate([(hour_index>=0)[:,:,None].astype(jnp.float32),
                                   (day_index>=0)[:,:,None].astype(jnp.float32)],axis=-1)
            joined=jnp.concatenate([minute_history,h_visible,d_visible,flags],axis=-1)
            tokens=self.sub('minute_conditioned',nn.Linear,minute.hidden)(nn.cast(joined))
            reset=jnp.zeros((B,T),dtype=jnp.bool_).at[:,0].set(True)
            actions={"market_dummy":jnp.zeros((B,T,1),jnp.float32)}
            slow_dim=0
            if conditional:
                actions["slow_context"]=jnp.concatenate(
                    [jnp.zeros_like(minute_context[:,:1]),minute_context[:,:-1]],axis=1)
                slow_dim=minute_context.shape[-1]
            dyn=self._dyn('minute_dyn',minute,slow_dim)
            carry,_,_=dyn.observe(dyn.initial(B),tokens,actions,reset,False)
            slow_condition=minute_context[:,-1]
            samples=[]
            means=[]
            scales=[]
            for _ in range(steps):
                action={"market_dummy":jnp.zeros((B,1),jnp.float32)}
                if conditional:
                    action["slow_context"]=slow_condition
                carry,(feat,_)=dyn.imagine(carry,action,length=1,
                                           training=False,single=True)
                z=jnp.concatenate([feat['deter'],feat['stoch'].reshape((B,-1))],axis=-1)
                mean=self.sub('next_minute_mean',nn.Linear,5)(z).astype(jnp.float32)
                raw=self.sub('next_minute_scale',nn.Linear,5)(z).astype(jnp.float32)
                scale=jax.nn.softplus(jnp.clip(raw,-12.,8.))+minute.min_scale
                sample=mean+scale*jax.random.normal(nj.seed(),mean.shape)
                samples.append(sample)
                means.append(mean)
                scales.append(scale)
            return {"samples":jnp.stack(samples,axis=1),
                    "means":jnp.stack(means,axis=1),
                    "scales":jnp.stack(scales,axis=1)}

    return ThreeScale(name='route_b_hierarchical_world_model')
