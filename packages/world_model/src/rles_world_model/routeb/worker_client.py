"""Python 3.14 host-side process client for isolated Python 3.12 JAX learner.

The caller MUST have already registered the Train access and exported approved
Train-only NPZ; this client grants no access permission and reads no raw data.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import subprocess
from dataclasses import dataclass
from ..json_utils import loads_unique_json

@dataclass(frozen=True)
class WorkerConfig:
    python_executable:Path
    upstream_checkout:Path
    timeout_seconds:int=1800
    def __post_init__(self):
        if self.timeout_seconds<1 or not self.python_executable.is_file():
            raise ValueError("missing JAX worker interpreter or invalid timeout")


class WorkerFailure(RuntimeError):
    pass


class WorkerClient:
    def __init__(self, config:WorkerConfig):
        self.config=config

    def preflight(self)->dict:
        """Verify the isolated worker can import its numeric runtime before data access."""
        env=os.environ.copy()
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
        src_dir=Path(__file__).resolve().parents[2]
        env["PYTHONPATH"]=str(src_dir)+os.pathsep+env.get("PYTHONPATH","")
        probe=("import json, sys; print(json.dumps({'executable':sys.executable,'path':sys.path}),flush=True); "
               "import numpy, jax; import rles_world_model.routeb.worker_cli; "
               "print(json.dumps({'numpy':numpy.__version__,'jax':jax.__version__,"
               "'devices':[str(x) for x in jax.devices()]}))")
        try:
            proc=subprocess.run([str(self.config.python_executable),"-c",probe],
                check=False,capture_output=True,text=True,timeout=min(60,self.config.timeout_seconds),env=env)
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise WorkerFailure(f"worker runtime preflight could not start: {exc}") from exc
        if proc.returncode!=0:
            raise WorkerFailure(
                f"worker runtime preflight failed for {self.config.python_executable} "
                f"with exit {proc.returncode}: stdout={proc.stdout[-1500:]} stderr={proc.stderr[-1500:]}"
            )
        try:
            result=loads_unique_json(proc.stdout.splitlines()[-1])
        except ValueError as exc:
            raise WorkerFailure(f"worker runtime preflight did not return JSON: {exc}") from exc
        if (not isinstance(result,dict) or not isinstance(result.get("numpy"),str)
                or not isinstance(result.get("jax"),str) or not result.get("devices")):
            raise WorkerFailure("worker runtime preflight returned an incomplete environment identity")
        return result

    def _run(self,args:list[str])->dict:
        env=os.environ.copy()
        # Do not allow JAX to claim the whole GPU just by starting a process.
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
        # Worker is Python 3.12 while the parent distribution requires 3.14.7.
        # Import this additive module via source path, NOT by installing the root wheel.
        src_dir=Path(__file__).resolve().parents[2]
        env["PYTHONPATH"]=str(src_dir)+os.pathsep+env.get("PYTHONPATH","")
        proc=subprocess.run([str(self.config.python_executable),"-m",
                             "rles_world_model.routeb.worker_cli",*args],
                            check=False,capture_output=True,text=True,
                            timeout=self.config.timeout_seconds,env=env)
        if proc.returncode!=0:
            raise WorkerFailure(f"worker failed with exit {proc.returncode}: {proc.stderr[-1500:]}")
        try:
            result=loads_unique_json(proc.stdout.splitlines()[-1])
        except ValueError as exc:
            raise WorkerFailure(f"worker stdout is not a single JSON result: {exc}") from exc
        if not isinstance(result,dict) or result.get('kind') not in {
            'TRAIN_DIAGNOSTIC_ONLY', 'F1_PRIOR_PATHS', 'E2_NEXT_DISTRIBUTION',
            'E2_RISK_DISTRIBUTION', 'E2_RISK_DISTRIBUTION_BATCH',
            'E1_TRAIN_DIAGNOSTIC_ONLY', 'E1_PRIOR_PATHS', 'E1_NEXT_DISTRIBUTION',
            'F1_BATCH_PRIOR_PATHS', 'E1_BATCH_PRIOR_PATHS'
        }:
            raise WorkerFailure('worker returned invalid output shape/kind')
        if result.get("status")!="COMPLETED":
            raise WorkerFailure("worker did not finish")
        return result

    def next_distribution(self, *, request:Path, checkpoint:Path,
                          output:Path, configuration:Path)->dict:
        """Request E2 location/std parameters plus their configured density family."""
        return self._next_distribution(command="next-distribution",
            kind="E2_NEXT_DISTRIBUTION",request=request,checkpoint=checkpoint,
            output=output,configuration=configuration)

    def train(self,*,sequences:Path,scope:str,checkpoint:Path,
              configuration:Path,resume:bool=False)->dict:
        args=["train","--sequences",str(sequences),"--scope",scope,
              "--checkout",str(self.config.upstream_checkout),
              "--checkpoint",str(checkpoint),"--configuration",str(configuration)]
        if resume:
            args.append("--resume")
        result=self._run(args)
        if result.get('kind')!='TRAIN_DIAGNOSTIC_ONLY' or not checkpoint.is_file() or not checkpoint.with_suffix('.json').is_file():
            raise WorkerFailure('worker training did not publish its checkpoint')
        return result

    def train_e1(self,*,sequences:Path,scope:str,checkpoint:Path,
                 configuration:Path,resume:bool=False)->dict:
        args=["e1-train","--sequences",str(sequences),"--scope",scope,
              "--checkout",str(self.config.upstream_checkout),
              "--checkpoint",str(checkpoint),"--configuration",str(configuration)]
        if resume:
            args.append("--resume")
        result=self._run(args)
        if (result.get('kind')!='E1_TRAIN_DIAGNOSTIC_ONLY' or not checkpoint.is_file()
                or not checkpoint.with_suffix('.json').is_file()):
            raise WorkerFailure('E1 worker training did not publish its checkpoint')
        return result

    def forecast(self,*,request:Path,checkpoint:Path,
                 output:Path,configuration:Path)->dict:
        return self._forecast(command="forecast",kind="F1_PRIOR_PATHS",request=request,
                              checkpoint=checkpoint,output=output,configuration=configuration)

    def forecast_e1(self,*,request:Path,checkpoint:Path,
                    output:Path,configuration:Path)->dict:
        return self._forecast(command="e1-forecast",kind="E1_PRIOR_PATHS",request=request,
                              checkpoint=checkpoint,output=output,configuration=configuration)

    def forecast_batch(self,*,request:Path,checkpoint:Path,output:Path,
                       configuration:Path,candidate:str)->dict:
        """Forecast a bounded batch of ordered anchors in one JAX worker process."""
        normalized=candidate.upper()
        if normalized not in {"E1","E2"}:
            raise ValueError("candidate must be E1 or E2")
        command="e1-forecast-batch" if normalized=="E1" else "forecast-batch"
        kind="E1_BATCH_PRIOR_PATHS" if normalized=="E1" else "F1_BATCH_PRIOR_PATHS"
        result=self._run([command,"--request",str(request),"--checkout",
                          str(self.config.upstream_checkout),"--checkpoint",str(checkpoint),
                          "--output",str(output),"--configuration",str(configuration)])
        reported_paths=result.get('paths')
        if (result.get('kind')!=kind or Path(result.get('output','')).resolve()!=output.resolve()
                or not output.is_file()
                or type(reported_paths) is not int
                or result.get('sampled_paths_per_anchor') != reported_paths * 4
                or result.get('anchor_chunk_size') != 4
                or result.get('chunk_seed_rule') != 'request_seed_plus_chunk_start_index'
                or type(result.get('rejected_paths')) is not int
                or result['rejected_paths'] < 0
                or not isinstance(result.get('rejected_paths_by_anchor'),list)
                or len(result['rejected_paths_by_anchor']) != result.get('anchors')
                or sum(result['rejected_paths_by_anchor']) != result['rejected_paths']):
            raise WorkerFailure('worker failed to produce a complete batched forecast/rejection report')
        if hashlib.sha256(output.read_bytes()).hexdigest()!=result.get('sha256'):
            raise WorkerFailure('worker batched forecast artifact checksum mismatch')
        import numpy as np
        with np.load(output,allow_pickle=False) as forecast:
            if set(forecast.files)!={'bars'}:
                raise WorkerFailure('worker batched forecast member schema mismatch')
            paths=np.asarray(forecast['bars'])
            anchors=result.get('anchors')
            expected_paths=result.get('paths')
            expected_horizon=result.get('horizon')
            if (type(anchors) is not int or anchors<1 or type(expected_paths) is not int
                or expected_paths<2 or type(expected_horizon) is not int or expected_horizon<1
                or paths.ndim!=4 or paths.shape!=(anchors,expected_paths,expected_horizon,5)
                or not np.isfinite(paths).all() or np.any(paths[...,0:4]<=0)
                or np.any(paths[...,1]<np.maximum(paths[...,0],paths[...,3]))
                or np.any(paths[...,2]>np.minimum(paths[...,0],paths[...,3]))
                or np.any(paths[...,4]<0)):
                raise WorkerFailure('worker batched forecast shape/geometry invalid')
        return result

    def forecast_risk_distribution_batch(self,*,request:Path,checkpoint:Path,
            output:Path,configuration:Path,risk_horizons:tuple[int,...])->dict:
        """Export E2 prior-native RV/extreme/barrier forecasts for batched anchors."""
        return self._forecast_risk_distribution(
            request=request,checkpoint=checkpoint,output=output,
            configuration=configuration,risk_horizons=risk_horizons,batch=True)

    def forecast_risk_distribution(self,*,request:Path,checkpoint:Path,
            output:Path,configuration:Path,risk_horizons:tuple[int,...])->dict:
        """Export E2 prior-native RV/extreme/barrier forecasts for one anchor."""
        return self._forecast_risk_distribution(
            request=request,checkpoint=checkpoint,output=output,
            configuration=configuration,risk_horizons=risk_horizons,batch=False)

    def _forecast_risk_distribution(self,*,request:Path,checkpoint:Path,
            output:Path,configuration:Path,risk_horizons:tuple[int,...],batch:bool)->dict:
        if (not risk_horizons or tuple(sorted(set(risk_horizons)))!=risk_horizons
                or any(type(horizon) is not int or horizon<1 or horizon>60
                       for horizon in risk_horizons)):
            raise ValueError('risk horizons must be sorted unique integers in [1,60]')
        command='risk-distribution-batch' if batch else 'risk-distribution'
        kind='E2_RISK_DISTRIBUTION_BATCH' if batch else 'E2_RISK_DISTRIBUTION'
        result=self._run([command,'--request',str(request),
                          '--checkout',str(self.config.upstream_checkout),
                          '--checkpoint',str(checkpoint),'--output',str(output),
                          '--configuration',str(configuration)])
        if (result.get('kind')!=kind
                or Path(result.get('output','')).resolve()!=output.resolve()
                or result.get('status')!='COMPLETED' or not output.is_file()
                or type(result.get('anchors')) is not int or result['anchors']<1):
            raise WorkerFailure('worker failed to produce a prior risk forecast')
        if hashlib.sha256(output.read_bytes()).hexdigest()!=result.get('sha256'):
            raise WorkerFailure('worker risk forecast artifact checksum mismatch')
        import numpy as np
        with np.load(output,allow_pickle=False) as artifact:
            if set(artifact.files)!={'risk_mean','risk_scale','barrier_logits','risk_horizons'}:
                raise WorkerFailure('worker risk forecast member schema mismatch')
            mean=np.asarray(artifact['risk_mean'],dtype=np.float64)
            scale=np.asarray(artifact['risk_scale'],dtype=np.float64)
            logits=np.asarray(artifact['barrier_logits'],dtype=np.float64)
            horizons=np.asarray(artifact['risk_horizons'])
            expected=(result['anchors'],len(risk_horizons),4)
            if (mean.shape!=expected or scale.shape!=expected
                    or logits.shape!=(result['anchors'],len(risk_horizons),2)
                    or horizons.shape!=(len(risk_horizons),)
                    or not np.issubdtype(horizons.dtype,np.integer)
                    or not np.array_equal(horizons,np.asarray(risk_horizons))
                    or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                    or np.any(scale<=0) or not np.isfinite(logits).all()):
                raise WorkerFailure('worker risk forecast shape, scale, or horizon invalid')
        return result

    def _forecast(self,*,command:str,kind:str,request:Path,checkpoint:Path,
                  output:Path,configuration:Path)->dict:
        result=self._run([command,"--request",str(request),"--checkout",
                          str(self.config.upstream_checkout),"--checkpoint",str(checkpoint),
                          "--output",str(output),"--configuration",str(configuration)])
        if result.get('kind')!=kind or Path(result.get('output','')).resolve()!=output.resolve() or not output.is_file():
            raise WorkerFailure('worker failed to produce requested forecast artifact')
        if hashlib.sha256(output.read_bytes()).hexdigest()!=result.get('sha256'):
            raise WorkerFailure('worker forecast artifact checksum mismatch')
        import numpy as np
        with np.load(output,allow_pickle=False) as forecast:
            if set(forecast.files)!={'bars'}:
                raise WorkerFailure('worker forecast member schema mismatch')
            paths=np.asarray(forecast['bars'])
            expected_paths=result.get('paths')
            expected_horizon=result.get('horizon')
            if (type(expected_paths) is not int or expected_paths<1
                or type(expected_horizon) is not int or expected_horizon<1
                or paths.ndim!=3 or paths.shape!=(expected_paths,expected_horizon,5)
                or not np.isfinite(paths).all() or np.any(paths[...,0:4]<=0)):
                raise WorkerFailure('worker forecast path shape/geometry invalid')
            if (np.any(paths[...,1]<np.maximum(paths[...,0],paths[...,3]))
                or np.any(paths[...,2]>np.minimum(paths[...,0],paths[...,3]))
                or np.any(paths[...,4]<0)):
                raise WorkerFailure('worker forecast OHLCV geometry invalid')
        return result

    def next_distribution_e1(self,*,request:Path,checkpoint:Path,
                             output:Path,configuration:Path)->dict:
        return self._next_distribution(command="e1-next-distribution",
            kind="E1_NEXT_DISTRIBUTION",request=request,checkpoint=checkpoint,
            output=output,configuration=configuration)

    def _next_distribution(self,*,command:str,kind:str,request:Path,checkpoint:Path,
                           output:Path,configuration:Path)->dict:
        result=self._run([command,"--request",str(request),"--checkout",
                          str(self.config.upstream_checkout),"--checkpoint",str(checkpoint),
                          "--output",str(output),"--configuration",str(configuration)])
        if (result.get('kind')!=kind or Path(result.get('output','')).resolve()!=output.resolve()
                or not output.is_file()):
            raise WorkerFailure('worker failed to produce requested forecast distribution')
        import numpy as np
        family=result.get("distribution_family")
        degrees=result.get("student_t_df")
        components=result.get("mixture_components")
        if (family not in {"gaussian","student_t","mixture_gaussian"}
                or result.get("scale_semantics")!="marginal_standard_deviation"
                or (family=="student_t" and (isinstance(degrees,bool)
                    or not isinstance(degrees,(int,float)) or not np.isfinite(degrees)
                    or degrees<=2.0))
                or (family!="student_t" and degrees is not None)
                or (family=="mixture_gaussian" and components!=2)
                or (family!="mixture_gaussian" and components is not None)):
            raise WorkerFailure('worker forecast distribution family metadata is invalid')
        if hashlib.sha256(output.read_bytes()).hexdigest()!=result.get('sha256'):
            raise WorkerFailure('worker distribution artifact checksum mismatch')
        with np.load(output,allow_pickle=False) as distribution:
            required={'standardized_mean','standardized_scale'}
            if family=="mixture_gaussian":
                required|={"mixture_means","mixture_scales","mixture_weights"}
            if set(distribution.files)!=required:
                raise WorkerFailure('worker distribution member schema mismatch')
            mean=np.asarray(distribution['standardized_mean'],dtype=np.float64)
            scale=np.asarray(distribution['standardized_scale'],dtype=np.float64)
            if (mean.shape!=(1,5) or scale.shape!=mean.shape or not np.isfinite(mean).all()
                    or not np.isfinite(scale).all() or np.any(scale<=0)):
                raise WorkerFailure('worker next-step distribution is invalid')
            if family=="mixture_gaussian":
                means=np.asarray(distribution['mixture_means'],dtype=np.float64)
                scales=np.asarray(distribution['mixture_scales'],dtype=np.float64)
                weights=np.asarray(distribution['mixture_weights'],dtype=np.float64)
                if (means.shape!=(1,2,5) or scales.shape!=means.shape or weights.shape!=(1,2)
                        or not np.isfinite(means).all() or not np.isfinite(scales).all()
                        or not np.isfinite(weights).all() or np.any(scales<=0)
                        or np.any(weights<0) or not np.allclose(weights.sum(-1),1.0,atol=1e-5)):
                    raise WorkerFailure('worker mixture components are invalid')
                expected_mean=(means*weights[...,None]).sum(-2)
                expected_var=(weights[...,None]*(scales**2+(means-expected_mean[...,None,:])**2)).sum(-2)
                if (not np.allclose(expected_mean,mean,rtol=1e-5,atol=1e-6)
                        or not np.allclose(np.sqrt(expected_var),scale,rtol=1e-5,atol=1e-6)):
                    raise WorkerFailure('worker mixture moments disagree with its components')
        return result
