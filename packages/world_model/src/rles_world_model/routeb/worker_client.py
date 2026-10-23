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
from rles_data.io import strict_json_loads
from rles_data.core import DataError

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
            result=strict_json_loads(proc.stdout, context='Route B worker response')
        except DataError as exc:
            raise WorkerFailure("worker stdout is not a single JSON result") from exc
        if not isinstance(result,dict) or result.get('kind') not in {
            'TRAIN_DIAGNOSTIC_ONLY', 'F1_PRIOR_PATHS', 'E2_NEXT_DISTRIBUTION',
            'E1_TRAIN_DIAGNOSTIC_ONLY', 'E1_PRIOR_PATHS', 'E1_NEXT_DISTRIBUTION',
            'F1_BATCH_PRIOR_PATHS', 'E1_BATCH_PRIOR_PATHS',
            'E1_BATCH_NEXT_DISTRIBUTION', 'E2_BATCH_NEXT_DISTRIBUTION'
        }:
            raise WorkerFailure('worker returned invalid output shape/kind')
        if result.get("status")!="COMPLETED":
            raise WorkerFailure("worker did not finish")
        return result

    def next_distribution(self, *, request:Path, checkpoint:Path,
                          output:Path, configuration:Path)->dict:
        """Request exact standardized one-step E2 feature distribution parameters."""
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
        if (result.get('kind')!=kind or Path(result.get('output','')).resolve()!=output.resolve()
                or not output.is_file()):
            raise WorkerFailure('worker failed to produce requested batched forecast artifact')
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

    def next_distribution_batch(self, *, request:Path, checkpoint:Path,
                                output:Path, configuration:Path,
                                candidate:str)->dict:
        """Export exact one-step heads for every ordered Validation anchor."""
        normalized=candidate.upper()
        if normalized not in {"E1","E2"}:
            raise ValueError("candidate must be E1 or E2")
        command=("e1-next-distribution-batch" if normalized=="E1"
                 else "next-distribution-batch")
        kind=f"{normalized}_BATCH_NEXT_DISTRIBUTION"
        return self._next_distribution(command=command,kind=kind,request=request,
            checkpoint=checkpoint,output=output,configuration=configuration,
            batch=True)

    def _next_distribution(self,*,command:str,kind:str,request:Path,checkpoint:Path,
                           output:Path,configuration:Path,batch:bool=False)->dict:
        result=self._run([command,"--request",str(request),"--checkout",
                          str(self.config.upstream_checkout),"--checkpoint",str(checkpoint),
                          "--output",str(output),"--configuration",str(configuration)])
        if (result.get('kind')!=kind or Path(result.get('output','')).resolve()!=output.resolve()
                or not output.is_file()):
            raise WorkerFailure('worker failed to produce requested forecast distribution')
        if hashlib.sha256(output.read_bytes()).hexdigest()!=result.get('sha256'):
            raise WorkerFailure('worker distribution artifact checksum mismatch')
        import numpy as np
        with np.load(output,allow_pickle=False) as distribution:
            if set(distribution.files)!={'standardized_mean','standardized_scale'}:
                raise WorkerFailure('worker distribution member schema mismatch')
            mean=np.asarray(distribution['standardized_mean'])
            scale=np.asarray(distribution['standardized_scale'])
            expected_anchors=result.get('anchors') if batch else 1
            expected_shape=((expected_anchors,5) if batch else (1,5))
            if (type(expected_anchors) is not int or expected_anchors<1
                    or mean.shape!=expected_shape or scale.shape!=mean.shape or not np.isfinite(mean).all()
                    or not np.isfinite(scale).all() or np.any(scale<=0)):
                raise WorkerFailure('worker next-step distribution is invalid')
        return result
