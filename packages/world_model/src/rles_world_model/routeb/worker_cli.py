"""Standalone JAX worker entry point, invoked by Python 3.14 host via subprocess.

This worker takes ONLY prepared Train NPZ or already-observable context NPZ.
It never imports NautilusTrader or reads curated raw market parquet.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import os
import tempfile
from pathlib import Path
import sys
import numpy as np
from rles_data.io import strict_json_loads
from .jax_model import RSSMConfig
from .jax_gru import GRUConfig
from .jax_train import (train, forecast_distribution_standardized,
                        forecast_standardized, TrainingConfig)
from .jax_gru_train import (train_gru, forecast_gru,
                            forecast_gru_distribution)
from .codec import OHLCVCodec,TrainOnlyStandardizer
from .market import MINUTE_NS


def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(prog="routeb-jax-worker")
    subs=ap.add_subparsers(dest="command",required=True)
    t=subs.add_parser("train")
    t.add_argument("--sequences",required=True)
    t.add_argument("--scope",required=True)
    t.add_argument("--checkout",required=True)
    t.add_argument("--checkpoint",required=True)
    t.add_argument("--configuration",required=True)
    t.add_argument("--resume",action="store_true")
    bt=subs.add_parser("b3-train")
    bt.add_argument("--sequences",required=True)
    bt.add_argument("--scope",required=True)
    bt.add_argument("--fold-sha256",required=True)
    bt.add_argument("--checkout",required=True)
    bt.add_argument("--checkpoint",required=True)
    bt.add_argument("--configuration",required=True)
    bt.add_argument("--resume",action="store_true")
    br=subs.add_parser("b3-rollout")
    br.add_argument("--request",required=True)
    br.add_argument("--context",required=True)
    br.add_argument("--checkout",required=True)
    br.add_argument("--checkpoint",required=True)
    br.add_argument("--output",required=True)
    br.add_argument("--configuration",required=True)
    bbr=subs.add_parser("b3-rollout-batch")
    bbr.add_argument("--request",required=True)
    bbr.add_argument("--checkout",required=True)
    bbr.add_argument("--checkpoint",required=True)
    bbr.add_argument("--output",required=True)
    bbr.add_argument("--configuration",required=True)
    f=subs.add_parser("forecast")
    f.add_argument("--request",required=True)
    f.add_argument("--checkout",required=True)
    f.add_argument("--checkpoint",required=True)
    f.add_argument("--output",required=True)
    f.add_argument("--configuration",required=True)
    d=subs.add_parser("next-distribution")
    d.add_argument("--request",required=True)
    d.add_argument("--checkout",required=True)
    d.add_argument("--checkpoint",required=True)
    d.add_argument("--output",required=True)
    d.add_argument("--configuration",required=True)
    et=subs.add_parser("e1-train")
    et.add_argument("--sequences",required=True)
    et.add_argument("--scope",required=True)
    et.add_argument("--checkout",required=True)
    et.add_argument("--checkpoint",required=True)
    et.add_argument("--configuration",required=True)
    et.add_argument("--resume",action="store_true")
    ef=subs.add_parser("e1-forecast")
    ef.add_argument("--request",required=True)
    ef.add_argument("--checkout",required=True)
    ef.add_argument("--checkpoint",required=True)
    ef.add_argument("--output",required=True)
    ef.add_argument("--configuration",required=True)
    bf=subs.add_parser("forecast-batch")
    bf.add_argument("--request",required=True)
    bf.add_argument("--checkout",required=True)
    bf.add_argument("--checkpoint",required=True)
    bf.add_argument("--output",required=True)
    bf.add_argument("--configuration",required=True)
    ebf=subs.add_parser("e1-forecast-batch")
    ebf.add_argument("--request",required=True)
    ebf.add_argument("--checkout",required=True)
    ebf.add_argument("--checkpoint",required=True)
    ebf.add_argument("--output",required=True)
    ebf.add_argument("--configuration",required=True)
    bd=subs.add_parser("next-distribution-batch")
    bd.add_argument("--request",required=True)
    bd.add_argument("--checkout",required=True)
    bd.add_argument("--checkpoint",required=True)
    bd.add_argument("--output",required=True)
    bd.add_argument("--configuration",required=True)
    ebd=subs.add_parser("e1-next-distribution-batch")
    ebd.add_argument("--request",required=True)
    ebd.add_argument("--checkout",required=True)
    ebd.add_argument("--checkpoint",required=True)
    ebd.add_argument("--output",required=True)
    ebd.add_argument("--configuration",required=True)
    ed=subs.add_parser("e1-next-distribution")
    ed.add_argument("--request",required=True)
    ed.add_argument("--checkout",required=True)
    ed.add_argument("--checkpoint",required=True)
    ed.add_argument("--output",required=True)
    ed.add_argument("--configuration",required=True)
    args=ap.parse_args(argv)
    try:
        config=strict_json_loads(
            Path(args.configuration).read_bytes(), context='Route B worker configuration',
        )
        if args.command == "b3-train":
            from .hierarchy_artifacts import (
                hierarchy_scaler_sha256, load_hierarchical_train_artifact,
            )
            from .jax_hierarchy_train import (
                HierarchicalModelConfig, HierarchicalTrainingConfig,
                train_hierarchical,
            )

            if (not isinstance(config, dict) or set(config) != {"model", "training"}
                    or not isinstance(config["model"], dict)
                    or set(config["model"]) != {"minute", "hour", "day"}
                    or any(not isinstance(config["model"][scale], dict)
                           for scale in ("minute", "hour", "day"))
                    or not isinstance(config["training"], dict)):
                raise ValueError("B3 config requires minute/hour/day models and training")
            model_config = HierarchicalModelConfig(**{
                scale: RSSMConfig(**config["model"][scale])
                for scale in ("minute", "hour", "day")
            })
            training_config = HierarchicalTrainingConfig(**config["training"])
            tensors, scalers, metadata = load_hierarchical_train_artifact(
                args.sequences, expected_scope=args.scope,
                expected_fold_sha256=args.fold_sha256)
            reports = train_hierarchical(
                tensors=tensors, train_scope_id=args.scope,
                dataset_id=metadata["dataset_id"],
                scaler_sha256=hierarchy_scaler_sha256(scalers),
                checkout=args.checkout, output_checkpoint=args.checkpoint,
                model_config=model_config, training_config=training_config,
                resume=args.resume)
            result = {
                "status": "COMPLETED",
                "kind": "B3_TRAIN_DIAGNOSTIC_ONLY",
                "checkpoint": args.checkpoint,
                "fold_sha256": args.fold_sha256,
                "hierarchy_tensor_sha256": metadata["hierarchy_tensor_sha256"],
                "scaler_sha256": metadata["scaler_sha256"],
                "reports": reports,
                "validation_scored": False,
                "test_scored": False,
                "economic_certification": False,
            }
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "b3-rollout":
            from .hierarchy_artifacts import load_hierarchical_inference_artifact
            from .jax_hierarchy_train import (
                HierarchicalModelConfig, HierarchicalTrainingConfig,
                generate_recursive_hierarchical_f2_paths,
            )

            if (not isinstance(config, dict) or set(config) != {"model", "training"}
                    or not isinstance(config["model"], dict)
                    or set(config["model"]) != {"minute", "hour", "day"}
                    or any(not isinstance(config["model"][scale], dict)
                           for scale in ("minute", "hour", "day"))
                    or not isinstance(config["training"], dict)):
                raise ValueError("B3 config requires minute/hour/day models and training")
            request = strict_json_loads(
                Path(args.request).read_bytes(), context="B3 rollout request")
            expected = {
                "schema", "train_scope_id", "dataset_id", "fold_sha256",
                "context_artifact_sha256", "scaler_sha256", "train_tensor_sha256",
                "checkpoint_sha256", "configuration_sha256", "anchor_ts_ns",
                "horizon", "n_paths", "forecast_seed", "rollout_mode",
                "hour_availability_delay_ns", "day_availability_delay_ns",
                "max_attempts_per_bar", "max_total_bar_attempts",
            }
            if (not isinstance(request, dict) or set(request) != expected
                    or request.get("schema") != "routeb.b3-rollout-request.v1"):
                raise ValueError("unexpected B3 rollout request schema")
            string_fields = ("train_scope_id", "dataset_id", "fold_sha256",
                             "context_artifact_sha256", "scaler_sha256",
                             "train_tensor_sha256", "checkpoint_sha256",
                             "configuration_sha256")
            if any(not isinstance(request[name], str) or not request[name]
                   for name in string_fields):
                raise ValueError("B3 rollout identities must be nonempty strings")
            rollout_sha_fields = ("fold_sha256", "context_artifact_sha256", "scaler_sha256",
                                  "train_tensor_sha256", "checkpoint_sha256",
                                  "configuration_sha256")
            if any(len(request[name]) != 64 or any(char not in "0123456789abcdef"
                   for char in request[name]) for name in rollout_sha_fields):
                raise ValueError("B3 rollout identities must be lowercase SHA-256")
            integer_fields = ("anchor_ts_ns", "horizon", "n_paths", "forecast_seed",
                              "hour_availability_delay_ns", "day_availability_delay_ns",
                              "max_attempts_per_bar", "max_total_bar_attempts")
            if any(type(request[name]) is not int for name in integer_fields):
                raise ValueError("B3 rollout integer fields must not be coerced")
            if (request["dataset_id"] == "" or request["anchor_ts_ns"] <= 0
                    or request["anchor_ts_ns"] % MINUTE_NS
                    or not 1 <= request["horizon"] <= 60
                    or not 2 <= request["n_paths"] <= 1024
                    or request["forecast_seed"] < 0
                    or request["hour_availability_delay_ns"] < 0
                    or request["day_availability_delay_ns"] < 0
                    or not 1 <= request["max_attempts_per_bar"] <= 100
                    or not 1 <= request["max_total_bar_attempts"] <= 1_000_000
                    or request["rollout_mode"] not in {"E3", "E4"}):
                raise ValueError("B3 rollout parameters are outside the supported contract")
            if (hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
                    != request["checkpoint_sha256"]
                    or hashlib.sha256(Path(args.configuration).read_bytes()).hexdigest()
                    != request["configuration_sha256"]):
                raise ValueError("B3 rollout checkpoint or configuration digest mismatch")
            tensors, scalers, observed, context_metadata = load_hierarchical_inference_artifact(
                args.context, expected_scope=request["train_scope_id"],
                expected_fold_sha256=request["fold_sha256"],
                expected_anchor_ts_ns=request["anchor_ts_ns"],
                expected_scaler_sha256=request["scaler_sha256"])
            if (context_metadata["dataset_id"] != request["dataset_id"]
                    or hashlib.sha256(Path(args.context).read_bytes()).hexdigest()
                    != request["context_artifact_sha256"]):
                raise ValueError("B3 rollout context artifact identity mismatch")
            model_config = HierarchicalModelConfig(**{
                scale: RSSMConfig(**config["model"][scale])
                for scale in ("minute", "hour", "day")
            })
            training_config = HierarchicalTrainingConfig(**config["training"])
            decisions = np.arange(
                request["anchor_ts_ns"] + MINUTE_NS,
                request["anchor_ts_ns"] + (request["horizon"] + 1) * MINUTE_NS,
                MINUTE_NS, dtype=np.int64)
            rollout = generate_recursive_hierarchical_f2_paths(
                tensors=tensors, train_scope_id=request["train_scope_id"],
                dataset_id=request["dataset_id"], scaler_sha256=request["scaler_sha256"],
                checkout=args.checkout, checkpoint=args.checkpoint,
                model_config=model_config, training_config=training_config,
                training_tensor_sha256=request["train_tensor_sha256"],
                inference_context_sha256=context_metadata["hierarchy_tensor_sha256"],
                future_minute_decisions_ns=decisions,
                hour_availability_delay_ns=request["hour_availability_delay_ns"],
                day_availability_delay_ns=request["day_availability_delay_ns"],
                minute_scaler=scalers.minute, hour_scaler=scalers.hour,
                day_scaler=scalers.day, observed_prefix=observed,
                paths=request["n_paths"], forecast_seed=request["forecast_seed"],
                advance_slow_priors=request["rollout_mode"] == "E4",
                max_attempts_per_bar=request["max_attempts_per_bar"],
                max_total_bar_attempts=request["max_total_bar_attempts"])
            output = Path(args.output)
            if output.exists() or not output.parent.is_dir():
                raise FileExistsError("B3 rollout output must be new in an existing directory")
            with tempfile.NamedTemporaryFile(
                    dir=output.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, bars=rollout.ensemble.minute,
                                    anchor_ts_ns=np.asarray(request["anchor_ts_ns"],
                                                           dtype=np.int64))
            try:
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            result = {
                "status": "COMPLETED", "kind": "B3_ROLLOUT_PATHS",
                "candidate": request["rollout_mode"],
                "rollout_mode": ("FROZEN_OBSERVED_SLOW_POSTERIOR"
                    if request["rollout_mode"] == "E3"
                    else "ADVANCE_PATH_CONDITIONED_SLOW_POSTERIOR"),
                "anchor_ts_ns": request["anchor_ts_ns"],
                "paths": request["n_paths"], "horizon": request["horizon"],
                "attempted_bar_samples": rollout.attempted_bar_samples,
                "rejected_bar_samples": rollout.rejected_bar_samples,
                "rejection_rate": rollout.rejection_rate,
                "context_artifact_sha256": request["context_artifact_sha256"],
                "output": str(output.resolve()),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "b3-rollout-batch":
            from .hierarchy_artifacts import load_hierarchical_inference_artifact
            from .jax_hierarchy_train import (
                HierarchicalModelConfig, HierarchicalTrainingConfig,
                create_stateful_hierarchical_f2_predictor,
                generate_recursive_hierarchical_f2_paths,
            )

            if (not isinstance(config, dict) or set(config) != {"model", "training"}
                    or not isinstance(config["model"], dict)
                    or set(config["model"]) != {"minute", "hour", "day"}
                    or any(not isinstance(config["model"][scale], dict)
                           for scale in ("minute", "hour", "day"))
                    or not isinstance(config["training"], dict)):
                raise ValueError("B3 config requires minute/hour/day models and training")
            request = strict_json_loads(
                Path(args.request).read_bytes(), context="B3 rollout batch request")
            expected = {
                "schema", "train_scope_id", "dataset_id", "fold_sha256",
                "scaler_sha256", "train_tensor_sha256", "checkpoint_sha256",
                "configuration_sha256", "inference_context_sha256",
                "context_artifacts", "horizon", "n_paths", "forecast_seed",
                "hour_availability_delay_ns", "day_availability_delay_ns",
                "max_attempts_per_bar", "max_total_bar_attempts",
            }
            if (not isinstance(request, dict) or set(request) != expected
                    or request.get("schema") != "routeb.b3-rollout-batch-request.v1"):
                raise ValueError("unexpected B3 rollout batch request schema")
            batch_sha_fields = ("fold_sha256", "scaler_sha256", "train_tensor_sha256",
                                "checkpoint_sha256", "configuration_sha256",
                                "inference_context_sha256")
            if (any(not isinstance(request.get(name), str) or len(request[name]) != 64
                    or any(char not in "0123456789abcdef" for char in request[name])
                    for name in batch_sha_fields)
                    or not isinstance(request.get("train_scope_id"), str)
                    or not request["train_scope_id"].strip()
                    or not isinstance(request.get("dataset_id"), str)
                    or not request["dataset_id"].strip()):
                raise ValueError("B3 rollout batch identity is invalid")
            batch_integer_fields = ("horizon", "n_paths", "forecast_seed",
                                    "hour_availability_delay_ns", "day_availability_delay_ns",
                                    "max_attempts_per_bar", "max_total_bar_attempts")
            if any(type(request.get(name)) is not int for name in batch_integer_fields):
                raise ValueError("B3 rollout batch integer fields must not be coerced")
            artifacts = request.get("context_artifacts")
            if (not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 64
                    or not 1 <= request["horizon"] <= 60
                    or not 2 <= request["n_paths"] <= 1024
                    or request["forecast_seed"] < 0
                    or request["hour_availability_delay_ns"] < 0
                    or request["day_availability_delay_ns"] < 0
                    or not 1 <= request["max_attempts_per_bar"] <= 100
                    or not 1 <= request["max_total_bar_attempts"] <= 100_000_000
                    or request["n_paths"] * request["horizon"]
                       * request["max_attempts_per_bar"] > 1_000_000
                    or len(artifacts) * request["n_paths"] * request["horizon"]
                       * request["max_attempts_per_bar"] > request["max_total_bar_attempts"]):
                raise ValueError("B3 rollout batch exceeds its explicit work budget")
            if (hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
                    != request["checkpoint_sha256"]
                    or hashlib.sha256(Path(args.configuration).read_bytes()).hexdigest()
                    != request["configuration_sha256"]):
                raise ValueError("B3 rollout batch checkpoint/configuration digest mismatch")
            if any(not isinstance(item, dict)
                   or set(item) != {"path", "artifact_sha256", "anchor_ts_ns"}
                   or not isinstance(item["path"], str)
                   or type(item["anchor_ts_ns"]) is not int
                   or item["anchor_ts_ns"] <= 0
                   or item["anchor_ts_ns"] % MINUTE_NS
                   or not isinstance(item["artifact_sha256"], str)
                   or len(item["artifact_sha256"]) != 64
                   or any(char not in "0123456789abcdef" for char in item["artifact_sha256"])
                   for item in artifacts):
                raise ValueError("B3 rollout batch context list is invalid")
            anchors = [item["anchor_ts_ns"] for item in artifacts]
            if any(right <= left for left, right in zip(anchors, anchors[1:])):
                raise ValueError("B3 rollout batch anchors must be strictly increasing")
            loaded = []
            for item in artifacts:
                context_path = Path(item["path"])
                if hashlib.sha256(context_path.read_bytes()).hexdigest() != item["artifact_sha256"]:
                    raise ValueError("B3 rollout batch context artifact digest mismatch")
                inference_artifact = load_hierarchical_inference_artifact(
                    context_path, expected_scope=request["train_scope_id"],
                    expected_fold_sha256=request["fold_sha256"],
                    expected_anchor_ts_ns=item["anchor_ts_ns"],
                    expected_scaler_sha256=request["scaler_sha256"])
                tensors, scalers, observed, metadata = inference_artifact
                if metadata["dataset_id"] != request["dataset_id"]:
                    raise ValueError("B3 rollout batch context dataset identity mismatch")
                loaded.append((tensors, scalers, observed, metadata))
            context_sha = hashlib.sha256(b"".join(
                bytes.fromhex(item[3]["hierarchy_tensor_sha256"])
                for item in loaded)).hexdigest()
            if context_sha != request["inference_context_sha256"]:
                raise ValueError("B3 rollout batch aggregate context identity mismatch")
            model_config = HierarchicalModelConfig(**{
                scale: RSSMConfig(**config["model"][scale])
                for scale in ("minute", "hour", "day")
            })
            training_config = HierarchicalTrainingConfig(**config["training"])
            outputs = {}
            sampling = {}
            for candidate, advance in (("E3", False), ("E4", True)):
                first_tensors, first_scalers, _, first_meta = loaded[0]
                predictor = create_stateful_hierarchical_f2_predictor(
                    tensors=first_tensors, train_scope_id=request["train_scope_id"],
                    dataset_id=request["dataset_id"], scaler_sha256=request["scaler_sha256"],
                    checkout=args.checkout, checkpoint=args.checkpoint,
                    model_config=model_config, training_config=training_config,
                    training_tensor_sha256=request["train_tensor_sha256"],
                    inference_context_sha256=first_meta["hierarchy_tensor_sha256"],
                    paths=request["n_paths"], forecast_seed=request["forecast_seed"],
                    advance_slow_priors=advance)
                rows = []
                attempted = rejected = 0
                for tensors, scalers, observed, metadata in loaded:
                    anchor = metadata["anchor_ts_ns"]
                    decisions = np.arange(anchor + MINUTE_NS,
                        anchor + (request["horizon"] + 1) * MINUTE_NS,
                        MINUTE_NS, dtype=np.int64)
                    rollout = generate_recursive_hierarchical_f2_paths(
                        tensors=tensors, train_scope_id=request["train_scope_id"],
                        dataset_id=request["dataset_id"], scaler_sha256=request["scaler_sha256"],
                        checkout=args.checkout, checkpoint=args.checkpoint,
                        model_config=model_config, training_config=training_config,
                        training_tensor_sha256=request["train_tensor_sha256"],
                        inference_context_sha256=metadata["hierarchy_tensor_sha256"],
                        future_minute_decisions_ns=decisions,
                        hour_availability_delay_ns=request["hour_availability_delay_ns"],
                        day_availability_delay_ns=request["day_availability_delay_ns"],
                        minute_scaler=scalers.minute, hour_scaler=scalers.hour,
                        day_scaler=scalers.day, observed_prefix=observed,
                        paths=request["n_paths"], forecast_seed=request["forecast_seed"],
                        advance_slow_priors=advance, stateful_predictor=predictor,
                        max_attempts_per_bar=request["max_attempts_per_bar"],
                        max_total_bar_attempts=1_000_000)
                    rows.append(rollout.ensemble.minute)
                    attempted += rollout.attempted_bar_samples
                    rejected += rollout.rejected_bar_samples
                outputs[candidate] = np.stack(rows)
                sampling[candidate] = {
                    "attempted": attempted, "rejected": rejected,
                    "rejection_rate": rejected / attempted,
                }
            output = Path(args.output)
            if output.exists() or not output.parent.is_dir():
                raise FileExistsError("B3 rollout batch output must be new in an existing directory")
            with tempfile.NamedTemporaryFile(
                    dir=output.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, bars_e3=outputs["E3"], bars_e4=outputs["E4"],
                                    anchor_ts_ns=np.asarray(anchors, dtype=np.int64))
            try:
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            result = {
                "status": "COMPLETED", "kind": "B3_ROLLOUT_BATCH_PATHS",
                "anchors": len(anchors), "paths": request["n_paths"],
                "horizon": request["horizon"],
                "inference_context_sha256": context_sha,
                "sampling": sampling, "output": str(output.resolve()),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        if set(config)!={"model","training"}:
            raise ValueError("unknown model/training configuration keys")
        e1=args.command.startswith('e1-')
        mc=(GRUConfig(**config['model']) if e1 else RSSMConfig(**config['model']))
        tc=TrainingConfig(**config['training'])
        if args.command in {'train','e1-train'}:
            if e1:
                assert isinstance(mc,GRUConfig)
                report=train_gru(dataset_npz=args.sequences,train_scope_id=args.scope,
                                 output_checkpoint=args.checkpoint,model_config=mc,
                                 training_config=tc,resume=args.resume)
            else:
                assert isinstance(mc,RSSMConfig)
                report=train(dataset_npz=args.sequences,train_scope_id=args.scope,
                             checkout=args.checkout,output_checkpoint=args.checkpoint,
                             model_config=mc,training_config=tc,resume=args.resume)
            result={"status":"COMPLETED",
                    "kind":"E1_TRAIN_DIAGNOSTIC_ONLY" if e1 else "TRAIN_DIAGNOSTIC_ONLY",
                    "checkpoint":args.checkpoint,"reports":report}
        else:
            request=strict_json_loads(
                Path(args.request).read_bytes(), context='Route B worker forecast request',
            )
            batch=args.command in {
                'forecast-batch','e1-forecast-batch',
                'next-distribution-batch','e1-next-distribution-batch'}
            expected={"schema","train_scope_id","dataset_id","history_npz","context_sha256","scaler_sha256","horizon",
                      "n_paths","seed","train_npz_sha256","instrument"}
            expected |= ({"anchor_timestamps_ns"} if batch else
                         {"anchor_ts_ns","anchor_close","first_future_close_ns"})
            schema="routeb.forecast-batch-request.v1" if batch else "routeb.forecast-request.v2"
            if set(request)!=expected or request['schema']!=schema:
                raise ValueError("unexpected forecast request schema")
            if batch and (not isinstance(request.get('anchor_timestamps_ns'),list)
                or not request['anchor_timestamps_ns']
                or any(type(value) is not int for value in request['anchor_timestamps_ns'])):
                raise ValueError('batched anchor timestamps must be a nonempty integer list')
            int_fields=('horizon','n_paths','seed') if batch else ('horizon','n_paths','seed','anchor_ts_ns','first_future_close_ns')
            if any(type(request[k]) is not int for k in int_fields):
                raise ValueError('forecast integer fields must not be coerced from strings or bools')
            h=request['horizon'];p=request['n_paths']
            if not batch:
                if int(request["first_future_close_ns"])!=int(request["anchor_ts_ns"])+MINUTE_NS:
                    raise ValueError("forecast time does not start after anchor close")
                if request['seed']<0 or request['anchor_ts_ns']<=0 or request['anchor_ts_ns']%MINUTE_NS:
                    raise ValueError('forecast seed or anchor timestamp invalid')
            elif request['seed']<0:
                raise ValueError('forecast seed invalid')
            if h<1 or h>60 or p<2 or p>1024:
                raise ValueError("forecast size exceeds supported research limits")
            context_path=Path(request['history_npz'])
            if hashlib.sha256(context_path.read_bytes()).hexdigest()!=request['context_sha256']:
                raise ValueError('forecast context checksum mismatch')
            with np.load(context_path,allow_pickle=False) as arr:
                expected_members=({'histories','scaler_mean','scaler_std','anchor_timestamps_ns',
                                   'close_timestamps','available_timestamps','anchor_closes'} if batch else
                                  {'history','scaler_mean','scaler_std','anchor_ns','close_timestamps','available_timestamps'})
                if set(arr.files)!=expected_members:
                    raise ValueError('invalid forecast context members')
                context=np.asarray(arr['histories' if batch else 'history'],dtype=np.float32)
                mean=np.asarray(arr['scaler_mean'],dtype=np.float64)
                std=np.asarray(arr['scaler_std'],dtype=np.float64)
                context_anchors=np.asarray(arr['anchor_timestamps_ns'],dtype=np.int64) if batch else np.asarray([int(arr['anchor_ns'])],dtype=np.int64)
                anchor_closes=np.asarray(arr['anchor_closes'],dtype=np.float64) if batch else np.asarray([request['anchor_close']],dtype=np.float64)
                closes=np.asarray(arr['close_timestamps'],dtype=np.int64)
                available=np.asarray(arr['available_timestamps'],dtype=np.int64)
            if batch:
                if (context.ndim!=3 or context.shape[0]<1 or context.shape[2]!=5 or context.shape[1]<2
                    or context_anchors.shape!=(context.shape[0],) or anchor_closes.shape!=(context.shape[0],)
                    or closes.shape!=context.shape[:2] or available.shape!=closes.shape
                    or not np.array_equal(closes[:,-1],context_anchors)
                    or not np.all(np.diff(closes,axis=1)==MINUTE_NS)
                    or np.any(available>context_anchors[:,None])
                    or np.any(available<closes) or np.any(np.diff(available,axis=1)<0)
                    or not np.isfinite(anchor_closes).all() or np.any(anchor_closes<=0)
                    or not np.array_equal(np.asarray(request['anchor_timestamps_ns'],dtype=np.int64),context_anchors)):
                    raise ValueError('batched forecast context violates causal timestamps or request anchors')
                if (np.any(context_anchors<=0) or np.any(context_anchors%MINUTE_NS)
                    or np.any(np.diff(context_anchors)<=0)):
                    raise ValueError('batched anchor timestamps must be ordered and minute aligned')
            else:
                if context.ndim!=2 or context.shape[1]!=5 or context.shape[0]<2:
                    raise ValueError("expected [T,5] standardized historical context")
                if context_anchors[0]!=int(request["anchor_ts_ns"]):
                    raise ValueError("context anchor differs from forecast request")
                if (closes.shape!=(len(context),) or available.shape!=closes.shape
                    or not np.all(np.diff(closes)==MINUTE_NS)
                    or closes[-1]!=context_anchors[0] or np.any(available>context_anchors[0])
                    or np.any(available<closes) or np.any(np.diff(available)<0)):
                    raise ValueError('forecast context violates causal timestamps')
            if (not np.isfinite(context).all() or mean.shape!=(5,) or std.shape!=(5,)
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any()):
                raise ValueError("invalid frozen scaler or forecast context")
            codec=TrainOnlyStandardizer(mean,std,request['dataset_id'],'train')
            if codec.fingerprint!=request['scaler_sha256']:
                raise ValueError('frozen Train scaler fingerprint mismatch')
            if args.command in {'forecast','e1-forecast','forecast-batch','e1-forecast-batch'}:
                if batch:
                    histories=np.repeat(context,p,axis=0)
                else:
                    histories=np.broadcast_to(context[None,:,:],(p,*context.shape))
                if e1:
                    assert isinstance(mc,GRUConfig)
                    std_paths=forecast_gru(
                        checkpoint=args.checkpoint,model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],history=histories,
                        steps=h,forecast_seed=request['seed'])
                else:
                    assert isinstance(mc,RSSMConfig)
                    std_paths=forecast_standardized(
                        checkout=args.checkout,checkpoint=args.checkpoint,
                        model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],history=histories,
                        steps=h,forecast_seed=request['seed'])
                market: list[np.ndarray]=[]
                if batch:
                    std_paths=std_paths.reshape((len(context),p,h,5))
                    for i,anchor_paths in enumerate(std_paths):
                        market_anchor: list[list[list[float]]]=[]
                        for x in anchor_paths:
                            bars=OHLCVCodec.decode(codec.inverse(x),
                                previous_close=float(anchor_closes[i]),
                                start_close_ns=int(context_anchors[i])+MINUTE_NS,
                                instrument=request['instrument'])
                            market_anchor.append([[b.open,b.high,b.low,b.close,b.volume] for b in bars])
                        market.append(np.asarray(market_anchor,dtype=np.float64))
                    members={"bars":np.asarray(market,dtype=np.float64)}
                else:
                    for x in std_paths:
                        bars=OHLCVCodec.decode(codec.inverse(x),
                                              previous_close=float(request['anchor_close']),
                                              start_close_ns=int(request['first_future_close_ns']),
                                              instrument=request['instrument'])
                        market.append(np.asarray(
                            [[b.open,b.high,b.low,b.close,b.volume] for b in bars],
                            dtype=np.float64))
                    members={"bars":np.asarray(market,dtype=np.float64)}
                kind=("E1_BATCH_PRIOR_PATHS" if batch else "E1_PRIOR_PATHS") if e1 else (
                    "F1_BATCH_PRIOR_PATHS" if batch else "F1_PRIOR_PATHS")
            else:
                if h!=1:
                    raise ValueError("next-distribution request horizon must equal one")
                if e1:
                    assert isinstance(mc,GRUConfig)
                    mean,scale=forecast_gru_distribution(
                        checkpoint=args.checkpoint,model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],
                        history=context if batch else context[None,:,:])
                else:
                    assert isinstance(mc,RSSMConfig)
                    mean,scale=forecast_distribution_standardized(
                        checkout=args.checkout,checkpoint=args.checkpoint,
                        model_config=mc,training_config=tc,
                        train_scope_id=request['train_scope_id'],dataset_id=request['dataset_id'],
                        train_npz_sha256=request["train_npz_sha256"],
                        scaler_sha256=request['scaler_sha256'],
                        history=context if batch else context[None,:,:])
                # DreamerV3 commonly emits bfloat16; NumPy otherwise serializes
                # that dtype as opaque |V2, which downstream scoring cannot read.
                members={"standardized_mean":np.asarray(mean,dtype=np.float32),
                         "standardized_scale":np.asarray(scale,dtype=np.float32)}
                kind=("E1_BATCH_NEXT_DISTRIBUTION" if batch else "E1_NEXT_DISTRIBUTION") if e1 else (
                    "E2_BATCH_NEXT_DISTRIBUTION" if batch else "E2_NEXT_DISTRIBUTION")
            output=Path(args.output).resolve()
            output.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.npz',delete=False) as tmp_file:
                temp_path=Path(tmp_file.name)
                np.savez_compressed(tmp_file,**members)  # type: ignore[arg-type]
            try:
                os.replace(temp_path,output)
            finally:
                temp_path.unlink(missing_ok=True)
            result={"status":"COMPLETED","kind":kind,
                    "output":str(output),
                    "paths":p if args.command in {'forecast','e1-forecast','forecast-batch','e1-forecast-batch'} else 1,
                    "anchors":len(context_anchors) if batch else 1,
                    "horizon":h,
                    "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
                    "economic_certification":False}
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED","error":type(exc).__name__,
                          "detail":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
