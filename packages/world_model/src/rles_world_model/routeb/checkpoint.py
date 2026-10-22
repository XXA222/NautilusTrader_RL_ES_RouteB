"""Fail-closed, non-pickle JAX optimizer/model checkpoint (format v2).

Stores the leaf arrays in a compressed NPZ and a small sidecar manifest. The
array checksum and exact *template* pytree signatures make a stale or mismatched
checkpoint fail before any training step. Training target-step/report cadence
are deliberately excluded from the immutable optimization recipe, so a run can
be extended without silently changing its algorithm.

A crash between the NPZ and JSON replace is detectable, not recoverable; callers
must retain the previous checkpoint if atomic multi-file rollback is required.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import tempfile
import numpy as np
from .randomness import validate_jax_seed
from ..json_utils import loads_unique_json

SCHEMA = "routeb.jax-checkpoint.v2"


def optimizer_recipe(model_config: Any, train_config: Any) -> dict:
    """Freeze algorithm parameters while allowing target steps/report cadence to grow."""
    from dataclasses import asdict

    t = asdict(train_config)
    t.pop("steps", None)
    t.pop("report_interval", None)
    model = asdict(model_config)
    for key in ("risk_horizons", "risk_horizon_weights",
                "path_log_rv_crps_horizons"):
        if key in model:
            model[key] = list(model[key])
    return {"model": model, "train": t}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _signature(tree) -> str:
    import jax
    # Exact JAX structure (including optimizer named tuples) is part of the
    # locked training runtime contract, not a portable cross-JAX schema.
    return str(jax.tree_util.tree_structure(tree))


def save_checkpoint(path: str | Path, *, params, opt_state, step: int,
                    seed: int, identity: dict, config: dict) -> None:
    import jax
    try:
        validate_jax_seed(seed, name="checkpoint seed")
    except ValueError as exc:
        raise ValueError('valid checkpoint step, seed, Train identity and config required') from exc
    if (type(step) is not int or step < 0
        or not isinstance(identity, dict) or not identity.get('train_scope_id')
        or not isinstance(config, dict)):
        raise ValueError('valid checkpoint step, seed, Train identity and config required')
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lp, _ = jax.tree_util.tree_flatten(params)
    lo, _ = jax.tree_util.tree_flatten(opt_state)
    arrays = {**{f'p{i}': np.asarray(v) for i, v in enumerate(lp)},
              **{f'o{i}': np.asarray(v) for i, v in enumerate(lo)}}
    if not arrays or any(a.dtype == object or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('checkpoint contains unsupported/nonfinite arrays')
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        tmp = Path(stream.name)
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    try:
        meta = {
            'schema': SCHEMA, 'step': step, 'seed': seed,
            'identity': identity, 'config': config,
            'param_leaves': len(lp), 'optim_leaves': len(lo),
            'param_tree': _signature(params), 'optim_tree': _signature(opt_state),
            'sha256': _sha_file(tmp),
        }
        # Validate JSON serializability *before* publishing the array file.
        payload = json.dumps(meta, sort_keys=True, separators=(',', ':'), allow_nan=False)
        os.replace(tmp, path)
        manifest = path.with_suffix('.json')
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, suffix='.json',
                                         delete=False) as stream:
            meta_tmp = Path(stream.name)
            stream.write(payload)
        try:
            os.replace(meta_tmp, manifest)
        finally:
            meta_tmp.unlink(missing_ok=True)
    finally:
        tmp.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, *, params_template, opt_template,
                    expected_identity: dict, expected_config: dict,
                    allow_seed_stream_metadata: bool = False):
    import jax
    path = Path(path)
    try:
        meta = loads_unique_json(path.with_suffix('.json').read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError('checkpoint metadata is unreadable or ambiguous') from exc
    saved_config = meta.get('config')
    config_matches = saved_config == expected_config
    if (not config_matches and allow_seed_stream_metadata
            and isinstance(saved_config, dict) and 'seed_streams' in saved_config):
        seed_streams = saved_config.get('seed_streams')
        expected_stream_keys = {'initialization_seed', 'sampler_seed', 'model_seed'}
        if (isinstance(seed_streams, dict)
                and set(seed_streams) == expected_stream_keys
                and all(type(value) is int for value in seed_streams.values())):
            try:
                for name, value in seed_streams.items():
                    validate_jax_seed(value, name=f'checkpoint {name}')
            except ValueError:
                pass
            else:
                base_config = {key: value for key, value in saved_config.items()
                               if key != 'seed_streams'}
                config_matches = base_config == expected_config
    if (meta.get('schema') != SCHEMA or meta.get('identity') != expected_identity
        or not config_matches):
        raise ValueError('checkpoint schema/train/config identity mismatch')
    if type(meta.get('step')) is not int or meta['step'] < 0:
        raise ValueError('malformed checkpoint step/seed')
    try:
        validate_jax_seed(meta.get('seed'), name='checkpoint seed')
    except ValueError as exc:
        raise ValueError('malformed checkpoint step/seed') from exc
    if _sha_file(path) != meta.get('sha256'):
        raise ValueError('checkpoint file corruption or partial publication')
    lp, pt = jax.tree_util.tree_flatten(params_template)
    lo, ot = jax.tree_util.tree_flatten(opt_template)
    if (len(lp) != meta.get('param_leaves') or len(lo) != meta.get('optim_leaves')
        or _signature(params_template) != meta.get('param_tree')
        or _signature(opt_template) != meta.get('optim_tree')):
        raise ValueError('checkpoint pytree structure differs from current runtime')
    keys = {f'p{i}' for i in range(len(lp))} | {f'o{i}' for i in range(len(lo))}
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != keys:
            raise ValueError('checkpoint NPZ members do not match declared tree')

        def restore(prefix, template):
            values = []
            for i, v in enumerate(template):
                arr = z[f'{prefix}{i}']
                if (arr.shape != np.shape(v) or arr.dtype != np.asarray(v).dtype
                    or not np.isfinite(arr).all()):
                    raise ValueError('checkpoint leaf shape/dtype/value mismatch')
                values.append(jax.numpy.asarray(arr))
            return values

        weights = restore('p', lp)
        optimizer = restore('o', lo)
    return pt.unflatten(weights), ot.unflatten(optimizer), meta['step'], meta['seed']
