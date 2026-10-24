"""Compare one default-size synthetic B3 update on the locked CPU and CUDA backends."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_profile(*, root: Path, profile: Path, checkout: Path,
                 output: Path, device: str) -> dict[str, object]:
    output.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(root / part) for part in (
        "packages/world_model/src", "packages/data/src", "packages/contracts/src",
        "packages/orchestrator/src", "packages/research/src",
    ))
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    subprocess.run([
        sys.executable, "-B", str(profile), "--checkout", str(checkout),
        "--output-dir", str(output), "--days", "30", "--updates", "1",
        "--device", device,
    ], cwd=root, env=env, check=True, timeout=300)
    metadata_path = output / f"b3_default_{device}_profile_30d.json"
    checkpoint_path = output / f"b3_default_{device}_profile_30d_checkpoint.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("status") != "COMPLETED"
            or metadata.get("device") != ("cpu:0" if device == "cpu" else "cuda:0")
            or metadata.get("synthetic_only") is not True
            or metadata.get("real_market_data_accessed") is not False):
        raise RuntimeError(f"{device} B3 profile failed its execution/data assertions")
    return {
        "metadata": metadata,
        "metadata_sha256": _sha256(metadata_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": _sha256(checkpoint_path),
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from qa.assert_canonical_workspace import assert_canonical_workspace

    root = assert_canonical_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True,
                        help="clean checkout of the exact pinned DreamerV3 commit")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    checkout = Path(args.checkout).resolve()
    if not checkout.is_dir():
        raise ValueError("pinned upstream checkout is missing")
    upstream = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if upstream != "e01491fad6434b2245a3b8ca201dd7faedcc458c":
        raise ValueError("DreamerV3 checkout does not match the pinned audit commit")
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(root) or output.exists() or not output.parent.is_dir():
        raise ValueError("device-parity output must be a new canonical-workspace path")
    output.mkdir()
    profile = root / "qa/world_model/routeb_b3_default_cpu_profile.py"
    cpu = _run_profile(root=root, profile=profile, checkout=checkout,
                        output=output / "cpu", device="cpu")
    cuda = _run_profile(root=root, profile=profile, checkout=checkout,
                         output=output / "cuda", device="cuda")
    with np.load(cpu["checkpoint_path"], allow_pickle=False) as cpu_archive, \
            np.load(cuda["checkpoint_path"], allow_pickle=False) as cuda_archive:
        cpu_params = sorted(key for key in cpu_archive.files if key.startswith("p"))
        cuda_params = sorted(key for key in cuda_archive.files if key.startswith("p"))
        if cpu_params != cuda_params or any(
                cpu_archive[key].shape != cuda_archive[key].shape for key in cpu_params):
            raise RuntimeError("CPU and CUDA checkpoint parameter structures differ")
        differences = [np.abs(cpu_archive[key] - cuda_archive[key]) for key in cpu_params]
        flat_differences = np.concatenate([value.reshape(-1) for value in differences])
        parameter_values = np.concatenate([
            np.abs(cpu_archive[key]).reshape(-1) for key in cpu_params
        ])
        max_abs_difference = float(np.max(flat_differences))
        mean_abs_difference = float(np.mean(flat_differences))
        tolerance = {"rtol": 0.0, "atol": 1e-3}
        within_tolerance = all(np.allclose(
            cpu_archive[key], cuda_archive[key], **tolerance) for key in cpu_params)

    payload = {
        "schema": "routeb.b3-synthetic-cpu-cuda-parity.v1",
        "status": "PASS" if within_tolerance else "FAIL",
        "scope": "one_default_model_optimizer_update_synthetic_only",
        "repository_commit": subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "upstream_commit": upstream,
        "profile_source_sha256": _sha256(profile),
        "training_source_sha256": _sha256(
            root / "packages/world_model/src/rles_world_model/routeb/jax_hierarchy_train.py"),
        "cpu": {key: value for key, value in cpu.items() if key != "checkpoint_path"},
        "cuda": {key: value for key, value in cuda.items() if key != "checkpoint_path"},
        "parameter_arrays": len(cpu_params),
        "parameter_count": int(parameter_values.size),
        "max_abs_parameter_difference": max_abs_difference,
        "mean_abs_parameter_difference": mean_abs_difference,
        "max_abs_difference_over_cpu_parameter_rms": float(
            max_abs_difference / np.sqrt(np.mean(np.square(parameter_values)))),
        "tolerance": tolerance,
        "data_boundary": "synthetic only; no market, Validation, Test or Holdout data",
        "economic_certification": False,
    }
    target = output / "result.json"
    target.write_text(json.dumps(payload, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
