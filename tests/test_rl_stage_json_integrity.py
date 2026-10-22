from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from rles_orchestrator.rl_stage import RLStageExecutor


def _executor(tmp_path: Path) -> RLStageExecutor:
    executor = object.__new__(RLStageExecutor)
    executor.inference_dir = tmp_path / "run" / "inference"
    executor.inference_dir.parent.mkdir(parents=True)
    executor.resume_from = tmp_path / "checkpoint"
    executor.train_steps = 10
    executor.config = SimpleNamespace(hash="config-sha", fingerprint=SimpleNamespace(hash="transition-sha"))
    return executor


def test_r1_transfer_resume_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    (executor.inference_dir.parent / "R1-transfer.json").write_text(
        '{"result":{},"evidence_hash":"a","evidence_hash":"b"}', encoding="utf-8"
    )
    context = SimpleNamespace(spec=SimpleNamespace(algorithm="PPO"))
    with pytest.raises(ValueError, match="duplicate JSON object key: .*evidence_hash"):
        executor._transfer_record(context)


def test_r1_learning_resume_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    payload = b'{"schema":"rles.r1-learning.v5","schema":"rles.r1-learning.v4"}'
    path = executor.inference_dir.parent / "R1-learning.json"
    path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(hashlib.sha256(payload).hexdigest(), encoding="ascii")
    context = SimpleNamespace(spec=SimpleNamespace(attempt_id="attempt-1"))
    with pytest.raises(ValueError, match="duplicate JSON object key: .*schema"):
        executor._training_diagnostics_record(context, (), initial_steps=0)

