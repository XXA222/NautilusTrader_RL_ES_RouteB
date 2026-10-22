from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ci" / "l6" / "src"))

from l6_ci_release.workflow_lane import (  # noqa: E402
    VALID_WORKFLOW_LANES,
    validate_workflow_lane,
)


@pytest.mark.parametrize("lane", sorted(VALID_WORKFLOW_LANES))
def test_reusable_workflow_accepts_each_declared_lane(lane: str) -> None:
    assert validate_workflow_lane(lane) == lane


@pytest.mark.parametrize("lane", [None, "", "PR", "relese", "gpu ", "unknown"])
def test_reusable_workflow_rejects_unsupported_lane(lane: object) -> None:
    with pytest.raises(ValueError, match="unsupported CI lane"):
        validate_workflow_lane(lane)


def test_every_reusable_workflow_job_depends_on_lane_validation() -> None:
    workflow = (ROOT / ".github" / "workflows" / "l6-ci-reusable.yml").read_text(
        encoding="utf-8"
    )
    jobs_yaml = workflow.split("\njobs:\n", maxsplit=1)[1]
    job_sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in jobs_yaml.splitlines():
        if line and not line.startswith(" "):
            break
        match = re.fullmatch(r"  ([A-Za-z0-9_-]+):", line)
        if match:
            if current is not None:
                job_sections[current] = "\n".join(lines)
            current = match.group(1)
            lines = [line]
        elif current is not None:
            lines.append(line)
    if current is not None:
        job_sections[current] = "\n".join(lines)

    assert "validate-lane" in job_sections
    assert "l6_ci_release.workflow_lane import main" in job_sections["validate-lane"]
    assert "ROUTEB_CI_LANE" in job_sections["validate-lane"]
    assert set(job_sections) == {"validate-lane", "cpu", "world-model-jax-cpu", "gpu", "release"}
    for job, section in job_sections.items():
        if job == "validate-lane":
            continue
        assert re.search(r"(?m)^    needs:.*\bvalidate-lane\b", section), job


def test_manual_dispatch_lanes_match_reusable_workflow_validator() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    options = workflow.split("options:", maxsplit=1)[1].split("\njobs:", maxsplit=1)[0]
    dispatch_lanes = set(re.findall(r"(?m)^\s+- ([a-z]+)\s*$", options))
    assert dispatch_lanes == VALID_WORKFLOW_LANES
