"""Validate a lane passed to the reusable GitHub Actions workflow."""
from __future__ import annotations

import os
import sys


VALID_WORKFLOW_LANES = frozenset({"pr", "nightly", "gpu", "release"})


def validate_workflow_lane(lane: object) -> str:
    """Return a supported lane or reject the request before any job is skipped."""
    if not isinstance(lane, str) or lane not in VALID_WORKFLOW_LANES:
        choices = ", ".join(sorted(VALID_WORKFLOW_LANES))
        raise ValueError(f"unsupported CI lane {lane!r}; expected one of: {choices}")
    return lane


def main() -> int:
    try:
        validate_workflow_lane(os.environ.get("ROUTEB_CI_LANE"))
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
