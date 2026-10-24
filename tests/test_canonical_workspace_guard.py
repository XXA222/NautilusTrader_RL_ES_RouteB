import pytest

from qa.assert_canonical_workspace import (
    CANONICAL_PROJECT_ROOT,
    CANONICAL_WSL_PROJECT_ROOT,
    _marker_matches,
    _platform_root,
)


def test_canonical_workspace_root_is_platform_specific_but_same_checkout() -> None:
    assert _platform_root(os_name="nt", platform="win32") == CANONICAL_PROJECT_ROOT
    assert _platform_root(os_name="posix", platform="linux") == CANONICAL_WSL_PROJECT_ROOT


def test_canonical_workspace_rejects_unsupported_platforms() -> None:
    with pytest.raises(RuntimeError, match="canonical Windows or WSL checkout"):
        _platform_root(os_name="posix", platform="darwin")


def test_canonical_marker_keeps_one_shared_project_identity() -> None:
    marker = str(CANONICAL_PROJECT_ROOT)
    assert _marker_matches(marker, expected=CANONICAL_PROJECT_ROOT, os_name="nt")
    assert _marker_matches(marker, expected=CANONICAL_WSL_PROJECT_ROOT, os_name="posix")
    assert not _marker_matches(str(CANONICAL_WSL_PROJECT_ROOT),
                               expected=CANONICAL_WSL_PROJECT_ROOT, os_name="posix")
