"""Test sessions run in temporary directories, usually on the root volume."""

import pytest


@pytest.fixture(autouse=True)
def allow_root_volume(monkeypatch, request):
    if "root_volume_check" not in request.keywords:
        monkeypatch.setenv("Q1_ALLOW_ROOT_VOLUME", "1")
