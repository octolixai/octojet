"""The Octojet fork never contacts TensorFold's releases: no background check, and `update` refuses."""

import subprocess
import threading
import urllib.request

import pytest

from tensorfold import update


@pytest.fixture(autouse=True)
def calls(monkeypatch):
    """Every trapped call is recorded, so a swallowed AssertionError (update.py catches Exception) still fails."""
    seen = []

    def trap(name):
        def stub(*args, **kwargs):
            seen.append(name)
            raise AssertionError(name)
        return stub

    monkeypatch.delenv("TENSORFOLD_NO_UPDATE_CHECK", raising=False)   # the conftest must not mask a regression
    monkeypatch.setattr(urllib.request, "urlopen", trap("network"))
    monkeypatch.setattr(threading.Thread, "start", trap("thread"))
    monkeypatch.setattr(subprocess, "run", trap("install"))
    monkeypatch.setattr(subprocess, "call", trap("install"))
    return seen


def test_background_check_is_off(calls):
    assert update.check_in_background() is None
    assert calls == []


def test_update_refuses(capsys, calls):
    assert update.update(check_only=False, force=True) == 2
    assert update.FORK_NOTICE in capsys.readouterr().out
    assert calls == []


def test_no_release_lookup(calls):
    assert update.latest_release(use_cache=False) is None
    assert calls == []
