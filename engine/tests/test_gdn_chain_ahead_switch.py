"""F14's switch (no GPU): OCTOJET_GDN_CHAIN_AHEAD picks the prompt chain's double-buffered kernel, on by default."""

import pytest

from tensorfold.cuda.kernels import gdn


@pytest.mark.parametrize("value,want", [(None, True), ("1", True), ("", True), ("on", True), (" TRUE ", True),
                                        ("yes", True), ("0", False), ("off", False), ("false", False), ("no", False)])
def test_the_switch(monkeypatch, value, want):
    if value is None:
        monkeypatch.delenv(gdn.AHEAD_ENV, raising=False)
    else:
        monkeypatch.setenv(gdn.AHEAD_ENV, value)
    assert gdn.chain_ahead() is want


def test_an_unknown_setting_is_refused(monkeypatch):
    monkeypatch.setenv(gdn.AHEAD_ENV, "maybe")
    with pytest.raises(ValueError, match=gdn.AHEAD_ENV):
        gdn.chain_ahead()
