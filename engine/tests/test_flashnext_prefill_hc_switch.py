"""F12's switch and dispatch (no GPU): OCTOJET_PREFILL_HC_FUSED picks the fusions; only a prompt chunk's read-out of
an MLX 4-bit hyper-connection (tiled down, group-32 prefill up, 4 streams, D a multiple of 256) takes them."""

from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import hc_prefill as hp


def _hc(n=4 * 2560, k=320, gs=32, down="tiled", up="frag"):
    return SimpleNamespace(down=SimpleNamespace(layout=down), prefill_up=SimpleNamespace(layout=up, gs=gs, n=n, k=k))


@pytest.mark.parametrize("value,want", [(None, (True, True)), ("1", (True, True)), ("", (True, True)),
                                        ("on", (True, True)), ("0", (False, False)), ("off", (False, False)),
                                        ("plain", (False, False)), ("norm", (True, False)), ("up", (False, True)),
                                        (" UP ", (False, True))])
def test_the_switch(monkeypatch, value, want):
    if value is None:
        monkeypatch.delenv(hp.ENV, raising=False)
    else:
        monkeypatch.setenv(hp.ENV, value)
    assert tuple(hp.parts()) == want


def test_an_unknown_setting_is_refused(monkeypatch):
    monkeypatch.setenv(hp.ENV, "maybe")
    with pytest.raises(ValueError, match=hp.ENV):
        hp.parts()


def test_the_dispatch(monkeypatch):
    monkeypatch.delenv(hp.ENV, raising=False)
    chunk, window = SimpleNamespace(prefill=True), SimpleNamespace(prefill=False)
    assert hp.choose(_hc(), chunk, 4) == hp.Parts(True, True)
    assert hp.choose(_hc(), window, 4) == hp.PLAIN                    # decode windows keep their own fused read-out
    assert hp.choose(_hc(), chunk, 2) == hp.PLAIN                     # the up+mix kernel is built for 4 streams
    assert hp.choose(_hc(down="frag"), chunk, 4) == hp.PLAIN
    assert hp.choose(SimpleNamespace(down=object(), prefill_up=None), chunk, 4) == hp.PLAIN   # an EXL3 pack
    assert hp.choose(_hc(up="tiled"), chunk, 4) == hp.PLAIN
    assert hp.choose(_hc(gs=64), chunk, 4) == hp.PLAIN
    assert hp.choose(_hc(n=4 * 1000), chunk, 4) == hp.PLAIN          # D not a multiple of 256 (the sums' blocks)
    assert hp.choose(_hc(n=4 * 1024), chunk, 4) == hp.Parts(True, True)   # the synthetic test model's D
    monkeypatch.setenv(hp.ENV, "0")
    assert hp.choose(_hc(), chunk, 4) == hp.PLAIN


def test_the_tile_setting(monkeypatch):
    monkeypatch.delenv(hp.ENV_TILE, raising=False)
    assert hp.tile() == 0
    monkeypatch.setenv(hp.ENV_TILE, "2")
    assert hp.tile() == 2
    monkeypatch.setenv(hp.ENV_TILE, "4")
    with pytest.raises(ValueError):
        hp.tile()
