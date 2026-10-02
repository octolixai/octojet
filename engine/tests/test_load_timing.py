"""Load-phase timing: accumulates only when enabled; the report names every phase and the remainder."""

import time

from tensorfold.cuda import load_timing


def test_disabled_records_nothing():
    p = load_timing.Phases(enabled=False)
    with p.phase("pack"):
        time.sleep(0.01)
    assert p.seconds == {} and p.counts == {} and p.layers == []


def test_enabled_accumulates_across_calls():
    p = load_timing.Phases(enabled=True)
    for _ in range(3):
        with p.phase("pack"):
            time.sleep(0.005)
    assert p.counts == {"pack": 3}
    assert 0.012 <= p.seconds["pack"] < 1.0


def test_layer_phase_records_each_layer_with_its_breakdown():
    p = load_timing.Phases(enabled=True)
    for _ in range(2):
        with p.phase("layer"):
            with p.phase("pack"):
                time.sleep(0.002)
            with p.phase("dense"):
                time.sleep(0.001)
    assert len(p.layers) == 2 and p.counts["layer"] == 2
    for d in p.layers:
        assert 0.002 < d["total"] < 1.0 and 0.0015 < d["pack"] < 1.0 and 0.0005 < d["dense"] < 1.0
        assert set(d) == {"total", "pack", "dense"}
    assert abs(sum(d["pack"] for d in p.layers) - p.seconds["pack"]) < 1e-3


def test_report_lists_phases_and_remainder():
    p = load_timing.Phases(enabled=True)
    p.seconds = {"read_routed": 120.0, "pack": 150.0, "upload": 12.0, "layer": 300.0}
    p.counts = {"read_routed": 48, "pack": 48, "upload": 48, "layer": 48}
    line = p.report(total=355.0)
    assert line.startswith("[octojet] load phases:")
    assert "read_routed 120.0 s (48)" in line and "pack 150.0 s (48)" in line and "upload 12.0 s (48)" in line
    assert "layer 300.0 s (48)" in line
    assert "other 73.0 s" in line and "total 355.0 s" in line      # layer is a wrapper: not subtracted


def test_env_switch(monkeypatch):
    monkeypatch.setenv("OCTOJET_LOAD_TIMING", "1")
    assert load_timing.Phases().enabled is True
    monkeypatch.delenv("OCTOJET_LOAD_TIMING")
    assert load_timing.Phases().enabled is False


def test_reset_clears():
    p = load_timing.Phases(enabled=True)
    with p.phase("layer"):
        pass
    p.reset()
    assert p.seconds == {} and p.counts == {} and p.layers == []
