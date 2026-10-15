"""F17: a default window leaves the memory-mapped n-gram tables their pages plus a safety margin
(OCTOJET_TABLE_MARGIN_GIB, default 2), so a long prompt does not re-read them from the SSD."""

import pytest

from tensorfold.cuda import capacity as cap

GIB = cap.GIB
GEO = cap.Geometry(lambda slots: slots * 10_000, 0)            # 10 kB a token of cache


def plan(margin=None, monkeypatch=None, explicit=False, requested=None):
    if margin is not None:
        monkeypatch.setenv("OCTOJET_TABLE_MARGIN_GIB", str(margin))
    weights = cap.Weights(resident=70 * GIB, staging=1 * GIB, mapped=30 * GIB)
    return cap.make_plan(10_000_000, requested, explicit, 106 * GIB, weights, GEO, room=110 * GIB)   # native: no cap


def test_the_margin_shrinks_the_default_window_by_its_bytes(monkeypatch):
    old = plan(0, monkeypatch)
    monkeypatch.delenv("OCTOJET_TABLE_MARGIN_GIB", raising=False)
    new = plan(monkeypatch=monkeypatch)
    assert old.keeps_tables is True and new.keeps_tables is True
    assert old.fitting - new.fitting == pytest.approx(2 * GIB / 10_000, abs=1)   # 2 GiB of cache tokens fewer
    assert (110 - 30) * GIB - new.weights.resident - GEO.needed(new.fitting) >= 2 * GIB - 10_000


def test_margin_zero_is_the_old_plan_and_explicit_windows_are_unchanged(monkeypatch):
    zero = plan(0, monkeypatch)
    assert zero.fitting == ((110 - 30) * GIB - 70 * GIB) // 10_000
    monkeypatch.setenv("OCTOJET_TABLE_MARGIN_GIB", "5")
    explicit = plan(monkeypatch=monkeypatch, explicit=True, requested=500_000)
    assert explicit.keeps_tables is None and explicit.fitting == 500_000           # an explicit request: as before


@pytest.mark.parametrize("bad", ["-1", "two"])
def test_a_bad_margin_is_refused(monkeypatch, bad):
    monkeypatch.setenv("OCTOJET_TABLE_MARGIN_GIB", bad)
    with pytest.raises(ValueError, match="OCTOJET_TABLE_MARGIN_GIB"):
        cap.table_margin()
