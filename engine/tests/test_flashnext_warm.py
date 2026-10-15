"""F10 on CPU fakes: the startup warm-up compiles every block-selector specialisation a context up to the capacity can
reach (``warmplan.select_launches``), on private tensors, with OCTOJET_WARM_LONG=0 skipping it; the concurrent
decoder's warm-up keeps nothing and reports both parts; the startup log line."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import warmplan
from tensorfold.families.qwen4_exp.cuda.engine import warm_line
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

REGS = 32768                         # attention.SELECT_REGS
PRODUCTION = 262_144 + 8             # a 262,144-token window plus the drafts' rows (order of magnitude is what matters)


def reached(capacity, ratio, rows_set=(1, 4, 17, 63, 64, 256)):
    """Every specialisation qsa_rows' ``blocks = min(nb, max(1, cdiv(context, ratio)))`` can produce."""

    nb = -(-capacity // ratio)
    blocks = {min(nb, max(1, math.ceil(c / ratio))) for c in range(1, capacity + 1, 7)} | {nb, 1}
    return {warmplan.specialisation(r, b, REGS) for r in rows_set for b in blocks}


@pytest.mark.parametrize("capacity", [2_500, 20_000, 140_000, PRODUCTION])
def test_the_plan_covers_every_specialisation_a_context_reaches(capacity):
    plan = warmplan.select_launches(capacity, 2048, 4, REGS)
    warmed = {warmplan.specialisation(r, b, REGS) for r, b in plan}
    assert reached(capacity, 4) <= warmed
    assert len(warmed) == len(plan)                                 # no launch repeats a specialisation


def test_the_production_plan_and_no_plan_without_qsa():
    plan = warmplan.select_launches(PRODUCTION, 2048, 4, REGS)
    widths = [b for r, b in plan if b <= REGS]
    assert widths == [2 ** i for i in range(16)]                    # 1 .. 32,768
    tiles = {warmplan.specialisation(r, b, REGS) for r, b in plan if b > REGS}
    assert tiles == {("tiles", 4096, 8), ("tiles", 8192, 16)}
    assert warmplan.select_launches(2048, 2048, 4, REGS) == []      # never past the budget: no QSA
    assert max(b for _, b in warmplan.select_launches(20_000, 2048, 4, REGS)) == 8192


def test_specialisation_follows_launch_select():
    assert warmplan.specialisation(1, 1, REGS) == ("select", 1)
    assert warmplan.specialisation(256, 1029, REGS) == ("select", 2048)
    assert warmplan.specialisation(256, REGS, REGS) == ("select", REGS)
    assert warmplan.specialisation(64, REGS + 1, REGS) == ("tiles", 4096, 8)
    assert warmplan.specialisation(63, REGS + 1, REGS) == ("tiles", 8192, 16)


def test_the_knob(monkeypatch):
    monkeypatch.delenv("OCTOJET_WARM_LONG", raising=False)
    assert warmplan.warm_long()
    monkeypatch.setenv("OCTOJET_WARM_LONG", "0")
    assert not warmplan.warm_long()
    monkeypatch.setenv("OCTOJET_WARM_LONG", "1")
    assert warmplan.warm_long()


class Kernel:
    """A jitted kernel's ``kernel[grid](...)`` launch, recorded."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getitem__(self, grid):
        return lambda *args, **kw: self.log.append((self.name, grid, args, kw))


def scratch(capacity, rows=256):
    nb = -(-capacity // 4)
    return SimpleNamespace(qsa=capacity > 2048, budget=2048, ratio=4, nb=nb, idw=2052,
                           scores=torch.full((rows, nb), 7.0), ids=torch.full((rows, 2052), 5, dtype=torch.int32),
                           nk=torch.full((rows,), 3, dtype=torch.int32), sparse=torch.ones((rows,), dtype=torch.int32))


@pytest.mark.torch
def test_warm_select_launches_the_plan_on_private_tensors(allocations, monkeypatch):  # noqa: F811
    attention = importlib.import_module("tensorfold.families.qwen4_exp.cuda.attention")
    log = []
    monkeypatch.setattr(attention, "_select", Kernel("select", log))
    monkeypatch.setattr(attention, "_select_tiles", Kernel("tiles", log))
    monkeypatch.setattr(attention, "LAUNCHED", set())
    sc = scratch(140_000)
    before = [t.clone() for t in (sc.scores, sc.ids, sc.nk, sc.sparse)]
    n = attention.warm_select(sc)
    plan = warmplan.select_launches(sc.nb * 4, 2048, 4, attention.SELECT_REGS)
    assert n == len(plan) == len(log)
    got = set()
    for name, grid, args, kw in log:
        got.add(("select", kw["BLOCK"]) if name == "select" else ("tiles", kw["TB"], kw["num_warps"]))
        scores, pos0, ids, nk, sparse, nb = args
        assert nb == sc.nb and kw["TOP"] == 512 and kw["IDW"] == 2052 and kw["RATIO"] == 4
        assert pos0.dtype == torch.int32 and int(pos0.sum()) == 0          # position 0: every row is dense
        assert grid[0] <= nk.shape[0] and grid[0] <= sparse.shape[0]
        assert all(t is not s for t, s in zip((scores, ids, nk, sparse), (sc.scores, sc.ids, sc.nk, sc.sparse)))
    assert got == attention.LAUNCHED == {warmplan.specialisation(r, b, attention.SELECT_REGS) for r, b in plan}
    assert all(torch.equal(a, b) for a, b in zip(before, (sc.scores, sc.ids, sc.nk, sc.sparse)))


@pytest.mark.torch
def test_warm_select_without_qsa_launches_nothing(allocations, monkeypatch):  # noqa: F811
    attention = importlib.import_module("tensorfold.families.qwen4_exp.cuda.attention")
    log = []
    monkeypatch.setattr(attention, "_select", Kernel("select", log))
    assert attention.warm_select(SimpleNamespace(qsa=False)) == 0 and log == []


@pytest.mark.torch
def test_launch_select_records_what_it_compiles(allocations, monkeypatch):  # noqa: F811
    attention = importlib.import_module("tensorfold.families.qwen4_exp.cuda.attention")
    log = []
    monkeypatch.setattr(attention, "_select", Kernel("select", log))
    monkeypatch.setattr(attention, "_select_tiles", Kernel("tiles", log))
    monkeypatch.setattr(attention, "LAUNCHED", set())
    sc = scratch(600_000, rows=4)
    attention._launch_select(sc, torch.zeros(1, dtype=torch.int32), 256, 8000)      # a 32k prompt's row block
    attention._launch_select(sc, torch.zeros(1, dtype=torch.int32), 4, 40_000)      # a decode window past 131k keys
    assert attention.LAUNCHED == {("select", 8192), ("tiles", 8192, 16)}
    assert log[0][3]["BLOCK"] == 8192 and log[1][3]["TB"] == 8192


@pytest.mark.torch
@pytest.mark.parametrize("knob", ["1", "0"])
def test_warm_long_select_honours_the_knob(allocations, monkeypatch, knob):  # noqa: F811
    decode = importlib.import_module("tensorfold.families.qwen4_exp.cuda.decode")
    attention = importlib.import_module("tensorfold.families.qwen4_exp.cuda.attention")
    calls = []
    monkeypatch.setattr(attention, "warm_select", lambda sc: calls.append(sc) or 18)
    monkeypatch.setattr(decode.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setenv("OCTOJET_WARM_LONG", knob)
    sc = SimpleNamespace(nb=65_538, ratio=4)
    info = decode.warm_long_select(sc)
    if knob == "0":
        assert calls == [] and info["select_skipped"] and info["select_launches"] == 0
    else:
        assert calls == [sc] and info["select_launches"] == 18 and info["select_keys"] == 262_152
        assert not info["select_skipped"] and info["select_s"] >= 0


@pytest.mark.torch
def test_the_concurrent_warm_up_keeps_nothing_and_reports_both_parts(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    from tests.test_prefix_checkpoints import decoder

    log, warmed = [], []
    dec = decoder(multi, 1, monkeypatch, log, n=3)
    dec.pbuf = SimpleNamespace(rows=4, attn="the prompt scratch")
    dec.round = lambda: log.append(("round",))
    monkeypatch.setattr(multi, "warm_long_select", lambda sc: warmed.append(sc) or {"select_launches": 18})
    info = dec.warm()
    assert warmed == ["the prompt scratch"] and info["select_launches"] == 18 and info["prompt_s"] >= 0
    assert log[0][:2] == ("prefill", 4 + multi.WARM_TAIL) and log[0][3] == []        # no checkpoint planned
    assert dec.kept == [] and len(dec.free) == 1 and dec.checkpoints == 3 and dec.streams == {}


def test_the_startup_line():
    on = warm_line({"prompt_s": 3.04, "select_launches": 18, "select_s": 0.41, "select_keys": 262_152,
                    "select_skipped": False})
    assert on == ("synthetic prompt 3.0s; block selector compiled for contexts up to 262,152 keys (18 launches) "
                  "in 0.4s")
    assert "skipped (OCTOJET_WARM_LONG=0)" in warm_line({"prompt_s": 1.0, "select_skipped": True})
    assert "never passes the indexer budget" in warm_line({"prompt_s": 1.0, "select_launches": 0})
