"""tools/diag_cut_compare.py --prefill-rows on CPU, with deterministic stand-ins for the CUDA engine.

The stand-in ``forward`` mirrors the real one: the prefill buffers (``prefill=True``) run the head for the chunk's LAST
row only (forward.finish), so a prefill chunk yields [1, V] logits; the decode buffers yield one row per token."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

V, TOKENS, ROWS = 50, 96, 64


def _load_tool():
    spec = importlib.util.spec_from_file_location("diag_cut_compare", Path(__file__).parents[1] / "tools/diag_cut_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _logits(token: int, pos: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(token * 1000 + pos)
    return torch.randn(V, generator=g)


class FakeState:
    capacity = 512
    pos = 0


class FakeBuf:
    def __init__(self, prefill=False):
        self.prefill = prefill


class FakeEngine:
    def __init__(self, w, capacity=512, max_rows=8, prefill_rows=64, graphs=False):
        self.capacity, self.prefill_rows = capacity, prefill_rows
        self.st, self.buf, self.pbuf = FakeState(), FakeBuf(), FakeBuf(prefill=True)

    def reset(self):
        self.st.pos = 0


def fake_forward(w, st, b, tokens):
    rows = torch.stack([_logits(t, st.pos + i) for i, t in enumerate(tokens)])
    return rows[-1:] if b.prefill else rows


def fake_commit(w, st, b, R, keep, at=0):
    st.pos += keep


@pytest.fixture
def tool(monkeypatch):
    mod = _load_tool()
    dec = types.ModuleType("tensorfold.families.qwen4_exp.cuda.decode")
    dec.Engine = FakeEngine
    fwd = types.ModuleType("tensorfold.families.qwen4_exp.cuda.forward")
    fwd.forward, fwd.commit = fake_forward, fake_commit
    monkeypatch.setitem(sys.modules, dec.__name__, dec)
    monkeypatch.setitem(sys.modules, fwd.__name__, fwd)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return mod


def test_prefill_pass_collects_a_row_per_chunk_at_the_chunk_ends(tool):
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, size=TOKENS)]
    out: dict = {}
    one, info = tool.one_row_logits(object(), 512, toks, ROWS, out)
    assert one.shape == (TOKENS, V)
    assert info["prefill_final_pos"] == TOKENS
    ends = tool.prefill_ends(TOKENS, ROWS)
    assert ends == [63, 95]                                   # chunks of 64 and 32
    assert out["prefill"].shape == (len(ends), V)             # the head runs on each chunk's last row only
    assert torch.equal(out["prefill"], one[ends])


def test_prefill_vs_onerow_reports_full_agreement_when_passes_match(tool):
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, size=TOKENS)]
    out: dict = {}
    one, _ = tool.one_row_logits(object(), 512, toks, ROWS, out)
    r = tool.prefill_vs_onerow(out["prefill"], one, TOKENS, ROWS)
    assert r["top1_agree"] == 1.0 and r["cos_min"] == 1.0 and r["first_disagree"] is None and r["rows"] == 2


def test_prefill_vs_onerow_flags_a_wrong_row(tool):
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, size=TOKENS)]
    out: dict = {}
    one, _ = tool.one_row_logits(object(), 512, toks, ROWS, out)
    bad = out["prefill"].clone()
    bad[1] = bad[1].roll(7)
    assert tool.prefill_vs_onerow(bad, one, TOKENS, ROWS)["top1_agree"] == 0.5


def test_run_reports_prefill_vs_onerow_for_both_models(tool, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import nvfp4

    monkeypatch.setattr(nvfp4, "sources", lambda d: types.SimpleNamespace(experts=Path("x"), base=Path("y")))
    monkeypatch.setattr(nvfp4, "is_mixed", lambda d: True)
    monkeypatch.setattr(tool, "load_cut", lambda d, layers: types.SimpleNamespace(cfg=types.SimpleNamespace(vocab=1000)))
    monkeypatch.setattr(tool, "meta", lambda w: {})
    a = types.SimpleNamespace(mixed="m", layers=4, tokens=TOKENS, capacity=512, capacity2=1024, seed=0, prefill_rows=ROWS,
                              self_check=False)
    report = {"errors": [], "seconds": {}}
    tool.run(a, report)
    assert report["errors"] == []
    for cap in ("512", "1024"):
        for name in ("mixed", "mlx"):
            r = report["prefill_vs_onerow"][cap][name]
            assert r["top1_agree"] == 1.0 and r["rows"] == 2
        assert report["prefill"][cap]["top1_agree"] == 1.0
