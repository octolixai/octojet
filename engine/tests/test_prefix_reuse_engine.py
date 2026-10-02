"""The single-stream path: exact hits, invalidation by the resume position, complete cleanup on failure, rank 0's
generate() → _share() dispatch and the tensor-parallel request protocol (kind + serial) on rank 1."""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch


def modules():
    return (importlib.import_module("tensorfold.families.qwen4_exp.cuda.engine"),
            importlib.import_module("tensorfold.families.qwen4_exp.cuda.decode"))


def bare(mod):
    eng = mod.FlashNextEngine.__new__(mod.FlashNextEngine)
    eng.cache, eng.next_serial, eng.depth, eng.confidence, eng.tp, eng.rank, eng.served = [], 0, 3, 0.3, 1, 0, 0
    eng.scheduler = eng.comm = None
    eng.eos, eng.max_len = (), 100
    eng.e = SimpleNamespace(st=SimpleNamespace(snapshot=lambda: {"pos": 0}), mbuf=object(), last_streams=None, last_logits=None)
    return eng


def entry(eng, ids):
    return eng._remember(list(ids), {"pos": len(ids)}, f"tail{len(ids)}", f"logits{len(ids)}")


def test_resume_matches_exact_then_extend_and_start_from_drops_overwritten_entries(allocations):  # noqa: F811
    eng = bare(modules()[0])
    P = [1, 2, 3]
    entry(eng, P); entry(eng, P + [4, 5, 6])
    m = eng._resume(P)
    assert m.kind == "exact" and m.cached == 3
    eng._start_from(m.cached, P)                                   # the reply writes above 3: P+X goes, P stays
    assert [k.ids for k in eng.cache] == [P]
    entry(eng, P + [4, 5, 6])
    m = eng._resume(P + [4, 5, 6, 7])
    assert m.kind == "extend" and m.cached == 6
    eng._start_from(m.cached, P + [4, 5, 6, 7])
    assert [k.ids for k in eng.cache] == [P, P + [4, 5, 6]]         # nothing has rows above 6: both stay
    eng._start_from(0, [9])
    assert eng.cache == []                                          # a fresh prefill overwrites everything


def test_exact_hit_drops_the_longer_entry(allocations):  # noqa: F811
    eng = bare(modules()[0])
    A = list(range(4_096)); entry(eng, A); entry(eng, A + [7] * 2_048)
    m = eng._resume(A)
    assert m.kind == "exact"
    eng._start_from(m.cached, A)
    assert [len(k.ids) for k in eng.cache] == [4_096]
    eng._start_from(2_000, A[:2_000] + [5])                          # a resume at 2,000: rows above it die
    assert eng.cache == []


def test_remember_keeps_at_most_two_entries_and_serials_increase(allocations):  # noqa: F811
    eng = bare(modules()[0])
    a = entry(eng, [1]); b = entry(eng, [1, 2]); c = entry(eng, [1, 2, 3])
    assert [k.ids for k in eng.cache] == [[1, 2], [1, 2, 3]] and (a.serial, b.serial, c.serial) == (0, 1, 2)


def test_decode_exact_hit_and_failure_cleanup(allocations, monkeypatch, capsys):  # noqa: F811
    mod, dec = modules()
    eng = bare(mod)
    k = entry(eng, [1, 2, 3])
    log = []
    monkeypatch.setattr(mod, "exact_hit", lambda e, ent, sampling: log.append(("exact", ent.serial, sampling)) or 9)
    monkeypatch.setattr(mod, "_admission", lambda fn, carrier: (fn(), {"ttft_s": 0.1}))
    monkeypatch.setattr(dec, "prefill", lambda *a, **kw: pytest.fail("an exact hit must not prefill"))
    monkeypatch.setattr(dec, "mtp_decode", lambda *a, **kw: SimpleNamespace(drafted=0, accepted=0, widths=[1], seconds=0.1,
                                                                             rounds=1, tokens_per_second=1.0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    stats = eng._decode([1, 2, 3], 4, "S2", lambda new: False, eng._resume([1, 2, 3]))
    assert log == [("exact", k.serial, "S2")] and stats["reuse"] == "exact" and stats["cached"] == 3 and stats["reuse_miss"] is None
    assert [x.serial for x in eng.cache] == [k.serial]              # an exact hit keeps its entry

    def boom(*a, **kw):
        raise RuntimeError("injected")

    monkeypatch.setattr(mod, "exact_hit", boom)
    with pytest.raises(RuntimeError, match="injected"):
        eng._decode([1, 2, 3], 4, None, lambda new: False, eng._resume([1, 2, 3]))
    assert eng.cache == [] and "[octojet] prefix reuse exact failed: injected" in capsys.readouterr().err


def test_a_draft_failure_after_a_fresh_prefill_clears_the_cache(allocations, monkeypatch, capsys):  # noqa: F811
    mod, dec = modules()
    eng = bare(mod)
    entry(eng, [1, 2])                                                # an extend source

    def fake_prefill(e, prompt, sampling, *, mtp=True, resume=None):
        e.last_streams, e.last_logits = SimpleNamespace(clone=lambda: "tailX"), "logitsX"
        return 5

    def boom(*a, **kw):
        raise RuntimeError("draft injected")

    monkeypatch.setattr(mod, "_admission", lambda fn, carrier: (fn(), {}))
    monkeypatch.setattr(dec, "prefill", fake_prefill)
    monkeypatch.setattr(dec, "mtp_decode", boom)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with pytest.raises(RuntimeError, match="draft injected"):
        eng._decode([1, 2, 3], 4, None, lambda new: False, eng._resume([1, 2, 3]))
    assert eng.cache == [] and "[octojet] prefix reuse extend failed: draft injected" in capsys.readouterr().err


def test_share_and_unpack_carry_kind_and_serial(allocations):  # noqa: F811
    mod = modules()[0]
    eng = bare(mod)
    eng.tp = 2
    store = {}
    eng.comm = SimpleNamespace(store=SimpleNamespace(set=lambda key, v: store.__setitem__(key, v)))
    k = entry(eng, [1, 2, 3])
    got = eng._share([1, 2, 3], 5, None, True, 3, "exact", k.serial)
    assert got == ([1, 2, 3], 5, None, True, 3, "exact", k.serial)
    body = json.loads(next(iter(store.values())))
    assert body["kind"] == "exact" and body["serial"] == k.serial and body["cached"] == 3
    assert mod.FlashNextEngine._unpack(json.dumps({"stop": True})) is None
    assert mod.FlashNextEngine._unpack(json.dumps({"prompt": [1], "max_tokens": 1, "sampling": None, "draft": True,
                                                   "cached": 0}))[4:] == (0, None, None)          # an older body


def test_rank0_generate_shares_the_reuse_decision_then_decodes(allocations, monkeypatch):  # noqa: F811
    mod = modules()[0]
    eng = bare(mod)
    eng.tp = 2
    store = {}
    eng.comm = SimpleNamespace(store=SimpleNamespace(set=lambda key, v: store.__setitem__(key, v)))
    seen = []
    monkeypatch.setattr(eng, "_decode", lambda prompt, n, s, cb, hit, carrier=None:
                        seen.append(("decode", list(prompt), None if hit is None else (hit.kind, hit.cached))) or {})
    monkeypatch.setattr(eng, "_serial", lambda prompt, n, s, cb, carrier=None: seen.append(("serial", list(prompt), None)) or {})
    eng.generate([1, 2, 3], 4, None, lambda new: False)                        # cold
    k = entry(eng, [1, 2, 3])
    eng.generate([1, 2, 3], 4, None, lambda new: False)                        # exact
    eng.generate([1, 2, 3, 4], 4, None, lambda new: False)                     # extend
    eng.generate([1, 2, 3], 4, None, lambda new: False, draft=False)           # the serial reference announces nothing
    bodies = [json.loads(v) for v in store.values()]
    assert [(b["prompt"], b["cached"], b["kind"], b["serial"], b["draft"]) for b in bodies] == [
        ([1, 2, 3], 0, None, None, True), ([1, 2, 3], 3, "exact", k.serial, True), ([1, 2, 3, 4], 3, "extend", k.serial, True),
        ([1, 2, 3], 0, None, None, False)]
    assert seen == [("decode", [1, 2, 3], None), ("decode", [1, 2, 3], ("exact", 3)), ("decode", [1, 2, 3, 4], ("extend", 3)),
                    ("serial", [1, 2, 3], None)]
    assert eng.served == 4


def test_follower_resolves_by_serial_refuses_inconsistency_and_dispatches(allocations, monkeypatch):  # noqa: F811
    mod = modules()[0]
    eng = bare(mod)
    k = entry(eng, [1, 2, 3])
    m = eng._resolve_shared([1, 2, 3], True, 3, "exact", k.serial)
    assert m.kind == "exact" and m.entry is k
    m = eng._resolve_shared([1, 2, 3, 4], True, 3, "extend", k.serial)
    assert m.kind == "extend" and m.cached == 3
    assert eng._resolve_shared([1, 2, 3], True, 0, None, None) is None              # a cold request
    assert eng._resolve_shared([1, 2, 3], False, 0, None, None) is None             # the serial reference
    for args, why in [(([1, 2, 3], True, 3, "exact", k.serial + 99), "no kept state"),       # unknown serial
                      (([1, 2, 9], True, 3, "exact", k.serial), "no kept state"),            # ids differ
                      (([1, 2, 3, 4], True, 3, "exact", k.serial), "no kept state"),         # kind differs (it is an extend)
                      (([1, 2, 3], True, 2, "exact", k.serial), "no kept state"),            # cached differs
                      (([1, 2, 3], True, 3, None, None), "cold request announcing"),         # cached without a kind
                      (([1, 2, 3], True, 0, None, k.serial), "cold request announcing"),     # a serial without a kind
                      (([1, 2, 3], False, 3, "exact", k.serial), "serial request announcing")]:
        with pytest.raises(RuntimeError, match=why):
            eng._resolve_shared(*args)
    requests = iter([([1, 2, 3], 4, None, True, 3, "exact", k.serial), ([5], 2, None, False, 0, None, None), None])
    seen = []
    monkeypatch.setattr(eng, "_receive", lambda: next(requests))
    monkeypatch.setattr(eng, "_decode", lambda prompt, n, s, cb, hit, carrier=None: seen.append(("decode", hit.kind, hit.entry.serial)) or {})
    monkeypatch.setattr(eng, "_serial", lambda *a, **kw: seen.append(("serial",)) or {})
    eng.follow()
    assert seen == [("decode", "exact", k.serial), ("serial",)] and eng.served == 2
