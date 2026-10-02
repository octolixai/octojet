"""The concurrent decoder's kept entries: exact hits skip the prefill and sample with the new request's parameters,
busy duplicates are copied and hit exact, busy extends fork beside their source (F7), the serial reference stays fresh, and a failed admission at any
step — slot set-up through the first token's emission — leaves nothing kept, no stream registered and the slot free."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.streams import Stream
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch


class FakeTensor:
    def __init__(self, tag):
        self.tag = tag

    def clone(self):
        return FakeTensor(self.tag)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and other.tag == self.tag


class FakeState:
    def __init__(self, name):
        self.name, self.pos, self.restored = name, 0, []

    def snapshot(self):
        return {"pos": self.pos, "who": self.name, "mtp_len": max(self.pos - 1, 0)}

    def restore(self, snap):
        self.restored.append(snap)
        self.pos = snap["pos"]

    def copy_prefix(self, src, rows, mtp_rows, ratio):
        COPIES.append((src.name, self.name, rows, mtp_rows, ratio))


COPIES = []                                               # every copy_prefix call: (from, to, rows, mtp rows, ratio)


def decoder(multi, slots, monkeypatch, log):
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.w = SimpleNamespace(mtp=object(), cfg=SimpleNamespace(eos=(), index_ratio=4))
    COPIES.clear()
    dec.depth, dec.confidence, dec.capacity, dec.eos = 3, 0.3, 10_000, ()
    dec.buf, dec.mbuf, dec.pbuf = object(), object(), object()             # MTP on: the admission clones and drafts
    dec.free = [FakeState(f"slot{i}") for i in range(slots)]
    dec.streams, dec.next_id, dec.kept, dec.keep, dec.next_serial = {}, 0, [], 8, 0
    dec.filling, dec.fills = [], {}

    def fake_slot(w, st, buf, mbuf, pbuf, capacity):
        e = SimpleNamespace(st=st, last_streams=None, last_logits=None, first=None)
        e.sample = lambda logits, positions, sampling: log.append(("sample", logits, positions, sampling)) or [42]
        return e

    def fake_prefill(e, prompt, sampling, *, mtp=True, resume=None):
        log.append(("prefill", tuple(prompt), None if resume is None else resume["state"]["pos"]))
        e.st.pos = len(prompt)
        e.last_streams, e.last_logits, e.first = FakeTensor(f"tail:{len(prompt)}"), FakeTensor(f"logits:{len(prompt)}"), 7
        return 7

    def fake_draft(e, streams, next_tokens, position, count, sampling, confidence):
        log.append(("draft", streams.tag, list(next_tokens), position, count))
        return [11]

    monkeypatch.setattr(multi, "_slot", fake_slot)
    monkeypatch.setattr(multi, "prefill", fake_prefill)
    monkeypatch.setattr(multi, "draft", fake_draft)
    return dec


def admit(dec, prompt, count=4, draft=True, sampling=None, emit=None):
    s = Stream(list(prompt), count, sampling, draft=draft, emit=emit)
    dec.admit(s)
    return s


def finish(dec, s):
    s.done = True
    dec.finish([s])


def test_exact_hit_skips_the_prefill_and_samples_with_the_new_request(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    log = []
    dec = decoder(multi, 2, monkeypatch, log)
    P = [1, 2, 3, 4]
    a = admit(dec, P, sampling="S1")
    assert a.reuse is None and a.cached == 0 and a.out == [7] and a.drafts == [11] and a.context == P + [7]
    finish(dec, a)
    assert len(dec.kept) == 1 and dec.kept[0].logits == FakeTensor("logits:4") and dec.kept[0].checkpoints == []
    log.clear()
    b = admit(dec, P, sampling="S2")
    assert b.reuse == "exact" and b.cached == 4 and b.reuse_miss is None and b.out == [42] and b.drafts == [11]
    assert not any(c[0] == "prefill" for c in log) and ("sample", FakeTensor("logits:4"), [4], "S2") in log
    assert ("draft", "tail:4", [42], 5, 3) in log                                     # drafts from the kept tail
    assert b.st is a.st and b.st.restored[-1] == {"pos": 4, "who": a.st.name, "mtp_len": 3}
    assert len(dec.kept) == 1 and dec.kept[0].slot is a.st and dec.kept[0].serial == 0    # kept, not re-remembered
    assert dec.streams == {b.sid: b} and b.sid != a.sid
    finish(dec, b)
    assert len(dec.kept) == 1 and a.st not in dec.free                                 # the slot stays pinned


def test_busy_duplicate_copies_the_twins_kept_state_and_hits_exact(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    log = []
    dec = decoder(multi, 2, monkeypatch, log)
    P = [1, 2, 3, 4]
    a = admit(dec, P); finish(dec, a)
    b = admit(dec, P)                                     # exact hit; the slot is busy now
    log.clear()
    c = admit(dec, P, sampling="S3")                      # identical while b decodes: b's kept state, copied
    assert c.reuse == "exact" and c.reuse_copy and c.cached == 4 and c.reuse_miss is None and c.st is not b.st
    assert c.stats()["reuse_copy"] is True and "reuse_copy" not in b.stats()
    assert COPIES == [(b.st.name, c.st.name, 4, 3, 4)] and not any(x[0] == "prefill" for x in log)
    assert ("sample", FakeTensor("logits:4"), [4], "S3") in log and ("draft", "tail:4", [42], 5, 3) in log
    assert c.st.restored[-1] == {"pos": 4, "who": a.st.name, "mtp_len": 3} and not dec.free
    assert len(dec.kept) == 2 and {k.slot for k in dec.kept} == {b.st, c.st}    # the copy is an entry of its own
    finish(dec, c)
    finish(dec, b)
    assert len(dec.kept) == 2 and not dec.free            # both slots stay kept


def test_a_busy_extend_entry_forks_beside_it(allocations, monkeypatch):  # noqa: F811
    """F7: a prompt extending a decoding stream's kept prompt copies its rows into the spare slot and resumes there
    (no busy miss, no cold fill); the source stays kept in its own slot."""

    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    log = []
    dec = decoder(multi, 2, monkeypatch, log)
    a = admit(dec, [1, 2, 3])                             # decoding: its entry is busy
    c = admit(dec, [1, 2, 3, 4])                          # extends it: a fork beside it
    assert c.reuse == "extend" and c.reuse_miss is None and c.cached == 3 and c.reuse_copy and c.st is not a.st
    assert COPIES and COPIES[0][:3] == (a.st.name, c.st.name, 3)
    assert not any(x[0] == "prefill" and x[2] is None for x in log[1:])     # resumed, not filled cold
    assert any(k.slot is a.st for k in dec.kept)          # the source's entry survives


def test_a_fork_needs_a_spare_slot_and_more_tokens_than_the_idle_match(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, 2, monkeypatch, [])
    admit(dec, [1, 2, 3])
    admit(dec, [7, 8, 9])                                 # both slots decode: no spare slot to fork into
    assert dec._fork([1, 2, 3, 4], dec._busy(), 0) is None
    dec = decoder(multi, 3, monkeypatch, [])
    admit(dec, [1, 2, 3])
    assert dec._fork([1, 2, 3, 4], dec._busy(), 3) is None          # an idle match reusing as much wins
    slot, m, miss = dec._fork([1, 2, 3, 4], dec._busy(), 2)
    assert m.kind == "extend" and m.forked and m.cached == 3 and miss is None and slot in dec.free + [slot]


def test_the_copy_evicts_the_least_recently_used_idle_entry_and_never_the_twin(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, 2, monkeypatch, [])
    V, P = [7, 8, 9], [1, 2, 3]
    v = admit(dec, V); finish(dec, v)                     # slot of V: idle and kept
    a = admit(dec, P)                                     # the other slot, decoding
    c = admit(dec, P)                                     # no free slot: V's goes, P's twin stays
    assert c.reuse_copy and c.st is v.st and [k.ids for k in dec.kept] == [P, P] and COPIES[-1][0] == a.st.name


def test_a_failed_copy_cleans_up_like_any_admission(allocations, monkeypatch, capsys):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, 2, monkeypatch, [])
    P = [1, 2, 3]
    a = admit(dec, P)
    monkeypatch.setattr(FakeState, "copy_prefix", boom)
    with pytest.raises(RuntimeError, match="injected"):
        admit(dec, P)
    assert [k.slot for k in dec.kept] == [a.st] and len(dec.free) == 1 and list(dec.streams) == [a.sid]
    assert "[octojet] prefix reuse exact failed: injected" in capsys.readouterr().err
    monkeypatch.setattr(FakeState, "copy_prefix", lambda self, *x: None)
    real = multi.exact_hit
    monkeypatch.setattr(multi, "exact_hit", boom)         # past the copy: its new entry goes too
    with pytest.raises(RuntimeError, match="injected"):
        admit(dec, P)
    assert [k.slot for k in dec.kept] == [a.st] and len(dec.free) == 1 and list(dec.streams) == [a.sid]
    monkeypatch.setattr(multi, "exact_hit", real)
    c = admit(dec, P)
    assert c.reuse_copy and len(dec.kept) == 2


def test_five_identical_requests_under_two_slots(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, 2, monkeypatch, [])
    P = list(range(10))
    kinds = []
    s1 = admit(dec, P); s2 = admit(dec, P); kinds += [s1.reuse, s2.reuse]          # both slots busy
    finish(dec, s1)                                                                   # slot 1 idle and kept
    s3 = admit(dec, P); kinds.append(s3.reuse)
    finish(dec, s2)                                                                   # slot 2 idle and kept (duplicate)
    s4 = admit(dec, P); kinds.append(s4.reuse)
    finish(dec, s3)
    s5 = admit(dec, P); kinds.append(s5.reuse)
    assert kinds == [None, "exact", "exact", "exact", "exact"]   # s2: s1's kept state, copied


def test_serial_reference_never_reuses_or_evicts(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    log = []
    dec = decoder(multi, 2, monkeypatch, log)
    P = [1, 2, 3]
    a = admit(dec, P); finish(dec, a)
    b = admit(dec, P, draft=False)
    assert b.reuse is None and b.reuse_miss is None and b.cached == 0 and ("prefill", (1, 2, 3), None) in log
    finish(dec, b)
    assert len(dec.kept) == 1 and dec.kept[0].slot is a.st                        # b took the free slot; a's entry intact


def test_extend_drops_the_source_entry_and_reports_extend(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    log = []
    dec = decoder(multi, 1, monkeypatch, log)
    a = admit(dec, [1, 2, 3]); finish(dec, a)
    b = admit(dec, [1, 2, 3, 4, 5])
    assert b.reuse == "extend" and b.cached == 3 and ("prefill", (1, 2, 3, 4, 5), 3) in log
    finish(dec, b)
    assert [k.ids for k in dec.kept] == [[1, 2, 3, 4, 5]]


def boom(*a, **k):
    raise RuntimeError("injected")


@pytest.mark.parametrize("stage,kind", [("slot", "exact"), ("restore", "exact"), ("prefill", "extend"), ("snapshot", "extend"),
                                        ("draft", "exact"), ("emit", "exact")])
def test_a_failed_admission_frees_the_slot_and_keeps_nothing(allocations, monkeypatch, capsys, stage, kind):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, 1, monkeypatch, [])
    a = admit(dec, [1, 2]); finish(dec, a)
    prompt = [1, 2] if kind == "exact" else [1, 2, 3]
    emit = None
    if stage == "slot":
        monkeypatch.setattr(multi, "_slot", boom)
    elif stage == "restore":
        monkeypatch.setattr(multi, "exact_hit", boom)
    elif stage == "prefill":
        monkeypatch.setattr(multi, "prefill", boom)
    elif stage == "snapshot":
        monkeypatch.setattr(FakeState, "snapshot", boom)
    elif stage == "draft":
        monkeypatch.setattr(multi, "draft", boom)
    else:
        emit = boom                                       # Stream.emit raises inside take(): the first token's emission
    with pytest.raises(RuntimeError, match="injected"):
        admit(dec, prompt, emit=emit)
    assert dec.kept == [] and len(dec.free) == 1 and dec.streams == {}
    assert f"[octojet] prefix reuse {kind} failed: injected" in capsys.readouterr().err
    monkeypatch.undo()
    ok = admit(decoder(multi, 1, monkeypatch, []), [1, 2])                        # a fresh decoder still admits
    assert ok.reuse is None and ok.out == [7]


def test_the_same_decoder_admits_again_after_a_failure(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    log = []
    dec = decoder(multi, 1, monkeypatch, log)
    a = admit(dec, [1, 2]); finish(dec, a)
    real = multi.draft
    monkeypatch.setattr(multi, "draft", boom)
    with pytest.raises(RuntimeError, match="injected"):
        admit(dec, [1, 2])
    monkeypatch.setattr(multi, "draft", real)
    b = admit(dec, [1, 2])                                # the slot is free again: a cold admission on the same decoder
    assert b.reuse is None and b.cached == 0 and b.out == [7] and dec.streams == {b.sid: b} and len(dec.free) == 0
    finish(dec, b)
    assert len(dec.kept) == 1 and dec.kept[0].ids == [1, 2]


def test_an_exact_hit_refreshes_the_entrys_recency(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, 2, monkeypatch, [])
    P, V, W = [1, 2, 3], [7, 8, 9], [5, 5, 5]
    a = admit(dec, P); finish(dec, a)                    # slot A keeps P
    b = admit(dec, V); finish(dec, b)                    # slot B keeps V; no slot is free from here on
    assert [k.ids for k in dec.kept] == [P, V]
    for _ in range(3):
        h = admit(dec, P); assert h.reuse == "exact" and h.st is a.st; finish(dec, h)
    assert [k.ids for k in dec.kept] == [V, P]           # the hits moved P behind V: V is now the least recently used
    c = admit(dec, W)                                    # unrelated and cold: needs a slot → evicts V, not the thrice-used P
    assert c.reuse is None and c.st is b.st and [k.ids for k in dec.kept] == [P, W]
    finish(dec, c)
    e = admit(dec, P)
    assert e.reuse == "exact" and e.st is a.st


def test_exact_hits_on_duplicate_entries_with_tensor_snapshots_do_not_compare_tensors(allocations, monkeypatch):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    monkeypatch.setattr(FakeState, "snapshot", lambda self: {"pos": self.pos, "rec": torch.zeros(4), "who": self.name,
                                                       "mtp_len": self.pos})
    dec = decoder(multi, 2, monkeypatch, [])
    P = list(range(10))
    kinds = []
    s1 = admit(dec, P); s2 = admit(dec, P); kinds += [s1.reuse, s2.reuse]          # both slots busy: two entries for P
    finish(dec, s1)
    s3 = admit(dec, P); kinds.append(s3.reuse)                                        # exact on slot A (B is busy)
    finish(dec, s2)                                                                   # slot B idle and kept: a duplicate of A's entry
    s4 = admit(dec, P); kinds.append(s4.reuse)                                        # exact on slot B (A is busy with s3)
    assert s4.st is s2.st and len(dec.kept) == 2 and dec.kept[-1].slot is s4.st
    finish(dec, s3)
    s5 = admit(dec, P); kinds.append(s5.reuse)
    assert kinds == [None, "exact", "exact", "exact", "exact"]   # s2: s1's kept state, copied
    finish(dec, s4); finish(dec, s5)                                                  # both idle: the larger serial (B) wins ties
    s6 = admit(dec, P); finish(dec, s6)                                               # B's entry was first; now it is last
    s7 = admit(dec, P)                                                                # B again, behind A's equal-ids entry
    assert s6.reuse == s7.reuse == "exact" and s6.st is s7.st is s2.st and dec.kept[-1].slot is s2.st
