"""F2d stage B on CPU fakes: the concurrent decoder plans chunk-boundary checkpoints for a drafting text prompt, keeps
them with its entry, resumes a diverging prompt from the last one at or before the divergence, moves the survivors to
the new entry (thinned, at most N, the rest released), keeps nothing on a failure, shares them with a twin's copy, and
takes none for images, the serial reference or the warm-up; prefill_steps aligns chunk ends to multiples of the rows
and takes a checkpoint after a listed chunk's commit (corrected MTP length, the pre-MTP streams row), skipping one on
an out-of-memory clone; --prefix-checkpoints parsing and the startup geometry."""

import argparse
import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.streams import Stream
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch

ROWS = 4
PROBES = []


class FakeState:
    def __init__(self, name):
        self.name, self.pos, self.restored, self.rope_delta = name, 0, [], 0

    def snapshot(self):
        return {"pos": self.pos, "who": self.name, "mtp_len": max(self.pos - 1, 0)}

    def restore(self, snap):
        self.restored.append(snap)
        self.pos = snap["pos"]

    def copy_prefix(self, src, rows, mtp_rows, ratio):
        self.copied = getattr(self, "copied", []) + [(src.name, rows, mtp_rows)]


def decoder(multi, slots, monkeypatch, log, n=3, fail_at=None):
    """A MultiDecoder with ``n`` checkpoints over fakes: the fake prefill records its resume point and the positions it
    was asked to checkpoint, and takes them (``fail_at``: raise after committing that position)."""

    from tensorfold.families.qwen4_exp.cuda.prefix import Checkpoint

    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.w = SimpleNamespace(mtp=object(), cfg=SimpleNamespace(eos=(), index_ratio=4))
    dec.depth, dec.confidence, dec.capacity, dec.eos = 3, 0.3, 10_000, ()
    dec.buf, dec.mbuf, dec.pbuf = object(), object(), SimpleNamespace(rows=ROWS)
    dec.free = [FakeState(f"slot{i}") for i in range(slots)]
    dec.streams, dec.next_id, dec.kept, dec.keep, dec.next_serial = {}, 0, [], 8, 0
    dec.filling, dec.fills, dec.unreplied, dec.checkpoints, dec.vision = [], {}, [], n, None

    def fake_slot(w, st, buf, mbuf, pbuf, capacity):
        e = SimpleNamespace(st=st, last_streams=None, last_logits=None, first=None)
        e.sample = lambda logits, positions, sampling: [42]
        return e

    def fake_prefill(e, prompt, sampling, *, mtp=True, resume=None, vision=None, checkpoints=None):
        begin = 0 if resume is None else resume["state"]["pos"]
        if resume is not None:
            e.st.restore(resume["state"])
        resume = None                                          # restored: as prefill_steps, this frame lets it go
        log.append(("prefill", len(prompt), begin, list(checkpoints or [])))
        for probe in PROBES:                                   # what is still alive while the checkpoints are taken
            probe()
        e.checkpoints = []
        for p in checkpoints or []:
            assert begin < p < len(prompt)
            e.st.pos = p
            if fail_at == p:
                raise RuntimeError("injected")
            e.checkpoints.append(Checkpoint(p, {"pos": p, "mtp_len": p - 1, "of": tuple(prompt[:p]), "t": torch.zeros(1)},
                                            f"tail:{p}"))
        e.st.pos = len(prompt)
        e.last_streams, e.last_logits = SimpleNamespace(clone=lambda: "tail"), "logits"
        return 7

    monkeypatch.setattr(multi, "_slot", fake_slot)
    monkeypatch.setattr(multi, "prefill", fake_prefill)
    monkeypatch.setattr(multi, "draft", lambda *a, **k: [11])
    return dec


def admit(dec, prompt, count=4, draft=True, **kw):
    s = Stream(list(prompt), count, None, draft=draft, **kw)
    dec.admit(s)
    return s


def finish(dec, s):
    s.done = True
    dec.finish([s])


def toks(seed, n):
    return [seed * 1000 + i for i in range(n)]


@pytest.fixture
def multi(allocations):  # noqa: F811
    return importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")


def test_a_cold_prompt_keeps_its_planned_checkpoints(multi, monkeypatch):
    log = []
    dec = decoder(multi, 2, monkeypatch, log, n=3)
    P = toks(1, 30)                                            # interior ends 4..28: stride 3 -> 12, 24, plus 28
    a = admit(dec, P)
    assert log == [("prefill", 30, 0, [12, 24, 28])] and a.reuse is None
    assert [c.pos for c in dec.kept[0].checkpoints] == [12, 24, 28]


@pytest.mark.parametrize("d,cached", [(24, 24), (25, 24), (20, 12), (11, 0), (12, 12)])
def test_a_diverging_prompt_resumes_from_the_last_checkpoint_at_or_before_the_divergence(multi, monkeypatch, d, cached):
    log = []
    dec = decoder(multi, 1, monkeypatch, log, n=3)            # one slot: the variant resumes in its source's slot
    P = toks(1, 30)
    a = admit(dec, P); finish(dec, a)
    Q = P[:d] + toks(2, 9)
    q = admit(dec, Q)
    if cached:
        assert q.reuse == "checkpoint" and q.cached == cached and q.st is a.st and log[-1][2] == cached
        assert q.st.restored[-1]["pos"] == cached and q.st.restored[-1]["of"] == tuple(P[:cached])
    else:                                                      # inside the first stride: no checkpoint below d
        assert q.reuse is None and q.cached == 0 and log[-1][2] == 0
    mine = [k for k in dec.kept if k.ids == Q]
    assert len(mine) == 1 and len(dec.kept) == 1              # the source is gone (rows overwritten, or evicted)
    kept = mine[0].checkpoints
    assert len(kept) <= 3 and all(c.pos < len(Q) for c in kept)
    assert all(c.snapshot["of"] == tuple(Q[:c.pos]) for c in kept)      # every survivor is a prefix of the new prompt


def test_a_variant_resumes_beside_its_source_so_a_resend_stays_exact(multi, monkeypatch):
    log = []
    dec = decoder(multi, 2, monkeypatch, log, n=3)
    P = toks(1, 30)
    a = admit(dec, P); finish(dec, a)
    source = dec.kept[0]
    points = [c.pos for c in source.checkpoints]
    q = admit(dec, P[:25] + toks(2, 9))
    assert q.reuse == "checkpoint" and q.cached == 24 and q.st is not a.st
    assert q.st.copied == [(a.st.name, 24, 23)]                # the source's rows below the checkpoint, then the resume
    assert q.st.restored[-1]["pos"] == 24 and log[-1][2] == 24
    assert source in dec.kept and [c.pos for c in source.checkpoints] == points   # the source and its checkpoints stay
    mine = next(k for k in dec.kept if k.ids == q.prompt)
    assert len(mine.checkpoints) <= 3 and all(c.snapshot["of"] == tuple(q.prompt[:c.pos]) for c in mine.checkpoints)
    finish(dec, q)
    r = admit(dec, P)                                          # the resend: an exact hit on the untouched source
    assert r.reuse == "exact" and r.cached == 30 and r.st is a.st and len(log) == 2


def test_a_variant_beside_its_source_evicts_the_least_recently_used_other_entry(multi, monkeypatch):
    log = []
    dec = decoder(multi, 3, monkeypatch, log, n=3)
    P1, P2, P3 = toks(1, 30), toks(2, 30), toks(3, 30)
    for p in (P1, P2, P3):
        finish(dec, admit(dec, p))
    slot1 = next(k.slot for k in dec.kept if k.ids == P1)
    q = admit(dec, P1[:25] + toks(4, 9))                      # no free slot: P1 is the source, P2 the oldest other
    assert q.reuse == "checkpoint" and q.st is not slot1
    assert {tuple(k.ids) for k in dec.kept} == {tuple(P1), tuple(P3), tuple(q.prompt)}


def test_a_shortened_prompt_resumes_from_the_previous_checkpoint(multi, monkeypatch):
    log = []
    dec = decoder(multi, 1, monkeypatch, log, n=3)
    P = toks(1, 30)
    a = admit(dec, P); finish(dec, a)
    q = admit(dec, P[:24])                                     # pos == len(prompt) never matches: 12 does
    assert q.reuse == "checkpoint" and q.cached == 12


def test_inherited_checkpoints_are_thinned_and_the_peak_stays_at_n(multi, monkeypatch):
    log = []
    dec = decoder(multi, 1, monkeypatch, log, n=3)
    P = toks(1, 30)
    a = admit(dec, P); finish(dec, a)
    source = dec.kept[0]
    X = P + toks(3, 40)                                        # extends P: inherits 12, 24, 28; boundaries above 30
    x = admit(dec, X)
    assert x.reuse == "extend" and x.cached == 30
    assert source.checkpoints == []                            # released from the dropped source at admission
    took = log[-1][3]
    kept = dec.kept[0].checkpoints
    assert len(kept) == 3 and all(p > 30 for p in took) and sorted(c.pos for c in kept) == [c.pos for c in kept]
    assert {c.pos for c in kept} - set(took) <= {12, 24, 28}   # survivors are inherited, the rest newly taken


def test_a_failed_checkpoint_prefill_keeps_nothing_and_frees_the_slot(multi, monkeypatch, capsys):
    log = []
    dec = decoder(multi, 1, monkeypatch, log, n=3)
    P = toks(1, 30)
    a = admit(dec, P); finish(dec, a)
    dec2 = decoder(multi, 1, monkeypatch, log, n=3, fail_at=48)
    dec2.free, dec2.kept, dec2.next_serial = dec.free, dec.kept, dec.next_serial
    with pytest.raises(RuntimeError, match="injected"):
        admit(dec2, P[:25] + toks(4, 30))                     # resumes at 24, fails while taking 48
    assert dec2.kept == [] and len(dec2.free) == 1 and dec2.streams == {}
    assert "[octojet] prefix reuse checkpoint failed: injected" in capsys.readouterr().err


def test_exact_hits_keep_the_entry_and_a_copy_shares_its_checkpoints(multi, monkeypatch):
    log = []
    dec = decoder(multi, 2, monkeypatch, log, n=3)
    P = toks(1, 30)
    a = admit(dec, P)                                          # decoding: its entry is busy
    b = admit(dec, P)                                          # item 2: a copied exact hit
    assert b.reuse_copy and len(log) == 1
    first, second = sorted(dec.kept, key=lambda k: k.serial)
    assert second.checkpoints == first.checkpoints and second.checkpoints is not first.checkpoints
    finish(dec, a); finish(dec, b)
    c = admit(dec, P[:20] + toks(5, 8))                        # either slot's checkpoint 12 serves it
    assert c.reuse == "checkpoint" and c.cached == 12


def test_no_checkpoints_for_the_serial_reference_images_n_zero_or_the_warm_up(multi, monkeypatch):
    log = []
    dec = decoder(multi, 3, monkeypatch, log, n=3)
    admit(dec, toks(1, 30), draft=False)
    assert log[-1][3] == []
    dec0 = decoder(multi, 1, monkeypatch, log, n=0)
    admit(dec0, toks(2, 30))
    assert log[-1][3] == [] and dec0.kept[0].checkpoints == []
    dec.vision = SimpleNamespace(encode=lambda pixels, ids: "features")
    admit(dec, toks(3, 30), vision="pixels")
    assert log[-1][3] == []


def test_prefill_steps_aligns_chunk_ends_and_takes_corrected_checkpoints(allocations, monkeypatch, capsys):  # noqa: F811
    decode = importlib.import_module("tensorfold.families.qwen4_exp.cuda.decode")
    calls = []

    class St:
        pos, mtp_len, mtp_drafted, rope_delta = 0, 0, 0, 0

        def set_mtp_len(self, n): self.mtp_len = n
        def set_rope_delta(self, d): self.rope_delta = d
        def restore(self, snap): self.pos, self.mtp_len = snap["pos"], snap["mtp_len"]
        def snapshot(self): return {"pos": self.pos, "mtp_len": self.mtp_len - self.mtp_drafted}

    def forward(w, st, pb, chunk, logits=True, features=None):
        calls.append(("fwd", st.pos, len(chunk)))
        pb.streams[:len(chunk)] = torch.arange(st.pos, st.pos + len(chunk), dtype=torch.float32)[:, None]
        return torch.zeros(1, 4)

    def mtp_forward(w, st, pb, nxt, streams):
        pb.streams.fill_(-1.0)                                 # the MTP forward overwrites the streams rows

    def commit(w, st, pb, R, keep, at=0):
        st.pos += R

    monkeypatch.setattr(decode, "forward", forward)
    monkeypatch.setattr(decode, "mtp_forward", mtp_forward)
    monkeypatch.setattr(decode, "commit", commit)
    st = St()
    pb = SimpleNamespace(rope_rows=None, streams=torch.zeros(16, 1), attn=SimpleNamespace(qsa=False, ratio=4))
    e = SimpleNamespace(w=SimpleNamespace(mtp=object()), st=st, pbuf=pb, mbuf=object(), prefill_rows=8,
                        reset=lambda: None, sample=lambda logits, pos, smp: [5])
    st.pos = 0
    decode.prefill(e, list(range(30)), None, resume={"state": {"pos": 5, "mtp_len": 4}, "tail": torch.zeros(1, 1)},
                   checkpoints=[16, 24, 29])
    # ends at 8, 16, 24 (aligned), and 29: an unaligned checkpoint (F7's turn start) cuts its chunk there
    assert [c[1:] for c in calls] == [(5, 3), (8, 8), (16, 8), (24, 5), (29, 1)]
    assert [(c.pos, c.snapshot["mtp_len"], float(c.tail[0, 0])) for c in e.checkpoints] == [(16, 15, 15.0),
                                                                                           (24, 23, 23.0),
                                                                                           (29, 28, 28.0)]
    assert decode.chunk_starts(0, 17, 8) == [0, 8, 16] and decode.chunk_starts(9, 10, 8) == [9]
    assert decode.chunk_starts(0, 17, 8, cuts=[3, 8, 17, 0, 20]) == [0, 3, 8, 16]   # only cuts inside the fill
    assert decode.chunk_starts(9, 30, 8, cuts=[12]) == [9, 12, 16, 24]

    def oom(self):
        raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(St, "snapshot", oom)
    calls.clear()
    assert decode.prefill(e, list(range(20)), None, checkpoints=[8]) == 5 and e.checkpoints == []
    assert "[octojet] prefix checkpoint skipped (OOM) at 8" in capsys.readouterr().err


def test_prefix_checkpoints_options_and_the_startup_geometry():
    from tensorfold.cuda.geometry import indexed_stream_geometry
    from tensorfold.serve_options import check
    from tests.test_cuda_capacity import small_config

    flash = SimpleNamespace(model_type="qwen4_exp", title="Flash Next", package=SimpleNamespace())
    dense = SimpleNamespace(model_type="qwen3_5", title="Qwen3.8 27B", package=SimpleNamespace())
    ns = lambda n: argparse.Namespace(prefix_checkpoints=n, kv_dtype="bf16")       # noqa: E731
    with pytest.raises(ValueError, match="0 or more"):
        check(ns(-1), flash, "cuda")
    with pytest.raises(ValueError, match="Flash Next CUDA feature"):
        check(ns(2), dense, "cuda")
    with pytest.raises(ValueError, match="Flash Next CUDA feature"):
        check(ns(2), flash, "mlx")
    check(ns(0), dense, "cuda")
    check(ns(8), flash, "cuda")
    t = small_config()
    base = indexed_stream_geometry(t, 3, 4, 8, mtp=True).bytes_at(4096)
    with4 = indexed_stream_geometry(t, 3, 4, 8, mtp=True, checkpoints=4).bytes_at(4096)
    lin, nv, dk, dv = 2, 4, 64, 64
    one = lin * nv * dk * dv * 4 + lin * 3 * (2 * 2 * 64 + nv * dv) * 2 + 3 * 3 * 1 * 512 * 2 + 512 * 2
    assert with4 - base == 3 * 4 * one


def test_a_thinned_out_resume_checkpoint_is_released_once_restored(multi, monkeypatch):
    """Peak <= N: Q resumes from P's checkpoint 12, which thinning drops for Q's own; after the admission nothing holds
    its snapshot (the match hands the resume state to the prefill alone)."""

    import gc
    import sys
    import weakref

    monkeypatch.setattr(FakeState, "restore", lambda self, snap: setattr(self, "pos", snap["pos"]))
    log = []
    dec = decoder(multi, 1, monkeypatch, log, n=3)
    P = toks(1, 30)
    a = admit(dec, P); finish(dec, a)
    twelve = next(c for c in dec.kept[0].checkpoints if c.pos == 12)
    gone = weakref.ref(twelve.snapshot["t"])
    del twelve
    alive = []
    monkeypatch.setattr(sys.modules[__name__], "PROBES", [lambda: (gc.collect(), alive.append(gone() is not None))])
    q = admit(dec, P[:12] + toks(6, 88))                      # 100 tokens: plans 44, 92, 96 and thins 12 away
    assert q.reuse == "checkpoint" and q.cached == 12 and log[-1][3] == [44, 92, 96]
    assert alive == [False]                                    # released before the new checkpoints are taken
    assert [c.pos for c in dec.kept[0].checkpoints] == [44, 92, 96]
