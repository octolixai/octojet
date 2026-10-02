"""Prompts inside rounds on the concurrent Flash Next decoder (CPU fakes): a queued prompt prefills one chunk a round
while the live streams keep decoding, every stream's tokens are its solo run, a burst fills oldest first, a failed
chunk fails only its own request and frees its slot, and the scheduler queues untimed admissions only."""

import importlib
import queue
from types import SimpleNamespace

import pytest

from tensorfold.cuda.streams import Stream
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch

ROWS = 4                                              # prompt rows a chunk
COPIES = []


class FakeTensor:
    def __init__(self, tag):
        self.tag = tag

    def clone(self):
        return FakeTensor(self.tag)


class FakeState:
    def __init__(self, name):
        self.name, self.pos, self.history = name, 0, []

    def snapshot(self):
        return {"pos": self.pos, "who": self.name, "history": list(self.history), "mtp_len": self.pos - 1}

    def copy_prefix(self, src, rows, mtp_rows, ratio):
        COPIES.append((src.name, self.name, rows))

    def restore(self, snap):
        self.pos, self.history = snap["pos"], list(snap["history"])


def token(st, position):
    """A row's sample: a function of its own stream's committed rows and position only (as every real row is)."""

    return 100 + (sum(st.history) * 31 + position * 7) % 97


def decoder(multi, slots, monkeypatch, log, fail=None, round_fails=None):
    """A ``MultiDecoder`` over fake kernels: rounds stage, compute, sample and commit with the stand-ins below; prompt
    chunks come from a fake ``prefill_steps`` that logs each chunk. ``fail`` = (prompt tag, chunk) raises there."""

    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.w = SimpleNamespace(mtp=None, cfg=SimpleNamespace(eos=(), index_ratio=4))
    COPIES.clear()
    dec.depth, dec.confidence, dec.capacity, dec.eos = 3, 0.3, 10_000, ()
    dec.buf, dec.mbuf, dec.pbuf = object(), None, object()            # no MTP head: one row a round, no drafts
    dec.free = [FakeState(f"slot{i}") for i in range(slots)]
    dec.streams, dec.next_id, dec.kept, dec.keep, dec.next_serial = {}, 0, [], 8, 0
    dec.filling, dec.fills, dec.vision, dec.unreplied = [], {}, None, []

    def fake_slot(w, st, buf, mbuf, pbuf, capacity):
        e = SimpleNamespace(st=st, last_streams=None, last_logits=None, first=None)
        e.sample = lambda logits, positions, sampling: [token(st, positions[0])]
        return e

    def fake_steps(e, prompt, sampling, *, mtp=True, resume=None, vision=None):
        st, tag = e.st, prompt[0]
        if resume is None:
            st.pos, st.history = 0, []
        else:
            st.restore(resume["state"])
        for start in range(st.pos, len(prompt), ROWS):
            if fail is not None and fail == (tag, start // ROWS):
                raise RuntimeError("injected chunk failure")
            chunk = prompt[start:start + ROWS]
            log.append(("chunk", tag, start))
            st.history += chunk
            st.pos += len(chunk)
            if st.pos < len(prompt):
                yield st.pos
        e.last_streams, e.last_logits = FakeTensor(f"tail:{tag}"), FakeTensor(f"logits:{tag}")
        first = token(st, len(prompt))
        e.first = first
        return first

    def fake_prefill(e, prompt, sampling, *, mtp=True, resume=None, vision=None):
        log.append(("sync", prompt[0]))
        from tensorfold.families.qwen4_exp.cuda.decode import run_steps

        return run_steps(fake_steps(e, prompt, sampling, mtp=mtp, resume=resume, vision=vision))

    staged = {}                                          # slot name -> the tokens of its window this round
    dec.buf = SimpleNamespace(staged=staged)

    def stage(w, buf, windows):
        log.append(("round", tuple(st.name for st, _ in windows)))
        if round_fails is not None and round_fails():
            raise RuntimeError("injected round failure")
        segs, at = [], 0
        for st, tokens in windows:
            staged[st.name] = list(tokens)
            segs.append((st, at, at + len(tokens)))
            at += len(tokens)
        return segs

    def compute(w, segs, buf):
        return segs                                      # the fake sampler reads each segment's state

    def sample_streams(logits, starts, positions, samplings):
        return [[token(st, p) for p in pos] for (st, _, _), pos in zip(logits, positions)]

    def commit(w, st, buf, R, keep, at=0):
        st.history += buf.staged[st.name][:keep]
        st.pos += keep

    monkeypatch.setattr(multi, "_slot", fake_slot)
    monkeypatch.setattr(multi, "prefill_steps", fake_steps)
    monkeypatch.setattr(multi, "prefill", fake_prefill)
    monkeypatch.setattr(multi, "stage", stage)
    monkeypatch.setattr(multi, "compute", compute)
    monkeypatch.setattr(multi, "sample_streams", sample_streams)
    monkeypatch.setattr(multi, "commit", commit)
    return dec


def prompt(tag, n):
    return [tag] + [tag * 10 + i for i in range(n - 1)]


def run_alone(multi, monkeypatch, p, count):
    dec = decoder(multi, 1, monkeypatch, [])
    s = Stream(list(p), count, None, draft=True)
    dec.admit(s, defer=True)
    while dec.live():
        dec.finish(dec.round())
    return s.out


@pytest.fixture
def multi(allocations):  # noqa: F811
    return importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")


def test_a_live_stream_decodes_between_the_chunks_of_another_prompt(multi, monkeypatch):
    A, B = prompt(1, 3), prompt(2, 22)                  # B: six chunks of four rows
    ref_a, ref_b = run_alone(multi, monkeypatch, A, 30), run_alone(multi, monkeypatch, B, 6)
    log = []
    dec = decoder(multi, 2, monkeypatch, log)
    a = Stream(list(A), 30, None)
    dec.admit(a, defer=True)
    assert dec.live() == 1 and dec.filling == [a] and not a.out and dec.streams == {}   # queued, nothing run yet
    dec.finish(dec.round())                            # nothing else decodes: the whole prompt, then a round
    assert len(a.out) == 2 and [x[0] for x in log] == ["chunk", "round"]
    b = Stream(list(B), 6, None)
    dec.admit(b, defer=True)
    assert dec.filling == [b] and dec.live() == 2
    grew = []
    while b in dec.filling:
        before = len(a.out)
        log.clear()
        dec.finish(dec.round())
        grew.append(len(a.out) - before)
        assert log[0][0] == "chunk" and log[0][1] == 2 and sum(1 for x in log if x[0] == "chunk") == 1
        assert any(x[0] == "round" and a.st.name in x[1] for x in log)    # A's window was in the round
    assert len(grew) == 6 and all(n == 1 for n in grew)                 # a chunk a round; A decoded every one of them
    assert b.out and b.prefill_s >= 0 and not dec.filling
    while dec.live():
        dec.finish(dec.round())
    assert a.out == ref_a and b.out == ref_b                             # each stream's solo run
    assert len(dec.free) + len({id(k.slot) for k in dec.kept}) == 2


def test_a_burst_fills_oldest_first_one_prompt_at_a_time(multi, monkeypatch):
    log = []
    dec = decoder(multi, 3, monkeypatch, log)
    a = Stream(prompt(1, 2), 40, None)
    dec.admit(a, defer=True)
    dec.finish(dec.round())
    b, c = Stream(prompt(2, 9), 4, None), Stream(prompt(3, 9), 4, None)
    dec.admit(b, defer=True)
    dec.admit(c, defer=True)
    while dec.filling:
        dec.finish(dec.round())
    chunks = [x[1] for x in log if x[0] == "chunk"]
    assert chunks == [1, 2, 2, 2, 3, 3, 3]                              # B's three chunks, then C's
    rounds = [i for i, x in enumerate(log) if x[0] == "round"]
    assert all(any(r > i for r in rounds) for i, x in enumerate(log) if x[0] == "chunk")   # a round after each chunk


def test_a_failed_chunk_fails_its_request_frees_its_slot_and_the_others_decode(multi, monkeypatch, capsys):
    log = []
    dec = decoder(multi, 2, monkeypatch, log, fail=(2, 2))
    a = Stream(prompt(1, 2), 20, None)
    dec.admit(a, defer=True)
    dec.finish(dec.round())
    b = Stream(prompt(2, 17), 5, None)
    dec.admit(b, defer=True)
    slot = b.st
    ended = []
    while b in dec.filling:
        done = dec.round()
        ended += done
        dec.finish(done)
    assert ended == [b] and b.done and isinstance(b.error, RuntimeError) and not b.out
    assert dec.free.count(slot) == 1 and all(k.slot is not slot for k in dec.kept) and b.sid not in dec.streams
    while dec.live():
        dec.finish(dec.round())
    assert len(a.out) == 20 and a.error is None


def test_a_failed_first_emission_of_a_filled_prompt_is_cleaned_up(multi, monkeypatch, capsys):
    dec = decoder(multi, 2, monkeypatch, [])
    a = Stream(prompt(1, 2), 20, None)
    dec.admit(a, defer=True)
    dec.finish(dec.round())

    def boom(new):
        raise RuntimeError("emit failed")

    b = Stream(prompt(2, 6), 5, None, emit=boom)
    dec.admit(b, defer=True)
    ended = []
    while b in dec.filling:
        done = dec.round()
        ended += done
        dec.finish(done)
    assert ended == [b] and str(b.error) == "emit failed"
    assert b.sid not in dec.streams and dec.kept and all(k.slot is not b.st for k in dec.kept)   # B kept nothing
    assert dec.free.count(b.st) == 1


def test_drop_ends_the_filling_prompts_too(multi, monkeypatch):
    dec = decoder(multi, 2, monkeypatch, [])
    a = Stream(prompt(1, 2), 20, None)
    dec.admit(a, defer=True)
    dec.finish(dec.round())
    b = Stream(prompt(2, 30), 5, None)
    dec.admit(b, defer=True)
    dec.finish(dec.round())                             # one chunk of B
    gen = dec.fills[b.sid].steps
    dropped = dec.drop()
    assert {s.sid for s in dropped} == {a.sid, b.sid} and not dec.filling and not dec.fills and not dec.streams
    assert gen.gi_frame is None                         # the generator closed
    assert len(dec.free) == 2 and dec.live() == 0


def test_an_identical_prompt_while_one_fills_is_a_busy_miss_and_an_exact_hit_is_admitted_at_once(multi, monkeypatch):
    log = []
    dec = decoder(multi, 3, monkeypatch, log)
    P = prompt(4, 9)
    a = Stream(list(P), 3, None)
    dec.admit(a, defer=True)
    b = Stream(list(P), 3, None)
    dec.admit(b, defer=True)                            # a still fills: nothing kept yet, so B prefills cold
    assert b.reuse is None and b.reuse_miss == "busy" and b.st is not a.st
    while dec.live():
        dec.finish(dec.round())
    log.clear()
    c = Stream(list(P), 3, None)
    dec.admit(c, defer=True)                            # kept and idle: an exact hit has no prompt to fill
    assert c.reuse == "exact" and c.out and not dec.filling and not any(x[0] == "chunk" for x in log)


def test_the_synchronous_admission_is_unchanged(multi, monkeypatch):
    log = []
    dec = decoder(multi, 1, monkeypatch, log)
    s = Stream(prompt(5, 9), 3, None)
    dec.admit(s)
    assert s.out and s.sid in dec.streams and not dec.filling and log[0] == ("sync", 5)


def test_the_scheduler_queues_untimed_admissions_and_rounds_fill_them(multi, monkeypatch):
    from tensorfold.cuda import scheduler as sched_mod

    log = []
    dec = decoder(multi, 3, monkeypatch, log)
    assert sched_mod.defer(dec, Stream([1], 1)) is True
    for flag in ("timing", "histogram", "profile"):
        assert sched_mod.defer(dec, Stream([1], 1, **{flag: True})) is False
    assert sched_mod.defer(SimpleNamespace(), Stream([1], 1)) is False          # decoders without ``defers``
    sched = sched_mod.Scheduler.__new__(sched_mod.Scheduler)                     # no worker thread: driven here
    sched.decoder, sched.max_streams, sched.waiting, sched.boxes, sched.held = dec, 3, queue.Queue(), {}, []
    streams = [Stream(prompt(6, 10), 8, None), Stream(prompt(7, 13), 4, None)]
    for s in streams:
        sched.waiting.put((s, queue.Queue()))
    assert sched._admit() == [] and dec.filling == streams and not log         # both queued, no prompt row run yet
    done = dec.round()                                   # the first fills whole (nothing decodes), then a round
    assert [x[0] for x in log] == ["chunk"] * 3 + ["round"] and streams[0].out and not streams[1].out
    log.clear()
    dec.finish(done)
    dec.finish(dec.round())                              # one chunk of the second beside the first's round
    assert [x[0] for x in log] == ["chunk", "round"]


@pytest.mark.parametrize("b_fails", [False, True])
def test_a_round_that_fails_after_a_fill_ended_still_answers_that_request(multi, monkeypatch, b_fails):
    """B ends in the round's fill (done at its first token, or its chunk fails), then A's decode raises: B's request
    gets its own reply (done, or its own error) and every slot comes back."""

    import threading

    from tensorfold.cuda.scheduler import Scheduler

    joined = []
    dec = decoder(multi, 2, monkeypatch, [], fail=(2, 0) if b_fails else None, round_fails=lambda: bool(joined))
    real_fill = dec._fill

    def fill():                                    # once B's fill has ended, the round's decode raises
        ended = real_fill()
        joined.extend(x for x in ended if x.prompt[0] == 2)
        return ended

    dec._fill = fill
    sched = Scheduler(dec, max_streams=2)
    flowing = threading.Event()
    got = {}

    def ask(name, p, count, on_tokens=None):
        def go():
            try:
                got[name] = ("done", sched.submit(p, count, None, True, on_tokens or (lambda new: False)))
            except Exception as exc:               # noqa: BLE001
                got[name] = ("error", exc)
        t = threading.Thread(target=go, daemon=True)
        t.start()
        return t

    ta = ask("a", prompt(1, 2), 10**6, lambda new: flowing.set() and False)
    assert flowing.wait(10)
    tb = ask("b", prompt(2, 3), 1)
    ta.join(10), tb.join(10)
    assert got["a"][0] == "error" and str(got["a"][1]) == "injected round failure"
    if b_fails:
        assert got["b"][0] == "error" and str(got["b"][1]) == "injected chunk failure"
    else:
        assert got["b"][0] == "done"
    assert sched.thread.is_alive() and dec.live() == 0 and not dec.unreplied
    assert len(dec.free) + len({id(k.slot) for k in dec.kept}) == 2 and len(set(map(id, dec.free))) == len(dec.free)


def scheduler(dec, streams):
    """A ``Scheduler`` without its worker thread (driven by the test), ``streams`` queued in order."""

    from tensorfold.cuda import scheduler as sched_mod

    sched = sched_mod.Scheduler.__new__(sched_mod.Scheduler)
    sched.decoder, sched.max_streams, sched.waiting, sched.boxes, sched.held = dec, 3, queue.Queue(), {}, []
    for s in streams:
        sched.waiting.put((s, queue.Queue()))
    return sched


def test_an_identical_request_waits_for_its_twins_fill_then_copies_its_state(multi, monkeypatch):
    """F4 item 2: B (A's twin) is held while A fills and C passes it; once A joins and decodes, B exact-hits A's kept
    state copied into a spare slot, and emits A's solo run."""

    P = prompt(3, 22)
    ref = run_alone(multi, monkeypatch, P, 12)
    log = []
    dec = decoder(multi, 3, monkeypatch, log)
    a, b, c = Stream(list(P), 12, None), Stream(list(P), 12, None), Stream(prompt(4, 5), 12, None)
    sched = scheduler(dec, [a, b, c])
    sched._admit()
    assert dec.filling == [a, c] and [h[0] for h in sched.held] == [b] and not b.out    # C passed the held B
    while a in dec.filling:
        sched._admit()                                   # B stays held while A fills
        assert [h[0] for h in sched.held] == [b]
        dec.finish(dec.round())
    log.clear()
    sched._admit()                                       # A joined (decoding): B copies its state
    assert not sched.held and b.reuse == "exact" and b.reuse_copy and b.cached == len(P) and b.reuse_miss is None
    assert COPIES == [(a.st.name, b.st.name, len(P))] and not any(x[0] in ("chunk", "sync") for x in log)
    assert b.out and b.st is not a.st and b.admitted_at >= b.queued_at
    while dec.live():
        dec.finish(dec.round())
    assert a.out == b.out == ref


def test_a_twin_wait_past_the_cap_prefills_cold_as_a_busy_miss(multi, monkeypatch):
    from tensorfold.cuda import scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "TWIN_WAIT_S", -1.0)                     # every wait is already past the cap
    dec = decoder(multi, 3, monkeypatch, [])
    P = prompt(5, 22)
    a, b = Stream(list(P), 6, None), Stream(list(P), 6, None)
    sched = scheduler(dec, [a])
    sched._admit()
    sched.held.append((b, queue.Queue(), 0.0))
    sched._admit()
    assert not sched.held and b in dec.filling and b.reuse is None and b.reuse_miss == "busy" and not COPIES


def test_a_profiled_request_never_waits_for_a_twin(multi, monkeypatch):
    dec = decoder(multi, 3, monkeypatch, [])
    P = prompt(6, 22)
    a, b = Stream(list(P), 6, None), Stream(list(P), 6, None, profile=True)
    sched = scheduler(dec, [a, b])
    sched._admit()                                       # B prefills at once, cold, beside A's fill
    assert not sched.held and b.out and b.reuse_miss == "busy" and b in dec.streams.values()


def test_held_requests_are_admitted_when_the_twin_fails(multi, monkeypatch):
    dec = decoder(multi, 3, monkeypatch, [], fail=(7, 1))
    P = prompt(7, 22)
    a, b = Stream(list(P), 6, None), Stream(list(P), 6, None)
    sched = scheduler(dec, [a, b])
    sched._admit()
    assert [h[0] for h in sched.held] == [b]
    done = dec.round()
    while a not in done:
        dec.finish(done)
        done = dec.round()
    dec.finish(done)
    assert a.error is not None and dec.live() == 0
    sched._admit()                                       # nothing live: the held request goes in, a cold fill
    assert not sched.held and b in dec.filling and b.reuse_miss is None


def test_held_requests_make_an_idle_fill_return_between_chunks(multi, monkeypatch):
    """With a request held for a twin and nothing decoding, a round runs one chunk, so the 120 s cap is checked between
    chunks rather than after the whole prompt."""

    from tensorfold.cuda import scheduler as sched_mod

    log = []
    dec = decoder(multi, 3, monkeypatch, log)
    P = prompt(8, 22)
    a, b = Stream(list(P), 6, None), Stream(list(P), 6, None)
    sched = scheduler(dec, [a, b])
    sched._admit()
    assert [h[0] for h in sched.held] == [b]
    dec.short_fill = bool(sched.held)                    # what the worker loop sets before each round
    dec.finish(dec.round())
    assert [x[0] for x in log] == ["chunk"] and a in dec.filling
    monkeypatch.setattr(sched_mod, "TWIN_WAIT_S", -1.0)  # the cap has passed: B goes in cold, a busy miss
    sched._admit()
    assert not sched.held and b.reuse_miss == "busy" and b in dec.filling
