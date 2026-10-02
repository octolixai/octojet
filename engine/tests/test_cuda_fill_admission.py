"""F7 on the concurrent Flash Next decoder (CPU fakes): a lone fill stops between chunks for a request the scheduler
could admit, the next chunk goes to the prompt with the fewest rows left (a prompt passed over FILL_GUARD chunks goes
first), every stream still emits its solo run, and a prompt sharing a checkpoint with a decoding stream forks beside it.
Idea from TensorFold 0.6.1 (short prompts admitted while a long one fills; forks that resume from a shared prefix);
the implementation and these tests are Octojet's own."""

import importlib
import queue

import pytest

from tensorfold.cuda.streams import Stream
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)
from tests.test_cuda_prompts_inside_rounds import COPIES, decoder, prompt, run_alone

pytestmark = pytest.mark.torch


@pytest.fixture
def multi(allocations):  # noqa: F811
    return importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")


def chunks(log):
    return [x[1] for x in log if x[0] == "chunk"]


def test_a_lone_fill_stops_between_chunks_for_an_admissible_request(multi, monkeypatch):
    log = []
    dec = decoder(multi, 3, monkeypatch, log)
    a = Stream(prompt(1, 40), 4, None)                   # ten chunks, nothing decoding
    dec.admit(a, defer=True)
    waiting = []
    dec.arrived = lambda: bool(waiting)
    calls = {"n": 0}
    real = dec._chunk

    def chunk():
        calls["n"] += 1
        if calls["n"] == 3:
            waiting.append(1)                            # a request arrives during the third chunk
        return real()

    dec._chunk = chunk
    assert dec.round() == [] and chunks(log) == [1, 1, 1] and a in dec.filling      # back to the scheduler
    waiting.clear()
    dec.finish(dec.round())                              # nothing waits: the rest fills, then a round
    assert chunks(log) == [1] * 10 and a.out


def test_the_fewest_rows_left_fill_next_and_every_stream_keeps_its_solo_run(multi, monkeypatch):
    L, S1, S2 = prompt(1, 40), prompt(2, 6), prompt(3, 9)
    refs = [run_alone(multi, monkeypatch, p, 5) for p in (L, S1, S2)]
    log = []
    dec = decoder(multi, 3, monkeypatch, log)
    long_ = Stream(list(L), 5, None)
    dec.admit(long_, defer=True)
    dec.arrived = lambda: True                           # stop after every chunk: the test admits in between
    dec.finish(dec.round())
    dec.finish(dec.round())                              # two chunks of the long prompt (32 rows left)
    s1, s2 = Stream(list(S1), 5, None), Stream(list(S2), 5, None)
    dec.admit(s1, defer=True)
    dec.admit(s2, defer=True)
    while dec.live():
        dec.finish(dec.round())
    order = chunks(log)
    assert order[:2] == [1, 1]
    first_long_again = order.index(1, 2)
    assert set(order[2:first_long_again]) == {2, 3}      # both short prompts fill before the long one resumes
    assert order[2:4] == [2, 2]                          # the 6-row prompt (fewest left) first
    assert [long_.out, s1.out, s2.out] == refs


def test_a_prompt_passed_over_fill_guard_chunks_takes_the_next(multi, monkeypatch):
    dec = decoder(multi, 3, monkeypatch, [])
    long_, short = Stream(prompt(1, 400), 2, None), Stream(prompt(2, 400), 2, None)
    dec.admit(long_, defer=True)
    dec.admit(short, defer=True)
    dec.fills[long_.sid].at = 0
    dec.fills[short.sid].at = 300                        # 100 rows left: it is chosen every time ...
    for _ in range(multi.FILL_GUARD):
        assert dec._next_fill() is short
        dec.fills[long_.sid].skipped += 1
    assert dec._next_fill() is long_                     # ... until the long one has waited FILL_GUARD chunks


def test_skips_count_and_reset_through_chunks(multi, monkeypatch):
    log = []
    dec = decoder(multi, 3, monkeypatch, log)
    a, b = Stream(prompt(1, 20), 2, None), Stream(prompt(2, 8), 2, None)
    dec.admit(a, defer=True)
    dec.admit(b, defer=True)
    dec._chunk()                                         # b: 8 rows < 20
    assert chunks(log) == [2] and dec.fills[a.sid].skipped == 1 and dec.fills[b.sid].skipped == 0
    assert dec.fills[b.sid].at == 4


def test_the_scheduler_hands_its_admission_check_to_the_decoder(multi, monkeypatch):
    from tensorfold.cuda import scheduler as sched_mod

    dec = decoder(multi, 2, monkeypatch, [])
    sched = sched_mod.Scheduler.__new__(sched_mod.Scheduler)
    sched.decoder, sched.max_streams, sched.waiting, sched.boxes, sched.held = dec, 2, queue.Queue(), {}, []
    assert sched.admissible() is False
    sched.waiting.put((Stream([1], 1), queue.Queue()))
    assert sched.admissible() is True
    dec.admit(Stream(prompt(1, 9), 2, None), defer=True)
    dec.admit(Stream(prompt(2, 9), 2, None), defer=True)
    assert sched.admissible() is False                  # every stream busy: nothing could be admitted


def test_an_exact_hit_during_a_long_fill_decodes_before_the_fill_ends(multi, monkeypatch):
    """The classifier case: a repeated prompt arriving while a long prompt fills alone is admitted between chunks and
    decodes beside the remaining chunks, instead of waiting for the whole fill."""

    import threading

    from tensorfold.cuda.scheduler import Scheduler

    log = []
    dec = decoder(multi, 3, monkeypatch, log)
    P = prompt(5, 6)
    dec.admit(Stream(list(P), 2, None))                  # kept: an exact hit from now on
    while dec.live():
        dec.finish(dec.round())
    gate = threading.Event()
    real = dec._chunk

    def slow_chunk():
        gate.wait(5)                                     # the long fill's chunks run only once the hit is queued
        return real()

    dec._chunk = slow_chunk
    sched = Scheduler(dec, max_streams=3)
    got = {}
    long_t = threading.Thread(target=lambda: got.setdefault("long", sched.submit(prompt(6, 400), 2, None, True,
                                                                                 lambda new: False)), daemon=True)
    long_t.start()
    hit_slot = dec.kept[0].slot.name
    hit_t = threading.Thread(target=lambda: got.setdefault("hit", sched.submit(list(P), 2, None, True,
                                                                               lambda new: False)), daemon=True)
    import time
    time.sleep(0.2)
    hit_t.start()
    time.sleep(0.2)
    gate.set()
    hit_t.join(10)
    long_t.join(10)
    assert got["hit"]["reuse"] == "exact"
    done_at = [i for i, x in enumerate(log) if x[0] == "chunk" and x[1] == 6]
    assert len(done_at) == 100 and got["long"]
    hit_round = next(i for i, x in enumerate(log) if x[0] == "round" and hit_slot in x[1] and i > done_at[0])
    assert hit_round < done_at[-1]                       # the hit decoded while the long prompt still filled


def test_a_variant_of_a_decoding_prompt_forks_from_its_checkpoint(multi, monkeypatch):
    """B shares A's first two chunks: while A decodes, B copies A's rows below the checkpoint into a spare slot and
    resumes there; B's tokens are its solo run and A's entry stays kept."""

    from tensorfold.families.qwen4_exp.cuda.prefix import Checkpoint

    A = prompt(1, 12)
    B = A[:8] + [999, 998, 997]
    ref_b = run_alone(multi, monkeypatch, B, 4)
    dec = decoder(multi, 3, monkeypatch, [])
    a = Stream(list(A), 50, None)
    dec.admit(a, defer=True)
    dec.finish(dec.round())                              # A filled and decoding
    entry = next(k for k in dec.kept if k.slot is a.st)
    entry.checkpoints = [Checkpoint(8, {"pos": 8, "who": a.st.name, "history": list(A[:8]), "mtp_len": 7}, None)]
    b = Stream(list(B), 4, None)
    dec.admit(b, defer=True)
    assert b.reuse == "checkpoint" and b.cached == 8 and b.reuse_copy and b.reuse_miss is None and b.st is not a.st
    assert COPIES == [(a.st.name, b.st.name, 8)]
    while not b.done:
        dec.finish(dec.round())
    assert b.out == ref_b and any(k.slot is a.st for k in dec.kept)


def test_turn_start_and_its_planned_checkpoint():
    from tensorfold.families.qwen4_exp.cuda import prefix as px

    assert px.turn_start([5, 1, 2, 5, 3, 4], 5) == 3 and px.turn_start([5, 1, 2], 5) is None
    assert px.turn_start([1, 2, 3], None) is None and px.turn_start([1, 2, 3], 9) is None
    # 30,000 tokens in 4,096-row chunks, 4 checkpoints, the last message starting at 29,500
    take, keep = px.plan_checkpoints(30_000, 4_096, 4, [], 0, turn=29_500)
    assert 29_500 in take and len(take) == 4 and 28_672 in take              # the tail's last chunk end stays
    take, _ = px.plan_checkpoints(30_000, 4_096, 4, [], 29_600, turn=29_500)    # already cached: not planned
    assert take == []
    assert px.plan_checkpoints(30_000, 4_096, 0, [], 0, turn=29_500) == ([], [])  # no checkpoints, no turn start
    take, _ = px.plan_checkpoints(30_000, 4_096, 4, [], 0, turn=28_672)          # on a chunk end: one position
    assert take.count(28_672) == 1 and len(take) == 4


def test_message_start_id_reads_the_tokenizer(tmp_path):
    import json

    from tensorfold.families.qwen4_exp.cuda.prefix import message_start_id

    assert message_start_id(tmp_path) is None
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 248044, "content": "<|endoftext|>"}, {"id": 248045, "content": "<|im_start|>"}]}))
    assert message_start_id(tmp_path) == 248045
    (tmp_path / "tokenizer.json").write_text("not json")
    assert message_start_id(tmp_path) is None


def test_the_admission_plans_the_turn_start(multi, monkeypatch):
    planned = []
    dec = decoder(multi, 2, monkeypatch, [])
    dec.checkpoints, dec.turn_marker = 4, 77
    dec.pbuf = type("P", (), {"rows": 4})()
    real = multi.plan_checkpoints
    monkeypatch.setattr(multi, "plan_checkpoints", lambda *a, **k: planned.append((a, k)) or real(*a, **k))
    seen = {}

    def steps(e, p, sampling, *, mtp=True, resume=None, vision=None, checkpoints=None):
        seen["checkpoints"] = checkpoints            # (the generator's body runs only with the rounds)
        return iter(())

    monkeypatch.setattr(multi, "prefill_steps", steps)
    p = [1, 2, 3, 4, 5, 6, 77, 8, 9, 10, 11]
    dec.admit(Stream(list(p), 2, None), defer=True)
    assert planned and planned[0][1]["turn"] == 6 and 6 in seen["checkpoints"]
