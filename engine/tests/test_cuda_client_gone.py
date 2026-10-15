"""A request whose client has disconnected stops costing GPU time: a queued one is never admitted, and a filling one
stops at its next prompt chunk and frees its slot, while the other streams go on with their solo runs (CPU fakes)."""

import importlib
import queue

import pytest

from tensorfold.cuda.streams import Stream, gone
from tensorfold.server.cancellation import RequestCancelled
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)
from tests.test_cuda_prompts_inside_rounds import decoder, prompt, run_alone


@pytest.fixture
def multi(allocations):  # noqa: F811
    return importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")


def scheduler(dec, slots):
    from tensorfold.cuda import scheduler as sched_mod

    sched = sched_mod.Scheduler.__new__(sched_mod.Scheduler)                     # no worker thread: driven here
    sched.decoder, sched.max_streams, sched.waiting, sched.boxes, sched.held = dec, slots, queue.Queue(), {}, []
    return sched


def test_gone_is_false_without_a_check_and_when_the_check_fails():
    assert gone(Stream([1], 1)) is False
    s = Stream([1], 1)
    s.client_gone = lambda: True
    assert gone(s) is True

    def broken():
        raise OSError("socket")

    s.client_gone = broken
    assert gone(s) is False


def test_a_queued_request_whose_client_left_is_never_admitted(multi, monkeypatch):
    log = []
    dec = decoder(multi, 2, monkeypatch, log)
    sched = scheduler(dec, 2)
    left, stays = Stream(prompt(6, 10), 8, None), Stream(prompt(7, 13), 4, None)
    left.client_gone = lambda: True
    boxes = [queue.Queue(), queue.Queue()]
    sched.waiting.put((left, boxes[0]))
    sched.waiting.put((stays, boxes[1]))
    sched._admit()
    assert dec.filling == [stays] and not log                                     # only the live client's request
    kind, err = boxes[0].get_nowait()
    assert kind == "error" and isinstance(err, RequestCancelled)


def test_a_filling_request_stops_at_its_next_chunk_when_its_client_leaves(multi, monkeypatch, capsys):
    A, B = prompt(1, 3), prompt(2, 22)                  # B: six chunks of four rows beside A's decoding
    ref_a = run_alone(multi, monkeypatch, A, 30)
    log = []
    dec = decoder(multi, 2, monkeypatch, log)
    a = Stream(list(A), 30, None)
    dec.admit(a, defer=True)
    dec.finish(dec.round())
    b = Stream(list(B), 6, None)
    leave = [False]
    b.client_gone = lambda: leave[0]
    dec.admit(b, defer=True)
    dec.finish(dec.round())                            # one chunk of B beside A
    chunks_before = sum(1 for x in log if x[0] == "chunk")
    leave[0] = True                                    # the client disconnects mid-fill
    ended = dec.round()
    assert b in ended and b.done and isinstance(b.error, RequestCancelled)
    assert b not in dec.filling and sum(1 for x in log if x[0] == "chunk") == chunks_before   # no further chunk
    dec.finish(ended)
    assert "the client left; prompt fill stopped at" in capsys.readouterr().err
    while dec.live():
        dec.finish(dec.round())
    assert a.out == ref_a                                                        # A's solo run, untouched
    assert len(dec.free) + len({id(k.slot) for k in dec.kept}) == 2              # B's slot is free again
