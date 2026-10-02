"""Scheduler arming: one timed admission at a time, the profiler range around admit(), timestamps and the summary
in the stream's stats; the lifecycle is exception-safe. The decoder is a fake; the recorder uses fake CUDA events."""

import threading
import time

import pytest

from tensorfold.cuda import prefill_timing as pt
from tensorfold.cuda import scheduler as sch_mod
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream


class FakeEvent:
    def __init__(self, enable_timing=True): self.t = None
    def record(self, stream=None): self.t = time.perf_counter()
    def synchronize(self): pass
    def elapsed_time(self, other): return (other.t - self.t) * 1000.0


class FakeDecoder:
    def __init__(self, fail=False): self.streams = []; self.fail = fail
    def live(self): return len([s for s in self.streams if not s.done])
    def admit(self, s):
        time.sleep(0.005)
        if pt.TIMER.armed:
            pt.TIMER.phase = "main"; i = pt.TIMER.begin("commit"); pt.TIMER.end(i)
        if self.fail:
            raise ValueError("boom")
        self.streams.append(s); s.take([1])
    def round(self):
        for s in self.streams:
            if not s.done: s.take([2])
        return [s for s in self.streams if s.done]
    def finish(self, done): pass
    def drop(self): return []


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(pt.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(pt, "_nvtx_push", lambda n: True); monkeypatch.setattr(pt, "_nvtx_pop", lambda: True)
    monkeypatch.setattr(sch_mod, "_nvtx_push", lambda n: True); monkeypatch.setattr(sch_mod, "_nvtx_pop", lambda: True)
    monkeypatch.setenv(pt.ENV, "1")
    pt.TIMER.__init__()
    pt.TIMER.configure(layers=2, attention_layers=1, prefill_rows=8, capacity=32, experts=4, top_k=2, slots=3, device="cpu")
    yield pt.TIMER
    pt.TIMER.__init__()


def test_timed_submit_returns_summary_and_timestamps(recorder, monkeypatch):
    calls = []
    monkeypatch.setattr(sch_mod, "_profiler_start", lambda: calls.append("start") or 0)
    monkeypatch.setattr(sch_mod, "_profiler_stop", lambda: calls.append("stop") or 0)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    t_recv = time.perf_counter()
    stats = sch.submit([5, 6, 7], 2, None, True, lambda new: False, timing=True, profile=True, histogram=False,
                       received_at=t_recv)
    assert stats["timing"]["spans"] == 1 and stats["timing"]["device_ms"]["main"]["commit"] >= 0
    assert stats["received_at"] == t_recv and stats["queued_at"] >= t_recv
    assert stats["admitted_at"] >= stats["queued_at"] and stats["first_token_at"] >= stats["admitted_at"]
    assert stats["ttft_s"] == stats["first_token_at"] - t_recv                  # unrounded
    assert stats["profiler_rc"] == [0, 0] and calls == ["start", "stop"]
    assert recorder.armed is False
    assert stats["timing"]["nvtx_failures"] == {"push": 0, "pop": 0} and stats["timing"]["notes"] == []
    assert "notes" not in stats                                                  # a pristine admission


def test_profile_only_waits_for_the_terminal_event(recorder, monkeypatch):
    waited = []
    real_terminal = recorder.terminal
    monkeypatch.setattr(sch_mod, "_profiler_start", lambda: 0); monkeypatch.setattr(sch_mod, "_profiler_stop", lambda: 0)
    monkeypatch.setattr(sch_mod, "_drain", lambda: waited.append(True))
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, profile=True, received_at=time.perf_counter())
    assert stats["profiler_rc"] == [0, 0] and waited == [True] and stats["timing"] is None


def test_untimed_submit_has_timestamps_but_no_summary(recorder):
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, received_at=time.perf_counter())
    assert stats["timing"] is None and stats["profiler_rc"] is None and stats["first_token_at"] > 0
    assert stats["ttft_s"] > 0


def test_busy_recorder_raises_typed(recorder):
    recorder.arm({"request": "other"})
    sch = Scheduler(FakeDecoder(), max_streams=2)
    with pytest.raises(pt.TimingBusy):
        sch.submit([5], 2, None, True, lambda new: False, timing=True, received_at=time.perf_counter())
    recorder.terminal(); recorder.resolve()
    assert sch.submit([5], 2, None, True, lambda new: False, timing=True, received_at=time.perf_counter())["timing"]


def test_admit_failure_disarms_and_reports(recorder, monkeypatch):
    monkeypatch.setattr(sch_mod, "_profiler_start", lambda: 0); monkeypatch.setattr(sch_mod, "_profiler_stop", lambda: 0)
    sch = Scheduler(FakeDecoder(fail=True), max_streams=2)
    with pytest.raises(ValueError, match="boom"):
        sch.submit([5], 2, None, True, lambda new: False, timing=True, profile=True, received_at=time.perf_counter())
    assert recorder.armed is False


def test_profiler_start_failure_disarms(recorder, monkeypatch):
    def bad(): raise RuntimeError("no cudart")
    monkeypatch.setattr(sch_mod, "_profiler_start", bad); monkeypatch.setattr(sch_mod, "_profiler_stop", lambda: 0)
    monkeypatch.setattr(sch_mod, "_drain", lambda: None)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, timing=True, profile=True, received_at=time.perf_counter())
    assert recorder.armed is False and stats["profiler_rc"] == [-1, 0] and stats["timing"] is None
    assert any("profiler" in n for n in stats.get("notes", []))


def test_terminal_wait_failure_still_stops_profiler_and_disarms(recorder, monkeypatch):
    calls = []
    monkeypatch.setattr(sch_mod, "_profiler_start", lambda: calls.append("start") or 0)
    monkeypatch.setattr(sch_mod, "_profiler_stop", lambda: calls.append("stop") or 0)
    class Bad:
        def record(self, stream=None): pass
        def synchronize(self): raise RuntimeError("device lost")
    recorder.terminal_event = Bad()
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, timing=True, profile=True, received_at=time.perf_counter())
    assert calls == ["start", "stop"] and recorder.armed is False and stats["timing"] is None
    assert any("terminal" in n for n in stats["notes"]) and stats["profiler_rc"] == [0, 0]


def test_profiler_start_nonzero_rc_disarms(recorder, monkeypatch):
    monkeypatch.setattr(sch_mod, "_profiler_start", lambda: 999); monkeypatch.setattr(sch_mod, "_profiler_stop", lambda: 0)
    monkeypatch.setattr(sch_mod, "_drain", lambda: None)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, timing=True, profile=True, received_at=time.perf_counter())
    assert stats["profiler_rc"] == [999, 0] and recorder.armed is False and stats["timing"] is None
    assert any("returned 999" in n for n in stats["notes"])


def test_profiler_stop_nonzero_rc_is_noted_and_timing_kept(recorder, monkeypatch):
    monkeypatch.setattr(sch_mod, "_profiler_start", lambda: 0); monkeypatch.setattr(sch_mod, "_profiler_stop", lambda: 999)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, timing=True, profile=True, received_at=time.perf_counter())
    assert stats["profiler_rc"] == [0, 999] and stats["timing"] is not None
    assert any("stop returned 999" in n for n in stats["notes"]) and recorder.armed is False


def test_summary_logging_failure_is_a_note_not_an_error(recorder, monkeypatch):
    def bad(summary): raise OSError("stderr closed")
    monkeypatch.setattr(sch_mod, "_log_summary", bad)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, timing=True, received_at=time.perf_counter())
    assert stats["timing"] is not None and any("bookkeeping" in n for n in stats["notes"]) and recorder.armed is False


def test_stream_take_sets_first_token_once():
    s = Stream([1, 2], 3, None)
    assert s.first_token_at == 0.0
    s.take([9]); t = s.first_token_at
    assert t > 0
    s.take([10])
    assert s.first_token_at == t


# ---- fix round 1: admission NVTX range, lifecycle failure paths, single-stream path ---------------------------------

def test_untimed_request_runs_no_admission_range(recorder, monkeypatch):
    calls = []
    monkeypatch.setattr(sch_mod, "_nvtx_push", lambda n: calls.append(("push", n)) or True)
    monkeypatch.setattr(sch_mod, "_nvtx_pop", lambda: calls.append(("pop",)) or True)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    sch.submit([5], 2, None, True, lambda new: False, received_at=time.perf_counter())
    assert calls == []
    sch.submit([5], 2, None, True, lambda new: False, timing=True, received_at=time.perf_counter())
    assert calls == [("push", "admission"), ("pop",)]


def test_admission_push_failure_is_noted_and_not_popped(recorder, monkeypatch):
    pops = []
    monkeypatch.setattr(sch_mod, "_nvtx_push", lambda n: False)
    monkeypatch.setattr(sch_mod, "_nvtx_pop", lambda: pops.append(1) or True)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, timing=True, received_at=time.perf_counter())
    assert pops == [] and "nvtx: admission range push failed" in stats["notes"] and stats["timing"] is not None


def test_admission_pop_failure_is_noted(recorder, monkeypatch):
    monkeypatch.setattr(sch_mod, "_nvtx_pop", lambda: False)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, timing=True, received_at=time.perf_counter())
    assert "nvtx: admission range pop failed" in stats["notes"] and recorder.armed is False


def test_queued_stream_busy_at_admission_fails_alone(recorder):
    class Gate(FakeDecoder):
        def __init__(self):
            super().__init__(); self.entered = threading.Event(); self.release = threading.Event()
        def admit(self, s):
            if s.prompt == [1]:
                self.entered.set(); assert self.release.wait(5)
            super().admit(s)

    dec = Gate()
    sch = Scheduler(dec, max_streams=3)
    results = {}

    def run(name, prompt, **kw):
        try:
            results[name] = sch.submit(prompt, 3, None, True, lambda new: False, received_at=time.perf_counter(), **kw)
        except Exception as exc:                 # noqa: BLE001
            results[name] = exc

    ta = threading.Thread(target=run, args=("a", [1])); ta.start()
    assert dec.entered.wait(5)                   # the worker is inside A's admission
    tb = threading.Thread(target=run, args=("b", [2]), kwargs={"timing": True}); tb.start()
    deadline = time.time() + 5
    while sch.waiting.qsize() < 1 and time.time() < deadline:
        time.sleep(0.001)
    assert sch.waiting.qsize() == 1              # B passed submit's early check and is queued
    recorder.arm({"request": "other"})           # another admission takes the recorder before B is admitted
    dec.release.set()
    ta.join(5); tb.join(5)
    assert isinstance(results["b"], pt.TimingBusy)
    assert isinstance(results["a"], dict) and results["a"]["first_token_at"] > 0      # the other stream went on
    recorder.terminal(); recorder.resolve()
    assert sch.submit([3], 2, None, True, lambda new: False, timing=True,
                      received_at=time.perf_counter())["timing"] is not None           # the worker loop is alive


def test_admit_and_terminal_wait_both_fail(recorder, monkeypatch):
    calls = []
    monkeypatch.setattr(sch_mod, "_profiler_start", lambda: calls.append("start") or 0)
    monkeypatch.setattr(sch_mod, "_profiler_stop", lambda: calls.append("stop") or 0)
    class Bad:
        def record(self, stream=None): pass
        def synchronize(self): raise RuntimeError("device lost")
    recorder.terminal_event = Bad()
    sch = Scheduler(FakeDecoder(fail=True), max_streams=2)
    with pytest.raises(ValueError, match="boom"):
        sch.submit([5], 2, None, True, lambda new: False, timing=True, profile=True, received_at=time.perf_counter())
    assert recorder.armed is False and calls == ["start", "stop"]


def test_drain_failure_on_profile_only_is_a_note(recorder, monkeypatch):
    def bad(): raise RuntimeError("sync failed")
    monkeypatch.setattr(sch_mod, "_profiler_start", lambda: 0); monkeypatch.setattr(sch_mod, "_profiler_stop", lambda: 0)
    monkeypatch.setattr(sch_mod, "_drain", bad)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, profile=True, received_at=time.perf_counter())
    assert stats["profiler_rc"] == [0, 0] and stats["timing"] is None
    assert any("terminal wait failed" in n for n in stats["notes"]) and stats["first_token_at"] > 0


def test_profiler_stop_raising_is_a_note_and_timing_kept(recorder, monkeypatch):
    def bad(): raise RuntimeError("no cudart")
    monkeypatch.setattr(sch_mod, "_profiler_start", lambda: 0); monkeypatch.setattr(sch_mod, "_profiler_stop", bad)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, timing=True, profile=True, received_at=time.perf_counter())
    assert stats["profiler_rc"] == [0, -1] and stats["timing"] is not None and recorder.armed is False
    assert any("profiler stop failed" in n for n in stats["notes"])


@pytest.mark.parametrize("step", ["resolve", "dump"])
def test_resolve_or_dump_failure_is_a_note(recorder, monkeypatch, step):
    def bad(*a, **k): raise OSError(f"{step} failed")
    monkeypatch.setattr(recorder, step, bad)
    sch = Scheduler(FakeDecoder(), max_streams=2)
    stats = sch.submit([5], 2, None, True, lambda new: False, timing=True, received_at=time.perf_counter())
    assert any("timing bookkeeping failed" in n for n in stats["notes"]) and recorder.armed is False
    assert stats["first_token_at"] > 0 and (stats["timing"] is None) == (step == "resolve")


def _single_stream_engine(tp=1):
    from tensorfold.families.qwen4_exp.cuda import engine as E
    eng = E.FlashNextEngine.__new__(E.FlashNextEngine)
    eng.max_len, eng.depth, eng.scheduler, eng.tp, eng.cache, eng.served = 1000, 0, None, tp, [], 0
    return E, eng


def test_single_stream_admitted_at_is_generate_entry(recorder, monkeypatch):
    E, eng = _single_stream_engine()
    seen = {}
    def fake_decode(prompt, max_tokens, sampling, on_tokens, hit, carrier=None):
        seen["at_decode"] = (carrier.admitted_at, carrier.queued_at)
        first, carried = E._admission(lambda: 9, carrier)       # the lifecycle must not restamp admitted_at
        seen["carried"] = carried
        return carried
    monkeypatch.setattr(eng, "_decode", fake_decode)
    before = time.perf_counter()
    eng.generate([1, 2, 3], 4, None, lambda new: False, received_at=before)
    after = time.perf_counter()
    admitted, queued = seen["at_decode"]
    assert before <= admitted <= after and queued == admitted
    assert seen["carried"]["admitted_at"] == admitted and seen["carried"]["queued_at"] == admitted
    assert seen["carried"]["first_token_at"] >= admitted


@pytest.mark.parametrize("state", ["disabled", "armed"])
def test_tp_refuses_before_share(recorder, monkeypatch, state):
    E, eng = _single_stream_engine(tp=2)
    shared = []
    monkeypatch.setattr(eng, "_share", lambda *a: shared.append(a) or a)
    if state == "disabled":
        recorder.enabled = False
    else:
        recorder.arm({"request": "other"})
    with pytest.raises(pt.TimingBusy):
        eng.generate([1, 2, 3], 4, None, lambda new: False, timing=True)
    assert shared == [] and eng.served == 0
