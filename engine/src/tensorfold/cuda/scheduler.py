"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Any, Callable

from .prefill_timing import ENV_OUT, TIMER, TimingBusy
from .streams import Stream


def _profiler_start() -> int:
    import torch
    return int(torch.cuda.cudart().cudaProfilerStart())


def _profiler_stop() -> int:
    import torch
    return int(torch.cuda.cudart().cudaProfilerStop())


def _drain() -> None:
    """Wait for the admission's GPU work when the recorder is not armed (profile-only): a plain device sync,
    outside any timed region."""

    import torch
    torch.cuda.synchronize()


def _nvtx_push(name: str) -> bool:
    """Push a host NVTX range; True on success, False when the torch call raised (never raises)."""

    try:
        import torch
        torch.cuda.nvtx.range_push(name)
    except Exception:                        # noqa: BLE001  a range is a marker; it never fails the request
        return False
    return True


def _nvtx_pop() -> bool:
    """Pop a host NVTX range; True on success, False when the torch call raised (never raises)."""

    try:
        import torch
        torch.cuda.nvtx.range_pop()
    except Exception:                        # noqa: BLE001
        return False
    return True


def defer(decoder, stream) -> bool:
    """Whether the admission only queues the prompt (a decoder with ``defers``, e.g. Flash Next's ``MultiDecoder``),
    so live streams keep decoding while it prefills. A timed, histogram or profiled admission prefills at once, inside
    its recorder / profiler range, exactly as before."""

    return bool(getattr(decoder, "defers", False)) and not (stream.timing or stream.histogram or stream.profile)


def run_admission(decoder, stream) -> None:
    """decoder.admit(stream) inside the Phase 1 lifecycle: arm the recorder / start the profiler before; after the
    admission, in every exit path and without masking admit()'s own exception: wait for the terminal event, stop the
    profiler, resolve. Any failure of the bookkeeping itself is a note on the stream, never an exception."""

    stream.admitted_at = stream.admitted_at or time.perf_counter()     # the single-stream path stamps it at entry
    timed = False
    rc = None
    if stream.timing or stream.histogram:
        if not TIMER.arm({"request": id(stream), "prompt_tokens": len(stream.prompt)}, histogram=stream.histogram):
            raise TimingBusy("prefill timing is busy with another request")
        timed = True
    if stream.profile:
        try:
            rc = [int(_profiler_start())]
        except Exception as exc:              # noqa: BLE001  a missing profiler must not fail the request
            rc = [-1]
            stream.notes.append(f"profiler start failed: {exc}")
        if rc[0] != 0:                        # no capture is running: the measurement would be meaningless
            if rc[0] != -1:
                stream.notes.append(f"profiler start returned {rc[0]}")
            if timed:
                TIMER.abort("profiler start failed"); timed = False
    pushed = False                            # an ordinary request runs no instrumentation code
    if timed or rc is not None:
        pushed = _nvtx_push("admission")
        if not pushed:
            stream.notes.append("nvtx: admission range push failed")
    try:
        if defer(decoder, stream):
            decoder.admit(stream, defer=True)          # the rounds prefill it a chunk at a time
        else:
            decoder.admit(stream)
    finally:
        # Bookkeeping only. Every step is guarded so that admit()'s own exception, when there is one, is the one
        # that propagates; a failing step becomes a note and disarms the recorder.
        if timed or rc is not None:
            try:
                if timed:
                    ev = TIMER.terminal()      # bounds a cut an exception left open, balances NVTX, records the terminal
                    if ev is not None:
                        ev.synchronize()
                else:
                    _drain()
            except Exception as exc:          # noqa: BLE001
                stream.notes.append(f"terminal wait failed: {exc}")
                if timed:
                    TIMER.abort("terminal wait failed"); timed = False
        if pushed and not _nvtx_pop():         # the admission range closes after the queued GPU work has completed
            stream.notes.append("nvtx: admission range pop failed")
        if rc is not None:
            try:
                rc.append(int(_profiler_stop()))
            except Exception as exc:          # noqa: BLE001
                rc.append(-1); stream.notes.append(f"profiler stop failed: {exc}")
            if rc[1] > 0:                     # the capture may be truncated; the device timing does not depend on it
                stream.notes.append(f"profiler stop returned {rc[1]}")
            stream.profiler_rc = rc
        if timed:
            try:
                stream.timing_summary = TIMER.resolve()
                TIMER.dump(stream.timing_summary, os.environ.get(ENV_OUT))
                _log_summary(stream.timing_summary)
            except Exception as exc:          # noqa: BLE001
                stream.notes.append(f"timing bookkeeping failed: {exc}")
                TIMER.abort("bookkeeping failed")


def _log_summary(s: dict | None) -> None:
    if s is None:
        return
    print(f"[octojet] prefill timing: {s['spans']} spans, device main {s['device_total_ms']['main'] / 1000:.1f} s,"
          f" mtp {s['device_total_ms']['mtp'] / 1000:.1f} s, draft {s['device_total_ms']['draft'] / 1000:.1f} s,"
          f" wall {s['wall_ms'] / 1000:.1f} s, overflow {s['overflow']}", file=sys.stderr, flush=True)


TWIN_WAIT_S = 120.0     # the longest a request waits for an identical prompt's fill before it prefills cold


def waits_for_twin(decoder, stream) -> bool:
    """Whether a queued request waits for a twin (a filling prompt with exactly its ids, F4): an untimed request on a
    decoder that queues prompts; a timed, histogram or profiled one is admitted as before."""

    check = getattr(decoder, "twin_pending", None)
    return check is not None and defer(decoder, stream) and bool(check(stream))


class Scheduler:
    def __init__(self, decoder: Any, *, max_streams: int = 4) -> None:
        self.decoder = decoder
        self.max_streams = max_streams
        self.waiting: queue.Queue = queue.Queue()
        self.held: list[tuple[Stream, queue.Queue, float]] = []   # requests waiting for a twin's fill, oldest first
        self.boxes: dict[int, queue.Queue] = {}
        if hasattr(decoder, "arrived"):              # a lone fill stops between chunks for a request it could admit
            decoder.arrived = self.admissible
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], timing: bool = False, profile: bool = False,
               histogram: bool = False, received_at: float = 0.0, *, vision: Any = None) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats."""

        if (timing or histogram) and TIMER.armed:        # early refusal; the definitive check is in run_admission
            raise TimingBusy("prefill timing is busy with another request")
        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, timing=timing, profile=profile,
                        histogram=histogram, received_at=received_at or time.perf_counter(), vision=vision)
        cancel = [False]
        stream.emit = lambda new: (box.put(("tokens", new)), cancel[0])[1]
        stream.queued_at = time.perf_counter()
        self.waiting.put((stream, box))
        while True:
            kind, value = box.get()
            if kind == "tokens":
                if not cancel[0] and emit(value):
                    cancel[0] = True                 # the client left: the stream ends after its next round
            elif kind == "error":
                raise value
            else:
                return value

    def admissible(self) -> bool:
        """Whether a waiting request could be admitted now: one is queued and a stream is free (held requests wait
        for a twin's fill and are checked through ``short_fill``)."""

        return not self.waiting.empty() and self.decoder.live() < self.max_streams

    def _next(self, first=None):
        """The next request to admit: a held one whose twin's fill has ended (or that waited TWIN_WAIT_S), else the
        next waiting one; a request with a twin still filling is held and later ones pass it. None: nothing to admit."""

        now = time.perf_counter()
        for i, (stream, box, since) in enumerate(self.held if first is None else []):
            if now - since > TWIN_WAIT_S or not waits_for_twin(self.decoder, stream):
                del self.held[i]
                return stream, box
        while True:
            if first is not None:
                item, first = first, None
            else:
                try:
                    item = self.waiting.get_nowait()
                except queue.Empty:
                    return None
            if waits_for_twin(self.decoder, item[0]):
                self.held.append((item[0], item[1], now))
                continue
            return item

    def _admit(self, first=None) -> list[Stream]:
        done = []
        while self.decoder.live() < self.max_streams:
            item = self._next(first)
            first = None
            if item is None:
                break
            stream, box = item
            self.boxes[id(stream)] = box
            try:
                run_admission(self.decoder, stream)
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if stream.done:
                done.append(stream)
        return done

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            box.put((kind, value))

    def _loop(self) -> None:
        while True:
            idle = not self.decoder.live() and not self.held
            done = self._admit(self.waiting.get() if idle else None)                  # idle: wait for a request
            if hasattr(self.decoder, "twin_pending"):  # held requests' deadlines are checked between chunks
                self.decoder.short_fill = bool(self.held)
            try:
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                for s in self.decoder.drop():        # a stream already done (or failed) in the round keeps its reply
                    if s.done and s.error is None and s.out:
                        self._reply(s, "done", s.stats())
                    else:
                        self._reply(s, "error", s.error if s.error is not None else exc)
            self.decoder.finish(done)
            for s in done:
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))
