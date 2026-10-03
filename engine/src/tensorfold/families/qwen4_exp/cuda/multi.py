"""Flash Next's concurrent rounds on one GPU: every stream keeps exactly its own accepted prefix.

Prompts inside rounds (a port of upstream TensorFold d23087c's lanes, its one-prompt-a-pass subset): on the scheduler
path an admission only picks the slot and queues the prompt; each round first runs the next chunk of a filling prompt
(see below for which), so live streams keep decoding a round per chunk instead of waiting out the whole prefill. With
nothing decoding the prompt fills chunk after chunk until it joins (or a request can be admitted). The chunks are ``prefill_steps``'s, the same kernels on the same rows
as a solo ``prefill``; a stream's rows never read another stream's state, so every stream emits its solo tokens.

Admission between chunks (F7; the idea of TensorFold 0.6.1's "short prompts admitted while a long one fills", Octojet's
own implementation): with nothing decoding, the chunks of a lone fill stop as soon as the scheduler holds a request it
could admit (``arrived``), so that request is admitted (an exact hit decodes at once) instead of waiting out the whole
prompt. The next chunk goes to the filling prompt with the fewest rows left, unless one has been passed over
``FILL_GUARD`` chunks in a row (it goes first, so a long prompt is never starved). Order moves only when a prompt's
rows run; every chunk is still its prompt's own ``prefill_steps`` chunk, so no stream's tokens change."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from tensorfold.cuda.prefill_timing import TIMER
from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import MARGIN, choose_rows

from .decode import PREFILL_ROWS, WARM_TAIL, Engine, draft, prefill, prefill_steps
from .forward import commit, compute, stage
from .mtp import mtp_compute, mtp_stage
from .prefix import Kept, Match, exact_hit, match, plan_checkpoints, snapshot_bytes, turn_start
from .state import Buffers, State
from ..cuda import CONFIDENCE, DEPTH

FILL_GUARD = 8           # a filling prompt passed over this many chunks in a row takes the next one (no starvation)


def _handoff(m: Match | None) -> dict | None:
    """The match's resume state, handed to the prefill alone: once restored, nothing holds a thinned-out checkpoint's
    snapshot any more (F2d 5.2: at most N checkpoint snapshots a slot at any time)."""

    if m is None:
        return None
    resume, m.resume = m.resume, None
    return resume


LIVE_PREFILL_ROWS = 1024    # F8: a prompt's chunk rows while other streams decode (their pause is one such chunk)


def live_prefill_rows(rows: int) -> int | None:
    """Chunk rows for a prompt that fills while other streams decode: OCTOJET_LIVE_PREFILL_ROWS (0 turns it off), else
    1,024; used only when it is a multiple of 256 that divides the full chunk size and is smaller than it, so every
    full-size chunk end (where checkpoints are taken) stays a chunk end."""

    value = os.environ.get("OCTOJET_LIVE_PREFILL_ROWS", "").strip()
    small = int(value) if value else LIVE_PREFILL_ROWS
    if small <= 0 or small >= rows or small % 256 or rows % small:
        return None
    return small


FILL_STREAM = os.environ.get("OCTOJET_FILL_STREAM", "1") != "0"
FILL_AHEAD = 2              # prompt chunks queued on the fill stream at most


ROUND_LOG = os.environ.get("OCTOJET_ROUND_LOG")   # a JSONL path: one line a round (fill / decode seconds, prompt positions)


def _round_log(row: dict) -> None:
    try:
        with open(ROUND_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


class _Fill:
    """A queued prompt's prefill: its engine, the ``prefill_steps`` generator, its prefix match and image features,
    the prompt rows committed so far (``at``) and the chunks run for other prompts since its own last one."""

    __slots__ = ("e", "mtp", "steps", "m", "image", "encoded", "at", "skipped", "order")

    def __init__(self, e, mtp, steps, m, image, encoded, at: int = 0, order: int = 0) -> None:
        self.e, self.mtp, self.steps, self.m, self.image, self.encoded = e, mtp, steps, m, image, encoded
        self.at, self.skipped, self.order = at, 0, order


def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, pbuf.rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    return e


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context."""

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16", prefill_rows: int = PREFILL_ROWS,
                 vision=None, prefix_checkpoints: int = 0, turn_marker: int | None = None) -> None:
        if w.comm is not None:
            raise ValueError("concurrent Flash Next runs on one GPU for now")
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        self.vision = vision                         # the image tower (``QwenCudaVision``) with --vision, else None
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = slots * (depth + 1)
        self.buf = Buffers(w, rows, capacity)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True)
        self.live_prefill_rows = live_prefill_rows(self.pbuf.rows)   # F8: shorter chunks while streams decode
        # F8: beside decoding streams a prompt's chunks run on their own CUDA stream, so decode steps never wait behind
        # a chunk and the next chunk is queued while one runs (at most FILL_AHEAD in flight)
        self.fill_stream = torch.cuda.Stream() if FILL_STREAM and torch.cuda.is_available() else None
        self.fill_events: list = []
        self.free = [State(w, capacity, depth + 1, kv_dtype) for _ in range(slots)]   # sized by the startup admission
        if prefix_checkpoints < 0:
            raise ValueError(f"--prefix-checkpoints is a count of 0 or more, not {prefix_checkpoints}")
        self.checkpoints = int(prefix_checkpoints)  # F2d stage B: chunk-boundary snapshots a kept entry holds, at most
        self.turn_marker = turn_marker               # F7: the message-start token one of those checkpoints follows
        # a stream's slot, plus the checkpoints its kept entry may hold (lazily allocated; in the startup estimate)
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0])) + \
            (self.checkpoints * (snapshot_bytes(w.cfg) + w.cfg.streams * w.cfg.hidden * 2) if self.checkpoints else 0)
        self.streams: dict[int, Stream] = {}
        self.filling: list[Stream] = []              # admitted, prompts still prefilling (oldest first)
        self.unreplied: list[Stream] = []            # a failed round's earlier fill endings, for drop() to return
        self.short_fill = False                      # one chunk a round even with nothing decoding (set by Scheduler)
        self.fills: dict[int, _Fill] = {}            # stream id -> its prefill
        self.next_id = 0
        self.draft_host = w.draft_ids.cpu().numpy() if w.draft_ids is not None else None
        self.kept: list[Kept] = []
        self.next_serial = 0
        self.keep = keep

    defers = True                                    # ``run_admission`` may queue prompts (``admit(defer=True)``)

    def _busy(self) -> set[int]:
        return {id(s.st) for s in [*self.streams.values(), *self.filling]}

    def _drop_kept(self, st: State) -> None:
        self.kept = [k for k in self.kept if k.slot is not st]

    def _slot_for(self, prompt: list[int], reuse: bool):
        """(slot, match, busy kind): the idle kept entry the prompt reuses most tokens from (its slot), else a free
        slot, else the oldest idle kept slot; ``reuse`` False (the serial reference) never matches."""

        busy = self._busy()
        # a filling prompt is a kept entry to be: a prompt it would serve reports the busy miss, as a decoding one does
        pending = [Kept(list(s.prompt), None, None, None, slot=s.st) for s in self.filling
                   if s.draft and not self.fills[s.sid].image] if reuse else []
        m, miss = match(prompt, self.kept + pending,
                        lambda k: id(k.slot) not in busy and k.snapshot is not None) if reuse else (None, None)
        if miss in ("extend", "checkpoint"):   # a decoding stream's entry reuses more: fork from it beside it
            fork = self._fork(prompt, busy, m.cached if m is not None else 0)
            if fork is not None:
                return fork
        if m is not None:
            if m.kind == "checkpoint":     # a variant: resume in another slot when one is spare, so a resend of the
                other = self._spare_except(m.entry.slot)   # source stays an exact hit (the source is likely asked again)
                if other is not None:
                    m.copy_from = m.entry.slot
                    self.kept = [k for k in self.kept if k is not m.entry] + [m.entry]
                    return other, m, miss
            if m.kind != "exact":          # the prefill overwrites the rows above cached: the entry goes
                self._drop_kept(m.entry.slot)
            else:                          # a hit refreshes recency: eviction takes the least recently used entry
                self.kept = [k for k in self.kept if k is not m.entry] + [m.entry]
            return m.entry.slot, m, miss
        if not self.free:
            idle = next((k.slot for k in self.kept if id(k.slot) not in busy), None)
            if idle is None:
                raise RuntimeError("no free stream slot")
            self._drop_kept(idle)
            self.free.append(idle)
        return self.free.pop(), None, miss

    def _fork(self, prompt: list[int], busy: set[int], idle_cached: int):
        """(slot, match, None) for a prompt that extends, or shares a checkpoint with, a kept entry whose slot a live
        stream is decoding in, when that reuses more than ``idle_cached`` tokens and a spare slot exists: the source's
        rows below the resume point are copied into the spare slot (``copy_from``) and the prompt resumes there, as a
        variant resumed beside an idle source does (F7; the idea of TensorFold 0.6.1's forks that resume from their
        shared prefix, Octojet's own implementation). The source's rows below its prompt end never change while it
        decodes, and its snapshots are read-only. None: no such entry or no spare slot (the prompt fills cold)."""

        live = [k for k in self.kept if id(k.slot) in busy and k.snapshot is not None]
        m, _ = match(prompt, live, lambda k: True)
        if m is None or m.kind not in ("extend", "checkpoint") or m.cached <= idle_cached:
            return None
        spare = self._spare_except(m.entry.slot)
        if spare is None:
            return None
        m.copy_from, m.forked = m.entry.slot, True
        self.kept = [k for k in self.kept if k is not m.entry] + [m.entry]      # recency: the source was used
        return spare, m, None

    def _busy_twin(self, prompt: list[int]) -> Kept | None:
        """The newest kept entry holding exactly ``prompt`` whose slot a live stream is using (its twin decodes), when
        no idle entry holds it (that is an ordinary exact hit); None otherwise."""

        busy = self._busy()
        same = [k for k in self.kept if k.ids == prompt and k.snapshot is not None and k.logits is not None]
        if not same or any(id(k.slot) not in busy for k in same):
            return None
        return max(same, key=lambda k: k.serial)

    def _spare(self) -> State | None:
        """A slot no live stream uses: a free one, else the least recently used idle kept one (its entry goes)."""

        if self.free:
            return self.free.pop()
        busy = self._busy()
        idle = next((k.slot for k in self.kept if id(k.slot) not in busy), None)
        if idle is not None:
            self._drop_kept(idle)
        return idle

    def _spare_except(self, keep: State) -> State | None:
        """A slot for a variant resumed beside its source: a free one, else the least recently used idle kept slot
        other than ``keep`` (its entry goes); None when only the source's slot is available."""

        if self.free:
            return self.free.pop()
        busy = self._busy()
        idle = next((k.slot for k in self.kept if id(k.slot) not in busy and k.slot is not keep), None)
        if idle is not None:
            self._drop_kept(idle)
        return idle

    def twin_pending(self, s: Stream) -> bool:
        """Whether a queued request should wait for a twin: a filling prompt with exactly its ids, which, once it
        joins, keeps the entry this request then exact-hits (copied while the twin decodes)."""

        if not s.draft or getattr(s, "vision", None) is not None:
            return False
        return any(x.draft and not self.fills[x.sid].image and x.prompt == s.prompt for x in self.filling)

    def _remember(self, ids: list[int], st: State, snap: dict, tail, logits=None, checkpoints=None) -> Kept:
        """Keep a prompt end in its slot (one entry per slot). An older entry with the same ids is replaced only when
        its slot is idle; a slot no entry holds any more goes back to the free list (unless a live stream uses it)."""

        busy = self._busy()

        def stale(k: Kept) -> bool:
            return k.slot is st or (k.ids == ids and id(k.slot) not in busy)

        gone = [k.slot for k in self.kept if stale(k) and k.slot is not st]
        self.kept = [k for k in self.kept if not stale(k)]
        entry = Kept(list(ids), snap, tail, logits, list(checkpoints or []), serial=self.next_serial, slot=st)
        self.next_serial += 1
        self.kept.append(entry)
        while len(self.kept) > self.keep:
            gone.append(self.kept.pop(0).slot)
        for old in gone:           # a displaced idle slot no kept entry holds goes back to the free list
            if old is not st and id(old) not in busy and all(k.slot is not old for k in self.kept) and \
                    all(f is not old for f in self.free):
                self.free.append(old)
        return entry

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    @torch.no_grad()
    def warm(self) -> None:
        """A synthetic greedy request through prefill, its drafts and one round, then forgotten, so no request compiles or loads a kernel."""

        s = Stream([0] * min(self.pbuf.rows + WARM_TAIL, self.capacity - self.depth - 2), 2)
        saved, self.checkpoints = self.checkpoints, 0    # warm-up takes no checkpoint (F2d 5.3) and keeps nothing
        try:
            self.admit(s)
        finally:
            self.checkpoints = saved
        if not s.done:
            self.round()
        self.streams.pop(s.sid, None)
        self._drop_kept(s.st)
        if all(f is not s.st for f in self.free):
            self.free.append(s.st)

    @torch.no_grad()
    def admit(self, s: Stream, *, defer: bool = False) -> None:
        """Prefill a request in a free slot, draft its first chain and emit its first token; ``defer`` (the scheduler
        path) only queues the prompt in its slot and the rounds prefill it a chunk at a time (an exact hit has no
        prompt to fill and is admitted at once)."""

        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context")
        s.count = max(1, min(s.count, room))
        t0 = time.perf_counter()
        encoded = None
        if getattr(s, "vision", None) is not None:       # an image prompt: its features and t/h/w positions
            from tensorfold.server.errors import RequestError

            if self.vision is None:
                raise RequestError("image inputs require starting this server with --vision")
            try:
                encoded = self.vision.encode(s.vision, s.prompt)
            except ValueError as exc:                    # a refused image is the request's error (400), not a crash
                raise RequestError(str(exc)) from exc
        image = encoded is not None
        # image placeholders look alike whatever the image: an image prompt never matches a kept entry (prefix.match
        # compares token ids only) and is never kept, so no prompt reuses another's image state
        reuse = s.draft and not image
        twin = self._busy_twin(list(s.prompt)) if reuse else None
        st = self._spare() if twin is not None else None
        if st is not None:                  # a decoding twin's kept prompt end, copied into a spare slot (F4)
            m, busy_kind = Match(twin, "exact", len(twin.ids), None), None
        else:
            twin = None
            st, m, busy_kind = self._slot_for(list(s.prompt), reuse)
        s.reuse_miss = "busy" if busy_kind is not None else None
        exact = m is not None and m.kind == "exact"
        # F2d stage B: the chunk ends this prefill keeps snapshots at, and the source entry's checkpoints that survive
        # (the source was dropped: its survivors move to the new entry, the rest go now, so at most N exist)
        take, inherit = [], []
        n = getattr(self, "checkpoints", 0)
        if reuse and n > 0 and not exact:
            cached = m.cached if m is not None else 0
            take, inherit = plan_checkpoints(len(s.prompt), self.pbuf.rows, n,
                                             m.entry.checkpoints if m is not None else [], cached,
                                             turn=turn_start(s.prompt, getattr(self, "turn_marker", None)))
        beside = m is not None and m.copy_from is not None
        if m is not None and not exact and not beside:   # the source was dropped: its checkpoints go (or moved over)
            m.entry.checkpoints = []
        registered = queued = False                      # (beside its source: the kept ones are shared, read-only)
        try:
            e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity)
            mtp = s.draft and self.depth > 0 and self.mbuf is not None
            if beside:                      # the source's rows below the checkpoint, then an ordinary resume here
                st.copy_prefix(m.copy_from, m.cached, m.resume["state"]["mtp_len"], self.w.cfg.index_ratio)
                s.reuse_copy = m.forked     # (forked: copied from a decoding stream's slot)
            if twin is not None:            # its rows, then an exact hit on the copy: the same bits as on the twin's
                st.copy_prefix(twin.slot, len(twin.ids), twin.snapshot["mtp_len"], self.w.cfg.index_ratio)
                # the twin's checkpoints are shared, not cloned: read-only snapshots, valid over the copied rows
                entry = self._remember(list(twin.ids), st, twin.snapshot, twin.tail, twin.logits,
                                       list(twin.checkpoints))
                first = exact_hit(e, entry, s.sampling)
                s.reuse_copy = True
            elif exact:
                first = exact_hit(e, m.entry, s.sampling)
            elif defer:                                  # the rounds fill it; its slot is busy from now on
                e.inherited = inherit
                small = getattr(self, "live_prefill_rows", None)
                live = (lambda: small if any(not x.done for x in self.streams.values()) else None) if small else None
                steps = prefill_steps(e, s.prompt, s.sampling, mtp=mtp, resume=_handoff(m),
                                      **({"vision": encoded} if image else {}), **({"checkpoints": take} if take else {}),
                                      **({"live_rows": live} if live else {}))
                s.cached = m.cached if m is not None else 0
                s.reuse = m.kind if m is not None else None
                s.sid, s.st = self.next_id, st
                self.next_id += 1
                self.fills[s.sid] = _Fill(e, mtp, steps, m, image, encoded, at=s.cached, order=s.sid)
                self.filling.append(s)
                queued = True
                s.prefill_s = time.perf_counter() - t0
                return
            else:
                # the text call is unchanged; only an image prompt hands prefill its encoded features, and a prompt
                # with checkpoints to take their positions
                self._join_fill_stream()                 # the shared prompt buffers: no chunk still reads them
                e.inherited = inherit
                first = prefill(e, s.prompt, s.sampling, mtp=mtp, resume=_handoff(m),
                                **({"vision": encoded} if image else {}), **({"checkpoints": take} if take else {}))
            s.cached = m.cached if m is not None else 0
            s.reuse = m.kind if m is not None else None
            s.sid, s.st = self.next_id, st
            self.next_id += 1
            registered = True                            # _start registers it; a failure there pops it again
            self._start(s, e, mtp, first, kept=s.draft and not exact and not image, t0=t0)
        except Exception as exc:
            self._abandon(s, st, m, exc, registered)
            raise
        finally:
            if image and not queued:       # the features are in the caches now; the tower's scratch goes back at once
                s.vision = encoded = None
                torch.cuda.empty_cache()
            elif image:                    # the pixels are encoded; the features stay with the fill until it ends
                s.vision = None

    def _start(self, s: Stream, e: Engine, mtp: bool, first: int, *, kept: bool, t0: float | None) -> None:
        """A prefilled prompt joins the rounds: keep its prompt end, draft its first chain, emit its first token (the
        first emission belongs to the admission: its failure is cleaned up by the caller too)."""

        st = s.st
        if kept:                           # the prompt's state; the MTP head has absorbed all but the last
            _t = TIMER.begin("snapshot")
            points = sorted([*getattr(e, "inherited", ()), *(getattr(e, "checkpoints", None) or ())],
                            key=lambda c: c.pos)         # the surviving inherited ones and the ones this prefill took
            self._remember(list(s.prompt), st, st.snapshot(), e.last_streams.clone() if mtp else None,
                           e.last_logits, points)
            TIMER.end(_t)
        s.context = list(s.prompt)
        s.drafts = draft(e, e.last_streams, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling,
                         self.confidence) if mtp and s.count > 1 else []
        if t0 is not None:
            s.prefill_s = time.perf_counter() - t0
        s.started = time.perf_counter()
        self.streams[s.sid] = s
        s.take([first], self.eos)

    def _abandon(self, s: Stream, st: State, m, exc: Exception, registered: bool) -> None:
        """A failed admission or fill: no stream registered, nothing kept in its half-restored or half-prefilled
        slot, the slot free again."""

        if m is not None:
            print(f"[octojet] prefix reuse {m.kind} failed: {exc}", file=sys.stderr, flush=True)
        if registered:
            self.streams.pop(s.sid, None)
        self._drop_kept(st)
        if all(f is not st for f in self.free):
            self.free.append(st)

    def _join_fill_stream(self) -> None:
        """The default stream waits for every chunk queued on the fill stream: before anything it ran is read or its
        slot or the shared prompt buffers are reused elsewhere."""

        fs = getattr(self, "fill_stream", None)
        if fs is not None and torch.cuda.current_stream() != fs:
            torch.cuda.current_stream().wait_stream(fs)
        self.fill_events = []

    def _fill_step(self) -> list[Stream]:
        """This round's prompt work. Beside decoding streams (F8): one chunk queued on the fill stream, unless
        FILL_AHEAD chunks are still running there; when a prompt joins or fails the default stream waits for it."""

        fs = getattr(self, "fill_stream", None)
        if not self.filling:
            if fs is not None and self.fill_events:
                self._join_fill_stream()
            return []
        if fs is None or not any(not x.done for x in self.streams.values()):
            if fs is not None and self.fill_events:
                self._join_fill_stream()                 # alone: the chunks run on the default stream again
            return self._fill()
        self.fill_events = [ev for ev in self.fill_events if not ev.query()]
        if len(self.fill_events) >= FILL_AHEAD:
            return []                                    # the queued chunks keep the GPU busy; decode only
        before = set(self.streams)
        with torch.cuda.stream(fs):
            ended = self._fill()
            ev = torch.cuda.Event()
            ev.record(fs)
        self.fill_events.append(ev)
        if ended or set(self.streams) != before:         # a prompt joined (or failed): its state is read next
            self._join_fill_stream()
        return ended

    def _release(self, s: Stream) -> _Fill:
        """Take a stream out of the filling queue; an image prompt's features go back at once."""

        if getattr(self, "fill_stream", None) is not None:
            self._join_fill_stream()                     # its slot may be reused at once (no-op inside a chunk)
        self.filling = [x for x in self.filling if x is not s]
        f = self.fills.pop(s.sid)
        if f.image:
            f.encoded = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return f

    arrived = staticmethod(lambda: False)            # a request waits that could be admitted (set by Scheduler)

    def _fill(self) -> list[Stream]:
        """The next prompt chunk(s) before a round: one chunk beside decoding streams, else chunks until a prompt
        joins or a waiting request can be admitted. Returns the streams that ended here (failed, or done at their
        first token)."""

        ended: list[Stream] = []
        while self.filling:
            ended += self._chunk()
            if getattr(self, "short_fill", False) or any(not x.done for x in self.streams.values()) or \
                    self.arrived():
                break                                    # (short_fill: the scheduler holds requests with deadlines)
        return ended

    def _next_fill(self) -> Stream:
        """The filling prompt whose chunk runs next: one passed over ``FILL_GUARD`` chunks in a row first (the longest
        passed over), then the fewest prompt rows left, then the earliest admitted."""

        def key(s: Stream):
            f = self.fills[s.sid]
            due = f.skipped >= FILL_GUARD
            return (not due, -f.skipped if due else 0, len(s.prompt) - f.at, f.order)

        return min(self.filling, key=key)

    def _chunk(self) -> list[Stream]:
        s = self._next_fill()
        f = self.fills[s.sid]
        for x in self.filling:                           # a chunk for s: every other filling prompt was passed over
            if x is not s:
                self.fills[x.sid].skipped += 1
        f.skipped = 0
        t0 = time.perf_counter()
        joining = False
        x3 = getattr(self.w, "x3", None) is not None    # EXL3 stages every buffer's n-gram rows in one pinned array:
        try:
            try:
                if x3:                                   # the round's copies out of it are done before the chunk's
                    self.buf.staged.synchronize()
                h0 = time.perf_counter()
                try:
                    f.at = next(f.steps)                 # one chunk, committed: the rows done so far
                finally:
                    if ROUND_LOG:                        # CPU time to stage and launch the chunk (no sync)
                        self._chunk_host = getattr(self, "_chunk_host", []) + [round(time.perf_counter() - h0, 4)]
                if x3:                                   # and the chunk's before the next round writes it
                    self.pbuf.staged.synchronize()
                s.prefill_s += time.perf_counter() - t0
                return []
            except StopIteration as stop:
                first = stop.value
            s.prefill_s += time.perf_counter() - t0
            self._release(s)
            joining = True
            self._start(s, f.e, f.mtp, first, kept=s.draft and not f.image, t0=None)
        except Exception as exc:                         # noqa: BLE001  (this request fails, the others go on)
            if not joining:
                self._release(s)
            self._abandon(s, s.st, f.m, exc, registered=joining)
            s.error, s.done = exc, True
            return [s]
        return [s] if s.done else []

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One round over the live streams; returns the ones that finished."""

        log = ROUND_LOG and (self.filling or self.streams)
        if log:                                          # measurement only (OCTOJET_ROUND_LOG): synchronised timings
            torch.cuda.synchronize()
            t0, pos = time.perf_counter(), [(len(s.prompt), int(s.st.pos)) for s in self.filling]
        ended = self._fill_step()                        # a prompt that ends here joins this round
        if log:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        try:
            out = ended + self._decode()
        except Exception:
            self.unreplied = ended                       # drop() hands them back: their requests still get a reply
            raise
        if log:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            host, self._chunk_host = getattr(self, "_chunk_host", []), []
            _round_log({"t": round(t0, 4), "fill_s": round(t1 - t0, 4), "decode_s": round(t2 - t1, 4), "chunk_host_s": host,
                        "filling": pos, "live": sum(1 for x in self.streams.values() if not x.done)})
        return out

    def _decode(self) -> list[Stream]:
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return []
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        segs = stage(self.w, self.buf, windows)
        logits = compute(self.w, segs, self.buf)
        starts = [a0 for _, a0, _ in segs] + [segs[-1][2]]
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        sampled = sample_streams(logits, starts, positions, [s.sampling for s in live])
        kept = []
        for s, (_, tokens), (st, a0, a1), rows in zip(live, windows, segs, sampled):
            path, end = accept(tokens, list(range(-1, len(tokens) - 1)), rows, s.count - len(s.out), self.eos)
            commit(self.w, st, self.buf, a1 - a0, len(path), at=a0)
            s.committed.extend(tokens[:len(path)])
            s.counted(len(tokens))
            new = [tokens[r] for r in path[1:]] + [end]
            last = len(s.out) + len(new) >= s.count or end in self.eos
            kept.append((s, a0, rows[:len(path)], new, last))
        self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if s.draft and not last])
        for s, _, _, new, _ in kept:
            s.take(new, self.eos)
        return [s for s in live if s.done]

    def _draft_all(self, streams: list) -> None:
        """Every drafting stream absorbs its kept rows and chains drafts, all streams in one step a depth."""

        for s, _, _ in streams:
            s.drafts = []
        room = {s.sid: min(self.depth, s.count - len(s.out) - len(keep)) for s, _, keep in streams}
        todo = [(s, a0, keep) for s, a0, keep in streams if room[s.sid] > 0 and self.mbuf is not None]
        if not todo:
            return
        for s, _, _ in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        windows = [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo]
        segs = mtp_stage(self.w, self.mbuf, windows)
        logits = mtp_compute(self.w, segs, self.mbuf)
        for (s, _, keep), (st, a0, a1) in zip(todo, segs):
            st.set_mtp_len(st.mtp_len + len(keep))
        active = [(s, a1 - 1) for s, (_, _, a1) in zip([t[0] for t in todo], segs)]
        for j in range(self.depth):
            picks = self._picks(logits, [s.st.pos + 1 + j for s, _ in active], [s.sampling for s, _ in active])
            nxt = []
            for (s, row), (d, p) in zip(active, picks):
                low = self.confidence > 0 and p < self.confidence
                if low and j > 0:
                    continue
                s.drafts.append(d)
                if not low and j + 1 < room[s.sid]:
                    nxt.append((s, row, d))
            if not nxt:
                return
            windows = [(s.st, [d], self.mbuf.streams[row:row + 1]) for s, row, d in nxt]
            segs = mtp_stage(self.w, self.mbuf, windows)
            logits = mtp_compute(self.w, segs, self.mbuf)
            for s, _, _ in nxt:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active = [(s, a0) for (s, _, _), (_, a0, _) in zip(nxt, segs)]

    def _picks(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """Each row's keyed draft and its probability at temperature 1, one read-back (drafts change speed only)."""

        row = logits.float()
        k = max([int(s.top_k) + MARGIN for s in samplings if s is not None and s.temperature > 0 and s.top_k] or [1])
        k = min(k, row.shape[1])
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        top, col = row.max(dim=-1, keepdim=True)
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        got = torch.cat([vals, idx.float(), top, col.float(), lse], dim=1).cpu().numpy()
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            g = got[i]
            lse_i = float(g[2 * k + 2])
            if smp is None or smp.temperature <= 0:
                c = int(g[2 * k + 1])
                out.append((int(self.draft_host[c]) if self.draft_host is not None else c,
                            float(np.exp(float(g[2 * k]) - lse_i))))
                continue
            cols = g[k:2 * k].astype(np.int64)
            ids = self.draft_host[cols] if self.draft_host is not None else cols
            tok = choose_rows(g[None, :k].astype(np.float32), ids[None, :], [pos], smp)[0]
            hit = np.nonzero(ids == tok)[0]
            out.append((int(tok), float(np.exp(float(g[hit[0]]) - lse_i)) if len(hit) else 0.0))
        return out

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams; a slot whose prompt end is kept stays with it, the rest are free again."""

        for s in done:
            self.streams.pop(s.sid, None)
            if s.st is not None and not any(k.slot is s.st for k in self.kept) and all(f is not s.st for f in self.free):
                self.free.append(s.st)              # a failed fill's slot is already back

    def drop(self) -> list[Stream]:
        """After a failed round: every live and filling stream ends (the caller replies with the error), and the
        streams that ended in that round's fill (done at their first token, or failed) are finished and returned."""

        self._join_fill_stream()                         # nothing queued there runs into reused slots or buffers
        ended, self.unreplied = self.unreplied, []
        self.finish(ended)
        live = [s for s in self.streams.values() if not s.done]
        for s in list(self.filling):               # their generators end here (their prompt buffers reset)
            self._release(s).steps.close()
            live.append(s)
        for s in live:
            self.streams.pop(s.sid, None)
            self._drop_kept(s.st)
            if all(f is not s.st for f in self.free):
                self.free.append(s.st)
        return ended + live


def _tensors(st: State):
    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))
