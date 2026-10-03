"""Verify MTP chains against the same keyed samples as serial decoding, committing only rows before the first mismatched draft so drafts never change emitted tokens."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.prefill_timing import TIMER
from tensorfold.cuda.sampling import sample_rows
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from . import CONFIDENCE, DEPTH
from .forward import commit, forward, prestage
from .state import CAND, Buffers, State
from .mtp import mtp_forward
from .weights import Weights


def sample_mapped(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                  id_map: torch.Tensor) -> list[int]:
    """Rows of logits over a token subset (column j is token id_map[j]) -> tokens, with the keyed rule on the real ids (the draft head over the draft vocabulary)."""

    if sampling is None or sampling.temperature <= 0:
        return [int(t) for t in id_map[logits.argmax(dim=-1)].cpu().tolist()]
    k = min(logits.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else logits.shape[1]
    vals, idx = torch.topk(logits.float(), k, dim=-1, sorted=False)
    return choose_rows(vals.cpu().numpy(), id_map[idx].cpu().numpy().astype(np.int64), positions, sampling)


def tp_sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                   offset: int = 0, id_map: torch.Tensor | None = None, with_prob: bool = False):
    """Gather each rank's global-id candidates and apply the same keyed draw on every rank; ``with_prob`` also returns temperature-1 probabilities from gathered log-sum-exps."""

    R = logits.shape[0]
    greedy = sampling is None or sampling.temperature <= 0
    k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
    if greedy:
        # argmax takes the first (lowest-id) maximum whatever the row count; topk promises no order among ties
        ids = logits.argmax(dim=-1, keepdim=True)
        vals = torch.gather(logits, 1, ids).float()
    else:
        vals, ids = torch.topk(logits.float(), k, dim=-1)
    ids = (id_map[ids] if id_map is not None else ids + int(offset)).to(torch.int32)
    parts = [vals, ids.view(torch.float32)]
    if with_prob:
        parts.append(torch.logsumexp(logits.float(), dim=-1, keepdim=True))
    packed = torch.cat(parts, dim=1).contiguous()
    width = packed.shape[1]
    world = int(w.meta["world"])
    got = torch.empty((world * packed.numel(),), dtype=torch.float32, device=logits.device)
    w.comm.all_gather(packed.view(-1), got)
    g = got.view(world, R, width).cpu()
    values = torch.cat([g[r, :, :k] for r in range(world)], dim=1).numpy().astype(np.float32)
    tokens = torch.cat([g[r, :, k:2 * k].contiguous().view(torch.int32) for r in range(world)], dim=1).numpy()
    tokens = tokens.astype(np.int64)
    if greedy:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if not with_prob:
        return chosen
    lse = g[:, :, 2 * k].numpy().astype(np.float64)                       # [world, R]
    top = lse.max(axis=0)
    total = top + np.log(np.exp(lse - top).sum(axis=0))
    probs = []
    for i, t in enumerate(chosen):
        hit = np.nonzero(tokens[i] == t)[0]
        probs.append(float(np.exp(float(values[i, hit[0]]) - total[i])) if len(hit) else 0.0)
    return chosen, probs


def choose_gathered(w: Weights, cand_all: torch.Tensor, R: int, positions: Sequence[int], sampling: Sampling | None,
                    with_prob: bool = False):
    """Apply keyed sampling to candidates gathered inside the step graph; ``with_prob`` also returns each selected token's probability."""

    world, width = int(w.meta["world"]), 2 * CAND + 1
    g = cand_all[:world * R * width].view(world, R, width).cpu().numpy()
    values = np.concatenate([g[r, :, :CAND] for r in range(world)], axis=1).astype(np.float32)
    tokens = np.concatenate([np.ascontiguousarray(g[r, :, CAND:2 * CAND]).view(np.int32) for r in range(world)],
                            axis=1).astype(np.int64)
    if sampling is None or sampling.temperature <= 0:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if not with_prob:
        return chosen
    lse = g[:, :, 2 * CAND].astype(np.float64)
    top = lse.max(axis=0)
    total = top + np.log(np.exp(lse - top).sum(axis=0))
    probs = []
    for i, t in enumerate(chosen):
        hit = np.nonzero(tokens[i] == t)[0]
        probs.append(float(np.exp(float(values[i, hit[0]]) - total[i])) if len(hit) else 0.0)
    return chosen, probs


def _gathered_fits(sampling: Sampling | None) -> bool:
    """Whether a step's gathered candidates (CAND a rank) cover the sampler's top-k plus its margin."""

    return sampling is None or sampling.temperature <= 0 or (bool(sampling.top_k) and sampling.top_k + MARGIN <= CAND)


PREFILL_ROWS = 2048      # rows of a prompt chunk (the server's engine: ``geometry.indexed_prefill_rows()``)


class Engine:
    """Weights, one sequence's state, buffers for decode windows (main model and MTP head) and for prompt chunks."""

    def __init__(self, w: Weights, *, capacity: int = 4096, max_rows: int = 8, prefill_rows: int = PREFILL_ROWS,
                 graphs: bool = False, kv_dtype: str = "bf16") -> None:
        self.w = w
        self.capacity = capacity
        self.rows, self.prefill_rows = max_rows, prefill_rows
        self.kv_dtype = kv_dtype
        self.buf = Buffers(w, max_rows, capacity)
        self.mbuf = Buffers(w, max_rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True)
        self.st = State(w, capacity, max_rows, kv_dtype)
        self.last_logits = None
        self.graphs = None
        if graphs:
            from .graphs import Graphs

            self.graphs = Graphs(self, max_rows=max_rows)

    def reset(self) -> None:
        self.st.reset(self.w)

    def twin(self) -> "Engine":
        """Share weights and scratch with an independent serial state, without MTP or graphs; requests run sequentially so scratch reuse leaves this engine's state and prefix cache intact."""

        other = object.__new__(Engine)
        other.w, other.capacity, other.rows, other.prefill_rows = self.w, self.capacity, self.rows, self.prefill_rows
        other.buf, other.mbuf, other.pbuf, other.graphs = self.buf, None, self.pbuf, None
        other.st = State(self.w, self.capacity, self.rows, self.st.kv_dtype)
        return other

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A decode step's forward (a CUDA graph when enabled): logits [R, V]."""

        if self.graphs is not None:
            return self.graphs.forward(tokens)
        return forward(self.w, self.st, self.buf, tokens)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False) -> list[int]:
        """Rows of logits at their positions -> tokens (``draft``: logits of the MTP's draft head)."""

        mapped = draft and self.w.draft_ids is not None
        if self.w.comm is not None:
            b = self.mbuf if draft else self.buf
            if logits.data_ptr() == b.logits.data_ptr() and _gathered_fits(sampling):
                return choose_gathered(self.w, b.cand_all, logits.shape[0], positions, sampling)
            return tp_sample_rows(self.w, logits, positions, sampling, offset=self.w.meta["vocab_offset"],
                                  id_map=self.w.draft_ids if mapped else None)
        if mapped:
            return sample_mapped(logits, positions, sampling, self.w.draft_ids)
        return sample_rows(logits, positions, sampling)

    def sample_draft(self, logits: torch.Tensor, position: int, sampling: Sampling | None) -> tuple[int, float]:
        """Return the position's keyed draft and its temperature-1 probability; this confidence ends draft chains early without changing output."""

        w = self.w
        mapped = w.draft_ids is not None
        if w.comm is not None:
            if logits.data_ptr() == self.mbuf.logits.data_ptr() and _gathered_fits(sampling):
                toks, probs = choose_gathered(w, self.mbuf.cand_all, 1, [position], sampling, with_prob=True)
            else:
                toks, probs = tp_sample_rows(w, logits[:1], [position], sampling, offset=w.meta["vocab_offset"],
                                             id_map=w.draft_ids if mapped else None, with_prob=True)
            return toks[0], probs[0]
        if mapped and getattr(self, "_draft_host", None) is None:
            self._draft_host = w.draft_ids.cpu().numpy()
        row = logits[:1].float()
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        if sampling is None or sampling.temperature <= 0:
            top, col = row.max(dim=-1, keepdim=True)            # the first maximum: argmax's (and serial's) choice
            got = torch.cat([top, lse, col.float()], dim=1).cpu().numpy()[0]       # one sync
            c = int(got[2])
            tok = int(self._draft_host[c]) if mapped else c
            return tok, float(np.exp(float(got[0]) - float(got[1])))
        k = min(row.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else row.shape[1]
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        got = torch.cat([vals, lse, idx.float()], dim=1).cpu().numpy()[0]          # one sync
        cols = got[k + 1:].astype(np.int64)
        ids = self._draft_host[cols] if mapped else cols
        tok = choose_rows(got[None, :k].astype(np.float32), ids[None, :], [position], sampling)[0]
        hit = np.nonzero(ids == tok)[0]
        return int(tok), float(np.exp(float(got[hit[0]]) - float(got[k]))) if len(hit) else 0.0

    def mtp_forward(self, next_tokens: Sequence[int], streams: torch.Tensor) -> torch.Tensor:
        if self.graphs is not None:
            return self.graphs.mtp_forward(next_tokens, streams)
        return mtp_forward(self.w, self.st, self.mbuf, next_tokens, streams)


def absorb(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """The MTP cache takes positions with main-model streams [n, S*D] and next tokens; logits of the last."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    logits = e.mtp_forward(next_tokens, streams)
    st.set_mtp_len(st.mtp_len + len(next_tokens))
    return logits


def draft(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb kept rows and chain drafts, always retaining the first even below ``confidence``, then stopping before later drafts below it or after a low-confidence first draft."""

    saved_phase = TIMER.phase
    TIMER.phase = "draft"
    saved_layer, saved_rows, saved_pos = TIMER.layer, TIMER.rows, TIMER.pos
    TIMER.layer, TIMER.rows = -1, 1
    try:
        st = e.st
        logits = absorb(e, streams, next_tokens)
        drafts: list[int] = []
        for j in range(count):
            low = False
            if confidence > 0:
                TIMER.pos = position + j
                _t = TIMER.begin("sample")
                d, p = e.sample_draft(logits, position + j, sampling)
                TIMER.end(_t)
                low = p < confidence
                if low and j > 0:
                    break
            else:
                TIMER.pos = position + j
                _t = TIMER.begin("sample")
                d = e.sample(logits[:1], [position + j], sampling, draft=True)[0]
                TIMER.end(_t)
            drafts.append(d)
            if low:
                break
            if j + 1 < count:
                prev = e.mbuf.streams[len(next_tokens) - 1:len(next_tokens)] if j == 0 else e.mbuf.streams[:1]
                logits = e.mtp_forward([d], prev)
                st.set_mtp_len(st.mtp_len + 1)
                st.mtp_drafted += 1
                next_tokens = [d]
        return drafts
    finally:
        TIMER.phase = saved_phase
        TIMER.layer, TIMER.rows, TIMER.pos = saved_layer, saved_rows, saved_pos


@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True,
            resume: dict | None = None, vision=None, checkpoints: Sequence[int] | None = None) -> int:
    """Commit the prompt in chunks, sample the first token; rows ignore chunking, so ``resume`` equals a fresh run.
    ``vision`` (``tensorfold.vision.qwen_cuda.EncodedVision``): the image rows take its features and every row its
    t/h/w rotary positions; later text rotates at its position plus the prompt's offset. The chunks are
    ``prefill_steps``'s, run back to back."""

    steps = prefill_steps(e, prompt, sampling, mtp=mtp, resume=resume, vision=vision, checkpoints=checkpoints)
    del resume                                       # the generator alone holds the resume state, until it restores
    return run_steps(steps)


def run_steps(steps) -> int:
    """Drive a ``prefill_steps`` generator to its end: the first token."""

    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


def chunk_starts(begin: int, total: int, rows: int, cuts=()) -> list[int]:
    """Chunk starts from ``begin``: chunk ends fall on absolute multiples of ``rows`` (F2d 5.1), so the checkpoints of
    prompts that share a prefix line up; after an unaligned resume the first chunk only reaches the next multiple.
    ``cuts`` (F7: a turn-start checkpoint) also end a chunk there; the chunks after it stay on the multiples. A row's
    bits never depend on its chunk, so a cut moves only where a snapshot can be taken."""

    starts, at = [], begin
    while at < total:
        starts.append(at)
        at = (at // rows + 1) * rows
    extra = sorted({int(c) for c in cuts if begin < int(c) < total} - set(starts))
    return sorted(starts + extra)


def take_checkpoint(e: Engine, end: int, tail: torch.Tensor, use_mtp: bool) -> None:
    """After the chunk ending at ``end`` committed: what a fresh prefill of prompt[:end] leaves (F2d 5): the snapshot
    with ``mtp_len = end - 1`` (the chunk's MTP forward absorbed prompt[end] already) and the pre-MTP last streams row.
    On one GPU an out-of-memory clone skips the checkpoint and never fails the request."""

    from .prefix import Checkpoint

    _t = TIMER.begin("checkpoint")
    try:
        snap = e.st.snapshot()
        if use_mtp:
            snap["mtp_len"] = end - 1
        e.checkpoints.append(Checkpoint(end, snap, tail))
    except torch.OutOfMemoryError:
        print(f"[octojet] prefix checkpoint skipped (OOM) at {end}", file=sys.stderr, flush=True)
    finally:
        TIMER.end(_t)


def next_chunk_end(at: int, total: int, rows: int, cuts=()) -> int:
    """Where the chunk starting at ``at`` ends: the next absolute multiple of ``rows`` (or the prompt's end), never
    past a cut. With ``rows`` the prompt chunk size this is exactly ``chunk_starts``' chunking; a smaller ``rows`` that
    divides it (F8: chunks while other streams decode) still ends a chunk on every multiple of the full size."""

    end = min(total, (at // rows + 1) * rows)
    inside = [int(c) for c in cuts if at < int(c) < end]
    return min(inside) if inside else end


def prefill_steps(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True,
                  resume: dict | None = None, vision=None, checkpoints: Sequence[int] | None = None,
                  live_rows=None):
    """``prefill`` one chunk at a time: a generator that yields after each chunk's commit and returns the first token.
    The concurrent decoder runs decode rounds between the yields (``MultiDecoder``'s prompts inside rounds); between
    two yields nothing is carried in the shared prompt buffers (every chunk restages them), so rounds or another
    admission may use them, and the chunks are the same kernels on the same rows as ``prefill``'s.

    ``live_rows`` (F8): called before each chunk; a row count (dividing the chunk size) while other streams decode, so
    their pause between tokens is one short chunk, or None for a full chunk. A row's bits never depend on its chunk."""

    with torch.no_grad():
        if not prompt:
            raise ValueError("prefill requires at least one token")
        if vision is not None and resume is not None:
            raise ValueError("an image prompt prefills from its start")
        w, st, pb = e.w, e.st, e.pbuf
        if vision is not None and pb.attn.qsa and e.prefill_rows % pb.attn.ratio:   # a block's first row is in its chunk
            raise ValueError(f"image prompts need prompt chunks (TENSORFOLD_PREFILL_ROWS) divisible by {pb.attn.ratio}")
        use_mtp = mtp and w.mtp is not None and e.mbuf is not None
        wanted = {int(p) for p in checkpoints or ()}     # chunk ends to keep a fresh-prefix snapshot at (F2d 5)
        if wanted and vision is not None:
            raise ValueError("an image prompt is never kept: it takes no checkpoints")
        e.checkpoints = []
        begin = 0
        if resume is None:
            e.reset()
        else:
            st.restore(resume["state"])
            begin = st.pos
            if not 0 < begin < len(prompt):
                raise ValueError("a resumed prompt must extend the cached tokens")
            if use_mtp and resume.get("tail") is not None:
                TIMER.chunk, TIMER.rows, TIMER.pos = begin // e.prefill_rows, 1, begin
                mtp_forward(w, st, pb, [prompt[begin]], resume["tail"])
                _t = TIMER.begin("prefill_other")
                st.set_mtp_len(st.mtp_len + 1)
                TIMER.end(_t)
            resume = None                                # restored: the resumed snapshot is no longer held here
        last = None
        chunk_index = begin // e.prefill_rows
        image_rows = positions = None
        if vision is not None:
            image_rows = torch.tensor(vision.rows, dtype=torch.int64, device=vision.features.device)
            positions = vision.positions.t().contiguous()            # [prompt, 3] int32
    start = begin
    while start < len(prompt):
        rows = (live_rows() if live_rows is not None else None) or e.prefill_rows
        end = next_chunk_end(start, len(prompt), rows, cuts=wanted)
        with torch.no_grad():
            try:
                chunk = list(prompt[start:end])
                R = len(chunk)
                chunk_index = start // e.prefill_rows            # (a turn-start cut shares its chunk's index)
                TIMER.chunk, TIMER.rows, TIMER.pos = chunk_index, R, start
                if 0 <= chunk_index < len(TIMER.chunk_rows):
                    TIMER.chunk_rows[chunk_index] = R
                final = start + R >= len(prompt)
                features = None
                if vision is not None:                   # an image prompt's chunk: its t/h/w rows and image features
                    pb.rope_rows = positions[start:start + R]
                    inside = ((image_rows >= start) & (image_rows < start + R)).nonzero().flatten()
                    if inside.numel():
                        features = (image_rows.index_select(0, inside) - start,
                                    vision.features.index_select(0, inside))
                # only the prompt's last row is sampled: the head runs on the final chunk alone
                logits = (forward(w, st, pb, chunk, logits=final) if features is None else
                          forward(w, st, pb, chunk, logits=final, features=features))
                if final:
                    _t = TIMER.begin("prefill_other")
                    last = logits.clone()
                    TIMER.end(_t)
                    e.last_logits = last
                _t = TIMER.begin("prefill_other")
                streams_last = pb.streams[R - 1:R].clone()
                TIMER.end(_t)
                if use_mtp:
                    nxt = list(prompt[start + 1:start + R + 1])
                    if nxt:
                        mtp_forward(w, st, pb, nxt, pb.streams[:len(nxt)])
                        _t = TIMER.begin("prefill_other")
                        st.set_mtp_len(st.mtp_len + len(nxt))
                        TIMER.end(_t)
                commit(w, st, pb, R, R)
                chunk_index += 1
                if not final and start + R in wanted:
                    take_checkpoint(e, start + R, streams_last, use_mtp)
            finally:
                pb.rope_rows = None                      # the shared prompt buffers serve text rows next
        if not final:
            # read the next chunk's table rows now, while the GPU still runs this one: the round that runs before it
            # waits for the GPU, and these host reads would otherwise run with the GPU idle (F8; same bytes)
            nrows = (live_rows() if live_rows is not None else None) or e.prefill_rows
            nend = next_chunk_end(start + R, len(prompt), nrows, cuts=wanted)
            prestage(w, pb, st, prompt[start + R:nend])
            yield start + R                              # rows committed; a round may run before the next chunk
        start += R
    with torch.no_grad():
        if vision is not None:                           # decode continues at the image prompt's rotary offset
            st.set_rope_delta(vision.rope_delta)
        _t = TIMER.begin("sample")
        first = e.sample(last, [len(prompt)], sampling)[0]
        TIMER.end(_t)
        e.last_streams = streams_last
        e.first = first
        return first


WARM_TAIL = 18      # a partial chunk after a full one: neither its rows nor the MTP head's 17 divide by 16


@torch.no_grad()
def warm(e: Engine) -> None:
    """Prefill a synthetic prompt (a full chunk, then a partial one) and empty the state, so no request compiles or loads a prompt kernel."""

    prefill(e, [0] * min(e.prefill_rows + WARM_TAIL, e.capacity), None)
    e.reset()


@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    keeps: list[int] = field(default_factory=list)      # tokens each round kept
    committed: list[int] = field(default_factory=list)  # the tokens now in the caches (all but the pending one)
    widths: list[int] = field(default_factory=list)     # rows each round verified

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, stop_eos: bool = False,
                  on_tokens=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token. ``on_tokens(new)`` hears each step's token; it returns True to stop early."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        logits = e.forward([out[-1]])
        tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
        commit(w, st, b, 1, 1)
        out.append(tok)
        if on_tokens is not None and on_tokens([tok]):
            break
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, committed=out[:-1], widths=[1] * (len(out) - 1))


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, depth: int = DEPTH,
               confidence: float = CONFIDENCE, stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Verify pending and drafted tokens from the prefill state, commit rows before the first mismatched draft, and call ``on_tokens(new)`` with kept tokens after pending, stopping on True."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    rounds = drafted = accepted = 0
    keeps: list[int] = []
    widths: list[int] = []
    pos0 = st.pos
    unabsorbed = None                                  # the last round's kept rows, not yet in the MTP cache
    torch.cuda.synchronize()
    start = time.perf_counter()
    drafts = draft(e, e.last_streams, [pending], st.pos + 1, min(depth, count - len(out)), sampling, confidence)
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens)
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        commit(w, st, b, R, keep)
        unabsorbed = (keep, sampled[:keep])
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        keeps.append(keep)
        widths.append(R)
        new = sampled[:keep][:max(0, count - len(out))]
        out.extend(sampled[:keep])
        if on_tokens is not None and new and on_tokens(new):
            break
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        n = min(depth, count - len(out))
        drafts = []
        if n > 0:
            drafts = draft(e, b.streams[:keep], sampled[:keep], st.pos + 1, n, sampling, confidence)
            unabsorbed = None
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if unabsorbed is not None:              # the MTP cache takes the last kept rows: it then covers the sequence
        absorb(e, b.streams[:unabsorbed[0]], unabsorbed[1])
    committed = out[:st.pos - pos0]
    return DecodeResult(out[:count], seconds, rounds, drafted, accepted, keeps, committed, widths)
