"""Prefix reuse for Flash Next on CUDA: kept prompt states, their checkpoints, and the one rule that matches a new
prompt to them (spec docs/superpowers/specs/2026-09-30-f2d-prefix-reuse-design.md sections 3 and 5).

A kept entry is a prompt's committed state at its end: the snapshot ``State.snapshot`` returns (recurrent state, conv
taps, PLE tail and history, MTP length; the KV rows stay in the slot), the main model's last streams row (``tail``,
what the initial drafts absorb from) and the final chunk's logits (``logits``, what an exact hit samples the first token
from). A checkpoint is the same data at an interior chunk boundary. ``match`` picks, among idle entries, the one that
lets a new prompt reuse the most tokens."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

KINDS = ("exact", "extend", "checkpoint")
_RANK = {k: i for i, k in enumerate(KINDS)}


@dataclass(eq=False)
class Checkpoint:
    pos: int                     # tokens committed at this checkpoint (a chunk end)
    snapshot: dict               # State.snapshot() with mtp_len = pos - 1 (a fresh prefill of prompt[:pos] stops there)
    tail: Any                    # the main model's last streams row before the MTP forward, [1, wide] bf16


@dataclass(eq=False)
class Kept:
    ids: list[int]
    snapshot: dict
    tail: Any | None
    logits: Any | None           # the final chunk's logits row, [1, vocab] bf16 (None for entries kept without it)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    serial: int = 0              # remember order; identical on both tensor-parallel ranks
    slot: Any = None             # the State whose KV rows the snapshot needs (scheduler path)


@dataclass
class Match:
    entry: Kept
    kind: str                    # one of KINDS
    cached: int                  # tokens reused
    resume: dict | None          # {"state", "tail"} for extend / checkpoint; None for exact
    copy_from: Any = None        # checkpoint resumed in another slot: the source's slot, whose rows below cached are
                                 # copied first so the source entry (its exact state) survives (scheduler path)
    forked: bool = False         # copy_from is a decoding stream's slot (a fork beside a busy source, F7)


def common_prefix(a, b) -> int:
    """Length of the longest common prefix of two id sequences."""

    n = min(len(a), len(b))
    if n == 0:
        return 0
    x = np.asarray(a[:n], dtype=np.int64)
    y = np.asarray(b[:n], dtype=np.int64)
    diff = np.flatnonzero(x != y)
    return int(diff[0]) if diff.size else n


def _candidate(prompt: list[int], entry: Kept) -> tuple[str, int, dict | None] | None:
    ids = entry.ids
    if len(ids) == len(prompt) and ids == prompt:
        return "exact", len(prompt), None
    if len(ids) < len(prompt) and prompt[:len(ids)] == ids:
        return "extend", len(ids), {"state": entry.snapshot, "tail": entry.tail}
    d = common_prefix(ids, prompt)
    best = None
    for c in entry.checkpoints:
        if c.pos <= d and c.pos < len(prompt) and (best is None or c.pos > best.pos):
            best = c
    if best is None:
        return None
    return "checkpoint", best.pos, {"state": best.snapshot, "tail": best.tail}


def match(prompt: list[int], entries: list[Kept], idle: Callable[[Kept], bool]) -> tuple[Match | None, str | None]:
    """The idle entry the prompt reuses most tokens from, and the kind of the busy entry that would have beaten it.

    Largest ``cached`` wins across kinds (an exact hit always wins); ties: exact > extend > checkpoint, then the largest
    ``serial``. Busy entries are never matched; the busy kind is reported only when that entry would have reused more
    tokens than the idle match (or when nothing idle matched) — an idle entry serving the same tokens is no miss."""

    best: Match | None = None
    best_key = None
    busy: str | None = None
    busy_key = None
    for e in entries:
        c = _candidate(prompt, e)
        if c is None:
            continue
        kind, cached, resume = c
        if not idle(e):
            key = (cached, -_RANK[kind])
            if busy_key is None or key > busy_key:
                busy, busy_key = kind, key
            continue
        key = (cached, -_RANK[kind], e.serial)
        if best_key is None or key > best_key:
            best, best_key = Match(e, kind, cached, resume), key
    if best is not None and busy_key is not None and busy_key[0] <= best_key[0]:
        busy = None
    return best, busy


def tail_count(n: int) -> int:
    """How many of ``n`` checkpoints go to the last interior chunk ends (F4: real traffic diverges near the end)."""

    return max(1, n // 2) if n > 0 else 0


def boundaries(total_len: int, rows: int, n: int) -> list[int]:
    """Checkpoint positions for a prompt of ``total_len`` tokens prefilled in ``rows``-row chunks, at most ``n``:
    the last ``tail_count(n)`` interior chunk ends (classifier variants and resent conversations diverge near the
    end), and the rest spread evenly over the interior ends before them (F4 amendment of spec 5.1). A prompt with at
    most ``n`` interior ends gets all of them."""

    if n <= 0 or total_len <= rows:
        return []
    chunks = math.ceil(total_len / rows)
    ends = [k * rows for k in range(1, chunks)]             # interior chunk ends, all < total_len
    if len(ends) <= n:
        return ends
    tail = tail_count(n)
    early, rest = ends[:-tail], n - tail
    spread = [early[(i + 1) * len(early) // rest - 1] for i in range(rest)]     # evenly, the last at the tail's edge
    return spread + ends[-tail:]


def thin(positions: list[int], n: int, protect=()) -> list[int]:
    """Drop positions until at most ``n`` remain, each time the one whose gap to its predecessor (or to 0) is
    smallest — ties: the smaller position — never one in ``protect`` while another can go. Deterministic."""

    pos = sorted(set(int(p) for p in positions))
    guard = {int(p) for p in protect}
    while len(pos) > max(n, 0):
        gaps = [(pos[i] - (pos[i - 1] if i else 0), pos[i]) for i in range(len(pos))]
        free = [g for g in gaps if g[1] not in guard] or gaps
        pos.remove(min(free)[1])
    return pos


MESSAGE_START = "<|im_start|>"


def message_start_id(model_dir) -> int | None:
    """The id of the chat template's message-start token (``<|im_start|>``) in the checkpoint's ``tokenizer.json``
    added tokens; None when the file or the token is missing (then no turn-start checkpoint is planned)."""

    try:
        data = json.loads((Path(model_dir) / "tokenizer.json").read_text())
    except (OSError, ValueError):
        return None
    for t in data.get("added_tokens", []) if isinstance(data, dict) else []:
        if isinstance(t, dict) and t.get("content") == MESSAGE_START and isinstance(t.get("id"), int):
            return t["id"]
    return None


def turn_start(prompt, marker: int | None) -> int | None:
    """The position of the prompt's last message start: the index of the last ``marker`` (``<|im_start|>``) token,
    so the tokens before it are a checkpoint's prefix (F7). A follow-up turn repeats the conversation up to and past
    that marker (its thinking-on generation prompt is what the next turn renders differently), and a variant that only
    changes its last message shares it too. None: no marker, or only at position 0."""

    if marker is None or len(prompt) < 2:
        return None
    ids = np.asarray(prompt, dtype=np.int64)
    hits = np.flatnonzero(ids[1:] == int(marker))
    return int(hits[-1]) + 1 if hits.size else None


def plan_checkpoints(total_len: int, rows: int, n: int, inherited: list[Checkpoint],
                     cached: int, turn: int | None = None) -> tuple[list[int], list[Checkpoint]]:
    """Before a prefill from ``cached``: the positions to take and the inherited checkpoints to keep, together at
    most ``n`` (spec 5.3). Inherited checkpoints above ``cached`` are invalid and never kept. ``turn`` (F7): the
    prompt's last message start (``turn_start``), planned and protected beside the tail when it lies inside the fill
    (above ``cached``, below the prompt's end); it need not be a chunk end (the prefill splits a chunk there)."""

    keep_pos = {c.pos for c in inherited if c.pos <= cached}
    planned = boundaries(total_len, rows, n)
    if n > 0 and turn is not None and cached < turn < total_len and turn not in planned:
        planned = sorted(planned + [turn])
    new = [p for p in planned if p > cached]
    tail = set(planned[-tail_count(n):]) if planned else set()   # the new prompt's end stays covered, new or inherited
    if n > 0 and turn is not None and cached < turn < total_len:
        tail.add(turn)
    final = set(thin(sorted(keep_pos | set(new)), n, protect=tail))
    take = sorted(p for p in new if p in final)
    keep = [c for c in inherited if c.pos <= cached and c.pos in final]
    return take, keep


def snapshot_bytes(cfg) -> int:
    """Bytes of one State.snapshot for this geometry: fp32 recurrent state [linear layers, nv, dv, dk], bf16 conv taps
    [linear layers, conv_kernel - 1, conv_dim], bf16 PLE tail [(ple_kernel - 1) * ngram_size, streams * hidden]."""

    lin = sum(1 for t in cfg.layer_types if t == "linear")
    rec = lin * cfg.nv * cfg.dv * cfg.dk * 4
    conv = lin * (cfg.conv_kernel - 1) * cfg.conv_dim * 2
    ple = (cfg.ple_kernel - 1) * cfg.ngram_size * cfg.streams * cfg.hidden * 2
    return rec + conv + ple


def exact_hit(e, entry: Kept, sampling) -> int:
    """Admit an identical prompt from its kept entry: restore the state, hand the drafts the kept tail, and sample the
    first token with the request's parameters — the same ``sample`` call ``prefill`` makes, on the kept logits."""

    e.st.restore(entry.snapshot)
    e.last_streams = entry.tail
    e.last_logits = entry.logits
    first = e.sample(entry.logits, [len(entry.ids)], sampling)[0]
    e.first = first
    return first
