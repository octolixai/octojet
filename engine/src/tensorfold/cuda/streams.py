"""Concurrent CUDA requests: a ``Stream`` each, the rows a verified window keeps, and prompt ends to resume."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class Stream:
    """One request. ``emit`` receives each round's new tokens and returns True to stop the stream."""

    prompt: list[int]
    count: int                                    # tokens to produce, the first sampled one included
    sampling: Any = None
    draft: bool = True                            # False: one row a round, the serial reference
    vision: Any = None                            # an image prompt's prepared pixels and positions (--vision)
    emit: Callable[[list[int]], bool | None] | None = None
    sid: int = 0
    st: Any = None                                # the committed model state
    snap: Any = None                              # the drafter's context
    copies: Any = None
    out: list[int] = field(default_factory=list)
    context: list[int] = field(default_factory=list)
    committed: list[int] = field(default_factory=list)     # tokens committed after the prompt, on every rank
    drafts: list[int] = field(default_factory=list)       # an MTP family's drafts for the next round
    stops: list[int] = field(default_factory=list)        # prompt positions whose states the prefill keeps
    error: Exception | None = None                        # why a stream ended without finishing
    done: bool = False
    rounds: int = 0
    min_rows: int = 0
    cached: int = 0
    prefill_s: float = 0.0
    started: float = 0.0
    finished: float = 0.0
    timing: bool = False                                  # arm the prefill recorder for this admission
    profile: bool = False                                 # a cudaProfilerStart/Stop range around this admission
    histogram: bool = False                               # the recorder's routing histogram (arms it)
    received_at: float = 0.0                              # perf_counter when the server received the request
    queued_at: float = 0.0
    admitted_at: float = 0.0
    first_token_at: float = 0.0
    timing_summary: dict | None = None
    profiler_rc: list | None = None                       # [cudaProfilerStart, cudaProfilerStop] return codes
    notes: list = field(default_factory=list)
    reuse: str | None = None                              # "exact" | "extend" | "checkpoint" | None: the kept state this admission used
    reuse_miss: str | None = None                         # "busy": a kept state matched but was decoding another request; else None
    reuse_copy: bool = False                              # the exact hit's state was copied from a decoding twin's slot

    def take(self, new: list[int], eos: Sequence[int] = ()) -> None:
        """Append a round's tokens and emit them; the stream ends at its count, an end token or a stop."""

        if self.first_token_at == 0.0 and new:
            self.first_token_at = time.perf_counter()
        self.out.extend(new)
        self.context.extend(new)
        stop = bool(self.emit(new)) if self.emit is not None else False
        if stop or len(self.out) >= self.count or self.out[-1] in eos:
            self.done = True
            self.finished = time.perf_counter()

    def counted(self, rows: int) -> None:
        self.rounds += 1
        self.min_rows = rows if self.min_rows == 0 else min(self.min_rows, rows)

    def stats(self) -> dict:
        out = {"prefill_s": round(self.prefill_s, 4), "decode_s": round(max(self.finished - self.started, 0.0), 4),
               "rounds": self.rounds, "drafts": self.draft, "cached": self.cached, "reuse": self.reuse,
               "reuse_miss": self.reuse_miss, "min_rows": self.min_rows,
               "received_at": self.received_at, "queued_at": self.queued_at, "admitted_at": self.admitted_at,
               "first_token_at": self.first_token_at,
               "ttft_s": (self.first_token_at - self.received_at) if self.received_at and self.first_token_at else None,
               "profiler_rc": self.profiler_rc, "timing": self.timing_summary}
        if self.reuse_copy:
            out["reuse_copy"] = True
        if self.notes:
            out["notes"] = list(self.notes)
        return out


def accept(tokens: Sequence[int], parents: Sequence[int], sampled: Sequence[int], room: int,
           eos: Sequence[int] = ()) -> tuple[list[int], int]:
    """The kept rows (root, then children equal to their parent's sample, <= ``room``, none past an end token)."""

    children: dict[tuple[int, int], int] = {}
    for row in range(1, len(tokens)):
        children.setdefault((parents[row], tokens[row]), row)
    path, terminal = [0], sampled[0]
    while len(path) < room and terminal not in eos:
        child = children.get((path[-1], terminal))
        if child is None:
            break
        path.append(child)
        terminal = sampled[child]
    return path, terminal


class PrefixCache:
    """Private prompt-end states by ids (never decoded rows: prefill and decode bits differ), newest last."""

    def __init__(self, keep: int = 8) -> None:
        self.keep = keep
        self.entries: list[tuple[list[int], Any, Any]] = []
        self.hit: set[tuple[int, ...]] = set()             # entries a later prompt resumed from

    def longest(self, prompt: Sequence[int]):
        """The longest entry the prompt strictly extends (one prompt token is always left to prefill), now newest."""

        best = None
        for entry in self.entries:
            ids = entry[0]
            if len(ids) < len(prompt) and list(prompt[:len(ids)]) == ids and (best is None or len(ids) > len(best[0])):
                best = entry
        return self._touch(best)

    def named(self, prompt: Sequence[int], length: int):
        """The entry holding the prompt's first ``length`` ids (a follower rank finds the leader's pick), now newest."""

        return self._touch(next((e for e in self.entries if len(e[0]) == length and list(prompt[:length]) == e[0]),
                                None))

    def _touch(self, entry):
        """A hit becomes the newest entry, so a shared system block outlives the prompts that reuse it."""

        if entry is not None:
            self.entries = [e for e in self.entries if e is not entry] + [entry]
            self.hit.add(tuple(entry[0]))
        return entry

    def add(self, ids: list[int], state: Any, snap: Any) -> None:
        """Newest last; past ``keep``, the oldest entry never resumed from goes first, else the oldest."""

        self.entries = [e for e in self.entries if e[0] != ids] + [(ids, state, snap)]
        while len(self.entries) > self.keep:
            cold = [e for e in self.entries[:-1] if tuple(e[0]) not in self.hit]
            gone = cold[0] if cold else self.entries[0]
            self.entries = [e for e in self.entries if e is not gone]
        self.hit &= {tuple(e[0]) for e in self.entries}
