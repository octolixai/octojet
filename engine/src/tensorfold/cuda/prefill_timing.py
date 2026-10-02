"""Where a prefill spends its time: CUDA events at frozen cut points, phase-qualified NVTX ranges, a routing histogram,
and a per-admission summary. Spec: docs/superpowers/specs/2026-09-30-f2c-prefill-design.md section 3.1.

Inert unless ``OCTOJET_PREFILL_TIMING=1`` at engine start (``configure``) and a request arms it (``arm``). Armed, every
cut point records one event pair from a preallocated pool and writes its metadata into preallocated slots; nothing is
read until ``resolve`` after the admission's terminal event. Nothing here synchronises, allocates or prints inside the
timed region.

NVTX ranges are markers and never fail a request: ``_nvtx_push``/``_nvtx_pop`` swallow errors and return whether the call
succeeded. The recorder pops only a range whose push succeeded, counts only successful pushes and pops in
``open_ranges``, counts failures, and ``resolve`` notes them (``nvtx_failures`` in the summary).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from typing import Any

import torch

ENV = "OCTOJET_PREFILL_TIMING"
ENV_OUT = "OCTOJET_PREFILL_TIMING_OUT"
ENV_NSYS = "OCTOJET_NSYS_CAPTURE"

PER_LAYER_CUTS = 16           # device cuts a decoder layer records at most (the GDN path uses 15)
ATTN_CUTS_PER_BLOCK = 3       # idx_scores, idx_select, attn_sparse per 256-row attention block
ATTN_ROWS = 256               # forward.ATT_ROWS
MTP_ALLOWANCE = 200           # the MTP tree's cuts per chunk
ADMISSION_ALLOWANCE = 64      # finish/commit/snapshot/sample/draft and slack, per admission
PHASES = ("main", "mtp", "draft")

HOST_BLOCKS = frozenset({"stage_wait", "stage_tokens", "stage_ple_gather", "stage_copy"})
DEVICE_BLOCKS = frozenset({
    "embed", "ple", "hc_readout", "dense_gdn_in", "dense_gdn_out", "dense_attn_in", "dense_attn_out", "dense_ple",
    "gdn_front", "gdn_recurrence", "gdn_back", "gdn_other", "attn_prep", "attn_other", "idx_pool", "idx_scores",
    "idx_select", "attn_sparse", "attn_gate", "router", "plan", "expert_up", "expert_down", "writeback", "finish",
    "commit", "snapshot", "checkpoint", "sample", "mtp_input", "prefill_other"})
BLOCKS = HOST_BLOCKS | DEVICE_BLOCKS
_NVTX = {(p, b): f"{p}:{b}" for p in PHASES for b in BLOCKS}       # precomputed: no string building when armed


class TimingBusy(RuntimeError):
    """Another admission holds the recorder."""


def pool_events(layers: int, attention_layers: int, prefill_rows: int, capacity: int) -> int:
    chunks = math.ceil(capacity / prefill_rows)
    per_chunk = (layers * PER_LAYER_CUTS
                 + (attention_layers + 1) * math.ceil(prefill_rows / ATTN_ROWS) * ATTN_CUTS_PER_BLOCK
                 + MTP_ALLOWANCE)
    return 2 * chunks * per_chunk + 2 * ADMISSION_ALLOWANCE


def percentile(values: list, p: float):
    """Nearest-rank percentile (ceil(p * n)-th smallest); None for an empty list."""

    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(p * len(s)))
    return s[k - 1]


def _nvtx_push(name: str) -> bool:
    """Push an NVTX range; True on success, False when the torch call raised (never raises)."""

    try:
        torch.cuda.nvtx.range_push(name)
    except Exception:                        # noqa: BLE001  a range is a marker; it never fails the request
        return False
    return True


def _nvtx_pop() -> bool:
    """Pop an NVTX range; True on success, False when the torch call raised (never raises)."""

    try:
        torch.cuda.nvtx.range_pop()
    except Exception:                        # noqa: BLE001
        return False
    return True


class Recorder:
    def __init__(self) -> None:
        self.enabled = False
        self.armed = False
        self.events: list = []
        self.span_meta: list = []          # preallocated: one 7-slot list per event pair
        self.host_meta: list = []          # preallocated: one 7-slot list per host cut
        self.hist = None
        self.idx_scratch = None
        self.ones = None
        self.terminal_event = None
        self.top_k = 0
        self.slots = 0
        self.layers = 0
        self.chunks = 0
        self._reset_context()

    def _reset_context(self) -> None:
        self.phase, self.layer, self.chunk, self.rows, self.pos = "main", -1, -1, 0, 0
        self.n = 0                          # events used
        self.nh = 0                         # host cuts used
        self.records: list[dict] = []
        self.overflow = False
        self.hist_overflow = False
        self.notes: list[str] = []
        self.meta: dict = {}
        self.want_hist = False
        self._t_arm = 0.0
        self.chunk_rows = [0] * self.chunks
        self.last_abort: str | None = None
        self.open_ranges = 0                # NVTX ranges successfully pushed and not yet successfully popped
        self.open_span = -1                 # event index of a device cut whose end() has not run, else -1
        self.dev_pushed = False             # the open device cut's push succeeded (end() pops only then)
        self.host_pushed = False            # the open host cut's push succeeded (host_end() pops only then)
        self.nvtx_push_failures = 0
        self.nvtx_pop_failures = 0

    # ---- lifecycle -------------------------------------------------------------------------------------------

    def configure(self, *, layers: int, attention_layers: int, prefill_rows: int, capacity: int, experts: int,
                  top_k: int, slots: int, device) -> bool:
        """Allocate the pool, the metadata slots and the histogram when the variable is set; False (inert) otherwise."""

        if os.environ.get(ENV) != "1":
            self.enabled = False
            return False
        n = pool_events(layers, attention_layers, prefill_rows, capacity)
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for e in self.events:               # a torch Event allocates its CUDA event on first record
            e.record()
        self.terminal_event = torch.cuda.Event(enable_timing=True)
        self.terminal_event.record()
        self.span_meta = [[None] * 7 for _ in range(n // 2)]
        self.host_meta = [[None] * 7 for _ in range(4 * math.ceil(capacity / prefill_rows) * 4 + ADMISSION_ALLOWANCE)]
        self.layers, self.chunks = layers, math.ceil(capacity / prefill_rows)
        self.hist = torch.zeros((layers, self.chunks, experts + 1), dtype=torch.int32, device=device)
        self.idx_scratch = torch.zeros((prefill_rows * slots,), dtype=torch.int64, device=device)
        self.ones = torch.ones((prefill_rows * slots,), dtype=torch.int32, device=device)
        self.top_k, self.slots = top_k, slots
        self.enabled = True
        self._reset_context()
        return True

    def arm(self, meta: dict, *, histogram: bool = False) -> bool:
        if not self.enabled or self.armed:
            return False
        self._reset_context()
        self.meta = dict(meta)
        self.want_hist = histogram
        self.hist.zero_()
        self.armed = True
        self._t_arm = time.perf_counter()
        return True

    def abort(self, reason: str) -> None:
        """Disarm after a lifecycle failure without reading anything, before or after resolve(): drop the records, pop
        NVTX ranges an exception left open, keep the reason; the request continues unmeasured."""

        self.last_abort = reason
        self.records = []
        try:
            self._close_open_ranges()
        except Exception:                    # noqa: BLE001  bookkeeping never escapes into the caller's exception path
            pass
        self.open_ranges = 0                 # given up: a range whose pop failed is no longer tracked
        self.dev_pushed = self.host_pushed = False
        self.open_span = -1
        self.armed = False

    def _close_open_ranges(self) -> int:
        """Attempt at most ``open_ranges`` pops; a failed pop is counted and stops the attempt. Returns how many closed."""

        closed = 0
        for _ in range(self.open_ranges):
            if not _nvtx_pop():
                self.nvtx_pop_failures += 1
                break
            self.open_ranges -= 1
            closed += 1
        if self.open_ranges == 0:
            self.dev_pushed = self.host_pushed = False
        return closed

    # ---- cut points (armed: one event record + an NVTX push/pop into preallocated slots; unarmed: return at once) --

    def begin(self, block: str, *, rows: int | None = None, pos: int | None = None) -> int:
        if not self.armed:
            return -1
        name = _NVTX.get((self.phase, block))
        if name is None:
            raise ValueError(f"prefill timing: unknown phase/block {self.phase!r}/{block!r}")
        if block in HOST_BLOCKS:
            raise ValueError(f"prefill timing: {block!r} is a host block; use host_begin/host_end")
        i = self.n
        if i + 2 > len(self.events):
            self.overflow = True
            return -1
        self.n = i + 2
        m = self.span_meta[i // 2]
        m[0], m[1], m[2], m[3], m[4], m[5], m[6] = (self.phase, self.layer, self.chunk, block,
                                                   self.rows if rows is None else rows,
                                                   self.pos if pos is None else pos, i)
        self.events[i].record()
        if _nvtx_push(name):
            self.dev_pushed = True
            self.open_ranges += 1
        else:
            self.dev_pushed = False
            self.nvtx_push_failures += 1
        self.open_span = i
        return i

    def end(self, idx: int) -> None:
        if idx < 0 or not self.armed:
            return
        self.events[idx + 1].record()
        if self.dev_pushed:
            self.dev_pushed = False
            if _nvtx_pop():
                self.open_ranges -= 1
            else:
                self.nvtx_pop_failures += 1
        self.open_span = -1

    def host_begin(self, block: str) -> float:
        if not self.armed:
            return 0.0
        name = _NVTX.get((self.phase, block))
        if name is None:
            raise ValueError(f"prefill timing: unknown phase/block {self.phase!r}/{block!r}")
        if block in DEVICE_BLOCKS:
            raise ValueError(f"prefill timing: {block!r} is a device block; use begin/end")
        if _nvtx_push(name):
            self.host_pushed = True
            self.open_ranges += 1
        else:
            self.host_pushed = False
            self.nvtx_push_failures += 1
        return time.perf_counter()

    def host_end(self, block: str, t0: float) -> None:
        if not self.armed:
            return
        if block not in HOST_BLOCKS:
            raise ValueError(f"prefill timing: {block!r} is not a host block")
        if self.host_pushed:
            self.host_pushed = False
            if _nvtx_pop():
                self.open_ranges -= 1
            else:
                self.nvtx_pop_failures += 1
        j = self.nh
        if j >= len(self.host_meta):
            self.overflow = True
            return
        self.nh = j + 1
        m = self.host_meta[j]
        m[0], m[1], m[2], m[3], m[4], m[5], m[6] = (self.phase, self.layer, self.chunk, block, self.rows, self.pos,
                                                   (time.perf_counter() - t0) * 1000.0)

    def histogram_add(self, pick: torch.Tensor, rows: int) -> None:
        """One main layer's chunk of picks [rows, slots] (the shared slot included; its bin is dropped in the summary)."""

        if not (self.armed and self.want_hist and self.phase == "main"):
            return
        if not (0 <= self.layer < self.layers and 0 <= self.chunk < self.chunks):
            self.hist_overflow = True
            return
        n = rows * self.slots
        self.idx_scratch[:n].copy_(pick[:rows].reshape(-1))          # contiguous view -> int64 scratch, no alloc
        self.hist[self.layer, self.chunk].index_add_(0, self.idx_scratch[:n], self.ones[:n])

    def terminal(self):
        """Record the terminal event; first bound a cut an exception left open and balance the NVTX stack."""

        if not self.armed:
            return None
        if self.open_span >= 0:
            self.events[self.open_span + 1].record()
            self.open_span = -1
            self.notes.append("a device cut was left open by an exception inside the timed region; its span ends at the terminal")
        left = self._close_open_ranges()
        if left:
            self.notes.append(f"{left} NVTX range(s) left open were popped at the terminal")
        self.terminal_event.record()
        return self.terminal_event

    # ---- after the admission -----------------------------------------------------------------------------------

    def resolve(self) -> dict | None:
        """Wait for the terminal event, turn spans into records and the summary, disarm."""

        if not self.armed:
            return None
        self.terminal_event.synchronize()
        wall_ms = (time.perf_counter() - self._t_arm) * 1000.0
        req = self.meta.get("request")
        records = []
        for k in range(self.n // 2):
            phase, layer, chunk, block, rows, pos, i = self.span_meta[k]
            records.append({"kind": "span", "request": req, "phase": phase, "chunk": chunk, "layer": layer,
                            "block": block, "rows": rows, "pos": pos,
                            "ms": self.events[i].elapsed_time(self.events[i + 1]), "clock": "device"})
        for k in range(self.nh):
            phase, layer, chunk, block, rows, pos, ms = self.host_meta[k]
            records.append({"kind": "span", "request": req, "phase": phase, "chunk": chunk, "layer": layer,
                            "block": block, "rows": rows, "pos": pos, "ms": ms, "clock": "host"})
        self.records = records
        if self.overflow:
            self.notes.append(f"event pool exhausted after {self.n // 2} spans; later cuts were not recorded")
        if self.hist_overflow:
            self.notes.append("histogram: chunks beyond the preallocated slots were not counted")
        if self.nvtx_push_failures or self.nvtx_pop_failures:
            self.notes.append(f"nvtx: {self.nvtx_push_failures} range push failure(s), {self.nvtx_pop_failures} pop "
                              "failure(s); range attribution for this admission is unreliable")
        summary = self._summary(records, wall_ms)
        self.armed = False
        return summary

    def _summary(self, records: list[dict], wall_ms: float) -> dict:
        device: dict[str, dict[str, float]] = {p: {} for p in PHASES}
        host: dict[str, float] = {}
        for r in records:
            if r["clock"] == "device":
                d = device[r["phase"]]
                d[r["block"]] = d.get(r["block"], 0.0) + r["ms"]
            else:
                host[r["block"]] = host.get(r["block"], 0.0) + r["ms"]
        return {"kind": "summary", "meta": self.meta, "wall_ms": wall_ms, "device_ms": device,
                "device_total_ms": {p: sum(v.values()) for p, v in device.items()}, "host_ms": host,
                "growth": self._growth(records), "histogram": self._histogram() if self.want_hist else None,
                "chunk_rows": list(self.chunk_rows), "spans": self.n // 2, "overflow": self.overflow,
                "nvtx_failures": {"push": self.nvtx_push_failures, "pop": self.nvtx_pop_failures},
                "notes": list(self.notes)}

    def _growth(self, records: list[dict]) -> dict:
        """ms per recorded row in the first vs last quarter of the prompt (by pos), per phase and block; a cut
        recorded with sub-block rows contributes those rows, so repeated sub-blocks do not inflate the denominator."""

        out: dict[str, dict[str, dict]] = {}
        dev = [r for r in records if r["clock"] == "device" and r["rows"] > 0]
        if not dev:
            return out
        end = max(r["pos"] + r["rows"] for r in dev)
        q = end / 4.0
        for phase in {r["phase"] for r in dev}:
            for block in {r["block"] for r in dev if r["phase"] == phase}:
                rs = [r for r in dev if r["phase"] == phase and r["block"] == block]
                first = [r for r in rs if r["pos"] + r["rows"] <= q]
                last = [r for r in rs if r["pos"] >= end - q]
                if not first or not last:
                    continue
                f = sum(r["ms"] for r in first) / sum(r["rows"] for r in first)
                l = sum(r["ms"] for r in last) / sum(r["rows"] for r in last)
                out.setdefault(phase, {})[block] = {"first_quarter_ms_per_row": f, "last_quarter_ms_per_row": l,
                                                    "ratio": (l / f) if f else None}
        return out

    def _histogram(self) -> dict:
        h = self.hist[:, :, :-1].cpu()                        # drop the shared expert's bin
        layers = []
        for li in range(h.shape[0]):
            chunks = []
            for ci in range(h.shape[1]):
                counts = [int(c) for c in h[li, ci].tolist() if c > 0]
                chunks.append({"rows": self.chunk_rows[ci], "touched": len(counts),
                               "rows_per_touched_expert": ({"median": statistics.median(counts),
                                                            "p10": percentile(counts, 0.10),
                                                            "p90": percentile(counts, 0.90)} if counts else None)})
            layers.append({"chunks": chunks})
        return {"layers": layers}

    def dump(self, summary: dict, path: str | None) -> None:
        if not path:
            return
        with open(path, "a") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps(summary) + "\n")


TIMER = Recorder()
