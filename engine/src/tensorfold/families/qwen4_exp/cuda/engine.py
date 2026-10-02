"""The Flash Next CUDA engine: MTP chains verified exactly on one GPU or two ranks in lockstep."""

from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from tensorfold.cuda.load_timing import PHASES

from . import CONFIDENCE, DEPTH
from .prefix import Kept, Match, exact_hit, match

MAX_DEPTH = 15           # a verify window of at most 16 rows
# the image tower's workspace with --vision: measured peaks 0.76 GiB (a 4,096-token image) and 0.83 GiB (a
# 256-frame video, encoded 16,384 patches at a time) over its 0.85 GiB of weights. TENSORFOLD_VISION_WORKSPACE_MIB
# sets what startup reserves for it; 0 draws it, for the moment of an encode, from the system reserve (the decoder
# returns it right after the image prompt's prefill)
VISION_WORKSPACE = 5 * 2**28


def vision_workspace() -> int:
    import os

    value = os.environ.get("TENSORFOLD_VISION_WORKSPACE_MIB")
    if value is None or value == "":
        return VISION_WORKSPACE
    if not value.isdecimal() or int(value) > 16384:
        raise ValueError(f"TENSORFOLD_VISION_WORKSPACE_MIB: 0 to 16,384 MiB, not {value!r}")
    return int(value) * 2**20


KEEP = 8                 # prompt ends a concurrent decoder keeps to resume from
CARRIED = ("received_at", "queued_at", "admitted_at", "first_token_at", "ttft_s", "profiler_rc", "timing", "notes")


class FlashNextEngine:
    """``eos``, ``generate`` (rank 0 or one GPU) and ``follow`` (rank 1), as ``tensorfold.cuda.server`` expects."""

    def __init__(self, model_dir: Path, *, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 draft_vocab: str | int | None = "default", max_len: int | None = None,
                 context_explicit: bool | None = None, tp: int = 1, rank: int = 0, master: str = "", port: int = 29551,
                 prefetch: bool = True, graphs: bool = True, streams: int = 1, ple_on_ssd: bool = False,
                 kv_dtype: str = "bf16", packed_cache: str | None = None, vision: bool = False,
                 vision_urls: bool = False, prefix_checkpoints: int = 0) -> None:
        from .nvfp4 import estimate_transform, is_mixed

        if tp == 2 and is_mixed(model_dir):
            raise ValueError("the NVFP4 mixed checkpoint runs on one GPU: start it without --tp 2")
        import torch

        from .exl3_pack import admission, extra_files, is_exl3

        exl3 = is_exl3(model_dir)
        if exl3 and tp != 1:
            raise ValueError("EXL3 packs of Flash Next run on one GPU: drop --tp 2, or serve the MLX checkpoint "
                             "(Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) on two")
        if vision and (exl3 or streams < 2 or tp != 1):
            raise ValueError("image input on Flash Next runs on one GPU with --parallel 2 or more, from the MLX "
                             "checkpoint (Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) or the NVFP4 mixed directory "
                             "that links it")
        if exl3 and ple_on_ssd:
            raise ValueError("--ple-on-ssd reads the MLX checkpoint's n-gram tables; an EXL3 pack maps its own table "
                             "from its file, so drop --ple-on-ssd")
        from .decode import Engine
        from .kvcache import BITS_OF, check as check_kv
        from .weights import draft_token_ids, load
        from tensorfold.cuda.capacity import admit, gather_ints
        from tensorfold.cuda.geometry import (PREFILL_ROWS, gdn_geometry, indexed_prefill_rows,
                                              indexed_stream_geometry, indexed_weights)

        if tp not in (1, 2) or rank not in range(tp):
            raise ValueError(f"rank {rank} of {tp}: Flash Next runs on one GPU or two")
        if streams > 1 and tp > 1:
            raise ValueError("--parallel decodes several Flash Next requests together on one GPU; with --tp 2 it "
                             "serves one request at a time for now, so drop --parallel")
        if not 0 <= int(depth) <= MAX_DEPTH:
            raise ValueError(f"MTP drafts a round: 0 to {MAX_DEPTH}, not {depth}")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"MTP draft confidence: a probability from 0 to 1, not {confidence}")
        if prefix_checkpoints < 0:
            raise ValueError(f"--prefix-checkpoints is a count of 0 or more, not {prefix_checkpoints}")
        # F2d stage B runs on the scheduler path (--parallel 2+); one stream keeps stage A's entries without them
        self.prefix_checkpoints = int(prefix_checkpoints) if streams > 1 else 0
        torch.cuda.set_device(0)
        self.tp, self.rank, self.depth, self.confidence = tp, rank, int(depth), float(confidence)
        self.kv_dtype = check_kv(kv_dtype)
        self.comm = None
        self.vision = None                   # the image tower (``QwenCudaVision``) with --vision
        ids = draft_token_ids(draft_vocab) if self.depth > 0 else None
        if tp == 2:
            from tensorfold.cuda.comm import NCCL

            if not master:
                raise ValueError("two ranks need rank 0's address (master)")
            self.comm = NCCL(rank, 2, master, port)
            self.comm.barrier()
        gather = (lambda values: gather_ints(torch, self.comm.all_gather, values)) if tp == 2 else None
        each, mtp, bits = self.depth + 1, self.depth > 0, BITS_OF[self.kv_dtype]
        # prompt chunk rows (TENSORFOLD_PREFILL_ROWS; 4,096, or 2,048 with --vision); an EXL3 pack's n-gram staging
        # holds 2,048. A row's bits never depend on its chunk: this moves speed and the buffers' memory only
        rows = PREFILL_ROWS if exl3 else indexed_prefill_rows(vision)
        self.prefill_rows = rows
        # one admission for one stream or many (every slot, the shared rows and kept snapshots), before any load
        points = self.prefix_checkpoints
        geometry = ((lambda text: indexed_stream_geometry(text, streams, each, KEEP, mtp=mtp, kv_bits=bits,
                                                          prefill_rows=rows, checkpoints=points))
                    if streams > 1 else
                    (lambda text: gdn_geometry(text, tp, each, indexed=True, mtp=mtp, kv_bits=bits,
                                               prefill_rows=rows)))
        if exl3:
            geometry = admission(geometry)
        transform = indexed_weights(tp, mtp, mapped_tables=not ple_on_ssd)
        release_files: tuple = ()
        if is_mixed(model_dir):
            from .nvfp4 import release_expert_files

            transform = estimate_transform(transform)
            release_files = release_expert_files(model_dir)   # the release layout keeps the experts outside the index
        # --vision: the tower's bf16 weights and its workspace join the one admission (unchanged without it)
        from tensorfold.vision.qwen_cuda import capacity_geometry, weight_transform as vision_weights

        geometry = capacity_geometry(geometry, model_dir, vision, rank, vision_workspace())
        transform = vision_weights(transform, vision, rank)
        self.capacity_plan = admit(model_dir, max_len, context_explicit, torch, geometry,
                                   transform, rank=rank, world=tp,
                                   gather=gather,
                                   extra_files=(extra_files(model_dir) if exl3 else ()) + release_files)
        self.max_len = self.capacity_plan["cache_slots"]
        if tp == 2:
            self._same_settings(torch, ids)
        w = load(model_dir, mtp=self.depth > 0, tp=(rank, 2) if tp == 2 else None,
                 draft_vocab=draft_vocab if self.depth > 0 else None, ple_on_ssd=ple_on_ssd,
                 packed_cache=packed_cache)
        w.comm = self.comm
        if self.depth > 0 and w.mtp is None:
            raise ValueError("this checkpoint has no MTP head, which Flash Next's CUDA engine drafts with: use one "
                             "that has it, or --no-drafts for the serial reference (one token a round)")
        self.w = w
        if vision:
            from tensorfold.vision.qwen_cuda import QwenCudaVision

            self.vision = QwenCudaVision(model_dir, torch.device("cuda", 0),
                                         allow_urls=vision_urls)
            torch.cuda.empty_cache()
            print(f"[octojet] vision: image{' and video' if self.vision.videos else ''} input, a "
                  f"{self.vision.weight_bytes / 2**30:.2f} GiB tower with {vision_workspace() / 2**30:.2f} GiB of "
                  f"workspace reserved{'; https URLs allowed' if vision_urls else ''}", flush=True)
        # ``streams`` > 1: up to that many requests decoded together, every stream's chain in one forward
        self.concurrent = streams > 1
        self.multi = self.scheduler = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder
            from .prefix import message_start_id

            self.e = None
            self.multi = MultiDecoder(w, slots=streams, capacity=self.max_len, depth=self.depth,
                                      confidence=self.confidence, keep=KEEP, kv_dtype=self.kv_dtype,
                                      prefill_rows=rows, vision=self.vision,
                                      prefix_checkpoints=self.prefix_checkpoints,
                                      turn_marker=message_start_id(model_dir))
            self.scheduler = Scheduler(self.multi, max_streams=streams)
        else:
            self.e = Engine(w, capacity=self.max_len, max_rows=max(8, self.depth + 1), prefill_rows=rows,
                            graphs=graphs, kv_dtype=self.kv_dtype)
        from tensorfold.cuda.prefill_timing import TIMER
        TIMER.configure(layers=w.cfg.layers, attention_layers=sum(t == "attention" for t in w.cfg.layer_types),
                        prefill_rows=rows, capacity=self.max_len, experts=w.cfg.experts, top_k=w.cfg.top_k,
                        slots=w.cfg.top_k + 1, device="cuda")
        started = time.perf_counter()
        locked = False
        if prefetch and not ple_on_ssd:               # the n-gram tables' pages, read now rather than by requests
            with PHASES.phase("ngram_prefetch"):
                tables = {id(layer.ple.table): layer.ple.table for layer in w.layers if layer.ple is not None}
                size = sum(a.nbytes for t in tables.values() for a in t.words + t.scales + t.biases)
                # pinned pages are no longer reclaimable: lock only what the startup budget leaves room for
                room = self.capacity_plan["budget_bytes"] - self.capacity_plan["total_bytes_estimate"]
                for table in tables.values():
                    locked = room >= size and table.lock()
                    if not locked:
                        table.prefetch()
        if PHASES.enabled:
            print(f"[octojet] load phases (engine): ngram_prefetch {PHASES.seconds.get('ngram_prefetch', 0.0):.1f} s",
                  flush=True)
        read_s = time.perf_counter() - started
        captured = self.e.graphs.warm(self.depth + 1) if self.e is not None and self.e.graphs is not None else 0
        started = time.perf_counter()
        if self.concurrent:
            self.multi.warm()
        else:
            from .decode import warm

            warm(self.e)
        if self.vision is not None:          # the tower's kernels load now, not on top of the first image request
            self.vision.warm()
            torch.cuda.empty_cache()
        warm_s = time.perf_counter() - started
        self.eos = tuple(w.cfg.eos)
        self.served = 0
        self.cache: list[Kept] = []                      # kept prompt states (spec section 3)
        self.next_serial = 0
        self.serial = None                                # the serial requests' engine, made on first use
        rule = (f"1 to {self.depth} MTP drafts a round, a chain stops before a later draft under "
                f"{self.confidence:.0%}" if self.depth else "no drafts: the serial reference, one token a round")
        where = (f"{streams} streams of {self.context_window} prompt/reply tokens "
                 f"({self.multi.slot_bytes / 2**20:.0f} MiB a stream, {self.prefix_checkpoints} prefix checkpoints "
                 f"each), eager" if self.concurrent else
                 f"{self.context_window}-token prompt/reply window; {self.max_len}-token cache")
        how = ("read from SSD at each lookup" if ple_on_ssd else
               f"{'locked in memory' if locked else 'read'} in {read_s:.1f}s")
        kv = "" if self.kv_dtype == "bf16" else f"; {self.kv_dtype} KV cache (fp16 scale per 32 values)"
        print(f"[octojet] Flash Next on CUDA: {rule}; {where}{kv}; n-gram tables {how}; {captured} "
              f"decode graphs captured; prompt kernels warmed in {warm_s:.1f}s", flush=True)

    def _same_settings(self, torch, ids) -> None:
        """Both ranks must decode with the same rule, context, draft vocabulary and KV cache, or they would fall out of step: refuse to start otherwise."""

        from .kvcache import BITS_OF

        total = int(ids.sum()) if ids is not None else -1
        mine = torch.tensor([self.depth, round(self.confidence * 1e6), self.max_len,
                             len(ids) if ids is not None else -1, total, BITS_OF[self.kv_dtype], int(getattr(self, "prefill_rows", 0))],
                            dtype=torch.int64, device="cuda")
        both = torch.empty((2 * mine.numel(),), dtype=torch.int64, device="cuda")
        self.comm.all_gather(mine, both)
        both = both.view(2, -1).cpu()
        if not torch.equal(both[0], both[1]):
            raise RuntimeError(f"the two ranks were started with different settings (drafts, confidence, context, "
                               f"draft vocabulary, KV cache, prompt chunk rows): rank 0 {both[0].tolist()}, rank 1 {both[1].tolist()}")

    def _key(self, n: int) -> str:
        return f"tensorfold/flashnext/request/{n}"

    def shutdown(self) -> None:
        """Rank 0: tell rank 1 to leave ``follow``."""

        if self.tp == 2 and self.rank == 0:
            self.comm.store.set(self._key(self.served), json.dumps({"stop": True}))

    def _share(self, prompt: list[int], max_tokens: int, sampling, draft: bool, cached: int, kind=None, serial=None) -> tuple:
        body = {"prompt": prompt, "max_tokens": max_tokens, "draft": bool(draft), "cached": int(cached), "kind": kind,
                "serial": serial,
                "sampling": None if sampling is None else [int(sampling.seed), float(sampling.temperature),
                                                           int(sampling.top_k), float(sampling.top_p)]}
        text = json.dumps(body)
        self.comm.store.set(self._key(self.served), text)
        return self._unpack(text)

    def _receive(self) -> tuple | None:
        from torch.distributed import DistNetworkError

        key = self._key(self.served)
        while True:
            try:
                self.comm.store.wait([key], timedelta(hours=1))
                break
            except DistNetworkError:                        # rank 0 is gone: leave ``follow``
                print("[octojet] rank 0 closed the connection; rank 1 stops", flush=True)
                return None
            except Exception:                               # noqa: BLE001  (no request within the hour: wait on)
                continue
        text = self.comm.store.get(key).decode()
        self.comm.store.delete_key(key)
        return self._unpack(text)

    @staticmethod
    def _unpack(text: str) -> tuple | None:
        from tensorfold.engine.exact_sampling import Sampling

        body = json.loads(text)
        if body.get("stop"):
            return None
        s = body["sampling"]
        return (body["prompt"], body["max_tokens"], None if s is None else Sampling(s[0], s[1], s[2], s[3]),
                body["draft"], body["cached"], body.get("kind"), body.get("serial"))

    @property
    def vocab_size(self) -> int:
        """Token ids the embedding takes (0 to vocab_size - 1); the server range-checks ``prompt_ids`` against it."""

        return int(self.w.cfg.vocab)

    @property
    def context_window(self) -> int:
        """Prompt and reply capacity after reserving speculative scratch positions."""

        return max(0, self.max_len - self.depth - 1)

    def _limit(self, prompt: list[int], max_tokens: int) -> int:
        room = self.max_len - len(prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in the {self.max_len}-token context")
        return max(1, min(max_tokens, room))

    def _resume(self, prompt: list[int]) -> Match | None:
        """The kept entry the prompt reuses most tokens from (spec section 3); nothing is busy on this path."""

        return match(prompt, self.cache, lambda k: True)[0]

    def _start_from(self, cached: int, prompt: list[int]) -> None:
        """Before decoding or prefilling from ``cached``: every entry with rows above ``cached`` on this prompt's
        prefix chain is overwritten next, so it goes (a fresh prefill, ``cached == 0``, drops them all)."""

        self.cache = [k for k in self.cache if not (len(k.ids) > cached and k.ids[:cached] == prompt[:cached])]

    def _remember(self, ids: list[int], snapshot: dict, tail, logits) -> Kept:
        entry = Kept(list(ids), snapshot, tail, logits, [], serial=self.next_serial)
        self.next_serial += 1
        self.cache = [k for k in self.cache if k.ids != ids][-1:] + [entry]
        return entry

    def _resolve_shared(self, prompt: list[int], draft: bool, cached: int, kind, serial) -> Match | None:
        """Rank 1: the kept entry rank 0 reused, by serial; refuse anything that does not match rank 0's decision
        (spec section 5.4): a missing or differently matching entry, a cold announcement carrying cached tokens or a
        serial, a serial request announcing reuse."""

        if not draft:
            if kind is not None or serial is not None or cached:
                raise RuntimeError(f"rank 1 received a serial request announcing reuse ({kind}, {cached} tokens, serial {serial})")
            return None
        if kind is None:
            if cached or serial is not None:
                raise RuntimeError(f"rank 1 received a cold request announcing {cached} cached tokens (serial {serial})")
            return None
        k = next((k for k in self.cache if k.serial == serial), None)
        m = match(prompt, [k], lambda e: True)[0] if k is not None else None
        if m is None or m.kind != kind or m.cached != cached:
            raise RuntimeError(f"rank 1 has no kept state for rank 0's {kind} reuse of {cached} tokens (serial {serial})")
        return m

    def _serial(self, prompt: list[int], max_tokens: int, sampling, on_tokens, carrier=None) -> dict[str, Any]:
        """One token a round from a fresh prefill in the serial engine's own state (no drafts, no kept states)."""

        import torch

        from .decode import prefill, serial_decode

        if self.serial is None:
            self.serial = self.e.twin()
        t0 = time.perf_counter()

        def admit() -> int:
            first = prefill(self.serial, prompt, sampling, mtp=False)
            torch.cuda.synchronize()
            return first

        first, carried = _admission(admit, carrier or self._carrier(prompt, max_tokens, sampling, False))
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": 0, "reuse": None,
                                 "reuse_miss": None, "drafts": False, **carried}
        if (on_tokens is not None and on_tokens([first])) or first in self.eos or max_tokens <= 1:
            return stats
        res = serial_decode(self.serial, first, max_tokens, sampling, stop_eos=True, on_tokens=on_tokens)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def _decode(self, prompt: list[int], max_tokens: int, sampling, on_tokens, hit: Match | None,
                carrier=None) -> dict[str, Any]:
        import torch

        from .decode import mtp_decode, prefill, serial_decode

        t0 = time.perf_counter()

        def admit() -> int:
            self._start_from(hit.cached if hit is not None else 0, prompt)
            if hit is not None and hit.kind == "exact":
                first = exact_hit(self.e, hit.entry, sampling)
            else:
                first = prefill(self.e, prompt, sampling, resume=hit.resume if hit is not None else None)
                # the prompt's state: the MTP head has absorbed every position but the last, whose streams resume needs
                self._remember(list(prompt), self.e.st.snapshot(),
                               self.e.last_streams.clone() if self.e.mbuf is not None else None, self.e.last_logits)
            torch.cuda.synchronize()
            return first

        try:
            first, carried = _admission(admit, carrier or self._carrier(prompt, max_tokens, sampling, True))
            stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4),
                                     "cached": hit.cached if hit is not None else 0,
                                     "reuse": hit.kind if hit is not None else None, "reuse_miss": None,
                                     "drafts": True, **carried}
            if (on_tokens is not None and on_tokens([first])) or first in self.eos or max_tokens <= 1:
                return stats
            if self.depth > 0:
                res = mtp_decode(self.e, first, max_tokens, sampling, depth=self.depth, confidence=self.confidence,
                                 stop_eos=True, on_tokens=on_tokens)
                stats.update(drafted=res.drafted, accepted=res.accepted, min_rows=min(res.widths, default=0))
            else:
                res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=True, on_tokens=on_tokens)
        except Exception as exc:
            if hit is not None:
                print(f"[octojet] prefix reuse {hit.kind} failed: {exc}", file=sys.stderr, flush=True)
            self.cache = []                    # no partial state is reusable
            raise
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    @staticmethod
    def _carrier(prompt: list[int], max_tokens: int, sampling, draft: bool, timing: bool = False,
                 profile: bool = False, histogram: bool = False, received_at: float = 0.0, admitted_at: float = 0.0):
        """The single-stream path's ``Stream``: it carries the request's flags and timestamps through the admission.
        Nothing queues on this path: ``queued_at`` is ``admitted_at``, the ``generate()`` entry (perf_counter, the
        server's clock)."""

        from tensorfold.cuda.streams import Stream

        now = time.perf_counter()
        admitted_at = admitted_at or now
        return Stream(list(prompt), max(1, max_tokens), sampling, draft=draft, timing=timing, profile=profile,
                      histogram=histogram, received_at=received_at or now, queued_at=admitted_at,
                      admitted_at=admitted_at)

    def generate(self, prompt: list[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None], draft: bool = True, timing: bool = False,
                 profile: bool = False, histogram: bool = False, received_at: float = 0.0, *,
                 vision=None) -> dict[str, Any]:
        """``draft=False``: one token a round with no MTP drafts, from a fresh prefill that leaves the kept states alone: the serial reference; ``vision``: an image prompt's prepared pixels and positions (the scheduler path only)."""

        entered = time.perf_counter()                 # the single-stream path's admitted_at (and queued_at)
        if timing or histogram:                       # every recorder refusal comes before rank 1 is handed the request
            from tensorfold.cuda.prefill_timing import ENV, TIMER, TimingBusy

            if not TIMER.enabled:
                raise TimingBusy(f"prefill timing is not configured: start the engine with {ENV}=1")
            if TIMER.armed:
                raise TimingBusy("prefill timing is busy with another request")
        max_tokens = self._limit(prompt, max_tokens)
        if vision is not None and (self.vision is None or self.scheduler is None):
            from tensorfold.server.errors import RequestError

            raise RequestError("image inputs require starting this server with --vision")
        if self.scheduler is not None:
            if vision is not None:                    # an image prompt: never matched against or kept for reuse
                return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, timing=timing,
                                             profile=profile, histogram=histogram, received_at=received_at,
                                             vision=vision)
            return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, timing=timing,
                                         profile=profile, histogram=histogram, received_at=received_at)
        hit = self._resume(prompt) if draft else None
        if self.tp == 2:                     # rank 0 decodes exactly what it hands rank 1
            prompt, max_tokens, sampling, draft, _, _, _ = self._share(
                prompt, max_tokens, sampling, draft, hit.cached if hit is not None else 0,
                hit.kind if hit is not None else None, hit.entry.serial if hit is not None else None)
            self.served += 1
            emit = on_tokens
            on_tokens = lambda new: (emit(new), False)[1]       # noqa: E731  both ranks decode to the end
        carrier = self._carrier(prompt, max_tokens, sampling, draft, timing, profile, histogram, received_at, entered)
        if not draft:
            return self._serial(prompt, max_tokens, sampling, on_tokens, carrier)
        return self._decode(prompt, max_tokens, sampling, on_tokens, hit, carrier)

    def follow(self) -> None:
        """Rank 1: decode every request rank 0 serves, until rank 0 stops."""

        while True:
            request = self._receive()
            if request is None:
                return
            prompt, max_tokens, sampling, draft, cached, kind, serial = request
            self.served += 1
            hit = self._resolve_shared(prompt, draft, cached, kind, serial)
            try:
                if draft:
                    self._decode(prompt, max_tokens, sampling, None, hit)
                else:
                    self._serial(prompt, max_tokens, sampling, None)
            except ValueError as exc:                       # rank 0 raised at the same point on the same input
                print(f"[octojet] request {self.served} failed on both ranks: {exc}", flush=True)


class _Admit:
    """The single-stream path's ``decoder`` for ``run_admission``: ``admit(carrier)`` runs ``prefill_fn`` (the prefill
    and whatever belongs to the admission) and hands its first token to the carrier; the caller reads ``first``."""

    def __init__(self, prefill_fn: Callable[[], int]) -> None:
        self.prefill_fn = prefill_fn
        self.first: int | None = None

    def admit(self, carrier) -> None:
        # the first MTP draft runs inside mtp_decode, after the admission: this path records no "draft" phase
        self.first = self.prefill_fn()
        carrier.take([self.first])          # no emit: the caller's on_tokens runs after the admission, as before


def _admission(prefill_fn: Callable[[], int], carrier) -> tuple[int, dict[str, Any]]:
    """The prefill inside the scheduler's admission lifecycle; its first token and the request-timing stats."""

    from tensorfold.cuda.scheduler import run_admission

    adapter = _Admit(prefill_fn)
    run_admission(adapter, carrier)                  # keeps the carrier's admitted_at (the generate() entry)
    full = carrier.stats()
    return adapter.first, {k: full[k] for k in CARRIED if k in full}

