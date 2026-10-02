"""Phase 1 instrumentation on the cut mixed model: armed and unarmed prefills are bit-identical in target logits, first
token, state (recurrence, conv, PLE tail/history, KV codes and scales over [:pos], indexer keys over [:pos], completed
pools, MTP caches over [:mtp_len]) and in the first drafts; the summary covers the blocks the model exercises; the
histogram counts rows x (top_k + 1) per layer per chunk; the final chunk's transients (routing picks and weights, the
indexer's key counts, sparse flags, selected ids and block scores) match over their valid regions; the pool overflow path; QSA active (capacity > 2048); int8 KV;
resume equals fresh under arming."""

import gc
import os
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

MODEL = os.environ.get("OCTOJET_NVFP4_FLASHNEXT", "")
LAYERS = int(os.environ.get("OCTOJET_NVFP4_FLASHNEXT_LAYERS", "4"))
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set OCTOJET_NVFP4_FLASHNEXT")
CAP = 20000                                       # QSA on (budget 2048); room for 8192-row chunks with a tail


def bits_equal(a, b):
    """Bit-for-bit equality (torch.equal calls -0.0 and 0.0 equal, so floats compare as their integer bits)."""

    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype in (torch.bfloat16, torch.float16):
        a, b = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    elif a.dtype == torch.float32:
        a, b = a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    return torch.equal(a.contiguous(), b.contiguous())


@pytest.fixture(scope="module")
def cut():
    from tensorfold.families.qwen4_exp.cuda import weights as W
    real = W.Config.read
    def read(d):
        c = real(d); c.layers = LAYERS; c.ple_layers = [i for i in c.ple_layers if i < LAYERS]; return c
    W.Config.read = staticmethod(read)
    try:
        w = W.load(MODEL, "cuda", mtp=True, draft_vocab="default", packed_cache="off")
    finally:
        W.Config.read = real
    yield w
    del w; torch.cuda.empty_cache()


def state_bytes(st, ratio):
    """Clones of the committed state over its valid regions, and its host scalars."""

    p = st.cur[0] if st.cur else 0
    parts = [st.rec[p].clone(), st.conv.clone(), st.ple_tail.clone()]
    for kc in st.kc:                                  # bf16 caches keep one-element scales: [:pos] is the whole (1,)
        for name in ("k", "v", "ks", "vs"):
            parts.append(getattr(kc, name)[:st.pos].clone())
    for t in st.ikc:
        parts.append(t[:st.pos].clone())
    for t in st.pooled:
        parts.append(t[:st.pos // ratio].clone())     # completed pools only
    if hasattr(st, "mtp_kc"):
        for name in ("k", "v", "ks", "vs"):
            parts.append(getattr(st.mtp_kc, name)[:st.mtp_len].clone())
        parts.append(st.mtp_ikc[:st.mtp_len].clone())
        parts.append(st.mtp_pooled[:st.mtp_len // ratio].clone())
    parts.append(st.mtp_pos.clone())
    parts.append(st.pos_dev.clone())
    return parts, (st.pos, st.mtp_len, st.mtp_drafted, list(st.cur),
                   None if st.ple_history is None else np.array(st.ple_history).tobytes())


def transients(e, w, prompt_len, resume_at):
    """Clones of the prefill scratch the final chunk leaves behind, over its valid regions only, taken right after
    prefill() returns (the next forward overwrites them).

    The last forward of prefill() on the final chunk is the prompt-MTP head (decode.prefill: ``mtp_forward(w, st, pb,
    nxt, ...)`` with ``nxt = prompt[start + 1:]``, R - 1 rows, an attention layer with its own MoE) when it has rows,
    else the main model's last layer (R rows). Both write the shared ``e.pbuf``: ``pbuf.moe.pick``/``wts`` [rows,
    top_k + 1] (moe._topk_rows stores every slot, the shared expert's included) and ``pbuf.attn`` for the last
    ATT_ROWS-row block (forward.attn_block -> attention.qsa_rows: ``_select`` stores nk and sparse for every row, the
    ids [:nk] of sparse rows only; ``_scores`` stores scores [:complete] of sparse rows only). The block's first
    position P is checked against what ``_select`` wrote (nk and the sparse flag of every row) before any region is
    sliced. Chunk-layout dependent: compare only runs with the same prefill_rows and resume point."""

    from tensorfold.families.qwen4_exp.cuda.state import ATT_ROWS
    rows = e.prefill_rows
    begin = 0 if resume_at is None else resume_at
    start = max(begin, (prompt_len - 1) // rows * rows)          # the final chunk's first position (aligned ends, F2d 5.1)
    R = prompt_len - start
    mtp_last = w.mtp is not None and e.mbuf is not None and R > 1
    n = R - 1 if mtp_last else R
    pb, top_k = e.pbuf, w.cfg.top_k
    assert pb.moe.slots == top_k + 1
    out = [pb.moe.pick[:n].clone(), pb.moe.wts[:n].clone()]
    if not mtp_last and "attention" not in w.cfg.layer_types[:LAYERS]:
        return out                                             # no attention ran on the final chunk
    a = pb.attn
    assert a.qsa                                             # CAP > budget: the indexer selects on every block
    r0 = (n - 1) // ATT_ROWS * ATT_ROWS
    m, P = n - r0, start + r0                                # the last block's rows and first position
    nk, sp = a.nk[:m].clone(), a.sparse[:m].clone()
    out += [nk, sp]
    top, ratio = a.budget // a.ratio, a.ratio
    for r, (k, s) in enumerate(zip(nk.tolist(), sp.tolist())):
        end = P + r + 1
        complete = end // ratio
        if complete <= top:                                  # dense: only its length is defined
            assert (s, k) == (0, end), (r, s, k, end)
        else:
            assert (s, k) == (1, top * ratio + end - ratio * complete), (r, s, k, end)
            out += [a.ids[r, :k].clone(), a.scores[r, :complete].clone()]
    return out


@torch.no_grad()                                  # as mtp_decode, which calls draft() in production
def run(w, prompt, *, armed, histogram=False, rows=2048, kv="int8", resume_at=None, drafts=6, graphs=False, sampling=None,
        confidence=0.3):
    """One fresh Engine's prefill and first drafts. Returns only clones and Python values:
    (first, logits, parts, scalars, drafts, summary, forwards, hist, transients); the Engine is released before
    returning."""

    from tensorfold.cuda import prefill_timing as pt
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, draft
    e = Engine(w, capacity=CAP, max_rows=8, prefill_rows=rows, graphs=graphs, kv_dtype=kv)
    try:
        if graphs:                                # production captures every window at startup (engine.py warm); a
            e.graphs.warm()                       # capture inside an armed admission would record events mid-capture
        e.reset()
        if armed:
            assert pt.TIMER.arm({"request": 1, "prompt_tokens": len(prompt)}, histogram=histogram)
        if resume_at is None:
            first = prefill(e, prompt, sampling, mtp=True)
        else:
            prefill(e, prompt[:resume_at], sampling, mtp=True)
            kept = {"state": e.st.snapshot(), "tail": e.last_streams.clone()}
            e.reset()
            first = prefill(e, prompt, sampling, mtp=True, resume=kept)
        logits = e.last_logits.clone()
        streams = e.last_streams.clone()
        trans = transients(e, w, len(prompt), resume_at)          # before draft(): the next forward overwrites them
        d, draft_logits = draft_with_spy(e, draft, first, drafts, sampling, confidence) if drafts else ([], [])
        summary = hist = None
        if armed:
            ev = pt.TIMER.terminal(); ev.synchronize(); summary = pt.TIMER.resolve()
            assert pt.TIMER.open_ranges == 0 and not pt.TIMER.armed
            if histogram:
                hist = pt.TIMER.hist.clone()     # raw bins, the shared slot's included, read before any reset
        parts, scalars = state_bytes(e.st, w.cfg.index_ratio)
        parts.append(streams)
        parts.extend(draft_logits)               # one tensor per MTP forward: absorb + each chained draft
        return first, logits, parts, scalars, list(d), summary, len(draft_logits), hist, trans
    except BaseException as exc:
        # the finished frames below this one (draft, mtp_forward, the kernels' callers) still hold the Engine through
        # the propagating traceback; clear their locals so the collection below frees it (the traceback text stays)
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        e = kept = None                          # the last references; Engine/Graphs form a cycle: collect it now
        gc.collect()
        torch.cuda.empty_cache()


def draft_with_spy(e, draft, first, count, sampling, confidence):
    """draft() with a spy on ``e.mtp_forward`` that clones every MTP forward's returned logits view (the valid
    [k, head.n] rows) before the next forward overwrites the shared buffer. Returns (drafts, clones) only: the
    override is removed and every local that reaches the Engine (the bound method, the spy, ``e``) is dropped in the
    ``finally``, so nothing here keeps the Engine alive, not even a traceback holding this frame."""

    captured: list[torch.Tensor] = []
    forward = e.mtp_forward
    def spy(tokens, streams):
        out = forward(tokens, streams)
        captured.append(out.clone())
        return out
    e.mtp_forward = spy
    try:
        return list(draft(e, e.last_streams, [first], e.st.pos + 1, count, sampling, confidence)), captured
    finally:
        del e.mtp_forward                        # the instance override (Engine -> spy -> bound method -> Engine)
        forward = spy = e = None                 # the closure cell and this frame's locals


def teardown(pt):
    """Pop any NVTX range a failed armed run left open, then forget the recorder; never masks the test's exception."""

    pt.TIMER.abort("test teardown")
    pt.TIMER.__init__()


def assert_idle(pt, *, configured):
    """Nothing was recorded while the recorder was unarmed (or unconfigured)."""

    assert pt.TIMER.n == 0 and pt.TIMER.nh == 0 and pt.TIMER.open_ranges == 0
    assert pt.TIMER.records == []
    if configured:
        assert int(pt.TIMER.hist.sum()) == 0


def check_summary(s, *, prompt_len, drafts, overflow=False):
    """What every armed admission's summary must show: balanced NVTX, no failures, the phases that ran."""

    assert s is not None
    assert s["nvtx_failures"] == {"push": 0, "pop": 0}
    assert not any("left open" in n or "nvtx" in n for n in s["notes"]), s["notes"]
    assert s["overflow"] is overflow
    if overflow:
        return
    if prompt_len > 1:                           # the prompt-MTP forward runs on every chunk with next tokens
        assert s["device_ms"]["mtp"], "no mtp-phase spans"
    if drafts:
        assert "sample" in s["device_ms"]["draft"], s["device_ms"]["draft"]


def configure(cut, monkeypatch, rows=2048):
    from tensorfold.cuda import prefill_timing as pt
    monkeypatch.setenv(pt.ENV, "1")
    pt.TIMER.__init__()
    assert pt.TIMER.configure(layers=LAYERS, attention_layers=sum(t == "attention" for t in cut.cfg.layer_types[:LAYERS]),
                              prefill_rows=rows, capacity=CAP, experts=cut.cfg.experts, top_k=cut.cfg.top_k,
                              slots=cut.cfg.top_k + 1, device="cuda")
    return pt


def same(a, b, *, ignore_cur=False):
    """Bit-identical outputs; ``ignore_cur`` drops the DeltaNet buffer parity, which depends on how many chunks ran
    (a resumed prompt runs a different number of chunks than a fresh one; the recurrence itself is compared) and the
    final chunk's transients, which depend on its layout (a resumed prompt's final chunk is a different row range)."""

    f0, l0, p0, s0, d0, _, n0, _, t0 = a
    f1, l1, p1, s1, d1, _, n1, _, t1 = b
    if ignore_cur:
        s0, s1 = s0[:3] + s0[4:], s1[:3] + s1[4:]
        t0 = t1 = []
    return (n0 == n1 and f0 == f1 and bits_equal(l0, l1) and len(p0) == len(p1)
            and all(bits_equal(x, y) for x, y in zip(p0, p1)) and s0 == s1 and d0 == d1
            and len(t0) == len(t1) and all(bits_equal(x, y) for x, y in zip(t0, t1)))


@pytest.mark.parametrize("rows,size", [(8192, 17000), (4096, 9000), (2048, 4500), (512, 1100), (64, 300)])
@needs_model
def test_armed_equals_unarmed(cut, monkeypatch, rows, size):
    pt = configure(cut, monkeypatch, rows)
    try:
        prompt = [int(t) for t in np.random.default_rng(size).integers(0, 1000, size=size)]
        a = run(cut, prompt, armed=False, rows=rows)
        b = run(cut, prompt, armed=True, histogram=True, rows=rows)
        assert same(a, b)
        s = b[5]
        check_summary(s, prompt_len=size, drafts=6)
        dev = s["device_ms"]["main"]
        for block in ("embed", "hc_readout", "router", "plan", "expert_up", "expert_down", "writeback", "finish", "commit", "sample", "prefill_other"):
            assert dev.get(block, 0.0) > 0.0, block
        types = cut.cfg.layer_types[:LAYERS]
        if "attention" in types:
            for block in ("attn_prep", "idx_pool", "idx_scores", "idx_select", "attn_sparse", "attn_gate", "attn_other", "dense_attn_in", "dense_attn_out"):
                assert dev.get(block, 0.0) > 0.0, block
        if "linear" in types:
            for block in ("gdn_front", "gdn_recurrence", "gdn_back", "gdn_other", "dense_gdn_in", "dense_gdn_out"):
                assert dev.get(block, 0.0) > 0.0, block
        assert s["device_total_ms"]["mtp"] > 0 and s["device_total_ms"]["draft"] > 0
        assert "mtp_input" in s["device_ms"]["draft"] and "draft" not in s["device_ms"]["draft"]
        assert s["host_ms"]["stage_wait"] >= 0 and s["overflow"] is False
        assert s["device_total_ms"]["main"] <= s["wall_ms"]
        chunks = -(-size // rows)
        for li in range(LAYERS):
            for ci in range(chunks):
                r = min(rows, size - ci * rows)
                h, experts = b[7], cut.cfg.experts
                assert int(h[li, ci, experts]) == r                             # the shared slot's bin
                assert int(h[li, ci, :experts].sum()) == r * cut.cfg.top_k        # routed picks only
                assert s["histogram"]["layers"][li]["chunks"][ci]["rows"] == r
    finally:
        teardown(pt)


@needs_model
def test_resume_equals_fresh_under_arming(cut, monkeypatch):
    pt = configure(cut, monkeypatch, 512)
    try:
        prompt = [int(t) for t in np.random.default_rng(77).integers(0, 1000, size=1300)]
        fresh = run(cut, prompt, armed=True, rows=512)
        resumed = run(cut, prompt, armed=True, rows=512, resume_at=700)
        assert same(fresh, resumed, ignore_cur=True)
        resumed_off = run(cut, prompt, armed=False, rows=512, resume_at=700)
        assert same(resumed_off, resumed)                        # the same layout: transients included
        check_summary(fresh[5], prompt_len=len(prompt), drafts=6)
        check_summary(resumed[5], prompt_len=len(prompt), drafts=6)
    finally:
        teardown(pt)


@needs_model
def test_bf16_kv_too(cut, monkeypatch):
    pt = configure(cut, monkeypatch, 512)
    try:
        prompt = [int(t) for t in np.random.default_rng(5).integers(0, 1000, size=900)]
        a = run(cut, prompt, armed=False, rows=512, kv="bf16")
        b = run(cut, prompt, armed=True, rows=512, kv="bf16")
        assert same(a, b)
        check_summary(b[5], prompt_len=len(prompt), drafts=6)
    finally:
        teardown(pt)


@needs_model
def test_graphs_and_sampled_under_arming(cut, monkeypatch):
    from tensorfold.engine.exact_sampling import Sampling
    pt = configure(cut, monkeypatch, 512)
    try:
        prompt = [int(t) for t in np.random.default_rng(9).integers(0, 1000, size=1300)]
        smp = Sampling(1234, 1.0, 20, 0.95)
        a = run(cut, prompt, armed=False, rows=512, graphs=True)
        b = run(cut, prompt, armed=True, rows=512, graphs=True)
        assert same(a, b)
        check_summary(b[5], prompt_len=len(prompt), drafts=6)     # replayed MTP graphs record no device cuts
        a = run(cut, prompt, armed=False, rows=512, sampling=smp)
        b = run(cut, prompt, armed=True, rows=512, sampling=smp)
        assert same(a, b)
        check_summary(b[5], prompt_len=len(prompt), drafts=6)
    finally:
        teardown(pt)


@needs_model
def test_overflow_and_profiler_arming(cut, monkeypatch):
    pt = configure(cut, monkeypatch, 64)
    try:
        from tensorfold.cuda.scheduler import _profiler_start, _profiler_stop
        pt.TIMER.events = pt.TIMER.events[:40]; pt.TIMER.span_meta = pt.TIMER.span_meta[:20]
        prompt = [int(t) for t in np.random.default_rng(3).integers(0, 1000, size=100)]
        a = run(cut, prompt, armed=False, rows=64, drafts=0)
        b = run(cut, prompt, armed=True, rows=64, drafts=0)
        assert same(a, b) and b[5]["overflow"] is True and b[5]["spans"] == 20 and any("pool" in n for n in b[5]["notes"])
        check_summary(b[5], prompt_len=len(prompt), drafts=0, overflow=True)
        assert _profiler_start() == 0 and _profiler_stop() == 0
    finally:
        teardown(pt)


@needs_model
def test_forced_six_draft_chain(cut, monkeypatch):
    pt = configure(cut, monkeypatch, 512)
    try:
        prompt = [int(t) for t in np.random.default_rng(11).integers(0, 1000, size=900)]
        a = run(cut, prompt, armed=False, rows=512, confidence=0.0)
        b = run(cut, prompt, armed=True, rows=512, confidence=0.0)
        for r in (a, b):
            assert len(r[4]) == 6 and r[6] == 6                  # six drafts: the absorb and five chained forwards
        assert same(a, b)
        check_summary(b[5], prompt_len=len(prompt), drafts=6)
    finally:
        teardown(pt)


@needs_model
def test_unarmed_runs_record_nothing(cut, monkeypatch):
    from tensorfold.cuda import prefill_timing as pt
    prompt = [int(t) for t in np.random.default_rng(13).integers(0, 1000, size=1100)]
    try:
        monkeypatch.delenv(pt.ENV, raising=False)                # (a) not configured: the recorder is inert
        pt.TIMER.__init__()
        assert pt.TIMER.configure(layers=LAYERS, attention_layers=1, prefill_rows=512, capacity=CAP,
                                  experts=cut.cfg.experts, top_k=cut.cfg.top_k, slots=cut.cfg.top_k + 1,
                                  device="cuda") is False
        assert pt.TIMER.enabled is False
        off = run(cut, prompt, armed=False, rows=512)
        assert_idle(pt, configured=False)
        configure(cut, monkeypatch, 512)                         # (b) configured, never armed
        idle = run(cut, prompt, armed=False, rows=512)
        assert_idle(pt, configured=True)
        armed = run(cut, prompt, armed=True, rows=512)           # (c) armed
        assert same(off, armed) and same(idle, armed)
        check_summary(armed[5], prompt_len=len(prompt), drafts=6)
    finally:
        teardown(pt)
