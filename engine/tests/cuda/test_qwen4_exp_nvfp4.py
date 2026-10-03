"""Flash Next's NVFP4 mixed checkpoint on CUDA (``families/qwen4_exp/cuda/nvfp4.py``): routed experts from the ModelOpt
NVFP4 export, everything else from the MLX checkpoint.

TensorFold's CUDA engine has no load-time row check, so the exactness contract is established here, as the EXL3 path
does. With ``OCTOJET_NVFP4_FLASHNEXT=<a mixed served directory>`` (``tools/make_mixed_dir.py``): a model cut to
``OCTOJET_NVFP4_FLASHNEXT_LAYERS`` layers (default 4, plus the head and the MTP head) through the real loader, its
decoder experts NVFP4 and the MTP head's affine; forward windows give one-row steps' logits bit for bit, MTP-drafted
decoding emits serial decoding's tokens (eager and with CUDA graphs), and prompts ignore chunking and resume as fresh.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

DEV = "cuda"
MODEL = os.environ.get("OCTOJET_NVFP4_FLASHNEXT", "")            # the mixed served directory (Task 5's make_mixed_dir)
LAYERS = int(os.environ.get("OCTOJET_NVFP4_FLASHNEXT_LAYERS", "4"))
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(),
                                 reason="set OCTOJET_NVFP4_FLASHNEXT to a mixed NVFP4 Flash Next directory")
WINDOWS = (1, 2, 3, 16, 17, 64, 128)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-for-bit equality (torch.equal calls -0.0 and 0.0 equal)."""

    itype = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[a.dtype]
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().view(itype), b.contiguous().view(itype)))


@pytest.fixture(scope="module")
def cut_model():
    """The mixed checkpoint cut to LAYERS decoder layers plus the head and the MTP head, through the real loader."""

    from tensorfold.families.qwen4_exp.cuda import weights as W

    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = LAYERS
        c.ple_layers = [i for i in c.ple_layers if i < LAYERS]
        return c

    W.Config.read = staticmethod(cut)
    try:
        w = W.load(MODEL, DEV, mtp=True, draft_vocab="default")
    finally:
        W.Config.read = real
    assert all(layer.moe.experts.fmt == "nvfp4" for layer in w.layers), "decoder experts must be NVFP4"
    assert w.mtp is not None and w.mtp.layer.moe.experts.fmt == "affine", "the MTP head's experts stay affine"
    yield w
    del w
    torch.cuda.empty_cache()


@needs_model
def test_cut_model_windows_equal_one_row_steps(cut_model):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward

    w = cut_model
    e = Engine(w, capacity=512, max_rows=max(WINDOWS), prefill_rows=128, graphs=False)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, w.cfg.vocab, size=150)]
    e.reset()
    ref = []
    for t in toks:
        ref.append(forward(w, e.st, e.buf, [t])[:1].clone())
        commit(w, e.st, e.buf, 1, 1)
    ref = torch.cat(ref)
    for rows in WINDOWS:
        e.reset()
        got = []
        for at in range(0, len(toks), rows):
            chunk = toks[at:at + rows]
            got.append(forward(w, e.st, e.buf, chunk)[:len(chunk)].clone())
            commit(w, e.st, e.buf, len(chunk), len(chunk))
        assert bits_equal(torch.cat(got), ref), rows
    e.reset()
    forward(w, e.st, e.buf, toks[:10])
    commit(w, e.st, e.buf, 10, 4)
    assert bits_equal(forward(w, e.st, e.buf, toks[4:12])[:8], ref[4:12])


@needs_model
@pytest.mark.parametrize("graphs", [False, True])
def test_cut_model_mtp_drafts_emit_serial_tokens(cut_model, graphs):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    w = cut_model
    assert w.mtp is not None
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=graphs)
    prompt = [int(t) for t in np.random.default_rng(4).integers(0, w.cfg.vocab, size=37)]
    for sampling in (None, Sampling(seed=1234, top_k=20, top_p=0.95)):
        s = serial_decode(e, prefill(e, prompt, sampling, mtp=False), 48, sampling)
        d = mtp_decode(e, prefill(e, prompt, sampling, mtp=True), 48, sampling, depth=6)
        assert s.tokens == d.tokens, sampling


@needs_model
def test_cut_model_prompts_ignore_chunking_and_resume_as_fresh(cut_model):
    """The prompt path: state and first token do not depend on chunk size; a kept prompt end resumes as fresh."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill

    w = cut_model
    prompt = [int(t) for t in np.random.default_rng(5).integers(0, w.cfg.vocab, size=150)]
    runs = []
    for rows in (150, 64, 17):
        e = Engine(w, capacity=512, max_rows=8, prefill_rows=rows, graphs=False)
        first = prefill(e, prompt, None)
        runs.append((first, e.st.snapshot(), e.last_streams.clone()))
    for first, snap, tail in runs[1:]:
        assert first == runs[0][0]
        assert bits_equal(tail, runs[0][2])
        for key in ("rec", "conv", "ple_tail"):
            assert bits_equal(snap[key], runs[0][1][key]), key
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=64, graphs=False)
    prefill(e, prompt[:90], None)
    kept = {"state": e.st.snapshot(), "tail": e.last_streams.clone()}
    first = prefill(e, prompt, None, resume=kept)
    assert first == runs[0][0]
    snap = e.st.snapshot()
    for key in ("rec", "conv", "ple_tail"):
        assert bits_equal(snap[key], runs[0][1][key]), key


@needs_model
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_cut_model_stream_decodes_bit_for_bit_while_a_long_prompt_prefills(cut_model, kv_dtype):
    """Prompts inside rounds (upstream TensorFold d23087c, ported): stream B's 1,500-token prompt prefills a 128-row
    chunk a round while stream A decodes; A's tokens equal A decoded alone, bit for bit, and B's equal B alone."""

    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    w = cut_model
    rng = np.random.default_rng(6)
    prompt_a = [int(t) for t in rng.integers(0, w.cfg.vocab, size=41)]
    prompt_b = [int(t) for t in rng.integers(0, w.cfg.vocab, size=1500)]             # 12 chunks of 128 rows
    smp_a, smp_b = Sampling(seed=77, top_k=20, top_p=0.95), None
    count_a, count_b = 160, 16

    def solo(prompt, smp, count):
        e = Engine(w, capacity=2048, max_rows=8, prefill_rows=128, graphs=False, kv_dtype=kv_dtype)
        out = serial_decode(e, prefill(e, prompt, smp, mtp=False), count, smp).tokens
        del e
        torch.cuda.empty_cache()
        return out

    ref_a, ref_b = solo(prompt_a, smp_a, count_a), solo(prompt_b, smp_b, count_b)

    def decoder():
        return MultiDecoder(w, slots=2, capacity=2048, depth=3, confidence=0.3, stop_eos=False, kv_dtype=kv_dtype,
                            prefill_rows=128)

    dec = decoder()                                                   # A in the concurrent decoder, nothing beside it
    alone = Stream(list(prompt_a), count_a, smp_a)
    dec.admit(alone, defer=True)
    while dec.live():
        dec.finish(dec.round())
    del dec
    torch.cuda.empty_cache()
    dec = decoder()
    a = Stream(list(prompt_a), count_a, smp_a)
    dec.admit(a, defer=True)
    dec.finish(dec.round())                                           # A's prompt, its first round
    b = Stream(list(prompt_b), count_b, smp_b)
    dec.admit(b, defer=True)
    grew = []
    while b in dec.filling:
        before = len(a.out)
        dec.finish(dec.round())
        grew.append(len(a.out) - before)
    assert len(grew) >= 12 and all(n > 0 for n in grew), grew        # A decoded every round of B's fill (F8: >= its chunks)
    while dec.live():
        dec.finish(dec.round())
    assert alone.out == ref_a and a.out == ref_a                      # drafted == serial, beside B or alone
    assert b.out == ref_b


@needs_model
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_cut_model_identical_prompts_together_give_cold_then_a_copied_exact_hit(cut_model, kv_dtype):
    """F4 item 2: two identical prompts queued together: A prefills cold, B waits for A's fill, then exact-hits A's
    kept state copied into a spare slot while A decodes; both replies equal a fresh run, bit for bit."""

    import queue

    from tensorfold.cuda import scheduler as sched_mod
    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    w = cut_model
    prompt = [int(t) for t in np.random.default_rng(8).integers(0, w.cfg.vocab, size=2502)]   # > index budget; main and MTP ends partial blocks
    smp_a, smp_b, count = Sampling(seed=5, top_k=20, top_p=0.95), Sampling(seed=6, top_k=20, top_p=0.95), 48

    def fresh(smp):
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=128, graphs=False, kv_dtype=kv_dtype)
        out = serial_decode(e, prefill(e, prompt, smp, mtp=False), count, smp).tokens
        del e
        torch.cuda.empty_cache()
        return out

    ref_a, ref_b = fresh(smp_a), fresh(smp_b)
    dec = MultiDecoder(w, slots=2, capacity=4096, depth=3, confidence=0.3, stop_eos=False, kv_dtype=kv_dtype,
                       prefill_rows=128)
    a, b = Stream(list(prompt), count, smp_a), Stream(list(prompt), count, smp_b)
    sched = sched_mod.Scheduler.__new__(sched_mod.Scheduler)          # no worker thread: driven here
    sched.decoder, sched.max_streams, sched.waiting, sched.boxes, sched.held = dec, 2, queue.Queue(), {}, []
    for s in (a, b):
        sched.waiting.put((s, queue.Queue()))
    sched._admit()
    assert dec.filling == [a] and [h[0] for h in sched.held] == [b]
    while a in dec.filling:
        dec.finish(dec.round())
    sched._admit()                                                    # A decodes: B copies its kept state
    assert b.reuse == "exact" and b.reuse_copy and b.cached == len(prompt) and b.st is not a.st and not a.done
    while dec.live():
        dec.finish(dec.round())
    assert a.reuse is None and a.out == ref_a and b.out == ref_b
