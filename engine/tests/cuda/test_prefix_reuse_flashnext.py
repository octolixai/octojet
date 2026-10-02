"""Identical-prompt reuse on the NVFP4 Flash Next cut (OCTOJET_NVFP4_FLASHNEXT, 4 layers: PLE at layer 1, the attention
layer at 3, QSA and pool construction on at CAP > index_budget): an exact hit equals a fresh drafting run and the serial
reference in the first token, the initial drafts, the reply and the committed state (PLE tail and history, completed
pools including the block the prompt end left partial, KV, indexer keys, recurrence), on both serving paths, greedy and
seeded; no prompt forward on a hit; the serial reference, warm-up and failed admissions (failing after the state was
mutated) leave nothing kept; stage A constructs no checkpoint; P/PX invalidation; an EOS first token; two chunk sizes
agree (stage-B evidence, run apart from the stage-A gate)."""

import gc

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_prefill_timing_flashnext import CAP, LAYERS, bits_equal, cut, needs_model, state_bytes  # noqa: E402,F401

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import decode as decode_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import engine as engine_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import multi as multi_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix as px  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402

pytestmark = needs_model
GREEDY, SEEDED = None, Sampling(seed=1234, top_k=20, top_p=0.95)
COUNT = 40                                  # the reply crosses ten pool blocks (ratio 4) after the unaligned prompt end
ROWS = 2048


def ids(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=n)]


P = ids(11, 2_501)                          # > index_budget 2,048: the sparse path runs; 2,501 % 4 = 1: the last pool block is partial
PX = P + ids(12, 300)
SHORT = ids(13, 37)                         # one chunk, still unaligned


def features(w):
    """The fixture the evidence needs: a PLE layer and an attention layer inside the cut, QSA on at this capacity."""

    assert w.cfg.ple_layers and all(i < LAYERS for i in w.cfg.ple_layers), w.cfg.ple_layers
    assert "attention" in w.cfg.layer_types[:LAYERS] and CAP > w.cfg.index_budget


def state_view(st, w):
    """(tensor parts, scalars without the parity selector): restore() resets the parity, so `cur` may differ while the
    recurrence, compared as the active parity, is the same."""

    parts, (pos, mtp_len, mtp_drafted, _cur, hist) = state_bytes(st, w.cfg.index_ratio)
    return parts, (pos, mtp_len, mtp_drafted, hist)


def same_state(a, b):
    (pa, sa), (pb, sb) = a, b
    return sa == sb and len(pa) == len(pb) and all(bits_equal(x, y) for x, y in zip(pa, pb))


def pools_built(st):
    return st.ple_history is not None and bool(st.pooled) and st.pooled[0][: st.pos // 4].abs().sum().item() > 0


def spy_drafts(monkeypatch, module):
    """Record every draft() result on ``module`` (multi or decode): the first call after an admission is the initial drafts."""

    seen = []
    real = module.draft

    def spy(*a, **k):
        d = real(*a, **k)
        seen.append(list(d))
        return d

    monkeypatch.setattr(module, "draft", spy)
    return seen


def failing_after(module, name):
    """The real function runs (so the state is mutated) and then the failure is raised."""

    real = getattr(module, name)

    def wrapped(*a, **k):
        real(*a, **k)
        raise RuntimeError("injected")

    return wrapped


def boom_emit(new):
    raise RuntimeError("injected")


def admit_and_run(dec, prompt, sampling, draft_on=True, emit=None):
    """Admit on the decoder, capture the first token and initial drafts, then decode to the end."""

    s = Stream(list(prompt), COUNT, sampling, draft=draft_on, emit=emit)
    dec.admit(s)
    first, drafts = s.out[0], list(s.drafts)
    if s.done:                                    # finished at the first token: round() would never see it
        dec.finish([s])
    while dec.live():
        dec.finish(dec.round())
    return s, first, drafts


def fresh_decoder(w, kv_dtype, slots=1):
    return MultiDecoder(w, slots=slots, capacity=CAP, depth=3, confidence=0.3, stop_eos=False, kv_dtype=kv_dtype)


def serial_oracle(w, prompt, sampling, kv_dtype, *, stop_eos):
    """The serial reference under the compared path's EOS policy: the scheduler path is built with stop_eos=False
    (MultiDecoder.eos empty), the single-stream path always stops at w.cfg.eos (mtp_decode(stop_eos=True))."""

    e = Engine(w, capacity=CAP, max_rows=8, prefill_rows=ROWS, kv_dtype=kv_dtype)
    first = prefill(e, prompt, sampling)
    tokens = serial_decode(e, first, COUNT, sampling, stop_eos=stop_eos).tokens
    del e; gc.collect(); torch.cuda.empty_cache()
    return tokens


def bare_engine(w, kv_dtype):
    eng = engine_mod.FlashNextEngine.__new__(engine_mod.FlashNextEngine)
    eng.w, eng.depth, eng.confidence, eng.max_len, eng.tp, eng.rank, eng.comm = w, 3, 0.3, CAP, 1, 0, None
    eng.served, eng.cache, eng.next_serial, eng.serial, eng.multi, eng.scheduler = 0, [], 0, None, None, None
    eng.eos, eng.kv_dtype = (), kv_dtype                    # only _decode's first-token check reads this; mtp_decode still stops at w.cfg.eos
    eng.e = Engine(w, capacity=CAP, max_rows=8, prefill_rows=ROWS, kv_dtype=kv_dtype)
    return eng


def run_single(eng, prompt, sampling, draft=True):
    out = []
    stats = eng.generate(list(prompt), COUNT, sampling, lambda new: out.extend(new) and False, draft=draft)
    return stats, out


@pytest.mark.parametrize("kv_dtype", ["int8", "bf16"])
@pytest.mark.parametrize("keep_with,hit_with", [(GREEDY, SEEDED), (SEEDED, GREEDY), (GREEDY, GREEDY)])
def test_exact_hit_equals_a_fresh_run(cut, kv_dtype, keep_with, hit_with):
    features(cut)
    dec = fresh_decoder(cut, kv_dtype, slots=2)
    cold, _, _ = admit_and_run(dec, P, keep_with)
    assert cold.reuse is None and cold.cached == 0 and dec.kept[0].checkpoints == [] and pools_built(cold.st)
    block = len(P) // 4                                    # the pool block the prompt end left partial
    hit, first, drafts = admit_and_run(dec, P, hit_with)
    assert hit.reuse == "exact" and hit.cached == len(P) and hit.reuse_miss is None and hit.st is cold.st
    ref = fresh_decoder(cut, kv_dtype)                     # the oracle: a fresh drafting admission and its rounds
    want, wfirst, wdrafts = admit_and_run(ref, P, hit_with)
    assert (first, drafts, hit.out) == (wfirst, wdrafts, want.out) and len(hit.out) == COUNT
    assert hit.out == serial_oracle(cut, P, hit_with, kv_dtype, stop_eos=False)   # and the serial reference agrees
    assert bits_equal(hit.st.pooled[0][block], want.st.pooled[0][block])   # rebuilt from the new reply, not carried over
    assert same_state(state_view(hit.st, cut), state_view(want.st, cut))


def test_exact_hit_runs_no_prompt_forward(cut, monkeypatch):
    dec = fresh_decoder(cut, "int8")
    admit_and_run(dec, SHORT, GREEDY)
    calls = []
    real = multi_mod.prefill
    monkeypatch.setattr(multi_mod, "prefill", lambda *a, **k: calls.append("prefill") or real(*a, **k))
    hit, _, _ = admit_and_run(dec, SHORT, GREEDY)
    assert hit.reuse == "exact" and calls == []


def test_serial_reference_never_reuses_and_leaves_the_entry(cut):
    dec = fresh_decoder(cut, "int8", slots=2)
    admit_and_run(dec, SHORT, GREEDY)
    ref, _, _ = admit_and_run(dec, SHORT, GREEDY, draft_on=False)
    assert ref.reuse is None and ref.cached == 0 and len(dec.kept) == 1 and dec.kept[0].slot is not ref.st
    again, _, _ = admit_and_run(dec, SHORT, GREEDY)
    assert again.reuse == "exact" and again.out == ref.out                    # drafted == serial


def test_warm_leaves_no_entry(cut):
    dec = fresh_decoder(cut, "int8")
    dec.warm()
    assert dec.kept == [] and len(dec.free) == 1


def test_stage_a_constructs_no_checkpoint(cut, monkeypatch):
    made = []
    real_init = px.Checkpoint.__init__
    monkeypatch.setattr(px.Checkpoint, "__init__", lambda self, *a, **k: (made.append(1), real_init(self, *a, **k))[1])
    planned = []
    monkeypatch.setattr(px, "plan_checkpoints", lambda *a, **k: planned.append(a) or ([], []))
    dec = fresh_decoder(cut, "int8")
    admit_and_run(dec, SHORT, GREEDY); admit_and_run(dec, SHORT, GREEDY); admit_and_run(dec, SHORT + [5, 6], GREEDY)
    eng = bare_engine(cut, "int8")
    run_single(eng, SHORT, GREEDY); run_single(eng, SHORT, GREEDY); run_single(eng, SHORT + [5, 6], GREEDY)
    assert made == [] and planned == [] and all(k.checkpoints == [] for k in dec.kept + eng.cache)


@pytest.mark.parametrize("stage,kind", [("restore", "exact"), ("prefill", "extend"), ("snapshot", "extend"),
                                        ("draft", "exact"), ("emit", "exact")])
def test_a_failed_admission_leaves_nothing_kept(cut, monkeypatch, stage, kind):
    dec = fresh_decoder(cut, "int8")
    admit_and_run(dec, SHORT, GREEDY)
    prompt = SHORT if kind == "exact" else SHORT + [7, 8]
    target = {"restore": (State, "restore"), "prefill": (multi_mod, "prefill"), "snapshot": (State, "snapshot"),
              "draft": (multi_mod, "draft")}.get(stage)
    if target is not None:
        monkeypatch.setattr(*target, failing_after(*target))          # the real step runs, then fails: the state was mutated
    with pytest.raises(RuntimeError, match="injected"):
        admit_and_run(dec, prompt, GREEDY, emit=boom_emit if stage == "emit" else None)
    assert dec.kept == [] and len(dec.free) == 1 and not dec.live() and dec.streams == {}
    monkeypatch.undo()
    cold, _, _ = admit_and_run(dec, SHORT, GREEDY)                                  # the same decoder admits again
    ref, _, _ = admit_and_run(fresh_decoder(cut, "int8"), SHORT, GREEDY)
    assert cold.reuse is None and cold.out == ref.out


def test_an_eos_first_token_ends_both_paths_at_the_admission(cut):
    dec = fresh_decoder(cut, "int8", slots=2)
    cold, first, _ = admit_and_run(dec, SHORT, GREEDY)
    dec.eos = (first,)                                     # the kept prompt's greedy first token is now an end token
    hit, hfirst, _ = admit_and_run(dec, SHORT, GREEDY)
    assert hit.reuse == "exact" and hit.out == [first] == [hfirst] and hit.done and not dec.live()
    eng = bare_engine(cut, "int8")
    _, out = run_single(eng, SHORT, GREEDY)
    eng.eos = (out[0],)
    stats, hout = run_single(eng, SHORT, GREEDY)
    assert stats["reuse"] == "exact" and hout == [out[0]]


@pytest.mark.parametrize("keep_with,hit_with", [(GREEDY, SEEDED), (SEEDED, GREEDY)])
def test_single_stream_exact_hit_equals_a_fresh_run(cut, monkeypatch, keep_with, hit_with):
    features(cut)
    eng = bare_engine(cut, "int8")
    cold, _ = run_single(eng, P, keep_with)
    assert cold["reuse"] is None and cold["cached"] == 0 and pools_built(eng.e.st)
    drafts = spy_drafts(monkeypatch, decode_mod)           # mtp_decode's first draft() call is the initial drafts
    hit, out = run_single(eng, P, hit_with)
    assert hit["reuse"] == "exact" and hit["cached"] == len(P) and hit["reuse_miss"] is None
    hit_drafts = drafts[0]
    drafts.clear()
    ref = bare_engine(cut, "int8")
    want, wout = run_single(ref, P, hit_with)
    assert out == wout and hit_drafts == drafts[0] and (len(out) == COUNT or out[-1] in cut.cfg.eos)
    assert out == serial_oracle(cut, P, hit_with, "int8", stop_eos=True)          # the single-stream path stops at the model's EOS
    assert same_state(state_view(eng.e.st, cut), state_view(ref.e.st, cut))


def test_single_stream_serial_reference_never_reuses(cut):
    eng = bare_engine(cut, "int8")
    run_single(eng, SHORT, GREEDY)
    ref, rout = run_single(eng, SHORT, GREEDY, draft=False)
    assert ref["reuse"] is None and ref["cached"] == 0 and [k.ids for k in eng.cache] == [SHORT]
    again, out = run_single(eng, SHORT, GREEDY)
    assert again["reuse"] == "exact" and out == rout


def test_single_stream_p_px_p_px(cut):
    eng = bare_engine(cut, "int8")
    run_single(eng, P, GREEDY); run_single(eng, PX, GREEDY)
    assert [len(k.ids) for k in eng.cache] == [len(P), len(PX)]
    hit, out = run_single(eng, P, SEEDED)
    assert hit["reuse"] == "exact" and [len(k.ids) for k in eng.cache] == [len(P)]        # PX dropped before decoding
    again, out2 = run_single(eng, PX, GREEDY)                                              # PX extends P: its tail prefills again
    assert again["reuse"] == "extend" and again["cached"] == len(P)
    ref = bare_engine(cut, "int8")
    _, wout = run_single(ref, PX, GREEDY)
    assert out2 == wout and same_state(state_view(eng.e.st, cut), state_view(ref.e.st, cut))


@pytest.mark.parametrize("stage,kind", [("restore", "exact"), ("prefill", "extend"), ("draft", "exact")])
def test_single_stream_failure_clears_the_cache(cut, monkeypatch, stage, kind):
    eng = bare_engine(cut, "int8")
    run_single(eng, SHORT, GREEDY)
    prompt = SHORT if kind == "exact" else SHORT + [5]
    target = {"restore": (State, "restore"), "prefill": (decode_mod, "prefill"), "draft": (decode_mod, "draft")}[stage]
    monkeypatch.setattr(*target, failing_after(*target))               # _decode imports prefill/mtp_decode from .decode at call time
    with pytest.raises(RuntimeError, match="injected"):
        run_single(eng, prompt, GREEDY)
    assert eng.cache == []
    monkeypatch.undo()
    cold, out = run_single(eng, SHORT, GREEDY)
    _, wout = run_single(bare_engine(cut, "int8"), SHORT, GREEDY)
    assert cold["reuse"] is None and out == wout


def test_two_chunk_sizes_prefill_the_same_prompt_bit_equal(cut):
    """Stage-B evidence (chunk ends aligned to prefill_rows): the exactness script runs it apart from the stage-A gate."""

    views = []
    for rows, chunks in ((512, 5), (2048, 2)):
        assert -(-len(P) // rows) == chunks
        e = Engine(cut, capacity=CAP, max_rows=8, prefill_rows=rows, kv_dtype="int8")
        first = prefill(e, P, GREEDY)
        views.append((first, e.last_logits.clone(), state_view(e.st, cut)))
        del e; gc.collect(); torch.cuda.empty_cache()
    (f1, l1, s1), (f2, l2, s2) = views
    assert f1 == f2 and bits_equal(l1, l2) and same_state(s1, s2)


# -- F2d stage B: chunk-boundary checkpoints (scheduler path) -------------------------------------------------------
B_ROWS, B_N = 512, 4
PB = ids(21, 4_100)                         # interior ends 512 .. 4,096 (8): n = 4 -> 1,536, 3,072 and the last two, 3,584, 4,096


def b_decoder(w, kv_dtype, slots=1, n=B_N):
    return MultiDecoder(w, slots=slots, capacity=CAP, depth=3, confidence=0.3, stop_eos=False, kv_dtype=kv_dtype,
                        prefill_rows=B_ROWS, prefix_checkpoints=n)


def b_fresh(w, prompt, sampling, kv_dtype):
    """The oracle: the same prompt admitted cold on a fresh decoder (no checkpoints), its reply and state."""

    dec = b_decoder(w, kv_dtype, n=0)
    s, first, drafts = admit_and_run(dec, prompt, sampling)
    view = state_view(s.st, w)
    del dec; gc.collect(); torch.cuda.empty_cache()
    return (first, drafts, s.out), view


def admit_deferred(dec, prompt, sampling):
    s = Stream(list(prompt), COUNT, sampling)
    dec.admit(s, defer=True)
    while dec.live():
        dec.finish(dec.round())
    return s


@pytest.mark.parametrize("kv_dtype", ["int8", "bf16"])
@pytest.mark.parametrize("d,cached", [(3_584, 3_584), (3_585, 3_584), (2_600, 1_536), (1_000, 0)])
def test_checkpoint_resume_equals_a_fresh_run(cut, kv_dtype, d, cached):
    """Stage-B cases 6 and 9: Q shares d tokens with the kept P (at a boundary, one past, mid-stride, inside the first
    stride): it resumes from the last checkpoint at or before d and equals a fresh prefill of Q bit for bit."""

    features(cut)
    dec = b_decoder(cut, kv_dtype)
    cold, _, _ = admit_and_run(dec, PB, GREEDY)
    assert [c.pos for c in dec.kept[0].checkpoints] == [1_536, 3_072, 3_584, 4_096]
    Q = PB[:d] + ids(22, 900)
    hit, first, drafts = admit_and_run(dec, Q, SEEDED)
    assert hit.reuse == ("checkpoint" if cached else None) and hit.cached == cached
    want, view = b_fresh(cut, Q, SEEDED, kv_dtype)
    assert (first, drafts, hit.out) == want and len(hit.out) == COUNT
    assert same_state(state_view(hit.st, cut), view)


@pytest.mark.parametrize("kv_dtype", ["int8", "bf16"])
def test_a_shortened_prompt_and_a_deferred_admission_resume_from_checkpoints(cut, kv_dtype):
    dec = b_decoder(cut, kv_dtype)
    admit_and_run(dec, PB, GREEDY)
    short = PB[:3_584]                                       # pos == len(prompt) never matches: 3,072 does
    s = admit_deferred(dec, short, GREEDY)                   # through the scheduler path's chunk-by-chunk fill
    assert s.reuse == "checkpoint" and s.cached == 3_072
    want, view = b_fresh(cut, short, GREEDY, kv_dtype)
    assert s.out == want[2] and same_state(state_view(s.st, cut), view)


@pytest.mark.parametrize("inherited", [True, False])
def test_chained_resume_extend_and_checkpoint_hit_on_inherited_thinned_checkpoints(cut, inherited):
    """Case 8: P cold, an extension of P (inherits P's checkpoints, thinned with new ones, at most N), then a prompt
    diverging inside the extension resumes from one of them; every step equals a fresh run."""

    dec = b_decoder(cut, "int8")
    admit_and_run(dec, PB, GREEDY)
    X = PB + ids(23, 3_000)
    x, _, _ = admit_and_run(dec, X, GREEDY)
    assert x.reuse == "extend" and x.cached == len(PB)
    points = [c.pos for c in dec.kept[0].checkpoints]
    assert points == [3_072, 5_632, 6_144, 6_656]           # P's 3,072 inherited; X's last ends protected from thinning
    d = (3_072 if inherited else 6_144) + 100               # an inherited checkpoint (P's), or one X took
    Y = X[:d] + ids(24, 500)
    y, first, drafts = admit_and_run(dec, Y, SEEDED)
    assert y.reuse == "checkpoint" and y.cached == max(p for p in points if p <= d)
    want, view = b_fresh(cut, Y, SEEDED, "int8")
    assert (first, drafts, y.out) == want and same_state(state_view(y.st, cut), view)
    assert x.out == b_fresh(cut, X, GREEDY, "int8")[0][2]


@pytest.mark.parametrize("stage", ["prefill", "draft"])
def test_a_failed_checkpoint_resume_leaves_nothing_kept(cut, monkeypatch, stage):
    """Cases 4 and 13 for a checkpoint resume: a failure inside the prefill (after it ran) or in the initial drafts
    frees the slot and leaves no entry or checkpoint behind."""

    dec = b_decoder(cut, "int8")
    admit_and_run(dec, PB, GREEDY)
    target = (multi_mod, stage)
    monkeypatch.setattr(*target, failing_after(*target))
    with pytest.raises(RuntimeError, match="injected"):
        admit_and_run(dec, PB[:2_600] + ids(25, 300), GREEDY)
    assert dec.kept == [] and len(dec.free) == 1 and not dec.live()


@pytest.mark.parametrize("kv_dtype", ["int8", "bf16"])
@pytest.mark.parametrize("deferred", [False, True])
def test_a_variant_resumed_beside_its_source_equals_fresh_and_the_resend_stays_exact(cut, kv_dtype, deferred):
    """Two slots: the variant Q resumes from P's checkpoint in the other slot (P's rows below it copied first), equals
    a fresh run of Q bit for bit, and P resent afterwards is still an exact hit equal to P's cold reply."""

    dec = b_decoder(cut, kv_dtype, slots=2)
    p, _, _ = admit_and_run(dec, PB, GREEDY)
    Q = PB[:3_585] + ids(26, 900)
    if deferred:
        q = admit_deferred(dec, Q, SEEDED)
        want, view = b_fresh(cut, Q, SEEDED, kv_dtype)
        assert q.out == want[2]
    else:
        q, first, drafts = admit_and_run(dec, Q, SEEDED)
        want, view = b_fresh(cut, Q, SEEDED, kv_dtype)
        assert (first, drafts, q.out) == want
    assert q.reuse == "checkpoint" and q.cached == 3_584 and q.st is not p.st
    assert same_state(state_view(q.st, cut), view)
    r, _, _ = admit_and_run(dec, PB, GREEDY)
    assert r.reuse == "exact" and r.cached == len(PB) and r.st is p.st and r.out == p.out
