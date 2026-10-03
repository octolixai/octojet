"""Flash Next concurrent rounds: each row gets its own stream's bits, and streams together emit what each does alone."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, forward, stage  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]


def test_segments_give_each_stream_its_own_rows():
    w = _model()
    b = Buffers(w, 32, 1024)
    chains = [[401, 33, 2048], [5], [77, 1500, 9, 10, 11], [3, 4]]
    states = []
    for prompt in PROMPTS:
        st = State(w, 1024, 16)
        forward(w, st, b, prompt)
        commit(w, st, b, len(prompt), len(prompt))
        states.append(st)
    alone = [st.clone() for st in states]
    ref = []
    for st, chain in zip(alone, chains):
        lg = forward(w, st, b, chain)
        ref.append((lg[:len(chain)].clone(), b.streams[:len(chain)].clone()))
        keep = max(1, len(chain) // 2)
        commit(w, st, b, len(chain), keep)
        ref[-1] += (forward(w, st, b, [chain[keep] if keep < len(chain) else 1])[0].clone(),)
    segs = stage(w, b, list(zip(states, chains)))
    lg = compute(w, segs, b)
    for (st, a0, a1), (rl, rs, _) in zip(segs, ref):
        assert torch.equal(lg[a0:a1], rl) and torch.equal(b.streams[a0:a1], rs), (a0, a1)
    for (st, a0, a1), chain in zip(segs, chains):
        commit(w, st, b, a1 - a0, max(1, (a1 - a0) // 2), at=a0)
    for st, chain, (_, _, nxt) in zip(states, chains, ref):
        keep = max(1, len(chain) // 2)
        assert torch.equal(forward(w, st, b, [chain[keep] if keep < len(chain) else 1])[0], nxt)


@pytest.mark.parametrize("confidence,vocab,kv_dtype", [(0.0, False, "bf16"), (0.3, False, "bf16"), (0.3, True, "bf16"),
                                                       (0.3, True, "int8"), (0.0, False, "int4")])
def test_streams_decoded_together_equal_each_alone(confidence, vocab, kv_dtype):
    w = _model()
    if vocab:                                            # drafts over a token subset (the real model's draft head)
        words, scales, biases = qmm.to_mlx(w.head)
        ids = torch.arange(1, V, 3, device="cuda")
        w.draft_ids = ids
        w.draft_head = qmm.make_q4(words[ids], scales[ids], biases[ids])
    samplings = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling in zip(PROMPTS, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        first = prefill(e, prompt, sampling)
        refs.append(serial_decode(e, first, 20, sampling).tokens)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=confidence, kv_dtype=kv_dtype)
    assert all(st.kv_dtype == kv_dtype and st.kc[0].dtype == kv_dtype for st in dec.free)
    streams = []
    for i, (prompt, sampling) in enumerate(zip(PROMPTS, samplings)):
        got: list[int] = []
        s = Stream(prompt, 20, sampling, draft=i != 3, emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        streams.append((s, got))
    while dec.live():
        dec.finish(dec.round())
    for i, (s, got) in enumerate(streams):
        assert got == refs[i] and s.out == refs[i], i
        assert s.min_rows >= (2 if s.draft else 1), (i, s.min_rows)
    assert len(dec.free) + len({id(k.slot) for k in dec.kept}) == 4 and not dec.live()     # every slot back or kept


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=31, top_k=20, top_p=0.95)])
def test_prompts_that_extend_a_finished_stream_resume_from_its_slot(sampling, kv_dtype):
    w = _model()
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)

    def run(prompt, count, draft=True):
        s = Stream(list(prompt), count, sampling, draft=draft)
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return s

    def fresh(prompt, count):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        return serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens

    first = run(PROMPTS[1], 12)
    longer = PROMPTS[1] + first.out[:-1] + [42, 43]          # the reply's committed tokens, then new ones
    warm = run(longer, 10)
    assert warm.cached == len(PROMPTS[1]) and warm.out == fresh(longer, 10)       # the reply prefills again
    ext = PROMPTS[0] + [7, 8]                                 # a prompt kept at admission, extended
    run(PROMPTS[0], 6)
    other = run(ext, 8)
    assert other.cached > 0 and other.out == fresh(ext, 8)
    serial = run(longer, 10, draft=False)
    assert serial.cached == 0 and serial.out == warm.out


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_prompts_fill_between_rounds_while_streams_decode(kv_dtype):
    """Ported from upstream TensorFold d23087c (one prompt a pass, the pass between rounds): a prompt queued beside a
    decoding stream prefills a chunk a round while it keeps decoding. Since F7 the prompt with the fewest rows left
    fills first, so the short prompts of a burst finish filling before the long one; every stream emits its solo
    run."""

    w = _model()
    g = torch.Generator().manual_seed(11)
    long = torch.randint(1, V, (70,), generator=g).tolist()           # five 16-row chunks
    prompts = [PROMPTS[0], long, PROMPTS[1], PROMPTS[0] + [7, 8]]
    samplings = [Sampling(seed=3, top_k=20, top_p=0.95), None, Sampling(seed=4, top_k=20, top_p=0.95), None]
    counts = [60, 24, 24, 24]                                         # the first outlasts the long prompt's fill
    refs = []
    for prompt, sampling, count in zip(prompts, samplings, counts):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=16,
                       stop_eos=False)
    first = Stream(prompts[0], counts[0], samplings[0])
    dec.admit(first, defer=True)
    assert dec.live() == 1 and not first.out                          # queued: the prompt fills in the rounds
    dec.finish(dec.round())                                           # alone: the whole prompt, then a round
    assert len(first.out) > 1
    rest = [Stream(p, 24, smp) for p, smp in zip(prompts[1:], samplings[1:])]
    for s in rest:
        dec.admit(s, defer=True)
    grew, left = [], []                                              # first's growth a round; order prompts finish filling
    while rest[0] in dec.filling:
        filling = [s for s in rest if s in dec.filling]
        before = len(first.out)
        dec.finish(dec.round())
        grew.append(len(first.out) - before)
        left += [rest.index(s) for s in filling if s not in dec.filling]
    assert all(n > 0 for n in grew)                                   # the first stream decodes every round
    assert len(grew) >= 5                                             # the long prompt's five chunks (F8: a round may
    assert left[-1] == 0 and set(left[:-1]) <= {1, 2}                 # skip a chunk while two run); short ones first
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in [first, *rest]] == refs
    assert len(dec.free) + len({id(k.slot) for k in dec.kept}) == 4


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_a_resumed_prompt_fills_between_rounds_as_fresh(kv_dtype):
    """A prompt that extends a kept prompt end resumes from it chunk by chunk beside a decoding stream; both streams
    emit their solo runs (F2d prefix reuse under prompts inside rounds)."""

    w = _model()
    g = torch.Generator().manual_seed(13)
    base = torch.randint(1, V, (40,), generator=g).tolist()
    longer = base + torch.randint(1, V, (45,), generator=g).tolist()
    sampling = Sampling(seed=21, top_k=20, top_p=0.95)

    def fresh(prompt, count, smp):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        return serial_decode(e, prefill(e, prompt, smp), count, smp).tokens

    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=16,
                       stop_eos=False)
    kept = Stream(list(base), 4, sampling)
    dec.admit(kept, defer=True)
    while dec.live():
        dec.finish(dec.round())
    a = Stream(list(PROMPTS[1]), 30, None)
    dec.admit(a, defer=True)
    dec.finish(dec.round())
    b = Stream(list(longer), 12, sampling)
    dec.admit(b, defer=True)
    assert b.reuse == "extend" and b.cached == len(base) and b in dec.filling
    while dec.live():
        dec.finish(dec.round())
    assert b.out == fresh(longer, 12, sampling) and a.out == fresh(PROMPTS[1], 30, None)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_a_twin_admitted_while_its_twin_decodes_copies_its_state(kv_dtype):
    """F4 item 2: an identical prompt admitted while its twin decodes exact-hits the twin's kept state copied into a
    spare slot (KV, indexer keys, pooled blocks, MTP caches): its reply equals a fresh run's, and the twin's too."""

    w = _model()
    g = torch.Generator().manual_seed(17)
    prompt = torch.randint(1, V, (2502,), generator=g).tolist()      # past the index budget: QSA pools
    samplings = [Sampling(seed=2, top_k=20, top_p=0.95), None]
    refs = []
    for sampling in samplings:
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=256, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 30, sampling).tokens)
    dec = MultiDecoder(w, slots=2, capacity=4096, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=256,
                       stop_eos=False)
    a = Stream(list(prompt), 30, samplings[0])
    dec.admit(a, defer=True)
    dec.finish(dec.round())
    b = Stream(list(prompt), 30, samplings[1])
    dec.admit(b, defer=True)
    assert b.reuse == "exact" and b.reuse_copy and b.st is not a.st and not a.done
    while dec.live():
        dec.finish(dec.round())
    assert a.out == refs[0] and b.out == refs[1]


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_shorter_chunks_while_streams_decode_keep_every_token(kv_dtype):
    """F8: while a stream decodes, a queued prompt fills in half-size chunks (more rounds, so the stream's pause is
    shorter); alone it would take full chunks. Every stream still emits its solo run, bit for bit."""

    w = _model()
    g = torch.Generator().manual_seed(29)
    long = torch.randint(1, V, (70,), generator=g).tolist()           # five 16-row chunks, nine 8-row ones
    prompts, counts = [PROMPTS[0], long], [80, 24]
    samplings = [Sampling(seed=5, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling, count in zip(prompts, samplings, counts):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens)
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=16,
                       stop_eos=False)
    dec.live_prefill_rows = 8                                          # the test model's half chunk (prod: 1,024 of 2,048)
    first = Stream(prompts[0], counts[0], samplings[0])
    dec.admit(first, defer=True)
    dec.finish(dec.round())
    second = Stream(prompts[1], counts[1], samplings[1])
    dec.admit(second, defer=True)
    rounds = 0
    while second in dec.filling:
        before = len(first.out)
        dec.finish(dec.round())
        rounds += 1
        assert len(first.out) > before                                # the stream decodes every round
    assert rounds >= 9                                                # 70 rows in 8-row chunks while the stream lives
    assert not dec.fill_events                                        # joined: nothing left on the fill stream
    while dec.live():
        dec.finish(dec.round())
    assert [first.out, second.out] == refs
