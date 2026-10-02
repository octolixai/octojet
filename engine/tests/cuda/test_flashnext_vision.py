"""Flash Next image input on CUDA (MiaAI-Lab patch 0008, ported): the rotary kernels' new modes reduce to the text
arithmetic bit for bit when their positions are the text positions, and an image prompt whose "image" rows carry the
very embeddings and positions of text prefills to the same bits as that text. MODE 0 is the original code path, so
these comparisons also pin that text requests are unchanged by the port."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model, _state  # noqa: E402

from tensorfold.families.qwen4_exp.cuda import attention as attn_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import glue  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache  # noqa: E402
from tensorfold.vision.qwen_cuda import EncodedVision  # noqa: E402

DEV = "cuda"


def _prep(w, rows, pos0, dtype, **mode):
    c = w.cfg
    a = [layer for layer in w.layers if layer.attn is not None][0].attn
    g = torch.Generator(device=DEV).manual_seed(5)
    p = (torch.randn((rows, c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim),
                     device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    pos = torch.tensor([pos0], dtype=torch.int32, device=DEV)
    cache = KVCache(1024, c.kv_heads, c.head_dim, DEV, dtype)
    q = torch.empty((rows, c.heads, c.head_dim), dtype=torch.bfloat16, device=DEV)
    iq = torch.empty((rows, c.index_heads, c.index_dim), dtype=torch.bfloat16, device=DEV)
    ikc = torch.zeros((1024, c.index_dim), dtype=torch.bfloat16, device=DEV)
    bits = {"bf16": 0, "int8": 8, "int4": 4}[dtype]
    glue.attn_prep(p, pos, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, q, cache.k, cache.v, iq, ikc, c.eps,
                   q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim, index_heads=c.index_heads,
                   index_dim=c.index_dim, ks=cache.ks, vs=cache.vs, bits=bits, **mode)
    rows_of = lambda t: t[pos0:pos0 + rows]                    # noqa: E731
    kept = [q, iq, rows_of(ikc), rows_of(cache.k), rows_of(cache.v)]
    if bits:
        kept += [rows_of(cache.ks), rows_of(cache.vs)]
    return kept


@pytest.mark.parametrize("dtype", ["bf16", "int8"])
def test_attn_prep_modes_reduce_to_the_text_rotation(dtype):
    w = _model()
    rows, pos0 = 24, 40
    text = _prep(w, rows, pos0, dtype)
    same = torch.arange(pos0, pos0 + rows, dtype=torch.int32, device=DEV)[:, None].repeat(1, 3).contiguous()
    zero = torch.zeros((1,), dtype=torch.int32, device=DEV)
    for got in (_prep(w, rows, pos0, dtype, rope=same), _prep(w, rows, pos0, dtype, delta=zero)):
        assert all(torch.equal(a, b) for a, b in zip(got, text))
    # text after images: rotated at pos + delta, stored at pos
    shifted = _prep(w, rows, pos0 + 7, dtype)
    moved = _prep(w, rows, pos0, dtype, delta=torch.tensor([7], dtype=torch.int32, device=DEV))
    assert all(torch.equal(a, b) for a, b in zip(moved, shifted))


def test_pool_modes_reduce_to_the_text_rotation():
    w = _model()
    c = w.cfg
    rows, pos0 = 32, 64
    g = torch.Generator(device=DEV).manual_seed(9)
    ikc = (torch.randn((1024, c.index_dim), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    a = [layer for layer in w.layers if layer.attn is not None][0].attn
    scratch = attn_mod.AttnScratch(rows, c.heads, c.head_dim, 1024, DEV, budget=c.index_budget, ratio=c.index_ratio)
    pos = torch.tensor([pos0], dtype=torch.int32, device=DEV)

    def pooled(**mode):
        out = torch.zeros((1024 // c.index_ratio, c.index_dim), dtype=torch.bfloat16, device=DEV)
        attn_mod.qsa_pool(ikc, out, pos, a.ik_scale, w.inv_freq, c.eps, scratch, rows, **mode)
        return out

    text = pooled()
    same = torch.arange(pos0, pos0 + rows, dtype=torch.int32, device=DEV)[:, None].repeat(1, 3).contiguous()
    assert torch.equal(pooled(rope=same), text)
    assert torch.equal(pooled(delta=torch.zeros((1,), dtype=torch.int32, device=DEV)), text)


@pytest.mark.parametrize("dtype", ["bf16", "int8"])
def test_an_image_prompt_of_text_embeddings_prefills_to_the_text_bits(dtype):
    """Rows 3..10 are "image" rows whose features are their own token embeddings and whose t/h/w positions are their
    text positions: the prompt must prefill (chunks of 16 rows, two of them) to exactly the text prompt's state."""

    w = _model()
    c = w.cfg
    prompt = [(17 * i + 5) % V for i in range(29)]
    text_e = Engine(w, capacity=1024, max_rows=16, prefill_rows=16, kv_dtype=dtype)
    first = prefill(text_e, prompt, None)
    text = [t.clone() for t in _state(text_e)]
    rows = tuple(range(3, 11))
    ids = torch.tensor([prompt[r] for r in rows], dtype=torch.int32, device=DEV)
    features = glue.embed(ids, *w.embed, c.hidden, copies=1)                 # one copy: the tower's width
    positions = torch.arange(len(prompt), dtype=torch.int32, device=DEV)[None].repeat(3, 1).contiguous()
    image_e = Engine(w, capacity=1024, max_rows=16, prefill_rows=16, kv_dtype=dtype)
    got = prefill(image_e, prompt, None, vision=EncodedVision(rows, features, positions, 0))
    assert got == first and image_e.st.rope_delta == 0 and image_e.pbuf.rope_rows is None
    assert all(torch.equal(a, b) for a, b in zip(_state(image_e), text))
    assert torch.equal(image_e.last_logits, text_e.last_logits)


def test_a_resumed_or_misaligned_image_prompt_is_refused():
    w = _model()
    e = Engine(w, capacity=1024, max_rows=16, prefill_rows=18)
    vision = EncodedVision((1,), torch.zeros((1, w.cfg.hidden), dtype=torch.bfloat16, device=DEV),
                           torch.zeros((3, 4), dtype=torch.int32, device=DEV), 0)
    if e.pbuf.attn.qsa:
        with pytest.raises(ValueError, match="divisible"):
            prefill(e, [1, 2, 3, 4], None, vision=vision)
    with pytest.raises(ValueError, match="from its start"):
        prefill(e, [1, 2, 3, 4], None, resume={"state": {}}, vision=vision)


def test_the_decode_offset_is_set_and_reset():
    w = _model()
    e = Engine(w, capacity=1024, max_rows=16, prefill_rows=16)
    prompt = [5, 6, 7, 8, 9]
    positions = torch.tensor([[0, 1, 1, 2, 3], [0, 1, 1, 2, 3], [0, 1, 1, 2, 3]], dtype=torch.int32, device=DEV)
    feats = glue.embed(torch.tensor([6, 7], dtype=torch.int32, device=DEV), *w.embed, w.cfg.hidden, copies=1)
    prefill(e, prompt, None, vision=EncodedVision((1, 2), feats, positions, -1))
    assert e.st.rope_delta == -1 and int(e.st.rope_delta_dev) == -1
    prefill(e, prompt, None)                                   # a text prompt in the same state: plain positions
    assert e.st.rope_delta == 0 and int(e.st.rope_delta_dev) == 0


@pytest.mark.parametrize("dtype", ["bf16", "int8"])
def test_prompt_chunk_rows_do_not_change_a_bit(dtype):
    """TENSORFOLD_PREFILL_ROWS (MiaAI-Lab 0006): the same 5,000-token prompt in 2,048- and 4,096-row chunks (and
    1,000, not a power of two) leaves the same state, caches and logits."""

    w = _model()
    prompt = [(31 * i + 7) % V for i in range(5000)]
    states = []
    for rows in (2048, 4096, 1000):
        e = Engine(w, capacity=8192, max_rows=16, prefill_rows=rows, kv_dtype=dtype)
        first = prefill(e, prompt, None)
        states.append((first, [t.clone() for t in _state(e)], e.last_logits.clone()))
        del e
        torch.cuda.empty_cache()
    for first, state, logits in states[1:]:
        assert first == states[0][0] and torch.equal(logits, states[0][2])
        assert all(torch.equal(a, b) for a, b in zip(state, states[0][1]))


@pytest.mark.parametrize("dtype", ["bf16", "int8"])
def test_an_image_prompt_fills_between_rounds_to_its_solo_bits(dtype):
    """Prompts inside rounds (upstream d23087c, ported): an image prompt queued beside a decoding text stream prefills
    a chunk a round with its t/h/w rows and features, the shared prompt buffers back to text rows between chunks; both
    streams emit their solo runs."""

    from types import SimpleNamespace

    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    w = _model()
    c = w.cfg
    prompt = [(13 * i + 3) % V for i in range(53)]                          # four 16-row chunks
    rows = tuple(range(5, 37))
    positions = torch.tensor([[min(i, 20) for i in range(53)], list(range(53)), list(range(53))],
                             dtype=torch.int32, device=DEV)
    features = (torch.randn((len(rows), c.hidden), generator=torch.Generator(device=DEV).manual_seed(3),
                            device=DEV) * 0.1).to(torch.bfloat16)
    vision = EncodedVision(rows, features, positions.contiguous(), -16)
    text, smp = [5, 17, 99, 250], Sampling(seed=9, top_k=20, top_p=0.95)
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=dtype)
    ref_image = serial_decode(e, prefill(e, prompt, None, vision=vision), 12, None).tokens
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=dtype)
    ref_text = serial_decode(e, prefill(e, text, smp), 40, smp).tokens
    tower = SimpleNamespace(encode=lambda pixels, ids: vision)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=dtype, prefill_rows=16,
                       vision=tower, stop_eos=False)
    a = Stream(list(text), 40, smp)
    dec.admit(a, defer=True)
    dec.finish(dec.round())
    b = Stream(list(prompt), 12, None, vision="pixels")
    dec.admit(b, defer=True)
    assert b in dec.filling and b.vision is None and b.reuse is None
    grew = []
    while b in dec.filling:
        before = len(a.out)
        dec.finish(dec.round())
        grew.append(len(a.out) - before)
        assert dec.pbuf.rope_rows is None                                   # text rows between the chunks
    assert len(grew) == 4 and all(n > 0 for n in grew)
    while dec.live():
        dec.finish(dec.round())
    assert a.out == ref_text and b.out == ref_image
    assert all(k.slot is not b.st for k in dec.kept)                        # an image prompt is never kept
