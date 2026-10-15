"""``--rope-yarn`` on CUDA (the cut model): the rotary kernels against a plain reference built from transformers'
YaRN formula (keys, queries, indexer queries, pooled indexer keys and image positions, cos/sin times the attention
factor on the rotary dims only), and the engine contract with YaRN on: MTP-drafted decoding emits serial
decoding's tokens and streams decoded together emit what each does alone. With YaRN off the kernels compile as
before (MSCALE 1.0 is a constexpr branch never taken); the existing suites hold those bits."""

import math
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402

from tensorfold import rope_scaling  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attention as att  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import glue  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import forward  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

DEV = "cuda"
TEXT = {"max_position_embeddings": 262144, "rope_parameters": {"rope_type": "default", "rope_theta": 1e7,
                                                                "partial_rotary_factor": 0.25}}
PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]


def reference_tables(factor: float):
    """transformers 5.18 ``_compute_yarn_parameters`` for Flash Next (dim 64, theta 1e7, original 262,144), fp32."""

    dim, base, original = 64, 1e7, 262144
    find = lambda rot: (dim * math.log(original / (rot * 2 * math.pi))) / (2 * math.log(base))   # noqa: E731
    low, high = max(math.floor(find(32)), 0), min(math.ceil(find(1)), dim - 1)
    pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
    keep = 1 - torch.clamp((torch.arange(dim // 2).float() - low) / (high - low), 0, 1)
    inv = (1.0 / (factor * pos_freqs)) * (1 - keep) + (1.0 / pos_freqs) * keep
    return inv.to(DEV), 0.1 * math.log(factor) + 1.0


PLAIN = SimpleNamespace(rope_theta=1e7, rotary_dim=64, yarn=None)


def _tables(w):
    """The engine's YaRN table, checked against the reference's (fp32 rounding apart), and the reference's factor.
    Rotations below use the engine's table: at positions near 400,000 one fp32 ulp of a frequency moves an angle
    by ~0.02 rad, plain RoPE's float64-rounded table and transformers' fp32 one differing the same way."""

    inv, scale = reference_tables(2.0)
    assert torch.allclose(w.inv_freq, inv, rtol=3e-7, atol=0) and scale == pytest.approx(w.rope_scale)
    return w.inv_freq, scale


def _yarn(w, factor: float = 2.0):
    w.cfg.yarn = rope_scaling.yarn_parameters(TEXT, factor)
    inv, scale = rope_scaling.rope_tables(w.cfg)
    w.inv_freq, w.rope_scale = inv.to(DEV), scale
    return w


def _rotate(xn: torch.Tensor, ang: torch.Tensor, scale: float) -> torch.Tensor:
    """xn [rows, heads, width] fp32 (bf16 values), ang [rows, 32]: the reference's q * cos + rotate_half(q) * sin
    with cos and sin times the attention factor, on the first 64 dims only."""

    cos, sin = (torch.cos(ang) * scale)[:, None], (torch.sin(ang) * scale)[:, None]
    out = xn.clone()
    out[..., :32] = xn[..., :32] * cos - xn[..., 32:64] * sin
    out[..., 32:64] = xn[..., 32:64] * cos + xn[..., :32] * sin
    return out.to(torch.bfloat16).float()


def _norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(torch.bfloat16).float()


def _axis(rope: torch.Tensor) -> torch.Tensor:
    """[rows, 32] positions per rotary pair: M-RoPE interleaved (h at pairs 1, 4, .. < 33; w at 2, 5, .. < 30)."""

    i = torch.arange(32, device=DEV)
    h = (i % 3 == 1) & (i < 33)
    wd = (i % 3 == 2) & (i < 30)
    return torch.where(h, rope[:, 1:2], torch.where(wd, rope[:, 2:3], rope[:, 0:1])).float()


@pytest.mark.parametrize("image", [False, True])
def test_attn_prep_rotates_as_the_reference_yarn(image):
    w = _yarn(_model())
    c = w.cfg
    a = [layer for layer in w.layers if layer.attn is not None][0].attn
    rows, pos0, cap = 8, 400_000, 400_016                 # past the 262,144-token native window
    g = torch.Generator(device=DEV).manual_seed(5)
    width = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
    p = (torch.randn((rows, width), device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    pos = torch.tensor([pos0], dtype=torch.int32, device=DEV)
    cache = KVCache(cap, c.kv_heads, c.head_dim, DEV, "bf16")
    q = torch.empty((rows, c.heads, c.head_dim), dtype=torch.bfloat16, device=DEV)
    iq = torch.empty((rows, c.index_heads, c.index_dim), dtype=torch.bfloat16, device=DEV)
    ikc = torch.zeros((cap, c.index_dim), dtype=torch.bfloat16, device=DEV)
    rope = None
    if image:                                             # an image prompt's t/h/w rows
        rope = torch.stack([torch.full((rows,), pos0, device=DEV), pos0 + torch.arange(rows, device=DEV) % 3,
                            pos0 + torch.arange(rows, device=DEV) // 3], 1).to(torch.int32)
    glue.attn_prep(p, pos, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, q, cache.k, cache.v, iq, ikc, c.eps,
                   q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim, index_heads=c.index_heads,
                   index_dim=c.index_dim, ks=cache.ks, vs=cache.vs, bits=0, rope=rope, sections=(11, 11, 10),
                   rope_scale=w.rope_scale)
    inv, scale = _tables(w)
    plain_inv = rope_scaling.rope_tables(PLAIN)[0].to(DEV)
    where = _axis(rope) if image else (pos0 + torch.arange(rows, device=DEV)).float()[:, None].expand(rows, 32)
    ang = where * inv[None, :]
    hd, nq, nkv = c.head_dim, c.heads, c.kv_heads
    qn = _norm(p[:, :nq * 2 * hd].view(rows, nq, 2 * hd)[..., :hd], a.q_scale, c.eps)
    kn = _norm(p[:, nq * 2 * hd:nq * 2 * hd + nkv * hd].view(rows, nkv, hd), a.k_scale, c.eps)
    i0 = nq * 2 * hd + 2 * nkv * hd
    iqn = _norm(p[:, i0:i0 + c.index_heads * c.index_dim].view(rows, c.index_heads, c.index_dim), a.iq_scale, c.eps)
    keys = cache.k.view(-1, nkv, hd)[pos0:pos0 + rows].float()
    for got, xn in ((q.float(), qn), (keys, kn), (iq.float(), iqn)):
        want = _rotate(xn, ang, scale)
        assert (got - want).abs().max().item() < 0.05
        # the unrotated dims are not scaled (an ulp of normalization rounding apart; 1.069x would show ~0.2)
        assert (got[..., 64:] - xn[..., 64:]).abs().max().item() < 0.04
        plain = _rotate(xn, where * plain_inv[None, :], 1.0)
        assert (got - plain).abs().max().item() > 0.1                    # and YaRN did change the rotary ones


@pytest.mark.parametrize("image", [False, True])
def test_pooled_indexer_keys_rotate_as_the_reference_yarn(image):
    torch.manual_seed(11)
    w = _yarn(_model())
    rows, di, p0 = 64, 128, 300_000
    cap = p0 + rows + 8
    scratch = att.AttnScratch(rows, 24, 256, cap, DEV)
    ikc = torch.randn((cap, di), device=DEV).to(torch.bfloat16)
    pooled = torch.zeros((cap // 4 + 1, di), dtype=torch.bfloat16, device=DEV)
    w_ = (1 + 0.1 * torch.randn((di,), device=DEV)).float()
    pos = torch.tensor([p0], dtype=torch.int32, device=DEV)
    rope = None
    if image:
        rope = torch.stack([torch.full((rows,), p0, device=DEV), p0 + torch.arange(rows, device=DEV) % 8,
                            p0 + torch.arange(rows, device=DEV) // 8], 1).to(torch.int32)
    att.qsa_pool(ikc, pooled, pos, w_, w.inv_freq, 1e-6, scratch, rows, rope=rope, rope_scale=w.rope_scale)
    b0, nb = p0 // 4, rows // 4
    blocks = ikc[p0:p0 + rows].float().reshape(nb, 4, di).sum(1) / 4
    xn = _norm(blocks.to(torch.bfloat16), w_, 1e-6)[:, None]
    inv, scale = _tables(w)
    first = rope[::4] if image else None
    where = _axis(first) if image else (torch.arange(b0, b0 + nb, device=DEV) * 4).float()[:, None].expand(nb, 32)
    want = _rotate(xn, where * inv[None, :], scale)[:, 0]
    assert (pooled[b0:b0 + nb].float() - want).abs().max().item() < 0.05


@pytest.mark.parametrize("sampling", [None, Sampling(seed=1234, top_k=20, top_p=0.95)])
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_yarn_drafts_give_serial_tokens(sampling, kv_dtype):
    w = _yarn(_model())
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    eager = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
    first = prefill(eager, prompt, sampling)
    ref = serial_decode(eager, first, 24, sampling).tokens
    graphs = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=True, kv_dtype=kv_dtype)
    assert prefill(graphs, prompt, sampling) == first
    assert serial_decode(graphs, first, 24, sampling).tokens == ref
    for depth in (1, 3):
        for e in (eager, graphs):
            prefill(e, prompt, sampling)
            assert mtp_decode(e, first, 24, sampling, depth=depth, confidence=0.0).tokens == ref, depth


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_yarn_streams_together_equal_each_alone(kv_dtype):
    w = _yarn(_model())
    samplings = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling in zip(PROMPTS, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 20, sampling).tokens)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)
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


def test_yarn_changes_the_logits_and_off_is_the_plain_model():
    plain, yarn = _model(), _yarn(_model())
    assert getattr(plain, "rope_scale", 1.0) == 1.0
    ref_inv = rope_scaling.rope_tables(PLAIN)[0]
    assert torch.equal(plain.inv_freq.cpu(), ref_inv)                  # the cut model's plain table is the loader's
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12]
    out = []
    for w in (plain, yarn):
        e = Engine(w, capacity=1024, max_rows=16, prefill_rows=16)    # the 10-row prompt fits one window
        out.append(forward(w, e.st, e.buf, prompt)[:len(prompt)].float().clone())
    assert not torch.equal(out[0], out[1])
