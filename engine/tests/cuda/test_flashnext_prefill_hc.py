"""F12: a prompt chunk's fused hyper-connection read-out (hc_prefill: the write-back with the norm, the up projection
with the mix) gives the separate kernels' bits: streams, partial sums, normed streams, mix, its group sums, the
activations and inject gates, at any row count and block shape, on synthetic weights and on the HC weights of the
mixed NVFP4 checkpoint (OCTOJET_NVFP4_FLASHNEXT) and the MLX 4-bit one (OCTOJET_MLX_FLASHNEXT); and a whole prefill
is the same bit for bit with OCTOJET_PREFILL_HC_FUSED on and off."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import forward as fwd  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import glue, hc_prefill, qmm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import HC  # noqa: E402
from test_prefill_timing_flashnext import bits_equal, cut, needs_model, run, same  # noqa: E402,F401

DEV = "cuda"
S, D, LOW = 4, 2560, 320
ROWS = [1, 15, 16, 17, 33, 333, 1024, 2048, 4096]
MLX = os.environ.get("OCTOJET_MLX_FLASHNEXT", "")
MLX_LAYERS = int(os.environ.get("OCTOJET_MLX_FLASHNEXT_LAYERS", "4"))
needs_mlx = pytest.mark.skipif(not MLX or not Path(MLX).is_dir(), reason="set OCTOJET_MLX_FLASHNEXT")


def _mlx(n: int, k: int, seed: int, kind: str = "random"):
    """MLX affine 4-bit (words, scales, biases): ``random`` (any biases), ``centred`` (the checkpoints' shape:
    b = -7.5 s), ``wide`` (20x the scales: the sigmoids saturate)."""

    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=g, device=DEV, dtype=torch.int64).to(torch.int32)
    s = torch.rand((n, k // 32), generator=g, device=DEV) * 0.02 + 0.001
    if kind == "wide":
        s = s * 20
    s = s.to(torch.bfloat16)
    if kind == "centred":
        b = (-(s.float() * 7.5)).to(torch.bfloat16)
    else:
        b = (torch.randn((n, k // 32), generator=g, device=DEV) * (0.4 if kind == "wide" else 0.02)).to(torch.bfloat16)
    return words, s, b


def _hc(seed: int, inject: bool, kind: str = "random", d: int = D) -> HC:
    parts = [_mlx(LOW, S * d, seed, kind)] + ([_mlx(S, S * d, seed + 1, kind)] if inject else [])
    up = _mlx(S * d, LOW, seed + 2, kind)
    scale = 1 + 0.05 * torch.randn((S * d,), generator=torch.Generator(device=DEV).manual_seed(seed + 3), device=DEV)
    return HC(qmm.stack_q4(parts, "tiled"), qmm.make_q4(*up, "tiled"), scale, inject,
              qmm.stack_q4(parts, "frag"), qmm.make_q4(*up, "frag"))


def _branch(mode: int, rows: int, d: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    inject = (torch.rand((rows, S), generator=g, device=DEV) * 2).to(torch.bfloat16)
    if mode == 0:
        return {}
    if mode == 1:
        return {"branch": (torch.randn((rows, d), generator=g, device=DEV)).to(torch.bfloat16), "inject": inject}
    if mode == 2:
        slots = 11
        y = (torch.randn((rows, slots, d), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
        wts = torch.softmax(torch.randn((rows, slots), generator=g, device=DEV), dim=1).float().contiguous()
        return {"y": y, "wts": wts, "inject": inject}
    return {"branch": torch.randn((2, rows, d), generator=g, device=DEV).contiguous(), "inject": inject}   # mode 4


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("mode", [0, 1, 2, 4])
def test_writeback_with_the_norm_gives_writeback_then_norm_bits(rows, mode):
    g = torch.Generator(device=DEV).manual_seed(rows * 7 + mode)
    h0 = (torch.randn((rows, S * D), generator=g, device=DEV) * 3).to(torch.bfloat16)
    scale = 1 + 0.05 * torch.randn((S * D,), generator=g, device=DEV)
    kw = _branch(mode, rows, D, rows + 11 * mode)
    f32, bf = torch.float32, torch.bfloat16
    h_ref, h_f = h0.clone(), h0.clone()
    pss_ref = torch.empty((rows, D // 256, S), dtype=f32, device=DEV)
    pss_f = torch.full_like(pss_ref, float("nan"))
    normed_ref = torch.empty((rows, S * D), dtype=bf, device=DEV)
    normed_f = torch.full_like(normed_ref, float("nan"))
    xs = torch.empty((rows, S * D // 32), dtype=f32, device=DEV)
    glue.hc_writeback(h_ref, h_ref, pss_ref, S, mode, **kw)
    glue.hc_normed(h_ref, pss_ref, scale, normed_ref, xs, S, 1e-6)
    glue.hc_writeback_normed(h_f, pss_f, S, mode, scale, normed_f, 1e-6, **kw)
    assert bits_equal(h_f, h_ref)
    assert bits_equal(pss_f, pss_ref)
    assert bits_equal(normed_f, normed_ref)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("tile", [0, 1, 2, 3])
@pytest.mark.parametrize("kind", ["random", "centred", "wide"])
def test_up_with_the_mix_gives_prefill_matmul_then_mix_bits(rows, tile, kind):
    q = qmm.make_q4(*_mlx(S * D, LOW, 31, kind), "frag")
    g = torch.Generator(device=DEV).manual_seed(rows + 1000 * tile)
    act = (torch.randn((rows, LOW), generator=g, device=DEV) * 0.7).to(torch.bfloat16)
    normed = (torch.randn((rows, S * D), generator=g, device=DEV)).to(torch.bfloat16)
    up = qmm.prefill_matmul(act, q)
    ref_mixed = torch.empty((rows, D), dtype=torch.bfloat16, device=DEV)
    ref_xs = torch.empty((rows, D // 32), dtype=torch.float32, device=DEV)
    glue.hc_mix(up, normed, ref_mixed, ref_xs, S)
    mixed = torch.full_like(ref_mixed, float("nan"))
    xs = torch.full_like(ref_xs, float("nan"))
    hc_prefill.upmix(act, q, normed, mixed, S, tile)
    glue.hc_mix_sums(mixed, xs)
    assert bits_equal(mixed, ref_mixed)
    assert bits_equal(xs, ref_xs)


def _buffers(rows: int, d: int, h0: torch.Tensor, prefill: bool = True):
    f32, bf = torch.float32, torch.bfloat16
    nan = float("nan")
    return SimpleNamespace(h=h0.clone(), pss=torch.full((rows, d // 256, S), nan, dtype=f32, device=DEV),
                           normed=torch.full((rows, S * d), nan, dtype=bf, device=DEV),
                           xs_normed=torch.full((rows, S * d // 32), nan, dtype=f32, device=DEV),
                           dn=torch.full((rows, LOW + S), nan, dtype=bf, device=DEV),
                           dn_mix=torch.full((rows, LOW), nan, dtype=bf, device=DEV),
                           act=torch.full((rows, LOW), nan, dtype=bf, device=DEV),
                           xs_act=torch.full((rows, LOW // 32), nan, dtype=f32, device=DEV),
                           up=torch.full((rows, S * d), nan, dtype=bf, device=DEV),
                           mixed=torch.full((rows, d), nan, dtype=bf, device=DEV),
                           xs_mixed=torch.full((rows, d // 32), nan, dtype=f32, device=DEV),
                           part=torch.empty((1,), dtype=f32, device=DEV),
                           inj=torch.full((rows, S), nan, dtype=bf, device=DEV), prefill=prefill)


def _block_outputs(b, hc):
    out = {"h": b.h, "pss": b.pss, "normed": b.normed, "mixed": b.mixed, "xs_mixed": b.xs_mixed, "act": b.act,
           "xs_act": b.xs_act}
    if hc.inject:
        out["inj"] = b.inj
    return out


def _block(monkeypatch, hc: HC, rows: int, d: int, mode: int, setting: str, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    h0 = (torch.randn((rows, S * d), generator=g, device=DEV) * 3).to(torch.bfloat16)
    kw = _branch(mode, rows, d, seed + 5)
    inject_prev = kw.pop("inject", None)
    b = _buffers(rows, d, h0)
    monkeypatch.setenv(hc_prefill.ENV, setting)
    fwd.hc_block(hc, b, rows, 1e-6, S, LOW, mode, inject_prev, b.inj, b.h, **kw)
    torch.cuda.synchronize()
    return _block_outputs(b, hc)


def _same_block(a: dict, b: dict) -> list[str]:
    return [k for k in a if not bits_equal(a[k], b[k])]


@pytest.mark.parametrize("rows", [1, 17, 1024, 4096])
@pytest.mark.parametrize("inject", [True, False])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_a_prompt_read_out_switch_keeps_every_bit(monkeypatch, rows, inject, mode):
    """forward.hc_block on a prompt chunk: plain (OCTOJET_PREFILL_HC_FUSED=0), each fusion alone, both."""

    hc = _hc(40 + inject, inject, "centred")
    plain = _block(monkeypatch, hc, rows, D, mode, "0", rows + mode)
    for setting in ("1", "norm", "up"):
        got = _block(monkeypatch, hc, rows, D, mode, setting, rows + mode)
        assert _same_block(got, plain) == [], setting


def test_the_switch_and_the_dispatch():
    hc = _hc(50, True)
    b = SimpleNamespace(prefill=True)
    assert hc_prefill.eligible(hc, b, S)
    assert not hc_prefill.eligible(hc, SimpleNamespace(prefill=False), S)        # decode windows: their own fusion
    assert not hc_prefill.eligible(HC(hc.down, hc.up, hc.scale, True), b, S)     # no prefill packing
    assert not hc_prefill.eligible(hc, b, 2)


def _real_hcs(w):
    return [w.layers[0].attn_hc, w.layers[0].mlp_hc, w.layers[1].attn_hc, w.layers[1].mlp_hc, w.mixer]


def _real_block_check(monkeypatch, w):
    d = w.cfg.hidden
    assert w.cfg.streams == S and w.cfg.low == LOW
    for i, hc in enumerate(_real_hcs(w)):
        assert hc_prefill.eligible(hc, SimpleNamespace(prefill=True), S), i
        for rows in (17, 2048):
            for mode in ((1, 2) if hc.inject else (0,)):
                plain = _block(monkeypatch, hc, rows, d, mode, "0", 7 * rows + i)
                got = _block(monkeypatch, hc, rows, d, mode, "1", 7 * rows + i)
                assert _same_block(got, plain) == [], (i, rows, mode)


@pytest.fixture(scope="module")
def mlx_cut():
    from tensorfold.families.qwen4_exp.cuda import weights as W
    real = W.Config.read
    def read(d, **kw):                                       # (rope_yarn and other Config.read options pass through)
        c = real(d, **kw); c.layers = MLX_LAYERS; c.ple_layers = [i for i in c.ple_layers if i < MLX_LAYERS]; return c
    W.Config.read = staticmethod(read)
    try:
        w = W.load(MLX, "cuda", mtp=True, draft_vocab="default", packed_cache="off")
    finally:
        W.Config.read = real
    yield w
    del w; torch.cuda.empty_cache()


@needs_model
def test_the_mixed_checkpoints_hc_weights(cut, monkeypatch):
    _real_block_check(monkeypatch, cut)


@needs_mlx
def test_the_mlx_checkpoints_hc_weights(mlx_cut, monkeypatch):
    _real_block_check(monkeypatch, mlx_cut)


def _prefill_both(monkeypatch, w, size: int, rows: int):
    prompt = [int(t) for t in np.random.default_rng(size).integers(0, 1000, size=size)]
    monkeypatch.setenv(hc_prefill.ENV, "0")
    plain = run(w, prompt, armed=False, rows=rows)
    monkeypatch.setenv(hc_prefill.ENV, "1")
    fused = run(w, prompt, armed=False, rows=rows)
    return plain, fused


@needs_model
@pytest.mark.parametrize("rows,size", [(2048, 4500), (1024, 2600), (64, 300)])
def test_a_whole_prefill_is_the_same_with_the_switch_on_and_off(cut, monkeypatch, rows, size):
    plain, fused = _prefill_both(monkeypatch, cut, size, rows)
    assert same(plain, fused)


@needs_mlx
def test_a_whole_mlx_prefill_is_the_same_with_the_switch_on_and_off(mlx_cut, monkeypatch):
    plain, fused = _prefill_both(monkeypatch, mlx_cut, 2600, 1024)
    assert same(plain, fused)
