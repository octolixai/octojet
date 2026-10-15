"""F14: the prompt chain's double-buffered kernel (OCTOJET_GDN_CHAIN_AHEAD, default on) gives the staged kernel's
bits: outputs and final states, at any row count (partial last stages included), wide and narrow head counts, and
chunked; bf16 keys and unaligned rows keep the staged kernel; a whole Flash Next prefill is the same bit for bit with
the switch on and off, on the 4-layer cut of the mixed NVFP4 checkpoint (OCTOJET_NVFP4_FLASHNEXT) and of the MLX 4-bit
one (OCTOJET_MLX_FLASHNEXT)."""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import gdn  # noqa: E402
from test_flashnext_prefill_hc import mlx_cut, needs_mlx  # noqa: E402,F401
from test_prefill_timing_flashnext import bits_equal, cut, needs_model, run, same  # noqa: E402,F401

DEV = "cuda"
ROWS = [1, 2, 31, 32, 33, 63, 64, 65, 300, 1000, 4096]


def _inputs(rows: int, hk: int, hv: int, keys=torch.float32, seed: int = 0, scale: float = 0.08):
    """Flash Next's shapes: L2-normed keys and queries (times scale), gates in (0, 1), betas in (0, 1)."""

    gen = torch.Generator(device=DEV).manual_seed(1000 * hv + rows + seed)
    q = torch.nn.functional.normalize(torch.randn((rows, hk, 128), generator=gen, device=DEV), dim=-1) * scale
    k = torch.nn.functional.normalize(torch.randn((rows, hk, 128), generator=gen, device=DEV), dim=-1)
    v = torch.randn((rows, hv, 128), generator=gen, device=DEV).bfloat16()
    g = torch.rand((rows, hv), generator=gen, device=DEV) * 0.5 + 0.5
    beta = torch.rand((rows, hv), generator=gen, device=DEV)
    state = torch.randn((hv, 128, 128), generator=gen, device=DEV) * 0.05
    return q.to(keys).contiguous(), k.to(keys).contiguous(), v, g, beta, state


def _both(args):
    q, k, v, g, beta, state = args
    la, lb = torch.empty_like(state), torch.empty_like(state)
    ya, ran_a = gdn.chain_form(q, k, v, g, beta, state, la, ahead=True)
    yb, ran_b = gdn.chain_form(q, k, v, g, beta, state, lb, ahead=False)
    return (ya, la, ran_a), (yb, lb, ran_b)


def _sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("hk,hv", [(16, 48), (8, 24), (2, 6), (16, 96)])
def test_ahead_gives_the_staged_bits(rows, hk, hv):
    (ya, la, ran_a), (yb, lb, ran_b) = _both(_inputs(rows, hk, hv))
    assert ran_a and not ran_b
    assert bits_equal(ya, yb) and bits_equal(la, lb)


def test_both_widths_run():
    """A head count at or past the SM count takes 128 value rows a block, fewer takes 64: both forms, both widths."""

    sms = _sms()
    for hv in (sms, max(1, sms // 2)):
        (ya, la, ran_a), (yb, lb, _) = _both(_inputs(777, 1, hv))
        assert ran_a and bits_equal(ya, yb) and bits_equal(la, lb), hv


@pytest.mark.parametrize("scale", [1e-3, 0.08, 1.0, 30.0])
def test_extreme_values_keep_the_bits(scale):
    """Large and tiny queries; gates near 0 and 1; zero rows of values."""

    q, k, v, g, beta, state = _inputs(333, 16, 48, scale=scale, seed=7)
    g[::5] = 1.0
    g[1::7] = 1e-6
    v[3::11] = 0
    (ya, la, ran_a), (yb, lb, _) = _both((q, k, v, g, beta, state))
    assert ran_a and bits_equal(ya, yb) and bits_equal(la, lb)


@pytest.mark.parametrize("size", [1, 7, 32, 33, 256, 1000])
def test_chunked_ahead_equals_the_whole_staged_chain(size):
    """The chunk invariance holds across the forms: a prompt chained in pieces by ahead_kernel gives the bits of one
    staged chain over the whole prompt."""

    rows = 2100
    q, k, v, g, beta, state = _inputs(rows, 16, 48, seed=3)
    last = torch.empty_like(state)
    whole, ran = gdn.chain_form(q, k, v, g, beta, state, last, ahead=False)
    assert not ran
    cur, parts = state, []
    for a in range(0, rows, size):
        nxt = torch.empty_like(state)
        y, ran = gdn.chain_form(*(x[a:a + size].contiguous() for x in (q, k, v, g, beta)), cur, nxt, ahead=True)
        assert ran
        parts.append(y)
        cur = nxt
    assert bits_equal(whole, torch.cat(parts)) and bits_equal(last, cur), size


def test_bf16_keys_keep_the_staged_kernel():
    (ya, la, ran_a), (yb, lb, ran_b) = _both(_inputs(300, 2, 6, keys=torch.bfloat16))
    assert not ran_a and not ran_b
    assert bits_equal(ya, yb) and bits_equal(la, lb)


def test_unaligned_rows_keep_the_staged_kernel():
    """A value tensor 2 bytes past a 16-byte boundary (contiguous, so it passes the shape checks) is not copied with
    16-byte cp.async: the staged kernel runs and the bits are those of an aligned copy."""

    q, k, v, g, beta, state = _inputs(200, 16, 48, seed=5)
    buf = torch.empty((v.numel() + 8,), dtype=v.dtype, device=DEV)
    vu = buf[1:1 + v.numel()].view(v.shape)
    vu.copy_(v)
    assert vu.is_contiguous() and vu.data_ptr() % 16 != 0
    l1, l2 = torch.empty_like(state), torch.empty_like(state)
    y1, ran = gdn.chain_form(q, k, vu, g, beta, state, l1, ahead=True)
    assert not ran
    y2, ran2 = gdn.chain_form(q, k, v, g, beta, state, l2, ahead=True)
    assert ran2 and bits_equal(y1, y2) and bits_equal(l1, l2)


def test_the_switch_picks_the_form(monkeypatch):
    args = _inputs(100, 16, 48, seed=9)
    q, k, v, g, beta, state = args
    monkeypatch.delenv(gdn.AHEAD_ENV, raising=False)
    assert gdn.chain_form(*args[:6], torch.empty_like(state))[1]
    monkeypatch.setenv(gdn.AHEAD_ENV, "0")
    assert not gdn.chain_form(*args[:6], torch.empty_like(state))[1]
    l_off = torch.empty_like(state)
    y_off = gdn.chain(q, k, v, g, beta, state, l_off)
    monkeypatch.setenv(gdn.AHEAD_ENV, "1")
    l_on = torch.empty_like(state)
    y_on = gdn.chain(q, k, v, g, beta, state, l_on)
    assert bits_equal(y_on, y_off) and bits_equal(l_on, l_off)


def _prefill_both(monkeypatch, w, size: int, rows: int):
    prompt = [int(t) for t in np.random.default_rng(size + 14).integers(0, 1000, size=size)]
    monkeypatch.setenv(gdn.AHEAD_ENV, "0")
    staged = run(w, prompt, armed=False, rows=rows)
    monkeypatch.setenv(gdn.AHEAD_ENV, "1")
    ahead = run(w, prompt, armed=False, rows=rows)
    return staged, ahead


@needs_model
@pytest.mark.parametrize("rows,size", [(4096, 9000), (2048, 4500), (1024, 2600), (64, 300)])
def test_a_whole_prefill_is_the_same_with_the_switch_on_and_off(cut, monkeypatch, rows, size):
    staged, ahead = _prefill_both(monkeypatch, cut, size, rows)
    assert same(staged, ahead)


@needs_mlx
@pytest.mark.parametrize("rows,size", [(2048, 4500), (1024, 2600)])
def test_a_whole_mlx_prefill_is_the_same_with_the_switch_on_and_off(mlx_cut, monkeypatch, rows, size):
    staged, ahead = _prefill_both(monkeypatch, mlx_cut, size, rows)
    assert same(staged, ahead)
