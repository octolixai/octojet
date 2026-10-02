"""Grouped-expert kernels, decode and prefill: a pair's bits ignore the call's other rows; outputs track float64."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts  # noqa: E402

DEV = "cuda"

# (group size, SwiGLU, limit, experts incl. shared, D, NI, shared slots): Flash Next, GLM and Nemotron in small
CASES = [(32, True, 0.0, 41, 256, 96, 1), (64, True, 10.0, 19, 512, 128, 1), (64, False, 0.0, 12, 384, 192, 2)]
# NVFP4 cases: (fmt, SwiGLU, limit, experts incl. shared, D, NI, shared slots); groups are 32 by definition.
# The third case is Flash Next's real shape with 8 routed experts + the shared one.
NVFP4_CASES = [("nvfp4", True, 0.0, 41, 256, 96, 1), ("nvfp4", True, 10.0, 19, 512, 128, 1),
               ("nvfp4", True, 0.0, 9, 2560, 640, 1)]
ALL_CASES = CASES + NVFP4_CASES


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-for-bit equality (torch.equal calls -0.0 and 0.0 equal)."""

    itype = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[a.dtype]
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().view(itype), b.contiguous().view(itype)))


def nvfp4(e: int, n: int, k: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    packed = torch.randint(0, 256, (e, n, k // 2), generator=g, device=DEV, dtype=torch.int64).to(torch.uint8)
    # scales from 2^-9 (subnormal) to ~0.02 so both UE4M3 ranges are exercised; every (column, block) distinct enough
    scales = (torch.rand((e, n, k // 16), generator=g, device=DEV) * 0.02 + 0.0015).to(torch.float8_e4m3fn)
    gscale = torch.rand((e,), generator=g, device=DEV) * 0.5 + 0.5
    return packed, scales, gscale


def gs_of(case) -> int:
    return 32 if case[0] == "nvfp4" else case[0]


def dequant_any(m, case) -> torch.Tensor:
    if case[0] == "nvfp4":
        return experts.dequant_nvfp4(*m).double()
    return dequant(*m, case[0])


def mlx(e: int, n: int, k: int, gs: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (e, n, k // 8), generator=g, device=DEV,
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((e, n, k // gs), generator=g, device=DEV) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((e, n, k // gs), generator=g, device=DEV) * 0.02).to(torch.bfloat16)
    return words, scales, biases


def dequant(words, scales, biases, gs: int) -> torch.Tensor:
    w = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=DEV, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], -1).double()
    return q * scales.double().repeat_interleave(gs, -1) + biases.double().repeat_interleave(gs, -1)


def build(case, seed: int):
    fmt, swiglu, limit, e, d, ni, _ = case
    if fmt == "nvfp4":
        up = [nvfp4(e, ni, d, seed + i) for i in range(2 if swiglu else 1)]
        down = nvfp4(e, d, ni, seed + 7)
        return experts.make_nvfp4(up, down, limit=limit), up, down
    gs = fmt
    up = [mlx(e, ni, d, gs, seed + i) for i in range(2 if swiglu else 1)]
    down = mlx(e, d, ni, gs, seed + 7)
    return experts.make(up, down, gs, limit=limit), up, down


def picks_for(rows: int, case, seed: int, *, hot: int | None = None) -> torch.Tensor:
    """Each row: distinct routed experts (the first ``hot`` if given, so every row shares them), then the shared."""

    _, _, _, e, _, _, shared = case
    routed = e - shared
    k = min(6, routed)
    g = torch.Generator().manual_seed(seed)
    rows_ = []
    for _ in range(rows):
        r = torch.arange(k) if hot is not None else torch.randperm(routed, generator=g)[:k]
        rows_.append(torch.cat([r, torch.arange(routed, e)]))
    return torch.stack(rows_).to(torch.int32).to(DEV)


def run(ex, x, picks, *, prefill: bool = False, y_dtype=torch.float32):
    rows, slots = picks.shape
    plan = experts.Plan(rows, slots, ex.count, DEV, prefill=prefill)
    experts.route(picks.contiguous(), plan)
    act = torch.empty((rows * slots, ex.width), dtype=torch.bfloat16, device=DEV)
    y = torch.empty((rows * slots, ex.dims), dtype=y_dtype, device=DEV)
    experts.gate_up(x, ex, plan, act, rows)
    experts.down(act, ex, plan, y, rows)
    return act, y, plan


@pytest.mark.parametrize("gs", [32, 64])
def test_pack_round_trip(gs):
    w = mlx(3, 96, 256, gs, 61)
    back = experts.unpack(experts.pack(*w, gs), gs)
    assert all(torch.equal(a, b) for a, b in zip(back, w))


def test_pack_nvfp4_round_trip():
    packed, scales, _ = nvfp4(3, 96, 256, 62)
    p2, s2 = experts.unpack_nvfp4(experts.pack_nvfp4(packed, scales))
    assert torch.equal(p2, packed) and torch.equal(s2.view(torch.uint8), scales.view(torch.uint8))


@pytest.mark.parametrize("case", ALL_CASES)
def test_pairs_do_not_depend_on_the_window(case):
    ex, _, _ = build(case, 11)
    rows = 40
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    picks = picks_for(rows, case, 3)
    slots = picks.shape[1]
    act, y, _ = run(ex, x, picks)
    alone = [run(ex, x[r:r + 1], picks[r:r + 1]) for r in range(rows)]
    assert bits_equal(act, torch.cat([a[0] for a in alone])) and bits_equal(y, torch.cat([a[1] for a in alone]))
    for m in (2, 3, 8, 9, 16, 17, 33):
        a_m, y_m, _ = run(ex, x[:m], picks[:m])
        assert bits_equal(a_m, act[:m * slots]) and bits_equal(y_m, y[:m * slots]), m
    perm = torch.randperm(rows, generator=torch.Generator().manual_seed(5)).to(DEV)
    a_p, y_p, _ = run(ex, x[perm], picks[perm])
    idx = (perm[:, None] * slots + torch.arange(slots, device=DEV)).reshape(-1)
    assert bits_equal(a_p, act[idx]) and bits_equal(y_p, y[idx])


@pytest.mark.parametrize("case", ALL_CASES)
def test_an_expert_shared_by_many_rows(case):
    """Every row picks the same experts: items of 16 pairs, several a expert, same bits as alone."""

    ex, _, _ = build(case, 21)
    rows = 37
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    picks = picks_for(rows, case, 4, hot=0)
    _, y, plan = run(ex, x, picks)
    slots = picks.shape[1]
    alone = torch.cat([run(ex, x[r:r + 1], picks[r:r + 1])[1] for r in range(rows)])
    assert bits_equal(y, alone)
    assert int(plan.counts[1]) == slots
    assert int(plan.counts[0]) == slots * 3                    # 37 pairs an expert: items of 16, 16 and 5


def want_plan(picks: torch.Tensor, e: int, tile: int):
    flat = picks.reshape(-1).tolist()
    items, members = [], []
    for ex in range(e):
        pairs = [p for p, v in enumerate(flat) if v == ex]
        for j in range(0, len(pairs), tile):
            items.append([ex, len(members) + j, min(tile, len(pairs) - j)])
        members += pairs
    return items, members, len(set(flat))


@pytest.mark.parametrize("rows,prefill", [(23, False), (23, True), (300, False), (300, True), (1500, True)])
def test_plan_groups_pairs_by_expert(rows, prefill):
    """One block up to 1,024 pairs, then 1,024-pair blocks in turn, in items of 16 pairs (decode) or 64 (prefill)."""

    slots, e = 7, 50
    g = torch.Generator().manual_seed(9)
    picks = torch.stack([torch.randperm(e, generator=g)[:slots] for _ in range(rows)]).to(torch.int32)
    picks[:, 0] = 3                                            # one expert with a pair in every row
    plan = experts.Plan(rows, slots, e, DEV, prefill=prefill)
    tile = plan.tile
    experts.route(picks.to(DEV), plan)
    want_items, want_members, distinct = want_plan(picks, e, tile)
    n = int(plan.counts[0])
    assert n == len(want_items) and int(plan.counts[1]) == distinct
    assert plan.items[:n].tolist() == want_items
    assert plan.members.tolist() == want_members


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("case", ALL_CASES)
def test_matches_fp64(case, prefill):
    """Against float64 over the dequantized weights (rounded to bf16 first for the prefill form)."""

    swiglu, limit = case[1], case[2]
    ex, up, down = build(case, 31)
    rows = 5
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    picks = picks_for(rows, case, 6)
    act, y, _ = run(ex, x, picks, prefill=prefill)
    slots = picks.shape[1]
    pe = picks.reshape(-1).long()
    xr = x.double().repeat_interleave(slots, 0)

    def weights(m):
        w = dequant_any(m, case)
        # the affine prefill form rounds bf16(fma(q, s, b)); NVFP4's bf16(e2m1 * s) is exact in both forms
        return w.to(torch.bfloat16).double() if (prefill and case[0] != "nvfp4") else w

    projs = [torch.einsum("pk,pnk->pn", xr, weights(m)[pe]) for m in up]
    if swiglu:
        g, u = (t.float().to(torch.bfloat16).double() for t in projs)
        if limit:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        ref = (g * torch.sigmoid(g)).to(torch.bfloat16).double() * u
    else:
        ref = projs[0].float().to(torch.bfloat16).double().clamp(min=0) ** 2
    assert (act.double() - ref).abs().max() <= 2 ** -6 * ref.abs().max()
    yref = torch.einsum("pk,pnk->pn", act.double(), weights(down)[pe])
    assert (y.double() - yref).abs().max() <= 1e-5 * yref.abs().max()


def test_graph_replay_follows_the_picks():
    case = CASES[1]
    ex, _, _ = build(case, 41)
    rows = 9
    picks = picks_for(rows, case, 7).contiguous()
    slots = picks.shape[1]
    plan = experts.Plan(rows, slots, ex.count, DEV)
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    act = torch.empty((rows * slots, ex.width), dtype=torch.bfloat16, device=DEV)
    y = torch.empty((rows * slots, ex.dims), dtype=torch.float32, device=DEV)

    def step():
        experts.route(picks, plan)
        experts.gate_up(x, ex, plan, act, rows)
        experts.down(act, ex, plan, y, rows)

    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for seed in (8, 9):
        picks.copy_(picks_for(rows, case, seed))
        g.replay()
        _, want, _ = run(ex, x, picks)
        assert torch.equal(y, want)


def test_graph_replay_follows_the_picks_nvfp4():
    case = NVFP4_CASES[1]
    ex, _, _ = build(case, 41)
    rows = 9
    picks = picks_for(rows, case, 7).contiguous()
    slots = picks.shape[1]
    plan = experts.Plan(rows, slots, ex.count, DEV)
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    act = torch.empty((rows * slots, ex.width), dtype=torch.bfloat16, device=DEV)
    y = torch.empty((rows * slots, ex.dims), dtype=torch.float32, device=DEV)

    def step():
        experts.route(picks, plan)
        experts.gate_up(x, ex, plan, act, rows)
        experts.down(act, ex, plan, y, rows)

    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for seed in (8, 9):
        picks.copy_(picks_for(rows, case, seed))
        g.replay()
        _, want, _ = run(ex, x, picks)
        assert bits_equal(y, want)


@pytest.mark.parametrize("case", ALL_CASES)
def test_prefill_rows_do_not_depend_on_the_chunk(case):
    """Prefill form past 1,024 pairs: any chunk or row order keeps each row's bits; bf16 down is fp32 rounded."""

    ex, _, _ = build(case, 71)
    rows = 200
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    picks = picks_for(rows, case, 13)
    slots = picks.shape[1]
    assert rows * slots > experts.SMALL
    act, y, _ = run(ex, x, picks, prefill=True)
    for a, b in ((0, 1), (77, 78), (199, 200), (0, 7), (5, 22), (0, 64), (100, 200)):
        a1, y1, _ = run(ex, x[a:b], picks[a:b], prefill=True)
        assert bits_equal(a1, act[a * slots:b * slots]) and bits_equal(y1, y[a * slots:b * slots]), (a, b)
    perm = torch.randperm(rows, generator=torch.Generator().manual_seed(5)).to(DEV)
    a_p, y_p, _ = run(ex, x[perm], picks[perm], prefill=True)
    idx = (perm[:, None] * slots + torch.arange(slots, device=DEV)).reshape(-1)
    assert bits_equal(a_p, act[idx]) and bits_equal(y_p, y[idx])
    _, y16, _ = run(ex, x, picks, prefill=True, y_dtype=torch.bfloat16)
    assert bits_equal(y16, y.to(torch.bfloat16))
    _, y_hot, _ = run(ex, x[:90], picks_for(90, case, 4, hot=0), prefill=True)        # an expert in every row
    alone = torch.cat([run(ex, x[r:r + 1], picks_for(90, case, 4, hot=0)[r:r + 1], prefill=True)[1]
                       for r in range(0, 90, 29)])
    assert bits_equal(alone, y_hot.view(90, slots, -1)[0:90:29].reshape(-1, ex.dims))


def test_many_units_a_warp_keep_the_bits():
    """A call with far more units than resident warps (each warp takes several in turn) against rows alone."""

    case = (64, True, 0.0, 257, 2048, 128, 1)
    ex, _, _ = build(case, 81)
    rows = 64
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(12)
    picks = torch.stack([torch.cat([torch.randperm(256, generator=g)[:8], torch.tensor([256])])
                         for _ in range(rows)]).to(torch.int32).to(DEV)
    _, y, plan = run(ex, x, picks)
    assert int(plan.counts[0]) * (ex.dims // experts.COLS) > 8192
    for r in (0, 31, 63):
        assert torch.equal(run(ex, x[r:r + 1], picks[r:r + 1])[1], y[9 * r:9 * r + 9]), r


def test_many_units_a_warp_keep_the_bits_nvfp4():
    """A call with far more units than resident warps (each warp takes several in turn) against rows alone."""

    case = ("nvfp4", True, 0.0, 257, 2048, 128, 1)
    ex, _, _ = build(case, 81)
    rows = 64
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(12)
    picks = torch.stack([torch.cat([torch.randperm(256, generator=g)[:8], torch.tensor([256])])
                         for _ in range(rows)]).to(torch.int32).to(DEV)
    _, y, plan = run(ex, x, picks)
    assert int(plan.counts[0]) * (ex.dims // experts.COLS) > 8192
    for r in (0, 31, 63):
        assert bits_equal(run(ex, x[r:r + 1], picks[r:r + 1])[1], y[9 * r:9 * r + 9]), r


@pytest.mark.parametrize("prefill", [False, True])
def test_nvfp4_every_code_and_scale_is_exact(prefill):
    """Basis inputs: y[r, n] = e2m1(code[n, r]) * ue4m3(scale[n, r // 16]) * gscale[e], one exact product each."""

    e_, n_, k_ = 8, 32, 32
    codes = ((torch.arange(k_)[None, None, :] + 5 * (torch.arange(k_) >= 16)[None, None, :]
              + torch.arange(n_)[None, :, None] + 3 * torch.arange(e_)[:, None, None]) % 16).to(torch.uint8)
    sb = (torch.arange(e_ * n_ * 2) % 127).reshape(e_, n_, 2).to(torch.uint8)     # every finite UE4M3 byte, 4+ times
    packed = experts.nvfp4_pack_codes(codes).to(DEV)
    scales = sb.view(torch.float8_e4m3fn).to(DEV)
    gscale = (torch.arange(e_).float() * 0.37 + 0.5).to(DEV)
    zero_up = (torch.zeros((e_, k_, n_ // 2), dtype=torch.uint8, device=DEV),
               torch.zeros((e_, k_, n_ // 16), dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn),
               torch.ones(e_, device=DEV))
    ex = experts.make_nvfp4([zero_up, zero_up], (packed, scales, gscale))
    act = torch.eye(k_, device=DEV).to(torch.bfloat16)                             # row r selects input r
    ref = experts.dequant_nvfp4(packed, scales, gscale) + 0.0                         # +0.0: -0 products sum to +0
    for e in range(e_):
        picks = torch.full((k_, 1), e, dtype=torch.int32, device=DEV)
        plan = experts.Plan(k_, 1, ex.count, DEV, prefill=prefill)
        experts.route(picks, plan)
        y = torch.empty((k_, n_), dtype=torch.float32, device=DEV)
        experts.down(act, ex, plan, y, k_)
        assert bits_equal(y, ref[e].t().contiguous()), e


@pytest.mark.parametrize("prefill", [False, True])
def test_nvfp4_gate_up_every_code_and_scale_is_exact(prefill):
    """The two-matrix path with its SwiGLU epilogue, made exact: every gate product is 16 (code 2.0 x scale 2.0 x
    gscale 4.0), and bf16(silu(16)) == 16, so act = bf16(16 * up) where up runs every code and every scale byte with
    a distinct gscale per expert. The reference is exact because bf16(silu(16)) == 16 and multiplying by a power of
    two commutes with bf16 rounding over this fixture's range (-0 products are normalized to +0 as accumulators
    start at +0)."""

    e_, n_, k_ = 8, 32, 32
    codes = ((torch.arange(k_)[None, None, :] + 5 * (torch.arange(k_) >= 16)[None, None, :]
              + torch.arange(n_)[None, :, None] + 3 * torch.arange(e_)[:, None, None]) % 16).to(torch.uint8)
    sb = (torch.arange(e_ * n_ * 2) % 127).reshape(e_, n_, 2).to(torch.uint8)
    up = (experts.nvfp4_pack_codes(codes).to(DEV), sb.view(torch.float8_e4m3fn).to(DEV),
          (torch.arange(e_).float() * 0.37 + 0.5).to(DEV))
    gate = (torch.full((e_, n_, k_ // 2), 0x44, dtype=torch.uint8, device=DEV),                # code 4 = 2.0 twice a byte
            torch.full((e_, n_, k_ // 16), 0x40, dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn),  # scale 2.0
            torch.full((e_,), 4.0, device=DEV))
    down = (torch.zeros((e_, k_, n_ // 2), dtype=torch.uint8, device=DEV),
            torch.zeros((e_, k_, n_ // 16), dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn),
            torch.ones(e_, device=DEV))
    ex = experts.make_nvfp4([gate, up], down)
    x = torch.eye(k_, device=DEV).to(torch.bfloat16)
    ref = (16.0 * (experts.dequant_nvfp4(*up) + 0.0)).to(torch.bfloat16)                # [E, N, K]; +0.0: -0 -> +0
    for e in range(e_):
        picks = torch.full((k_, 1), e, dtype=torch.int32, device=DEV)
        plan = experts.Plan(k_, 1, ex.count, DEV, prefill=prefill)
        experts.route(picks, plan)
        act = torch.empty((k_, n_), dtype=torch.bfloat16, device=DEV)
        experts.gate_up(x, ex, plan, act, k_)
        assert bits_equal(act, ref[e].t().contiguous()), e


def test_nvfp4_pairs_independent_past_128_rows():
    """Items beyond the first 128 pairs of an expert, and the wide (> SMALL pairs) decode routing plan, keep bits."""

    case = NVFP4_CASES[0]
    ex, _, _ = build(case, 12)
    rows = 150
    x = (torch.randn((rows, ex.dims), device=DEV) * 0.5).to(torch.bfloat16)
    picks = picks_for(rows, case, 8, hot=0)                                          # every row shares expert 0..5
    slots = picks.shape[1]
    assert rows * slots > experts.SMALL                                              # the wide plan path
    act, y, _ = run(ex, x, picks)
    for r in (0, 63, 64, 127, 128, 129, 149):
        a1, y1, _ = run(ex, x[r:r + 1], picks[r:r + 1])
        assert bits_equal(a1, act[r * slots:(r + 1) * slots]) and bits_equal(y1, y[r * slots:(r + 1) * slots]), r
