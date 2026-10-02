"""Grouped MoE experts on MLX 4-bit weights; a (row, slot) pair's bits never depend on the other rows of the call."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

NTW = 4                  # n8 tiles a warp
COLS = 8 * NTW           # output columns a warp
TILE = 16                # pairs an item holds (decode form)
PREFILL_TILE = 64        # pairs an item holds (prefill form)
SMALL = 1024             # pairs the one-block plan takes; wider plans rank in blocks of 1024 pairs
PACK_VERSION = 1         # the packed block layout and packer; bump when either changes (the disk cache keys on it)
NVFP4_BLOCK = 32 * NTW + 16      # int32 words a block: 128 of codes, 16 of UE4M3 scales (byte nt of word gq*2 + sub)
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="octojet_experts_v1", sources=[str(here / "experts.cpp"), str(here / "experts.cu"),
                                                      str(here / "experts_prefill.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def _nibbles(w: torch.Tensor) -> torch.Tensor:
    """Nibbles to (i0, i2, i4, i6, i1, i3, i5, i7), so one shift and mask give two adjacent inputs as a bf16 pair."""

    w = w.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(w)
    for i in range(8):
        out |= ((w >> (4 * i)) & 0xF) << (4 * (i // 2 + 4 * (i % 2)))
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)


def _pack_words(words: torch.Tensor, gs: int, chunk: int = 32) -> torch.Tensor:
    """MLX words [E, N, K/8] -> B-fragments [E, N/32, K/gs, 32*wpl] int32, lane gq*4+t holding column nt*8+gq's
    inputs [8t, 8t+8) of the group in word nt (nibble order c0 c2 c4 c6 c1 c3 c5 c7)."""

    e, n, k8 = words.shape
    k = k8 * 8
    if n % COLS or k % gs:
        raise ValueError(f"experts: shape {tuple(words.shape)} does not pack in groups of {gs}")
    kg, nb, h = k // gs, n // COLS, gs // 32
    wpl = NTW * h
    out = torch.empty((e, nb, kg, 32 * wpl), dtype=torch.int32, device=words.device)
    for e0 in range(0, e, chunk):
        w = _nibbles(words[e0:e0 + chunk].view(torch.int32))
        c = w.shape[0]
        w = w.view(c, nb, NTW, 8, kg, 4, h).permute(0, 1, 4, 2, 6, 3, 5).reshape(c, nb, kg, wpl // 4, 4, 32)
        out[e0:e0 + c] = w.permute(0, 1, 2, 3, 5, 4).reshape(c, nb, kg, 32 * wpl)
    return out


def _unpack_words(frags: torch.Tensor, gs: int) -> torch.Tensor:
    """``_pack_words``'s inverse: [E, N/32, K/gs, 32*wpl] -> MLX words [E, N, K/8]."""

    e, nb, kg, _ = frags.shape
    h = gs // 32
    wpl = NTW * h
    w = frags.reshape(e, nb, kg, wpl // 4, 32, 4).permute(0, 1, 2, 3, 5, 4)
    w = w.reshape(e, nb, kg, NTW, h, 8, 4).permute(0, 1, 3, 5, 2, 6, 4).reshape(e, nb * COLS, kg * 4 * h)
    v = w.to(torch.int64) & 0xFFFFFFFF
    words = torch.zeros_like(v)
    for s in range(8):                                  # nibble slot s holds input 2 (s % 4) + s // 4
        words |= ((v >> (4 * s)) & 0xF) << (4 * (2 * (s % 4) + s // 4))
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)


def pack(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, gs: int, chunk: int = 32) -> torch.Tensor:
    """MLX words [E, N, K/8], scales and biases -> [E, N/32, K/gs, block]: B-fragments, then scales and biases."""

    if gs not in (32, 64):
        raise ValueError(f"groups of 32 or 64 inputs, not {gs}")
    e, n, k8 = words.shape
    k = k8 * 8
    if n % COLS or k % gs or scales.shape != (e, n, k // gs) or biases.shape != scales.shape:
        raise ValueError(f"experts: shape {tuple(words.shape)} with scales {tuple(scales.shape)} does not pack")
    kg, nb, h = k // gs, n // COLS, gs // 32
    wpl = NTW * h
    out = torch.empty((e, nb, kg, 32 * wpl + 8 * NTW), dtype=torch.int32, device=words.device)
    out[..., :32 * wpl] = _pack_words(words, gs, chunk)
    for e0 in range(0, e, chunk):
        c = min(chunk, e - e0)
        sb = []
        for t in (scales, biases):
            v = t[e0:e0 + c].reshape(c, nb, NTW, 4, 2, kg).permute(0, 1, 5, 3, 2, 4).contiguous()
            sb.append(v.view(torch.int32).reshape(c, nb, kg, 4, NTW))
        out[e0:e0 + c, :, :, 32 * wpl:] = torch.cat(sb, dim=-1).reshape(c, nb, kg, 8 * NTW)
    return out


def unpack(blocks: torch.Tensor, gs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``pack``'s inverse: blocks [E, N/32, K/gs, block] -> MLX words [E, N, K/8], scales and biases [E, N, K/gs]."""

    e, nb, kg, _ = blocks.shape
    h = gs // 32
    wpl = NTW * h
    words = _unpack_words(blocks[..., :32 * wpl], gs)
    sb = blocks[..., 32 * wpl:].contiguous().view(torch.bfloat16).reshape(e, nb, kg, 4, 2, NTW, 2)
    scales, biases = (sb[:, :, :, :, i].permute(0, 1, 4, 3, 5, 2).reshape(e, nb * COLS, kg) for i in (0, 1))
    return words, scales.contiguous(), biases.contiguous()


@dataclass
class Experts:
    """One layer's experts (the shared ones included): gate and up (SwiGLU) or up (relu^2), and down."""

    up: torch.Tensor          # [E, NI/32, D/gs, M, block] int32, M = 2 (gate, then up) or 1
    down: torch.Tensor        # [E, D/32, NI/gs, 1, block]
    gs: int
    width: int                # NI: the expert's (or this rank's) intermediate width
    dims: int                 # D
    limit: float = 0.0        # SwiGLU clip (0: none)
    fmt: str = "affine"       # "affine" (MLX 4-bit) or "nvfp4" (E2M1 codes, UE4M3 scales, fp32 global scale)
    gscale_up: torch.Tensor | None = None     # nvfp4: [E, M] fp32
    gscale_down: torch.Tensor | None = None   # nvfp4: [E, 1] fp32

    @property
    def count(self) -> int:
        return self.up.shape[0]

    @property
    def swiglu(self) -> bool:
        return self.up.shape[3] == 2

    def bytes_per_expert(self) -> int:
        return (self.up[0].numel() + self.down[0].numel()) * 4

    def to(self, device) -> "Experts":
        """The same table with its tensors on ``device``."""

        return Experts(self.up.to(device), self.down.to(device), self.gs, self.width, self.dims, self.limit,
                       fmt=self.fmt, gscale_up=None if self.gscale_up is None else self.gscale_up.to(device),
                       gscale_down=None if self.gscale_down is None else self.gscale_down.to(device))


def make(up: list[tuple], down: tuple, gs: int, *, limit: float = 0.0) -> Experts:
    """One layer's MLX arrays: ``up`` [gate, up] (SwiGLU) or [up] (relu^2), ``down``, each (words, scales, biases)."""

    if len(up) not in (1, 2):
        raise ValueError("experts take a gate and an up projection, or an up projection alone")
    width, dims = up[0][0].shape[1], up[0][0].shape[2] * 8
    u = torch.stack([pack(*m, gs) for m in up], dim=3)
    d = pack(*down, gs).unsqueeze(3)
    return Experts(u, d, gs, width, dims, float(limit))


def nvfp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] u8 (two E2M1 codes a byte, low nibble = even input) -> [..., K] u8 codes."""

    return torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(*packed.shape[:-1], -1)


def nvfp4_pack_codes(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def ue4m3_to_float(codes: torch.Tensor) -> torch.Tensor:
    """UE4M3 (bias 7, no sign) bytes -> fp32; 0x7F would be NaN and never occurs in a scale."""

    c = codes.long()
    e, m = (c >> 3) & 15, c & 7
    return torch.where(e > 0, (1.0 + m / 8.0) * torch.pow(2.0, (e - 7).float()), m * 2.0 ** -9).float()


def dequant_nvfp4(packed: torch.Tensor, scales: torch.Tensor, gscale: torch.Tensor) -> torch.Tensor:
    """fp32 [E, N, K] = e2m1(code) * fp8(scale) * gscale[e]."""

    c = nvfp4_codes(packed).long()
    mag = E2M1_VALUES.to(c.device)[c & 7]
    val = torch.where((c & 8) != 0, -mag, mag)
    s = scales.float().repeat_interleave(16, dim=-1)
    return val * s * gscale.float().reshape(-1, 1, 1)


def _words_from_codes(codes: torch.Tensor) -> torch.Tensor:
    """u8 codes [E, N, K] -> MLX-style int32 words [E, N, K/8], code i of each eight in nibble i."""

    e, n, k = codes.shape
    c = codes.reshape(e, n, k // 8, 8).to(torch.int64)
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * 4
    w = (c << shifts).sum(-1)
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)


def pack_nvfp4(packed: torch.Tensor, scales: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """NVFP4 [E, N, K/2] codes + [E, N, K/16] fp8 scales -> [E, N/32, K/32, NVFP4_BLOCK] int32: the B-fragments,
    then 16 scale words (word gq*2 + sub, byte nt = the UE4M3 scale of column nt*8+gq for inputs [16 sub, 16 sub + 16))."""

    e, n, k2 = packed.shape
    k = 2 * k2
    if n % COLS or k % 32 or tuple(scales.shape) != (e, n, k // 16):
        raise ValueError(f"nvfp4: shape {tuple(packed.shape)} with scales {tuple(scales.shape)} does not pack")
    nb, kg = n // COLS, k // 32
    out = torch.empty((e, nb, kg, NVFP4_BLOCK), dtype=torch.int32, device=packed.device)
    for e0 in range(0, e, chunk):                       # 32 experts at a time: the int64 code expansion of a whole
        c = min(chunk, e - e0)                          # 513-expert projection would take ~13 GB
        codes = nvfp4_codes(packed[e0:e0 + c])
        out[e0:e0 + c, :, :, :32 * NTW] = _pack_words(_words_from_codes(codes), 32)
        sc = scales[e0:e0 + c].view(torch.uint8).reshape(c, nb, NTW, 8, kg, 2)   # (c, nb, nt, gq, kg, sub)
        sc = sc.permute(0, 1, 4, 3, 5, 2).contiguous()                             # (c, nb, kg, gq, sub, nt)
        out[e0:e0 + c, :, :, 32 * NTW:] = sc.reshape(c, nb, kg, 16 * 4).view(torch.int32)  # byte nt of word gq*2+sub
    return out


def unpack_nvfp4(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    e, nb, kg, _ = blocks.shape
    words = _unpack_words(blocks[..., :32 * NTW].contiguous(), 32)
    v = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=blocks.device, dtype=torch.int64) * 4
    codes = ((v[..., None] >> shifts) & 0xF).reshape(e, nb * COLS, kg * 32).to(torch.uint8)
    sc = blocks[..., 32 * NTW:].contiguous().view(torch.uint8).reshape(e, nb, kg, 8, 2, NTW)
    scales = sc.permute(0, 1, 5, 3, 2, 4).reshape(e, nb * COLS, kg * 2).contiguous().view(torch.float8_e4m3fn)
    return nvfp4_pack_codes(codes), scales


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value, ties to the even code; saturate at 6; NaN -> 0."""

    a = x.abs()
    c = ((a > 0.25).to(torch.uint8) + (a >= 0.75).to(torch.uint8) + (a > 1.25).to(torch.uint8)
         + (a >= 1.75).to(torch.uint8) + (a > 2.5).to(torch.uint8) + (a >= 3.5).to(torch.uint8)
         + (a > 5.0).to(torch.uint8))
    return c | ((x < 0).to(torch.uint8) << 3)


def quantize_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ModelOpt 0.46.0's dynamic NVFP4 weight path (qtensor/nvfp4_tensor.py): gscale = amax / (6 * 448); block scale
    = amax(block) / (6 * gscale), all-zero blocks set to 1.0, clamped to [2^-9, 448], cast to fp8 (RNE); codes =
    e2m1_rne(w / (fp8(scale) * gscale)), ties at 0.75, 1.75, 3.5 rounding up. An all-zero matrix gives codes 0,
    scales 1.0 and gscale 0 (ModelOpt would divide by zero there)."""

    n, k = w.shape
    if k % 16:
        raise ValueError("nvfp4: K must be a multiple of 16")
    x = w.float()
    amax = x.abs().amax()
    blocks = x.reshape(n, k // 16, 16)
    if amax == 0:
        return (torch.zeros((n, k // 2), dtype=torch.uint8), torch.ones((n, k // 16)).to(torch.float8_e4m3fn),
                torch.zeros(()))
    gscale = amax / (6.0 * 448.0)
    per_block = blocks.abs().amax(-1) / (6.0 * gscale)
    per_block[per_block == 0] = 1.0
    scales = per_block.clamp(min=2.0 ** -9, max=448.0).to(torch.float8_e4m3fn)
    q = blocks / (scales.float() * gscale).unsqueeze(-1)
    codes = e2m1_encode(q).reshape(n, k)
    return nvfp4_pack_codes(codes), scales, gscale.reshape(())


def make_nvfp4(up: list[tuple], down: tuple, *, limit: float = 0.0) -> Experts:
    """One layer's NVFP4 experts: ``up`` = [gate, up], ``down``; each (packed codes, fp8 scales, gscale [E])."""

    if len(up) != 2:   # F1 dispatches NVFP4 for SwiGLU experts (gate, up) only; relu^2 experts would need (32, 1, 1, 1)
        raise ValueError("nvfp4 experts take a gate and an up projection")
    width, dims = up[0][0].shape[1], up[0][0].shape[2] * 2
    u = torch.stack([pack_nvfp4(p, s) for p, s, _ in up], dim=3)
    d = pack_nvfp4(down[0], down[1]).unsqueeze(3)
    gu = torch.stack([g.float().reshape(-1) for _, _, g in up], dim=1).contiguous()
    gd = down[2].float().reshape(-1, 1).contiguous()
    return Experts(u, d, 32, width, dims, float(limit), fmt="nvfp4", gscale_up=gu, gscale_down=gd)


def max_items(pairs: int, experts: int, tile: int = TILE) -> int:
    """Items a plan of ``pairs`` can hold: an item per used expert, plus one per ``tile`` pairs past its first."""

    return min(pairs, experts) + pairs // tile


class Plan:
    """Scratch grouping pairs by expert: members, items (expert, first, count), counts [items, distinct experts]."""

    def __init__(self, rows: int, slots: int, experts: int, device: torch.device | str, *,
                 prefill: bool = False) -> None:
        pairs = rows * slots
        self.rows, self.slots, self.experts, self.prefill = rows, slots, experts, prefill
        self.tile = PREFILL_TILE if prefill else TILE
        self.members = torch.zeros((pairs,), dtype=torch.int32, device=device)
        self.items = torch.zeros((max_items(pairs, experts, self.tile), 3), dtype=torch.int32, device=device)
        self.counts = torch.zeros((2,), dtype=torch.int32, device=device)
        wide = pairs > SMALL
        self.rank = torch.zeros((pairs if wide else 1,), dtype=torch.int32, device=device)
        self.hist = torch.zeros((-(-pairs // 1024) * experts if wide else 1,), dtype=torch.int32, device=device)


def route(picks: torch.Tensor, plan: Plan) -> None:
    """``picks`` [R, slots] int32, contiguous: each (row, slot) pair's expert id (shared experts included)."""

    rows, slots = picks.shape
    if slots != plan.slots or rows > plan.rows:
        raise ValueError(f"picks {tuple(picks.shape)} do not fit a plan of {plan.rows} x {plan.slots}")
    _ext().plan(picks, rows * slots, plan.experts, plan.tile, plan.members, plan.items, plan.counts, plan.rank,
                plan.hist)


def _fmt(ex: Experts) -> int:
    return {"affine": 0, "nvfp4": 1}[ex.fmt]


def _g(t: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    return t if t is not None else torch.empty((0,), dtype=torch.float32, device=like.device)


def gate_up(x: torch.Tensor, ex: Experts, plan: Plan, out: torch.Tensor, rows: int) -> None:
    """x [R, D] bf16 (unit-stride rows) -> out [R * slots, NI] bf16, each pair's activation."""

    items = max_items(rows * plan.slots, plan.experts, plan.tile)
    epi = 2 if ex.swiglu else 1
    args = (_fmt(ex), ex.gs, epi, x, plan.slots, ex.up, _g(ex.gscale_up, x), ex.dims // ex.gs, ex.width // COLS,
            plan.items, plan.counts, plan.members, out, ex.width, ex.limit)
    if plan.prefill:
        _ext().prefill(*args, items)
    else:
        _ext().run(*args, items * (ex.width // COLS))


def down(act: torch.Tensor, ex: Experts, plan: Plan, out: torch.Tensor, rows: int) -> None:
    """act [R * slots, NI] bf16 -> out [R * slots, D] per pair: fp32, or bf16 on a prefill plan when ``out`` is bf16."""

    items = max_items(rows * plan.slots, plan.experts, plan.tile)
    if plan.prefill:
        args = (_fmt(ex), ex.gs, 3 if out.dtype == torch.bfloat16 else 0, act, 0, ex.down, _g(ex.gscale_down, act),
                ex.width // ex.gs, ex.dims // COLS, plan.items, plan.counts, plan.members, out, ex.dims, 0.0)
        _ext().prefill(*args, items)
    else:
        args = (_fmt(ex), ex.gs, 0, act, 0, ex.down, _g(ex.gscale_down, act), ex.width // ex.gs, ex.dims // COLS,
                plan.items, plan.counts, plan.members, out, ex.dims, 0.0)
        _ext().run(*args, items * (ex.dims // COLS))
