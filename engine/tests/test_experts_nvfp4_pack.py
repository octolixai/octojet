"""NVFP4 packers, reference dequantization and quantizer, on the CPU (no CUDA needed)."""

import pytest
import torch

from tensorfold.cuda import experts

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def fp8(vals):
    return torch.tensor(vals, dtype=torch.float32).to(torch.float8_e4m3fn)


def random_nvfp4(e, n, k, seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (e, n, k // 2), generator=g, dtype=torch.int64).to(torch.uint8)
    scales = (torch.rand((e, n, k // 16), generator=g) * 0.02 + 0.001).to(torch.float8_e4m3fn)
    gscale = torch.rand((e,), generator=g) * 0.5 + 0.5
    return packed, scales, gscale


def test_codes_low_nibble_first():
    packed = torch.tensor([[0x21, 0xF0]], dtype=torch.uint8)
    assert experts.nvfp4_codes(packed).tolist() == [[1, 2, 0, 15]]
    assert torch.equal(experts.nvfp4_pack_codes(experts.nvfp4_codes(packed)), packed)


def test_dequant_reference_values():
    # one row, 16 inputs: codes 0..15 (0..7 positive, 8..15 negative), scale 1.0 (0x38), gscale 2.0
    codes = torch.arange(16, dtype=torch.uint8).reshape(1, 1, 16)
    packed = experts.nvfp4_pack_codes(codes)
    scales = fp8([[[1.0]]])
    out = experts.dequant_nvfp4(packed, scales, torch.tensor([2.0]))
    want = [2 * v for v in E2M1] + [-2 * v for v in E2M1]
    assert out.reshape(-1).tolist() == want


def test_ue4m3_to_bf16_table():
    # every finite UE4M3 code decodes exactly and survives the round trip through bf16 (<= 4 significant bits)
    codes = torch.arange(0, 0x7F, dtype=torch.uint8)
    vals = experts.ue4m3_to_float(codes)
    e, m = (codes.long() >> 3) & 15, codes.long() & 7
    want = torch.where(e > 0, (1 + m / 8) * 2.0 ** (e - 7), m * 2.0 ** -9)
    assert torch.equal(vals, want.float())
    assert torch.equal(vals.to(torch.bfloat16).float(), vals)
    assert vals[1].item() == 2 ** -9 and vals[0x7E].item() == 448.0


def test_pack_round_trip():
    packed, scales, _ = random_nvfp4(3, 96, 256, 61)
    blocks = experts.pack_nvfp4(packed, scales)
    assert blocks.shape == (3, 3, 8, experts.NVFP4_BLOCK) and blocks.dtype == torch.int32
    p2, s2 = experts.unpack_nvfp4(blocks)
    assert torch.equal(p2, packed) and torch.equal(s2.view(torch.uint8), scales.view(torch.uint8))


def test_pack_nvfp4_layout():
    """Lane (gq, t) of a block holds column nt*8+gq, inputs [8t, 8t+8) in TensorFold's nibble order; its scale bytes
    are byte nt of word 128 + gq*2 + sub."""

    e, n, k = 1, 32, 32
    codes = torch.zeros((e, n, k), dtype=torch.uint8)
    codes[0, 13, 16:24] = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.uint8)   # column 13: nt 1, gq 5; t 2
    scales = torch.zeros((e, n, k // 16), dtype=torch.uint8)
    scales[0, 13, 0], scales[0, 13, 1] = 0x40, 0x38
    blocks = experts.pack_nvfp4(experts.nvfp4_pack_codes(codes), scales.view(torch.float8_e4m3fn))
    lane, nt = 5 * 4 + 2, 1
    words = blocks[0, 0, 0]
    assert (words[lane * 4 + nt].item() & 0xFFFFFFFF) == 0x86427531       # nibbles (c0 c2 c4 c6 c1 c3 c5 c7)
    assert int(words[:128].ne(0).sum()) == 1
    assert ((words[128 + 5 * 2 + 0].item() >> (8 * nt)) & 0xFF) == 0x40
    assert ((words[128 + 5 * 2 + 1].item() >> (8 * nt)) & 0xFF) == 0x38
    assert int(words[128:].ne(0).sum()) == 2


def test_e2m1_encode_rounding():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 100.0, -100.0, -0.6, 0.0, 0.3])
    assert experts.e2m1_encode(x).tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 15, 9, 0, 1]


def test_quantize_reproduces_grid_tensors():
    """A tensor already on the NVFP4 grid, with a 448 scale somewhere and a magnitude-6 code in every block,
    quantizes back to exactly the same codes, scales and global scale."""

    g = torch.Generator().manual_seed(5)
    n, k = 64, 256
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.int64)
    codes[codes == 8] = 0                                                   # -0 is not a value the quantizer emits
    codes[:, 0::16] = 7                                                     # every block holds a +6
    scales_u8 = torch.randint(0x20, 0x60, (n, k // 16), generator=g, dtype=torch.int64)
    scales_u8[0, 0] = 0x7E                                                  # 448: the block with the tensor amax
    scales = scales_u8.to(torch.uint8).view(torch.float8_e4m3fn)
    gscale = torch.tensor(0.0031)
    w = experts.dequant_nvfp4(experts.nvfp4_pack_codes(codes.to(torch.uint8))[None], scales[None], gscale[None])[0]
    p2, s2, g2 = experts.quantize_nvfp4(w)
    assert torch.equal(experts.nvfp4_codes(p2).long(), codes)
    assert torch.equal(s2.view(torch.uint8), scales.view(torch.uint8))
    assert torch.isclose(g2, gscale, rtol=1e-6)


def test_quantize_zero_and_tiny_blocks_match_modelopt():
    # amax 1 -> gscale 1/(6*448); an all-zero block gets scale 1.0 (0x38) and codes 0; a block of 1e-6 values would
    # underflow to scale 0 but ModelOpt clamps it to 2^-9 (0x01) and keeps the values as nonzero codes
    w = torch.zeros((1, 48))
    w[0, 0] = 1.0
    w[0, 32:48] = 1e-6
    p, s, g = experts.quantize_nvfp4(w)
    sb = s.view(torch.uint8)[0].tolist()
    assert sb[0] == 0x7E and sb[1] == 0x38 and sb[2] == 0x01
    codes = experts.nvfp4_codes(p)[0]
    assert codes[16:32].eq(0).all() and codes[32:48].ne(0).all()
    assert torch.isclose(g, torch.tensor(1.0 / (6 * 448)))


def test_quantize_all_zero_matrix():
    p, s, g = experts.quantize_nvfp4(torch.zeros((2, 32)))
    assert p.eq(0).all() and s.view(torch.uint8).eq(0x38).all() and g.item() == 0.0
    assert experts.dequant_nvfp4(p[None], s[None], g[None]).eq(0).all()


def test_quantize_error_is_bounded():
    g = torch.Generator().manual_seed(6)
    w = torch.randn((128, 640), generator=g) * 0.05
    p, s, gs = experts.quantize_nvfp4(w)
    back = experts.dequant_nvfp4(p[None], s[None], gs[None])[0]
    assert ((back - w).norm() / w.norm()).item() < 0.2
    assert torch.isfinite(back).all()


def test_make_nvfp4_shapes():
    up = [random_nvfp4(5, 96, 256, i) for i in (1, 2)]
    down = random_nvfp4(5, 256, 96, 3)
    ex = experts.make_nvfp4(up, down)
    assert ex.fmt == "nvfp4" and ex.gs == 32 and ex.count == 5 and ex.width == 96 and ex.dims == 256
    assert ex.up.shape == (5, 3, 8, 2, experts.NVFP4_BLOCK) and ex.down.shape == (5, 8, 3, 1, experts.NVFP4_BLOCK)
    assert ex.gscale_up.shape == (5, 2) and ex.gscale_down.shape == (5, 1)
    assert ex.swiglu
    with pytest.raises(ValueError):
        experts.make_nvfp4(up[:1], down)


@pytest.mark.parametrize("gs", [32, 64])
def test_affine_pack_round_trip_cpu(gs):
    """The affine ``pack``/``unpack`` (built on ``_pack_words``/``_unpack_words``) round-trip MLX arrays exactly."""

    g = torch.Generator().manual_seed(71 + gs)
    e, n, k = 3, 64, 256
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (e, n, k // 8), generator=g, dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((e, n, k // gs), generator=g) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((e, n, k // gs), generator=g) * 0.02).to(torch.bfloat16)
    w2, s2, b2 = experts.unpack(experts.pack(words, scales, biases, gs), gs)
    assert torch.equal(w2, words) and torch.equal(s2, scales) and torch.equal(b2, biases)
