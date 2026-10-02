"""The mixed loader on synthetic safetensors: expert stacking, shared-expert quantization, marker, estimate."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tensorfold.cuda import experts
from tensorfold.families.qwen4_exp.cuda import nvfp4

D, I, E = 64, 32, 4      # tiny Flash Next: hidden 64, width 32, 4 routed experts (2 a shard)


def write_export(root: Path, shared_fp8: str | None = None, shared_mult: float = 1.0):
    """``shared_fp8``: None writes the shared expert in bf16 (HF main); "tensor" writes it as fp8 e4m3 with a
    per-tensor [] fp32 ``weight_scale``; "noscale" writes fp8 without a companion scale."""

    root.mkdir()
    g = torch.Generator().manual_seed(1)
    wm = {}
    for a in (0, 2):
        tensors = {}
        for e in range(a, a + 2):
            for proj, (n, k) in {"gate_proj": (I, D), "up_proj": (I, D), "down_proj": (D, I)}.items():
                p = f"model.language_model.layers.0.mlp.experts.{e}.{proj}"
                tensors[p + ".weight"] = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.int64).to(torch.uint8)
                tensors[p + ".weight_scale"] = (torch.rand((n, k // 16), generator=g) * 0.02 + 0.002).to(torch.float8_e4m3fn)
                tensors[p + ".weight_scale_2"] = torch.rand((), generator=g) + 0.5
                tensors[p + ".input_scale"] = torch.tensor(1.0)
        name = f"layer-00000-experts-{a:04d}-{a + 1:04d}.safetensors"
        save_file(tensors, str(root / name))
        wm.update({k: name for k in tensors})
    shared = {f"model.language_model.layers.0.mlp.shared_expert.{proj}.weight": (torch.randn((n, k), generator=g) * 0.05 * shared_mult).to(torch.bfloat16)
              for proj, (n, k) in {"gate_proj": (I, D), "up_proj": (I, D), "down_proj": (D, I)}.items()}
    truth = dict(shared)
    if shared_fp8 is not None:
        for name, w in truth.items():
            w = w.float()
            s = w.abs().amax() / 448
            shared[name] = (w / s).to(torch.float8_e4m3fn)
            if shared_fp8 != "noscale":
                shared[name.removesuffix(".weight") + ".weight_scale"] = s.float().contiguous()
    shared["model.language_model.layers.0.mlp.gate.weight"] = torch.randn((E, D), generator=g).to(torch.bfloat16)
    save_file(shared, str(root / "model-bf16-00001.safetensors"))
    wm.update({k: "model-bf16-00001.safetensors" for k in shared})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    return truth if shared_fp8 is not None else shared


def test_marker_and_sources(tmp_path):
    base, export, out = tmp_path / "mlx", tmp_path / "export", tmp_path / "mixed"
    base.mkdir(); export.mkdir(); out.mkdir()
    (base / "config.json").write_text("{}")
    assert not nvfp4.is_mixed(out)
    (out / nvfp4.MARKER).write_text(json.dumps({"format": "nvfp4-mixed", "experts": str(export), "base": str(base)}))
    assert nvfp4.is_mixed(out)
    src = nvfp4.sources(out)
    assert src.experts == export and src.base == base


def test_routed_experts_stack_in_order(tmp_path):
    write_export(tmp_path / "export")
    rd = nvfp4.SafetensorsDir(tmp_path / "export")
    packed, scales, g = nvfp4.routed_experts(rd, 0, "gate_proj", E)
    assert packed.shape == (E, I, D // 2) and scales.shape == (E, I, D // 16) and g.shape == (E,)
    assert scales.dtype == torch.float8_e4m3fn and g.dtype == torch.float32
    want = rd.get("model.language_model.layers.0.mlp.experts.3.gate_proj.weight")
    assert torch.equal(packed[3], want)
    assert g[3].item() == rd.get("model.language_model.layers.0.mlp.experts.3.gate_proj.weight_scale_2").item()


def test_shared_expert_is_quantized_from_bf16(tmp_path):
    shared = write_export(tmp_path / "export")
    rd = nvfp4.SafetensorsDir(tmp_path / "export")
    packed, scales, g = nvfp4.shared_expert(rd, 0, "down_proj")
    w = shared["model.language_model.layers.0.mlp.shared_expert.down_proj.weight"].float()
    back = experts.dequant_nvfp4(packed, scales, g)[0]
    assert packed.shape == (1, D, I // 2)
    assert ((back - w).norm() / w.norm()).item() < 0.2


def _shared_rel(tmp_path, mode):
    truth = write_export(tmp_path / "export", shared_fp8=mode)
    rd = nvfp4.SafetensorsDir(tmp_path / "export")
    name = "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
    packed, scales, g = nvfp4.shared_expert(rd, 0, "gate_proj")
    back = experts.dequant_nvfp4(packed, scales, g)[0]
    w, raw = truth[name].float(), rd.get(name).float()
    return ((back - w).norm() / w.norm()).item(), ((back - raw).norm() / raw.norm()).item()


def test_shared_expert_fp8_is_dequantized_with_its_scale(tmp_path, capsys):
    rel, rel_raw = _shared_rel(tmp_path, "tensor")
    assert rel < 0.2
    assert rel_raw > 0.2                  # not a quantization of the raw fp8 codes
    out = capsys.readouterr()
    assert out.out == ""                  # stdout stays clean for the diagnostic tools' JSON
    assert "shared expert dequantized from fp8 (scale" in out.err


def write_block_fp8_shared(root: Path, n: int, k: int, extra: dict | None = None):
    """An export holding only layer 0's shared gate_proj in the local fp8hybrid layout: F8_E4M3 weight and an F32
    ``weight_scale_inv`` of 128x128 block scales [ceil(N/128), ceil(K/128)] that MULTIPLY the codes. Each block gets
    its own magnitude, so a misindexed block scale shows. Returns (truth, raw fp8 as float)."""

    root.mkdir()
    g = torch.Generator().manual_seed(3)
    nb, kb = -(-n // 128), -(-k // 128)
    mag = torch.rand((nb, kb), generator=g) * 0.9 + 0.1
    w = torch.randn((n, k), generator=g) * 0.05 * mag.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    pad = torch.zeros((nb * 128, kb * 128))
    pad[:n, :k] = w.abs()
    s = pad.reshape(nb, 128, kb, 128).amax(dim=(1, 3)) / 448
    w8 = (w / s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]).to(torch.float8_e4m3fn)
    p = "model.language_model.layers.0.mlp.shared_expert.gate_proj"
    tensors = {p + ".weight": w8, p + ".weight_scale_inv": s.float(), **{p + k_: v for k_, v in (extra or {}).items()}}
    save_file(tensors, str(root / "model-fp8-00001.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {t: "model-fp8-00001.safetensors" for t in tensors}}))
    return w, w8.float()


@pytest.mark.parametrize("n,k", [(256, 256), (200, 208)])       # (200, 208): partial last block in both dims
def test_shared_expert_fp8_block_scale_inv_is_multiplied(tmp_path, capsys, n, k):
    w, raw = write_block_fp8_shared(tmp_path / "export", n, k)
    rd = nvfp4.SafetensorsDir(tmp_path / "export")
    packed, scales, g = nvfp4.shared_expert(rd, 0, "gate_proj")
    back = experts.dequant_nvfp4(packed, scales, g)[0]
    assert back.shape == (n, k)
    assert ((back - w).norm() / w.norm()).item() < 0.2
    assert ((back - raw).norm() / raw.norm()).item() > 0.2         # not the raw fp8 codes (~448x larger)
    out = capsys.readouterr()
    assert out.out == ""
    assert "(scale model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_scale_inv)" in out.err


SHARED_P = "model.language_model.layers.0.mlp.shared_expert.gate_proj"


def write_shared_only(root: Path, tensors: dict):
    """An export holding only layer 0's shared gate_proj, as the given {suffix: tensor} (".weight", ".weight_scale", ...)."""

    root.mkdir()
    named = {SHARED_P + k: v for k, v in tensors.items()}
    save_file(named, str(root / "model-00001.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {t: "model-00001.safetensors" for t in named}}))
    return nvfp4.SafetensorsDir(root)


def _rel_to(rd, w):
    packed, scales, g = nvfp4.shared_expert(rd, 0, "gate_proj")
    back = experts.dequant_nvfp4(packed, scales, g)[0]
    return ((back - w).norm() / w.norm()).item()


def _truth(n=64, k=64, seed=5):
    return torch.randn((n, k), generator=torch.Generator().manual_seed(seed)) * 0.05


@pytest.mark.parametrize("shape", ["n", "n1"])
def test_shared_expert_fp8_per_row_scale_is_multiplied(tmp_path, shape):
    w = _truth() * torch.linspace(0.2, 1.0, 64)[:, None]                 # rows of different magnitude
    s = w.abs().amax(dim=1) / 448
    sc = s if shape == "n" else s[:, None]
    rd = write_shared_only(tmp_path / "e", {".weight": (w / s[:, None]).to(torch.float8_e4m3fn), ".weight_scale": sc.contiguous()})
    assert _rel_to(rd, w) < 0.2


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_shared_expert_fp16_fp32_pass_through(tmp_path, dtype):
    w = _truth().to(dtype)
    rd = write_shared_only(tmp_path / "e", {".weight": w})
    assert _rel_to(rd, w.float()) < 0.2


def test_shared_expert_e5m2_with_per_tensor_scale(tmp_path):
    w = _truth()
    s = w.abs().amax() / 448
    rd = write_shared_only(tmp_path / "e", {".weight": (w / s).to(torch.float8_e5m2), ".weight_scale": s.float().contiguous()})
    assert _rel_to(rd, w) < 0.2


def test_shared_expert_malformed_scale_shape_raises(tmp_path):
    w = _truth()
    rd = write_shared_only(tmp_path / "e", {".weight": w.to(torch.float8_e4m3fn), ".weight_scale": torch.ones(65)})
    with pytest.raises(ValueError, match="does not fit"):
        nvfp4.shared_expert(rd, 0, "gate_proj")


def test_shared_expert_weight_scale_inv_wins_over_weight_scale(tmp_path):
    w, _ = write_block_fp8_shared(tmp_path / "e", 256, 256, extra={".weight_scale": torch.tensor(1e-3)})
    assert _rel_to(nvfp4.SafetensorsDir(tmp_path / "e"), w) < 0.2       # the per-tensor 1e-3 would leave ~0.001x the truth


def test_shared_expert_non_finite_raises(tmp_path):
    w = _truth().to(torch.bfloat16)
    w[3, 4] = float("nan")
    rd = write_shared_only(tmp_path / "e", {".weight": w})
    with pytest.raises(ValueError, match=r"shared_expert\.gate_proj.*non-finite"):
        nvfp4.shared_expert(rd, 0, "gate_proj")


def test_shared_expert_fp8_without_scale_raises(tmp_path):
    write_export(tmp_path / "export", shared_fp8="noscale")
    rd = nvfp4.SafetensorsDir(tmp_path / "export")
    with pytest.raises(ValueError, match="weight_scale"):
        nvfp4.shared_expert(rd, 0, "up_proj")


def test_shared_expert_out_of_range_raises(tmp_path):
    write_export(tmp_path / "export", shared_mult=448 / 0.05 / 3)       # bf16 values ~O(448), like raw fp8 codes
    rd = nvfp4.SafetensorsDir(tmp_path / "export")
    with pytest.raises(ValueError, match="fp8"):
        nvfp4.shared_expert(rd, 0, "down_proj")


def test_make_layer_puts_shared_last(tmp_path):
    write_export(tmp_path / "export")
    mixed = nvfp4.Mixed(experts=tmp_path / "export", base=tmp_path / "export")
    ex = nvfp4.make_layer(mixed, 0, "cpu", n_experts=E, hidden=D, width=I, shared_width=I)
    assert ex.fmt == "nvfp4" and ex.count == E + 1 and ex.width == I and ex.dims == D
    rd = nvfp4.SafetensorsDir(tmp_path / "export")
    p2, s2 = experts.unpack_nvfp4(ex.up[:, :, :, 0].contiguous())
    assert torch.equal(p2[2], rd.get("model.language_model.layers.0.mlp.experts.2.gate_proj.weight"))
    sp, ss, sg = nvfp4.shared_expert(rd, 0, "gate_proj")                     # expert E (last) is the shared one
    assert torch.equal(p2[E], sp[0]) and torch.equal(s2[E].view(torch.uint8), ss[0].view(torch.uint8))
    assert ex.gscale_up[E, 0].item() == sg[0].item()
    assert ex.gscale_up[3, 1].item() == rd.get("model.language_model.layers.0.mlp.experts.3.up_proj.weight_scale_2").item()


def test_agreement_catches_a_lost_matrix_scale():
    g = torch.Generator().manual_seed(2)
    w = torch.randn((64, 256), generator=g) * 0.05
    p, s, gs = experts.quantize_nvfp4(w)
    w4 = experts.dequant_nvfp4(p[None], s[None], gs[None])[0]                   # a 4-bit copy: agrees
    assert nvfp4.agreement(w, w4)["ok"]
    assert not nvfp4.agreement(w, 2 * w4)["ok"]                                  # magnitude doubled: fails
    assert not nvfp4.agreement(w, w4[torch.randperm(64, generator=g)])["ok"]   # rows shuffled: fails


def test_estimate_transform_scales_switch_mlp():
    inner = lambda name, info: (1600, 7)  # noqa: E731
    t = nvfp4.estimate_transform(inner)
    assert t("language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight", {}) == (1440, 7)
    assert t("language_model.model.layers.3.mlp.gate.weight", {}) == (1600, 7)
    assert t("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight", {}) == (1600, 7)   # MTP head: affine
