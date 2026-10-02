"""Tiny synthetic sources for the self-contained mixed checkpoint: an MLX-layout base (two shards, one weight whose
scales and biases sit in the other shard) and an NVFP4 export in RadixArk's layout (routed shards, a bf16 shard with
the shared experts, routers and unneeded tensors), plus the local symlink layout built from them."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

D, I, E, LAYERS = 128, 64, 16, 2           # hidden, expert width, routed experts, decoder layers
PROJ = {"gate_proj": (I, D), "up_proj": (I, D), "down_proj": (D, I)}


def _affine(g, prefix: str, n: int, k: int, lead: tuple = ()) -> dict:
    return {prefix + ".weight": torch.randint(0, 2**31 - 1, (*lead, n, k // 8), generator=g, dtype=torch.int64).to(torch.uint32),
            prefix + ".scales": torch.randn((*lead, n, k // 32), generator=g).to(torch.bfloat16),
            prefix + ".biases": torch.randn((*lead, n, k // 32), generator=g).to(torch.bfloat16)}


def write_mlx(root: Path, *, layers: int = LAYERS, experts: int = E) -> dict[str, torch.Tensor]:
    root.mkdir(parents=True)
    g = torch.Generator().manual_seed(7)
    text = {"num_hidden_layers": layers, "num_experts": experts, "hidden_size": D, "moe_intermediate_size": I,
            "shared_expert_intermediate_size": I}
    (root / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": text,
                                                  "quantization": {"bits": 4, "group_size": 32}}))
    for name, body in {"tokenizer.json": "{}", "tokenizer_config.json": "{}", "generation_config.json": "{}",
                       "chat_template.jinja": "{{ messages }}", "README.md": "# mlx", "LICENSE": "license",
                       ".gitattributes": "*"}.items():
        (root / name).write_text(body)
    shards: list[dict] = [{}, {}]
    for layer in range(layers):
        t = shards[0] if layer == 0 else shards[1]
        p = f"language_model.model.layers.{layer}"
        for proj, (n, k) in PROJ.items():
            t.update(_affine(g, f"{p}.mlp.switch_mlp.{proj}", n, k, (experts,)))
            t.update(_affine(g, f"{p}.mlp.shared_expert.{proj}", n, k))
        t.update(_affine(g, f"{p}.mlp.shared_expert_gate", 1, D))
        t[f"{p}.mlp.gate.weight"] = torch.randn((experts, D), generator=g).to(torch.bfloat16)
        t.update(_affine(g, f"{p}.self_attn.q_proj", 128, D))
        t.update(_affine(g, f"{p}.ple.ple_embedding.ngram_embedding.shard_0", 64, D))
        t[f"{p}.input_layernorm.weight"] = torch.randn((D,), generator=g).to(torch.bfloat16)
    embed = _affine(g, "language_model.model.embed_tokens", 256, D)
    shards[0]["language_model.model.embed_tokens.weight"] = embed.pop("language_model.model.embed_tokens.weight")
    shards[1].update(embed)                      # a group split across the source shards
    for proj, (n, k) in PROJ.items():
        shards[1].update(_affine(g, f"language_model.mtp.layers.0.mlp.switch_mlp.{proj}", n, k, (experts,)))
        shards[1].update(_affine(g, f"language_model.mtp.layers.0.mlp.shared_expert.{proj}", n, k))
    shards[1].update(_affine(g, "language_model.lm_head", 256, D))
    wm, all_ = {}, {}
    for i, t in enumerate(shards, 1):
        name = f"model-{i:05d}-of-00002.safetensors"
        save_file(t, str(root / name), metadata={"format": "mlx"})
        wm.update({k: name for k in t})
        all_.update(t)
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return all_


def write_export(root: Path, *, layers: int = LAYERS, experts: int = E, junk: int = 4096,
                 shared_fp8: bool = False) -> dict[str, torch.Tensor]:
    root.mkdir(parents=True)
    g = torch.Generator().manual_seed(9)
    (root / "config.json").write_text("{}")
    wm, all_ = {}, {}
    half = experts // 2
    for layer in range(layers):
        for a in (0, half):
            t = {}
            for e in range(a, a + half):
                for proj, (n, k) in PROJ.items():
                    p = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
                    t[p + ".weight"] = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.int64).to(torch.uint8)
                    t[p + ".weight_scale"] = (torch.rand((n, k // 16), generator=g) * 0.02 + 0.002).to(torch.float8_e4m3fn)
                    t[p + ".weight_scale_2"] = torch.rand((), generator=g) + 0.5
                    t[p + ".input_scale"] = torch.tensor(1.0)
            name = f"layer-{layer:05d}-experts-{a:04d}-{a + half - 1:04d}.safetensors"
            save_file(t, str(root / name), metadata={"format": "pt"})
            wm.update({k: name for k in t})
            all_.update(t)
    t = {}
    for layer in range(layers):
        for proj, (n, k) in PROJ.items():
            p = f"model.language_model.layers.{layer}.mlp.shared_expert.{proj}"
            w = torch.randn((n, k), generator=g) * 0.05
            if shared_fp8:
                t[p + ".weight"] = (w / (w.abs().amax() / 448)).to(torch.float8_e4m3fn)
                t[p + ".weight_scale"] = (w.abs().amax() / 448).float().contiguous()
                t[p + ".input_scale"] = torch.tensor(1.0)
            else:
                t[p + ".weight"] = w.to(torch.bfloat16)
        t[f"model.language_model.layers.{layer}.mlp.gate.weight"] = torch.randn((experts, D), generator=g).to(torch.bfloat16)
    t["model.language_model.embed_tokens.weight"] = torch.randn((junk, D), generator=g).to(torch.bfloat16)
    save_file(t, str(root / "model-bf16-00001.safetensors"), metadata={"format": "pt"})
    wm.update({k: "model-bf16-00001.safetensors" for k in t})
    all_.update(t)
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return all_


def symlink_layout(mlx: Path, export: Path, out: Path) -> Path:
    """What tools/make_mixed_dir.py builds: links to every MLX file plus a marker with absolute paths."""

    out.mkdir(parents=True)
    for f in sorted(mlx.iterdir()):
        os.symlink(f.resolve(), out / f.name)
    (out / "octojet.json").write_text(json.dumps({"format": "nvfp4-mixed", "experts": str(export.resolve()),
                                                  "base": str(mlx.resolve())}))
    return out


def load_tool(name: str):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
