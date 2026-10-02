#!/usr/bin/env python3
"""One decoder layer's experts, mixed NVFP4 table vs the MLX checkpoint, through the real loader path.

  python tools/diag_expert_layer.py MIXED_DIR --layer L [--experts 0,7,511] [--rows 16] [--gpu]

Reports JSON on stdout and always exits 0 (a diagnostic; errors are captured into the report's "errors").

CPU ("table"): ``nvfp4.make_layer(sources(MIXED_DIR), L, "cpu")`` (the loader's own call), then each matrix of the
table unpacked (``unpack_nvfp4``) and dequantized (``dequant_nvfp4`` with its gscale column) for the requested routed
experts and the shared expert (index count - 1), compared with the MLX dequantized weight of the same layer
(``nvfp4.agreement`` plus the largest difference and where it is). Also: the unpacked table against the export's raw
tensors (bit equality), the shared expert's bf16 copy in the export against MLX's shared expert and against the table,
gscale ranges, and the fraction of zero codes per matrix.

GPU (``--gpu``): the table on CUDA and the MLX table built exactly as weights.py's moe() builds it (world 1, the
shared expert appended as the last expert); random x [rows, D] bf16 (seed 0) and picks [rows, top_k + 1] (distinct
routed experts, the first row starting with the requested ones, then the shared expert); decode (fp32 y) and prefill
(bf16 y) plans; per slot ||y_mixed - y_mlx|| / ||y_mlx|| averaged over rows (routed slots and the shared slot
separately; ~0.1-0.2 is two 4-bit quantizations of one weight, ~1 or more is a wrong table), the same for the gate/up
activation, and each kernel's row 0 against a torch reference over its own table (separates a kernel fault from a
table fault).
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback

import torch

ROUTED_OK = 0.3          # mixed-vs-MLX relative error of an output slot: above this the tables disagree
WRONG = 0.6
DEV = "cuda"             # the --gpu device (a module constant so a CPU harness can dry-run the GPU logic)


def rnd(x, n=5):
    x = float(x)
    return x if not math.isfinite(x) else round(x, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mixed")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--experts", default="0,7,511")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    report: dict = {"mixed": a.mixed, "layer": a.layer, "experts_requested": a.experts, "rows": a.rows,
                    "gpu": a.gpu, "errors": [], "seconds": {}}
    try:
        run(a, report)
    except BaseException as exc:                      # noqa: BLE001 - a diagnostic reports everything
        report["errors"].append({"step": "top", "error": repr(exc), "traceback": traceback.format_exc()[-4000:]})
    print(json.dumps(report, indent=1, default=str))
    raise SystemExit(0)


def step(report: dict, name: str):
    """Context manager: time a step and capture its exception into the report (the tool keeps going)."""

    class _Step:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, kind, exc, tb):
            report["seconds"][name] = round(time.time() - self.t0, 1)
            if exc is not None:
                report["errors"].append({"step": name, "error": repr(exc),
                                         "traceback": "".join(traceback.format_exception(kind, exc, tb))[-4000:]})
                return not isinstance(exc, KeyboardInterrupt)
            return False

    return _Step()


def run(a, report: dict) -> None:
    from tensorfold.cuda import experts as grouped
    from tensorfold.families.qwen4_exp.cuda import nvfp4
    from tensorfold.families.qwen4_exp.cuda import weights

    L = a.layer
    src = nvfp4.sources(a.mixed)
    report["sources"] = {"experts": str(src.experts), "base": str(src.base)}
    cfg = weights.Config.read(a.mixed)
    report["config"] = {"experts": cfg.experts, "top_k": cfg.top_k, "hidden": cfg.hidden, "moe_width": cfg.moe_width,
                        "shared_width": cfg.shared_width, "group_size": cfg.group_size, "bits": cfg.bits}
    E = cfg.experts
    want = [int(x) for x in a.experts.split(",") if x.strip()]
    routed = [e for e in dict.fromkeys(want) if 0 <= e < E]
    if len(routed) != len(want):
        report["experts_ignored"] = [e for e in want if not (0 <= e < E)]
    mlx = weights._Reader(src.base, "cpu")
    pre = "language_model." if mlx.has("language_model.model.embed_tokens.weight") else ""
    mlx_layer = f"{pre}model.layers.{L}.mlp."
    exp = nvfp4.SafetensorsDir(src.experts)
    kept: dict = {}                                    # MLX triples kept for the GPU table (read once)

    def mlx_triple(name: str):
        w = mlx.get(name + ".weight")
        return (w.view(torch.int32) if w.dtype != torch.int32 else w), mlx.get(name + ".scales"), mlx.get(name + ".biases")

    ex = None
    with step(report, "make_layer"):
        ex = nvfp4.make_layer(src, L, "cpu", n_experts=E, hidden=cfg.hidden, width=cfg.moe_width,
                              shared_width=cfg.shared_width)
        info = {"count": ex.count, "width": ex.width, "dims": ex.dims, "fmt": ex.fmt, "gs": ex.gs,
                "limit": ex.limit, "up_shape": list(ex.up.shape), "down_shape": list(ex.down.shape),
                "gscale_up_shape": list(ex.gscale_up.shape), "gscale_down_shape": list(ex.gscale_down.shape)}
        for key, g in (("gate", ex.gscale_up[:, 0]), ("up", ex.gscale_up[:, 1]), ("down", ex.gscale_down[:, 0])):
            info[f"gscale_{key}_routed_min"] = rnd(g[:-1].min(), 9)
            info[f"gscale_{key}_routed_max"] = rnd(g[:-1].max(), 9)
            info[f"gscale_{key}_shared"] = rnd(g[-1], 9)
        report["table"] = info

    if ex is not None:
        shared = ex.count - 1
        mats = {"gate_proj": (ex.up[:, :, :, 0], ex.gscale_up[:, 0]), "up_proj": (ex.up[:, :, :, 1], ex.gscale_up[:, 1]),
                "down_proj": (ex.down[:, :, :, 0], ex.gscale_down[:, 0])}
        report["zero_codes"] = {}
        with step(report, "zero_codes"):
            for proj, (m, _) in mats.items():
                report["zero_codes"][proj] = zero_code_fraction(m, shared)
        report["pairs"] = []
        report["raw_equal"] = []
        report["shared_bf16"] = []
        for proj, (m, gcol) in mats.items():
            with step(report, f"cpu_{proj}"):
                sel = routed + [shared]
                packed, scales = grouped.unpack_nvfp4(m[sel].contiguous())
                mixed_w = grouped.dequant_nvfp4(packed, scales, gcol[sel])            # [len(sel), N, K] fp32
                t3 = mlx_triple(mlx_layer + f"switch_mlp.{proj}")                     # one read of the stack
                s3 = mlx_triple(mlx_layer + f"shared_expert.{proj}")
                if a.gpu:
                    kept[proj] = (t3, s3)
                for i, e in enumerate(sel):
                    if e == shared:
                        ref = weights.dequantize(*s3).float()
                    else:
                        ref = weights.dequantize(t3[0][e], t3[1][e], t3[2][e]).float()
                    report["pairs"].append({"expert": e, "shared": e == shared, "proj": proj,
                                            "shape_mlx": list(ref.shape), "shape_mixed": list(mixed_w[i].shape),
                                            **compare(nvfp4, ref, mixed_w[i])})
                # the table against the export's own tensors (the pack/unpack path on real data)
                for i, e in enumerate(sel):
                    if e == shared:
                        sp, ss, sg = nvfp4.shared_expert(exp, L, proj)
                        rp, rs, rg = sp[0], ss[0], sg[0]
                    else:
                        p = f"{nvfp4.PREFIX}{L}.mlp.experts.{e}.{proj}"
                        rp, rs, rg = exp.get(p + ".weight"), exp.get(p + ".weight_scale"), exp.get(p + ".weight_scale_2")
                    report["raw_equal"].append({
                        "expert": e, "proj": proj, "packed": bool(torch.equal(packed[i], rp)),
                        "scales": bool(torch.equal(scales[i].view(torch.uint8), rs.view(torch.uint8))),
                        "gscale": bool(float(gcol[e]) == float(rg.float().reshape(()))),
                        "export_dtypes": [str(rp.dtype), str(rs.dtype), str(rg.dtype)]})
                # the shared expert's bf16 copy in the export: against MLX's shared expert and against the table
                bf = exp.get(f"{nvfp4.PREFIX}{L}.mlp.shared_expert.{proj}.weight")
                mlx_shared = weights.dequantize(*s3).float()
                entry = {"proj": proj, "export_dtype": str(bf.dtype), "export_shape": list(bf.shape),
                         "export_absmax": rnd(bf.float().abs().max(), 6), "mlx_absmax": rnd(mlx_shared.abs().max(), 6)}
                if tuple(bf.shape) == tuple(mlx_shared.shape):
                    entry["export_vs_mlx"] = nvfp4.agreement(mlx_shared, bf.float())
                    entry["export_vs_table"] = nvfp4.agreement(bf.float(), mixed_w[-1])
                elif tuple(bf.t().shape) == tuple(mlx_shared.shape):
                    entry["export_T_vs_mlx"] = nvfp4.agreement(mlx_shared, bf.t().float())
                report["shared_bf16"].append(entry)
                del t3, s3, mixed_w, packed, scales
        report["cpu_ok"] = bool(report["pairs"]) and all(p["ok"] for p in report["pairs"]) \
            and all(r["packed"] and r["scales"] and r["gscale"] for r in report["raw_equal"])

    if a.gpu and ex is not None:
        with step(report, "gpu"):
            report["gpu_result"] = gpu_check(a, cfg, ex, kept, mlx, mlx_layer, mlx_triple, routed, grouped, weights)
    report["verdict"] = verdict(report)


def compare(nvfp4, ref: torch.Tensor, got: torch.Tensor) -> dict:
    out = dict(nvfp4.agreement(ref, got))
    if ref.shape != got.shape:
        out["ok"] = False
        out["shape_mismatch"] = True
        return out
    d = (got - ref).abs()
    at = int(d.reshape(-1).argmax())
    n, k = divmod(at, ref.shape[-1])
    out.update(max_abs=rnd(d.reshape(-1)[at], 6), argmax=[n, k], mlx_at=rnd(ref[n, k], 6), mixed_at=rnd(got[n, k], 6),
               mlx_absmax=rnd(ref.abs().max(), 6), mixed_absmax=rnd(got.abs().max(), 6))
    return out


def zero_code_fraction(m: torch.Tensor, shared: int, chunk: int = 32) -> dict:
    """Fraction of E2M1 codes equal to 0 (and to 8, i.e. -0) in a packed matrix [E, N/32, K/32, 144]; the code
    words are the first 128 of a block and the count does not depend on the fragment layout."""

    zero = neg0 = total = 0
    for e0 in range(0, shared, chunk):
        w = m[e0:min(e0 + chunk, shared), :, :, :128].to(torch.int64) & 0xFFFFFFFF
        for s in range(8):
            c = (w >> (4 * s)) & 0xF
            zero += int((c == 0).sum())
            neg0 += int((c == 8).sum())
        total += w.numel() * 8
    w = m[shared, :, :, :128].to(torch.int64) & 0xFFFFFFFF
    sz = sum(int((((w >> (4 * s)) & 0xF) == 0).sum()) for s in range(8))
    return {"routed_zero": rnd(zero / max(total, 1)), "routed_neg_zero": rnd(neg0 / max(total, 1)),
            "shared_zero": rnd(sz / max(w.numel() * 8, 1))}


def gpu_check(a, cfg, ex, kept, mlx, mlx_layer, mlx_triple, routed, grouped, weights) -> dict:
    dev = DEV
    out: dict = {}
    nv = grouped.Experts(ex.up.to(dev), ex.down.to(dev), ex.gs, ex.width, ex.dims, ex.limit, fmt=ex.fmt,
                         gscale_up=ex.gscale_up.to(dev).contiguous(), gscale_down=ex.gscale_down.to(dev).contiguous())
    E, width = cfg.experts, cfg.moe_width

    # weights.py moe(), world 1: gu = rows [0, width), dn = input groups [0, width / 32): the whole matrices
    def triple(proj):
        if proj in kept:
            return kept[proj]
        return mlx_triple(mlx_layer + f"switch_mlp.{proj}"), mlx_triple(mlx_layer + f"shared_expert.{proj}")

    def table(routed_t, shared_t):
        return tuple(torch.cat([r.to(dev), t.to(dev)[None]]) for r, t in zip(routed_t, shared_t))

    g3, u3, d3 = triple("gate_proj"), triple("up_proj"), triple("down_proj")
    rows_w = weights._rows(g3[0], 0, width), weights._rows(g3[1], 0, width)
    gate_t = table(*rows_w)
    rows_w = weights._rows(u3[0], 0, width), weights._rows(u3[1], 0, width)
    up_t = table(*rows_w)
    down_t = table(weights._groups(d3[0], 0, width // 32), weights._groups(d3[1], 0, width // 32))
    ml = grouped.make([gate_t, up_t], down_t, 32)
    del gate_t, up_t, down_t, g3, u3, d3, rows_w
    kept.clear()
    out["mlx_table"] = {"count": ml.count, "width": ml.width, "dims": ml.dims, "fmt": ml.fmt,
                        "up_shape": list(ml.up.shape), "down_shape": list(ml.down.shape)}
    if (ml.count, ml.width, ml.dims) != (nv.count, nv.width, nv.dims):
        out["shape_mismatch"] = True
        return out

    rows, top_k = a.rows, cfg.top_k
    slots = top_k + 1
    g = torch.Generator().manual_seed(0)
    x = torch.randn((rows, nv.dims), generator=g).to(torch.bfloat16).to(dev).contiguous()
    picks = []
    for r in range(rows):
        first = routed[:top_k] if r == 0 else []
        rest = [int(e) for e in torch.randperm(E, generator=g) if int(e) not in first]
        picks.append(first + rest[:top_k - len(first)] + [E])
    picks = torch.tensor(picks, dtype=torch.int32).to(dev).contiguous()
    out["picks_row0"] = picks[0].tolist()

    for prefill in (False, True):
        key = "prefill" if prefill else "decode"
        plan = grouped.Plan(rows, slots, nv.count, dev, prefill=prefill)
        grouped.route(picks, plan)
        res = {}
        for name, t in (("mixed", nv), ("mlx", ml)):
            act = torch.empty((rows * slots, t.width), dtype=torch.bfloat16, device=dev)
            y = torch.empty((rows * slots, t.dims), dtype=torch.bfloat16 if prefill else torch.float32, device=dev)
            grouped.gate_up(x, t, plan, act, rows)
            grouped.down(act, t, plan, y, rows)
            if dev == "cuda":
                torch.cuda.synchronize()
            res[name] = (act.float().view(rows, slots, -1), y.float().view(rows, slots, -1))
        o = {}
        for i, what in ((0, "act"), (1, "y")):
            m, r = res["mixed"][i], res["mlx"][i]
            rel = (m - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)                  # [rows, slots]
            ratio = m.norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
            per_slot = rel.mean(0)
            o[f"{what}_rel_err_per_slot"] = [rnd(v, 4) for v in per_slot.tolist()]
            o[f"{what}_rel_err_routed"] = rnd(per_slot[:top_k].mean(), 4)
            o[f"{what}_rel_err_shared"] = rnd(per_slot[top_k], 4)
            o[f"{what}_norm_ratio_routed"] = rnd(ratio[:, :top_k].mean(), 4)
            o[f"{what}_norm_ratio_shared"] = rnd(ratio[:, top_k].mean(), 4)
            o[f"{what}_mlx_norm_routed"] = rnd(r[:, :top_k].norm(dim=-1).mean(), 4)
            o[f"{what}_mlx_norm_shared"] = rnd(r[:, top_k].norm(dim=-1).mean(), 4)
            o[f"{what}_nonfinite_mixed"] = int((~torch.isfinite(m)).sum())
            worst = torch.argsort(rel.reshape(-1).nan_to_num(float("inf")), descending=True)[:5].tolist()
            o[f"{what}_worst"] = [{"row": i // slots, "slot": i % slots, "expert": int(picks[i // slots, i % slots]),
                                   "rel_err": rnd(rel.reshape(-1)[i], 4)} for i in worst]
        o["self_check_row0"] = {name: self_check(x[0], picks[0].tolist(), t, res[name], grouped, weights)
                                for name, t in (("mixed", nv), ("mlx", ml))}
        out[key] = o
    return out


def self_check(x0, picks0, t, res, grouped, weights) -> dict:
    """Row 0 of a kernel run against a torch reference over the same table's own weights (unpacked from the table):
    act = bf16(bf16(silu(bf16(g))) * bf16(u)), y = W_down act. Small (~1e-2 or less) unless the kernel is wrong."""

    act_k, y_k = res
    xs = x0.float()
    acts, ys = [], []
    for s, e in enumerate(picks0):
        mats = []
        for blocks, gs in ((t.up[e:e + 1, :, :, 0], None if t.fmt == "affine" else t.gscale_up[e:e + 1, 0]),
                           (t.up[e:e + 1, :, :, 1], None if t.fmt == "affine" else t.gscale_up[e:e + 1, 1]),
                           (t.down[e:e + 1, :, :, 0], None if t.fmt == "affine" else t.gscale_down[e:e + 1, 0])):
            b = blocks.contiguous()
            if t.fmt == "affine":
                w, sc, bi = grouped.unpack(b, t.gs)
                mats.append(weights.dequantize(w[0], sc[0], bi[0]).float())
            else:
                p, sc = grouped.unpack_nvfp4(b)
                mats.append(grouped.dequant_nvfp4(p, sc, gs)[0])
        wg, wu, wd = mats
        gv = (wg @ xs).to(torch.bfloat16).float()
        uv = (wu @ xs).to(torch.bfloat16).float()
        act = ((gv * torch.sigmoid(gv)).to(torch.bfloat16).float() * uv).to(torch.bfloat16).float()
        acts.append((act_k[0, s] - act).norm() / act.norm().clamp_min(1e-30))
        y = wd @ act_k[0, s]                                   # the kernel's own activation, so down is checked alone
        ys.append((y_k[0, s] - y).norm() / y.norm().clamp_min(1e-30))
    return {"act_rel_err_max": rnd(max(float(v) for v in acts), 5),
            "y_rel_err_max": rnd(max(float(v) for v in ys), 5)}


def verdict(report: dict) -> dict:
    v = {}
    pairs = report.get("pairs") or []
    bad = [f"{p['proj']}[{p['expert']}]" for p in pairs if not p["ok"]]
    v["cpu_bad_pairs"] = bad
    v["cpu_shared_ok"] = all(p["ok"] for p in pairs if p["shared"]) if pairs else None
    v["cpu_routed_ok"] = all(p["ok"] for p in pairs if not p["shared"]) if pairs else None
    raw = report.get("raw_equal") or []
    v["raw_equal_all"] = all(r["packed"] and r["scales"] and r["gscale"] for r in raw) if raw else None
    gr = report.get("gpu_result") or {}
    for key in ("decode", "prefill"):
        o = gr.get(key)
        if not o:
            continue
        r_, s_ = o["y_rel_err_routed"], o["y_rel_err_shared"]
        v[f"{key}_routed"] = "agree" if r_ <= ROUTED_OK else "WRONG" if r_ >= WRONG else "suspect"
        v[f"{key}_shared"] = "agree" if s_ <= ROUTED_OK else "WRONG" if s_ >= WRONG else "suspect"
        sc = o["self_check_row0"]
        v[f"{key}_kernel_self_check_max"] = max(sc[n]["y_rel_err_max"] for n in sc)
    v["errors"] = len(report.get("errors") or [])
    return v


if __name__ == "__main__":
    main()
