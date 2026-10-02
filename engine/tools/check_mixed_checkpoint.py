#!/usr/bin/env python3
"""CPU consistency checks of a mixed NVFP4 Flash Next directory against the original MLX tensors and the NVFP4 export,
before any GPU time is spent. Works on the local symlink layout (tools/make_mixed_dir.py, absolute marker paths) and
on the self-contained release layout (tools/build_release_checkpoint.py, relative marker paths).

  python tools/check_mixed_checkpoint.py MIXED_DIR [--reference-mlx DIR] [--reference-nvfp4 DIR]
      [--layers all|0,3,24,47] [--experts 0,7,511] [--source-bytes] [--no-numeric]

References: the MLX checkpoint and the export the directory was built from. Required when the marker's sources are
relative (the release layout no longer holds the MLX routed experts or the export's routers); otherwise they default
to the marker's absolute paths.

Reports JSON: routers equal in every layer (MIXED_DIR's base vs the reference export); reference-MLX vs MIXED_DIR's
NVFP4 dequantized experts agree (nvfp4.agreement: cosine, norm ratio and relative error, so a lost per-matrix scale
fails too) and a shifted pair does not (cosine < 0.5), which catches a wrong nibble order or misaligned expert indices;
the export's block-scale byte and value ranges (no 0x7F = NaN) and per-matrix scale extrema; the shared expert
(expert "shared": MLX dequant vs nvfp4.shared_expert, which dequantizes an fp8 export copy with its scale first) and
its export dtype per layer (shared_source_dtype); the peak host memory of packing one layer. Each MLX projection is
read once per layer (~0.5 GB) and sliced in memory.

--source-bytes (spec 4.3): every tensor in MIXED_DIR's base and expert indices has the dtype, shape and payload sha256
(streamed) of the same-named tensor in its reference; in the release layout every dropped name (decoder routed and
shared experts) is absent from the base, every other reference-MLX name present, and every needed export name present.
--no-numeric skips the dequantization checks and the packing probe (bytes only; no GPU stack imported).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tensorfold.families.qwen4_exp import release as R


def parse(argv=None):
    ap = argparse.ArgumentParser(description="mixed NVFP4 Flash Next checkpoint checks")
    ap.add_argument("mixed")
    ap.add_argument("--reference-mlx", type=Path, help="the MLX checkpoint (required when the marker is relative)")
    ap.add_argument("--reference-nvfp4", type=Path, help="the NVFP4 export (required when the marker is relative)")
    ap.add_argument("--layers", default="0,3,24,47", help="'all' or a comma list of decoder layers")
    ap.add_argument("--experts", default="0,7,511")
    ap.add_argument("--source-bytes", action="store_true", help="every tensor's payload equals its reference's")
    ap.add_argument("--no-numeric", action="store_true", help="skip dequantization checks and the packing probe")
    return ap, ap.parse_args(argv)


def references(mixed: Path, ref_mlx: Path | None, ref_nvfp4: Path | None) -> tuple[Path, Path, bool]:
    """(reference MLX, reference export, the marker's paths are relative). Raises ValueError when a relative marker
    lacks a reference: the directory itself would be compared with itself."""

    raw = json.loads((mixed / "octojet.json").read_text())
    relative = not (Path(raw["base"]).is_absolute() and Path(raw["experts"]).is_absolute())
    if relative and (ref_mlx is None or ref_nvfp4 is None):
        raise ValueError("the marker's sources are relative (a release layout): pass --reference-mlx and "
                         "--reference-nvfp4 (the original MLX checkpoint and NVFP4 export)")
    from tensorfold.families.qwen4_exp.cuda.nvfp4 import sources

    src = sources(mixed)
    mlx, export = ref_mlx or src.base, ref_nvfp4 or src.experts
    for p, what in ((mlx, "--reference-mlx"), (export, "--reference-nvfp4")):
        if not (Path(p) / R.INDEX).is_file():
            raise ValueError(f"{what} {p}: no {R.INDEX}")
    return Path(mlx), Path(export), relative


def layer_list(spec: str, count: int) -> list[int]:
    if spec == "all":
        return list(range(count))
    try:
        layers = [int(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"--layers: 'all' or a comma list of layers, not {spec!r}") from None
    if not layers or any(not 0 <= x < count for x in layers):
        raise ValueError(f"--layers: each layer in 0..{count - 1}, got {spec!r}")
    return layers


def layer_count(mixed: Path) -> int:
    raw = json.loads((mixed / "config.json").read_text())
    return int((raw.get("text_config") or raw).get("num_hidden_layers", 48))


class _Tensors:
    """Headers of a sharded directory, read once each."""

    def __init__(self, root: Path) -> None:
        self.root, self.where, self.heads = root, R.read_index(root), {}

    def header(self, name: str):
        shard = self.where[name]
        if shard not in self.heads:
            self.heads[shard] = R.read_header(self.root / shard)
        return shard, self.heads[shard]

    def describe(self, name: str):
        shard, head = self.header(name)
        info = head[1][name]
        return info["dtype"], list(info["shape"]), R.tensor_sha256(self.root / shard, name, head)


def source_bytes(mixed: Path, ref_mlx: Path, ref_nvfp4: Path, experts_dir: Path, workers: int = 8) -> dict:
    out = {"layout": None, "base": {}, "experts": {}, "ok": True}
    base, eout = _Tensors(mixed), _Tensors(experts_dir)
    rmlx, rexp = _Tensors(ref_mlx), _Tensors(ref_nvfp4)
    release = not any(R.DECODER_ROUTED.match(n) for n in base.where)
    out["layout"] = "release" if release else "symlink"
    for label, mine, ref in (("base", base, rmlx), ("experts", eout, rexp)):
        names = sorted(mine.where)
        absent = [n for n in names if n not in ref.where]
        common = [n for n in names if n in ref.where]
        for shard in {mine.where[n] for n in common}:      # headers first: the threads only hash
            mine.header(next(n for n in common if mine.where[n] == shard))
        for shard in {ref.where[n] for n in common}:
            ref.header(next(n for n in common if ref.where[n] == shard))

        def compare(n, mine=mine, ref=ref):
            return n, mine.describe(n) == ref.describe(n)

        with ThreadPoolExecutor(workers) as pool:
            bad = [n for n, same in pool.map(compare, common) if not same]
        R.drop_cache(*(mine.root / s for s in mine.heads), *(ref.root / s for s in ref.heads))
        out[label] = {"checked": len(common), "mismatched": len(bad), "mismatched_names": bad[:20],
                      "not_in_reference": len(absent), "not_in_reference_names": absent[:20]}
        out["ok"] &= not bad and not absent
    if release:
        dropped = [n for n in rmlx.where if R.dropped_from_base(n)]
        kept = [n for n in rmlx.where if not R.dropped_from_base(n)]
        needed = [n for n in rexp.where if R.needed_from_export(n)]
        present = [n for n in dropped if n in base.where]
        missing = [n for n in kept if n not in base.where]
        lacking = [n for n in needed if n not in eout.where]
        out["dropped"] = {"count": len(dropped), "present": len(present), "present_names": present[:20]}
        out["kept"] = {"count": len(kept), "missing": len(missing), "missing_names": missing[:20]}
        out["needed_experts"] = {"count": len(needed), "missing": len(lacking), "missing_names": lacking[:20]}
        out["ok"] &= bool(dropped) and not present and not missing and not lacking
    return out


def numeric(mixed: Path, ref_mlx: Path, ref_nvfp4: Path, layers: list[int], experts_list: list[int], n_layers: int,
            report: dict) -> None:
    import subprocess

    import torch

    from tensorfold.cuda import experts
    from tensorfold.families.qwen4_exp.cuda import nvfp4
    from tensorfold.families.qwen4_exp.cuda.weights import _Reader, dequantize

    src = nvfp4.sources(mixed)
    mine = _Reader(mixed, "cpu")                  # the routers the engine serves
    mlx = _Reader(ref_mlx, "cpu")                 # the original MLX experts
    exp = nvfp4.SafetensorsDir(src.experts)       # the NVFP4 tables the engine packs
    ref = nvfp4.SafetensorsDir(ref_nvfp4)         # the export's routers
    report.update({"routers_equal": True, "pairs": [], "gscale_min": None, "gscale_max": None,
                   "scale_byte_min": None, "scale_byte_max": None, "scale_value_min": None, "scale_value_max": None,
                   "shared_source_dtype": {}, "layer0_pack_peak_rss_gib": None, "layer0_table_gib": None,
                   "layer0_pack_error": None})

    def mlx_projection(layer, proj):
        p = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}"
        return mlx.get(p + ".weight"), mlx.get(p + ".scales"), mlx.get(p + ".biases")

    def mlx_expert(t3, e):
        w, sc, b = t3
        return dequantize(w[e][None], sc[e][None], b[e][None]).float()[0]

    def nv_expert(layer, proj, e):
        p = f"{nvfp4.PREFIX}{layer}.mlp.experts.{e}.{proj}"
        return experts.dequant_nvfp4(exp.get(p + ".weight")[None], exp.get(p + ".weight_scale")[None],
                                     exp.get(p + ".weight_scale_2").reshape(1))[0]

    def mlx_shared(layer, proj):
        p = f"language_model.model.layers.{layer}.mlp.shared_expert.{proj}"
        return dequantize(mlx.get(p + ".weight"), mlx.get(p + ".scales"), mlx.get(p + ".biases")).float()

    def nv_shared(layer, proj):
        with contextlib.redirect_stdout(sys.stderr):                        # keep stdout pure JSON
            return experts.dequant_nvfp4(*nvfp4.shared_expert(exp, layer, proj))[0]

    def cos(x, y):
        return torch.nn.functional.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).item()

    for layer in range(n_layers):
        a_ = mine.get(f"language_model.model.layers.{layer}.mlp.gate.weight").to(torch.bfloat16)
        b_ = ref.get(f"{nvfp4.PREFIX}{layer}.mlp.gate.weight")
        if not torch.equal(a_, b_):
            report["routers_equal"] = False
            report["ok"] = False
    gmin, gmax = float("inf"), 0.0
    smin, smax = 255, 0
    agree_layers = {"routed": set(), "shared": set(), "routed_bad": set(), "shared_bad": set()}
    for layer in layers:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            t3 = mlx_projection(layer, proj)                                # one read of the 512-expert stack
            for e in experts_list:
                refw = mlx_expert(t3, e)
                agree = nvfp4.agreement(refw, nv_expert(layer, proj, e))
                other = cos(refw, nv_expert(layer, proj, (e + 1) % t3[0].shape[0]))
                g = exp.get(f"{nvfp4.PREFIX}{layer}.mlp.experts.{e}.{proj}.weight_scale_2").item()
                sb = exp.get(f"{nvfp4.PREFIX}{layer}.mlp.experts.{e}.{proj}.weight_scale").view(torch.uint8)
                smin, smax = min(smin, int(sb.min())), max(smax, int(sb.max()))
                gmin, gmax = min(gmin, g), max(gmax, g)
                ok = agree["ok"] and other < 0.5 and int(sb.max()) <= 0x7E
                report["pairs"].append({"layer": layer, "expert": e, "proj": proj, **agree,
                                        "cos_shifted": round(other, 4), "ok": ok})
                agree_layers["routed" if ok else "routed_bad"].add(layer)
                report["ok"] &= ok
            del t3
            dt = str(exp.get(f"{nvfp4.PREFIX}{layer}.mlp.shared_expert.{proj}.weight").dtype).removeprefix("torch.")
            report["shared_source_dtype"].setdefault(str(layer), {})[proj] = dt
            try:
                agree = nvfp4.agreement(mlx_shared(layer, proj), nv_shared(layer, proj))
            except ValueError as err:                                       # fp8 without a scale / out of range
                agree = {"ok": False, "error": str(err)}
            report["pairs"].append({"layer": layer, "expert": "shared", "proj": proj, **agree})
            agree_layers["shared" if agree["ok"] else "shared_bad"].add(layer)
            report["ok"] &= agree["ok"]
        mlx.release()                                                       # drop the read shards' page cache
        exp.release()
    report["routed_layers_agree"] = f"{len(agree_layers['routed'] - agree_layers['routed_bad'])}/{len(layers)}"
    report["shared_layers_agree"] = f"{len(agree_layers['shared'] - agree_layers['shared_bad'])}/{len(layers)}"
    report["gscale_min"], report["gscale_max"] = gmin, gmax
    report["scale_byte_min"], report["scale_byte_max"] = smin, smax
    report["scale_value_min"] = experts.ue4m3_to_float(torch.tensor([smin], dtype=torch.uint8)).item()
    report["scale_value_max"] = experts.ue4m3_to_float(torch.tensor([smax], dtype=torch.uint8)).item()
    # Peak host memory of packing one real layer on the CPU (the loader does this 48 times): checks the chunked
    # packing. Measured in a fresh subprocess so ru_maxrss is that process's own high-water mark, not this one's.
    probe = (
        "import json, resource, sys, torch\n"
        "from tensorfold.families.qwen4_exp.cuda import nvfp4\n"
        "src = nvfp4.sources(sys.argv[1])\n"
        "ex0 = nvfp4.make_layer(src, 0, 'cpu')\n"
        "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "peak_gib = peak / 2 ** 20 if sys.platform.startswith('linux') else peak / 2 ** 30   # KiB on Linux, bytes on macOS\n"
        "print(json.dumps({'layer0_pack_peak_rss_gib': round(peak_gib, 2),\n"
        "                  'layer0_table_gib': round(ex0.bytes_per_expert() * ex0.count / 2 ** 30, 2)}))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe, str(mixed)], capture_output=True, text=True)
    if out.returncode == 0:
        report.update(json.loads(out.stdout.strip().splitlines()[-1]))
    else:
        report["layer0_pack_error"] = out.stderr.strip()[-2000:]
        report["ok"] = False


def main(argv=None) -> int:
    ap, a = parse(argv)
    mixed = Path(a.mixed)
    try:
        ref_mlx, ref_nvfp4, relative = references(mixed, a.reference_mlx, a.reference_nvfp4)
        n_layers = layer_count(mixed)
        layers = layer_list(a.layers, n_layers)
        experts_list = [int(x) for x in a.experts.split(",")]
    except (ValueError, OSError, KeyError) as err:
        ap.error(str(err))
    from tensorfold.families.qwen4_exp.cuda.nvfp4 import sources

    report = {"mixed": str(mixed), "reference_mlx": str(ref_mlx), "reference_nvfp4": str(ref_nvfp4),
              "marker_relative": relative, "layers": layers if a.layers != "all" else "all", "ok": True}
    if a.source_bytes:
        report["source_bytes"] = source_bytes(mixed, ref_mlx, ref_nvfp4, sources(mixed).experts)
        report["ok"] &= report["source_bytes"]["ok"]
    if not a.no_numeric:
        numeric(mixed, ref_mlx, ref_nvfp4, layers, experts_list, n_layers, report)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
