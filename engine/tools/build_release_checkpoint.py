#!/usr/bin/env python3
"""Build the self-contained mixed Flash Next checkpoint from the two public sources (spec 2026-09-29-f2-release-design.md
section 4.1): one directory a Hugging Face upload can carry as is.

  python tools/build_release_checkpoint.py --mlx MLX_DIR --nvfp4 HF_MAIN_SNAPSHOT --out OUT [--max-shard-gib 4]
      [--mlx-revision SHA] [--nvfp4-revision SHA] [--hash] [--report PATH]

OUT/
  every non-weight file of MLX_DIR (config, generation config, tokenizer, chat template, ...) except README.md,
    LICENSE* and NOTICE* (the release writes its own)
  model.safetensors.index.json + model-XXXXX-of-NNNNN.safetensors   the MLX tensors less the decoder layers' routed
    experts (switch_mlp) and shared-expert projections; the MTP head kept whole. Resharded at <= --max-shard-gib with
    a weight's scales and biases in its shard; payload bytes, dtypes and names unchanged; each source shard's
    __metadata__ carried over
  experts/model.safetensors.index.json + experts/<shard>   the export's routed experts (weight, weight_scale,
    weight_scale_2) and shared-expert weights (plus their scales in an fp8 export): a shard is copied whole when the
    unneeded tensors in it are under 1 GiB, otherwise rewritten with the needed tensors only (bytes unchanged); the
    index lists the needed names only
  octojet.json   {"format": "nvfp4-mixed", "experts": "experts", "base": ".", "sources": {...repos, revisions},
                  "built_utc", "octojet"}, written last

The build report (sizes from the indices, wasted bytes, per-shard sha256 of output and source with --hash) is printed
and written beside OUT (OUT.build-report.json by default), so the uploaded directory carries no local paths. Needs no
torch: payloads are streamed as raw bytes. Refuses a non-empty OUT.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tensorfold.families.qwen4_exp import release as R

WHOLE_COPY_WASTE = 1 << 30                      # copy an export shard whole below this many unneeded bytes
SKIPPED_FILES = re.compile(r"^(README\.md|LICENSE.*|NOTICE.*|\..*)$")


def _revision(path: Path, given: str | None) -> str:
    """The given revision, else a Hugging Face snapshot directory's name, else "unknown"."""

    if given:
        return given
    real = path.resolve()
    if real.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", real.name):
        return real.name
    return "unknown"


def _octojet_version() -> str:
    from tensorfold import __version__

    try:
        sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = ""
    return __version__ + (f"+g{sha}" if sha else "")


def _text_config(mlx: Path) -> dict:
    raw = json.loads((mlx / "config.json").read_text())
    return dict(raw.get("text_config") or raw)


def _headers(root: Path, shards) -> dict[str, tuple[int, dict]]:
    return {s: R.read_header(root / s) for s in sorted(set(shards))}


def plan_base(mlx: Path, max_bytes: int):
    """[(output shard tensors [(name, src, base, info)], metadata)], {dropped name: bytes}, kept names, notes."""

    where = R.read_index(mlx)
    heads = _headers(mlx, where.values())
    order = sorted(where, key=lambda n: (where[n], heads[where[n]][1][n]["data_offsets"][0]))
    dropped = {n: R.payload_size(heads[where[n]][1][n]) for n in order if R.dropped_from_base(n)}
    kept = [n for n in order if not R.dropped_from_base(n)]
    groups: dict[str, list[str]] = {}
    for n in kept:
        groups.setdefault(R.group_of(n), []).append(n)
    shards, cur, cur_bytes, notes = [], [], 0, []

    def cost(n):                                  # payload plus a generous header entry
        return R.payload_size(heads[where[n]][1][n]) + len(n) + 160

    for g, names in groups.items():
        size = sum(cost(n) for n in names)
        if cur and cur_bytes + size > max_bytes - 4096:
            shards.append(cur)
            cur, cur_bytes = [], 0
        if size > max_bytes - 4096:
            notes.append(f"group {g} ({size / R.GIB:.2f} GiB) exceeds the shard limit: it gets a shard of its own")
        cur.extend(names)
        cur_bytes += size
    if cur:
        shards.append(cur)
    out = []
    for names in shards:
        meta: dict = {}
        for src in dict.fromkeys(where[n] for n in names):
            for k, v in (heads[src][1].get("__metadata__") or {}).items():
                if k in meta and meta[k] != v:
                    notes.append(f"__metadata__ {k!r} differs between source shards; kept {meta[k]!r}")
                meta.setdefault(k, v)
        out.append(([(n, mlx / where[n], heads[where[n]][0], heads[where[n]][1][n]) for n in names], meta))
    return out, dropped, kept, notes


def plan_experts(export: Path, cfg: dict):
    """{shard: (mode "copy"|"rewrite", needed names, needed bytes, wasted bytes)}, missing names."""

    where = R.read_index(export)
    needed = [n for n in where if R.needed_from_export(n)]
    missing = []
    layers, experts = cfg.get("num_hidden_layers"), cfg.get("num_experts")
    if layers and experts:
        want = [f"model.language_model.layers.{l}.mlp.experts.{e}.{p}.{s}" for l in range(int(layers))
                for e in range(int(experts)) for p in ("gate_proj", "up_proj", "down_proj")
                for s in ("weight", "weight_scale", "weight_scale_2")]
        want += [f"model.language_model.layers.{l}.mlp.shared_expert.{p}.weight" for l in range(int(layers))
                 for p in ("gate_proj", "up_proj", "down_proj")]
        have = set(needed)
        missing = [n for n in want if n not in have]
    by_shard: dict[str, list[str]] = {}
    for n in needed:
        by_shard.setdefault(where[n], []).append(n)
    heads = _headers(export, by_shard)
    plan = {}
    for shard, names in by_shard.items():
        _, entries = heads[shard]
        keep = set(names)
        need = sum(R.payload_size(entries[n]) for n in names)
        waste = sum(R.payload_size(i) for k, i in entries.items() if k != "__metadata__" and k not in keep)
        plan[shard] = ("copy" if waste < WHOLE_COPY_WASTE else "rewrite", names, need, waste)
    return plan, missing, heads


def build(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mlx", required=True, type=Path, help="the MLX 4-bit checkpoint (Vontra, with the MTP head)")
    ap.add_argument("--nvfp4", required=True, type=Path, help="the NVFP4 export's HF main snapshot (RadixArk)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-shard-gib", type=float, default=4.0)
    ap.add_argument("--mlx-revision")
    ap.add_argument("--nvfp4-revision")
    ap.add_argument("--mlx-repo", default=R.REPOS["base"])
    ap.add_argument("--nvfp4-repo", default=R.REPOS["experts"])
    ap.add_argument("--hash", action="store_true", help="sha256 of every output shard and its source")
    ap.add_argument("--report", type=Path, help="build report path (default: OUT.build-report.json beside OUT)")
    a = ap.parse_args(argv)
    mlx, export, out = a.mlx, a.nvfp4, a.out
    for p, what in ((mlx, "--mlx"), (export, "--nvfp4")):
        if not (p / R.INDEX).is_file():
            raise SystemExit(f"{what} {p}: no {R.INDEX}")
    if not (mlx / "config.json").is_file():
        raise SystemExit(f"--mlx {mlx}: no config.json")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"{out} exists and is not empty: build into a new directory")
    real_out = out.resolve()
    for p in (mlx, export):
        if real_out == p.resolve() or p.resolve() in real_out.parents:
            raise SystemExit(f"{out} lies inside a source ({p}); choose another directory")
    if a.max_shard_gib <= 0:
        raise SystemExit("--max-shard-gib must be positive")
    max_bytes = int(a.max_shard_gib * R.GIB)
    cfg = _text_config(mlx)

    base_plan, dropped, kept, notes = plan_base(mlx, max_bytes)
    exp_plan, missing, exp_heads = plan_experts(export, cfg)
    if missing:
        raise SystemExit(f"the export lacks {len(missing)} needed tensors, e.g. {missing[:3]}")
    if not exp_plan:
        raise SystemExit(f"--nvfp4 {export}: no routed-expert tensors in its index")

    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"tool": "build_release_checkpoint", "out": str(out), "max_shard_gib": a.max_shard_gib,
                    "sources": {"base": {"path": str(mlx), "repo": a.mlx_repo, "revision": _revision(mlx, a.mlx_revision)},
                                "experts": {"path": str(export), "repo": a.nvfp4_repo,
                                            "revision": _revision(export, a.nvfp4_revision)}},
                    "copied_files": [], "skipped_files": [], "notes": notes}
    # non-weight files
    for f in sorted(mlx.iterdir()):
        if f.name == R.INDEX or f.name.endswith(".safetensors"):
            continue
        if SKIPPED_FILES.match(f.name) or not f.is_file():
            report["skipped_files"].append(f.name)
            continue
        shutil.copyfile(f, out / f.name)
        report["copied_files"].append(f.name)

    # base shards
    n = len(base_plan)
    weight_map, base_rows, total = {}, [], 0
    for i, (tensors, meta) in enumerate(base_plan, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = R.write_shard(out / name, tensors, meta)
        total += size
        weight_map.update({t[0]: name for t in tensors})
        row = {"shard": name, "tensors": len(tensors), "bytes": size, "file_bytes": (out / name).stat().st_size,
               "sources": sorted({t[1].name for t in tensors})}
        if a.hash:
            row["sha256"] = R.file_sha256(out / name)
        base_rows.append(row)
        print(f"[build] base {name}: {len(tensors)} tensors, {size / R.GIB:.2f} GiB", flush=True)
    (out / R.INDEX).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    mlx_where = R.read_index(mlx)
    report["base"] = {"total_size": total, "shards": base_rows, "kept_tensors": len(kept),
                      "dropped_tensors": len(dropped),
                      "dropped_bytes": sum(dropped.values())}
    if a.hash:
        report["base"]["source_sha256"] = {s: R.file_sha256(mlx / s) for s in sorted(set(mlx_where.values()))}

    # expert shards
    edir = out / R.EXPERTS_DIR
    edir.mkdir()
    e_map, e_rows, e_total, wasted = {}, [], 0, 0
    for shard in sorted(exp_plan):
        mode, names, need, waste = exp_plan[shard]
        dst = edir / shard
        dst.parent.mkdir(parents=True, exist_ok=True)
        base, entries = exp_heads[shard]
        if mode == "copy":
            shutil.copyfile(export / shard, dst)
            R.drop_cache(dst, export / shard)
            wasted += waste
        else:
            order = sorted(names, key=lambda k: entries[k]["data_offsets"][0])
            R.write_shard(dst, [(k, export / shard, base, entries[k]) for k in order], entries.get("__metadata__"))
        e_total += need
        e_map.update({k: shard for k in names})
        row = {"shard": shard, "mode": mode, "tensors": len(names), "needed_bytes": need,
               "unneeded_bytes_in_source": waste, "file_bytes": dst.stat().st_size}
        if a.hash:
            row["sha256"] = R.file_sha256(dst)
            row["source_sha256"] = R.file_sha256(export / shard)
            row["identical"] = row["sha256"] == row["source_sha256"]
        e_rows.append(row)
        print(f"[build] experts {shard}: {mode}, {len(names)} tensors, {need / R.GIB:.2f} GiB"
              + (f", {waste / R.GIB:.2f} GiB unneeded kept" if mode == "copy" and waste else ""), flush=True)
    (edir / R.INDEX).write_text(json.dumps({"metadata": {"total_size": e_total}, "weight_map": e_map}, indent=2))
    report["experts"] = {"total_size": e_total, "wasted_bytes": wasted, "shards": e_rows,
                         "tensors": len(e_map), "copied_whole": sum(r["mode"] == "copy" for r in e_rows),
                         "rewritten": sum(r["mode"] == "rewrite" for r in e_rows)}

    marker = {"format": "nvfp4-mixed", "experts": R.EXPERTS_DIR, "base": ".",
              "sources": {k: {"repo": v["repo"], "revision": v["revision"]} for k, v in report["sources"].items()},
              "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "octojet": _octojet_version()}
    (out / "octojet.json").write_text(json.dumps(marker, indent=1) + "\n")
    report["marker"] = marker
    report["total_size"] = total + e_total
    report["out_file_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    path = a.report or out.parent / f"{out.name}.build-report.json"
    report["report"] = str(path)
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("total_size", "out_file_bytes")} |
                     {"base_total": total, "base_shards": len(base_rows), "dropped_tensors": len(dropped),
                      "experts_total": e_total, "experts_wasted": wasted, "experts_shards": len(e_rows),
                      "report": str(path), "notes": notes}, indent=1), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    build(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
