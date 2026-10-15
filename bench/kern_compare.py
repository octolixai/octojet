#!/usr/bin/env python3
"""Side-by-side GPU kernel time of several nsys captures (bench/spark/f12-profile.sh): which kernels each engine
spends its prompt time in, and where two engines differ most. Measurement of behaviour only.

  kern_compare.py LABEL=KERN.csv [LABEL=KERN.csv ...] [--top 25] [--json OUT.json]

Each KERN.csv is ``nsys stats --report cuda_gpu_kern_sum --format csv`` output (any preamble lines before the
"Time (%)" header are skipped). Kernel names are shortened to the function and its template arguments (no
parameter list, no ``void``); rows with the same short name are summed. Printed: the per-capture totals, a table of
the kernels by their largest total in any capture (ms and launches per capture), and for each later capture the
kernels it spends the most more / less time in than the first."""
from __future__ import annotations

import argparse
import csv
import json
import sys


def short_name(name: str) -> str:
    """'void <unnamed>::k<(int)32, (bool)0>(const T *, int)' -> '<unnamed>::k<(int)32, (bool)0>'."""

    s = name.strip().strip('"')
    if s.startswith("void "):
        s = s[5:]
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "(" and depth == 0 and i > 0:
            return s[:i].strip()
    return s.strip()


def read_kern(path: str) -> dict[str, dict]:
    """{short name: {"ns": total ns, "n": launches}} from a cuda_gpu_kern_sum CSV."""

    raw = open(path, encoding="utf-8", errors="replace").read().splitlines()
    start = next((i for i, line in enumerate(raw) if line.startswith("Time (%)") or line.startswith('"Time (%)"')), None)
    if start is None:
        raise ValueError(f"{path}: no 'Time (%)' header line")
    rows = list(csv.DictReader([line for line in raw[start:] if line.strip()]))
    if not rows:
        raise ValueError(f"{path}: no kernel rows")
    cols = rows[0].keys()
    total = next(k for k in cols if k.startswith("Total Time"))
    inst = next(k for k in cols if k.startswith("Instances"))
    out: dict[str, dict] = {}
    for r in rows:
        name = short_name(r["Name"])
        e = out.setdefault(name, {"ns": 0, "n": 0})
        e["ns"] += int(float(r[total]))
        e["n"] += int(float(r[inst]))
    return out


def compare(caps: dict[str, dict[str, dict]], top: int = 25) -> dict:
    labels = list(caps)
    names = sorted({n for c in caps.values() for n in c}, key=lambda n: -max(c.get(n, {"ns": 0})["ns"] for c in caps.values()))
    totals = {lb: sum(e["ns"] for e in caps[lb].values()) / 1e6 for lb in labels}
    table = [{"kernel": n, **{lb: {"ms": caps[lb].get(n, {"ns": 0})["ns"] / 1e6, "n": caps[lb].get(n, {"n": 0})["n"]}
                               for lb in labels}} for n in names]
    diffs = {}
    base = labels[0]
    for lb in labels[1:]:
        d = [(n, (caps[lb].get(n, {"ns": 0})["ns"] - caps[base].get(n, {"ns": 0})["ns"]) / 1e6) for n in names]
        diffs[f"{lb} - {base}"] = {"more": sorted([x for x in d if x[1] > 0], key=lambda x: -x[1])[:top],
                                   "less": sorted([x for x in d if x[1] < 0], key=lambda x: x[1])[:top]}
    return {"labels": labels, "totals_ms": totals, "table": table[:max(top * 2, top)], "diffs": diffs}


def render(res: dict, width: int = 70) -> str:
    labels = res["labels"]
    out = ["GPU kernel time per capture (ms): " + ", ".join(f"{lb} {res['totals_ms'][lb]:,.0f}" for lb in labels), ""]
    head = f"{'kernel':<{width}} | " + " | ".join(f"{lb + ' ms':>14} {'launches':>9}" for lb in labels)
    out += [head, "-" * len(head)]
    for row in res["table"]:
        cells = " | ".join(f"{row[lb]['ms']:>14,.1f} {row[lb]['n']:>9,}" for lb in labels)
        out.append(f"{row['kernel'][:width]:<{width}} | {cells}")
    for key, d in res["diffs"].items():
        out += ["", f"{key}: kernels with the most MORE time (ms)"]
        out += [f"  {ms:>+12,.1f}  {n[:width + 30]}" for n, ms in d["more"]] or ["  (none)"]
        out += [f"{key}: kernels with the most LESS time (ms)"]
        out += [f"  {ms:>+12,.1f}  {n[:width + 30]}" for n, ms in d["less"]] or ["  (none)"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("captures", nargs="+", help="LABEL=KERN.csv")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    caps = {}
    for item in a.captures:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            ap.error(f"expected LABEL=KERN.csv, got {item!r}")
        caps[label] = read_kern(path)
    res = compare(caps, a.top)
    print(render(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
