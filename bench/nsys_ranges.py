#!/usr/bin/env python3
"""Reduce an nsys sqlite export to per-NVTX-range GPU busy time (kernel + memcpy + memset union) and DRAM percent-of-peak.

  nsys_ranges.py EXPORT.sqlite --out FILE.json

A GPU activity belongs to the innermost block range ("phase:block" NVTX text) that contains the start of its launching
runtime call (same correlationId) on the same globalTid. Capture bounds are the "admission" NVTX range when present, else
[first GPU start, last GPU end]. Output "counters" is "available" (at least one metric in METRICS matched: per range, and over the capture bounds as
"capture_metrics", the mean of the samples inside the attributed intervals of dram_read_pct, dram_write_pct,
sms_active_pct, tensor_active_pct, sm_issue_pct, gr_active_pct, warps_in_flight_pct, each null when that metric is absent),
"no_metric_match" (both metrics tables present but none of those metrics; metric_names lists all names) or "unavailable" (a
metrics table is absent). Exit 2 when a required table or column is missing (named on stderr). Output "valid" is false (with "reason"; the JSON is
still written, exit 3) when there is no "admission" range, no "phase:block" range, or no launch attributed to any block range.
"""
import argparse, bisect, json, sqlite3, sys

REQUIRED = {"NVTX_EVENTS": ("start", "end", "text", "globalTid"),
            "CUPTI_ACTIVITY_KIND_RUNTIME": ("start", "end", "correlationId", "globalTid"),
            "CUPTI_ACTIVITY_KIND_KERNEL": ("start", "end", "correlationId"),
            "CUPTI_ACTIVITY_KIND_MEMCPY": ("start", "end", "correlationId")}
OPTIONAL = {"CUPTI_ACTIVITY_KIND_MEMSET": ("start", "end", "correlationId"),
            "GPU_METRICS": ("timestamp", "typeId", "metricId", "value"),
            "TARGET_INFO_GPU_METRICS": ("typeId", "metricId", "metricName")}
GPU_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET")
NS_PER_MS = 1e6


class Missing(Exception):
    pass


def union(intervals):
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def span(merged):
    return sum(e - s for s, e in merged)


def clip(merged, lo, hi):
    return [[max(s, lo), min(e, hi)] for s, e in merged if e > lo and s < hi]


def schema(db):
    tables = {}
    for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        tables[name] = [r[1] for r in db.execute(f'PRAGMA table_info("{name}")')]
    for name, cols in REQUIRED.items():
        if name not in tables:
            raise Missing(f"required table {name} is missing")
    for name, cols in {**REQUIRED, **OPTIONAL}.items():
        if name in tables:                  # an absent optional table is tolerated; a present one must be well-formed
            gone = [c for c in cols if c not in tables[name]]
            if gone:
                raise Missing(f"table {name} lacks column(s) {', '.join(gone)}; it has {', '.join(tables[name])}")
    return tables


METRICS = {"dram_read_pct": lambda n: "DRAM Read Bandwidth" in n,
           "dram_write_pct": lambda n: "DRAM Write Bandwidth" in n,
           "sms_active_pct": lambda n: n == "SMs Active [Throughput %]",
           "tensor_active_pct": lambda n: "Tensor Active [Throughput %]" in n,
           "sm_issue_pct": lambda n: "SM Issue [Throughput %]" in n,
           "gr_active_pct": lambda n: "GR Active [Throughput %]" in n,
           "warps_in_flight_pct": lambda n: "Compute Warps in Flight [Throughput %]" in n}


def metric_means(db, tables):
    """(mean, metric_names, counters): mean(merged) gives {key: mean or None} over the samples inside the merged intervals.

    counters is "unavailable" (a metrics table is absent; mean is None), "no_metric_match" (both present, no METRICS name
    among them) or "available" (at least one matched). metric_names lists every metric name in the capture."""
    if not all(t in tables for t in ("GPU_METRICS", "TARGET_INFO_GPU_METRICS")):
        return None, [], "unavailable"
    present = sorted({r[0] for r in db.execute("SELECT metricName FROM TARGET_INFO_GPU_METRICS") if r[0]})
    samples = {k: [] for k in METRICS}
    for name, ts, value in db.execute(
            "SELECT i.metricName, m.timestamp, m.value FROM GPU_METRICS m JOIN TARGET_INFO_GPU_METRICS i "
            "ON m.typeId = i.typeId AND m.metricId = i.metricId"):
        for key, match in METRICS.items():
            if name and match(name):
                samples[key].append((ts, value))
    matched = any(n and any(m(n) for m in METRICS.values()) for n in present)
    if not matched:
        return None, present, "no_metric_match"
    for key in samples:
        samples[key].sort()
    keys = {k: [t for t, _ in v] for k, v in samples.items()}

    def mean(merged):
        out = {}
        for key in METRICS:
            vals = []
            for s, e in merged:
                lo, hi = bisect.bisect_left(keys[key], s), bisect.bisect_right(keys[key], e)
                vals += [v for _, v in samples[key][lo:hi]]
            out[key] = sum(vals) / len(vals) if vals else None
        return out

    return mean, present, "available"


class RangeIndex:
    """Innermost block range containing a point, per globalTid: O(log n) bisect plus a walk up the enclosing chain.

    Ranges of one thread are sorted by (start, -end); parent[i] is the nearest earlier range still open at range i's start.
    The latest-starting range with start <= t is the innermost candidate; if it ends before t, an enclosing range (an
    ancestor along the parent chain) is the only possible container."""

    def __init__(self, blocks):
        per = {}
        for s, e, t, g in blocks:
            per.setdefault(g, []).append((s, e, t))
        self.tids = {}
        for g, rs in per.items():
            rs.sort(key=lambda r: (r[0], -r[1]))
            parent, stack = [], []
            for s, e, _ in rs:
                while stack and rs[stack[-1]][1] < s:
                    stack.pop()
                parent.append(stack[-1] if stack else -1)
                stack.append(len(parent) - 1)
            self.tids[g] = ([r[0] for r in rs], rs, parent)

    def innermost(self, tid, t):
        entry = self.tids.get(tid)
        if entry is None:
            return None
        starts, rs, parent = entry
        i = bisect.bisect_right(starts, t) - 1
        while i >= 0:
            if rs[i][1] >= t:
                return rs[i][2]
            i = parent[i]
        return None


def reduce(db):
    tables = schema(db)
    nvtx = [(s, e, t, g) for s, e, t, g in db.execute("SELECT start, end, text, globalTid FROM NVTX_EVENTS")
            if e is not None and t]
    admission = [(s, e) for s, e, t, _ in nvtx if t == "admission"]
    blocks = [(s, e, t, g) for s, e, t, g in nvtx if t != "admission" and ":" in t]
    launch = {c: (s, g) for s, c, g in db.execute("SELECT start, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    index = RangeIndex(blocks)
    attributed = {b[2]: [] for b in blocks}
    everything = []
    for table in GPU_TABLES:
        if table not in tables:
            continue
        for s, e, corr in db.execute(f"SELECT start, end, correlationId FROM {table}"):
            everything.append((s, e))
            call = launch.get(corr)
            if call is None:
                continue
            name = index.innermost(call[1], call[0])
            if name is not None:
                attributed[name].append((s, e))
    if admission:
        lo, hi = min(s for s, _ in admission), max(e for _, e in admission)
        bounds = "admission"
    elif everything:
        lo, hi = min(s for s, _ in everything), max(e for _, e in everything)
        bounds = "activity"
    else:
        lo = hi = 0
        bounds = "activity"
    merged_all = union(everything)
    ranges = {n: union(iv) for n, iv in attributed.items()}
    mean, names, counters = metric_means(db, tables)
    out_ranges = {}
    for name, merged in ranges.items():
        entry = {"busy_ms": span(merged) / NS_PER_MS, "launches": len(attributed[name])}
        if mean is not None:
            entry.update(mean(merged))
        out_ranges[name] = entry
    reasons = []
    if bounds != "admission":
        reasons.append('no "admission" NVTX range in the capture (capture bounds fell back to GPU activity)')
    if not blocks:
        reasons.append('no "phase:block" NVTX range (text of the form phase:block) in the capture')
    elif not any(attributed.values()):
        reasons.append("no launches attributed to any phase:block range")
    return {"valid": not reasons, "reason": "; ".join(reasons) or None, "capture_bounds": bounds, "capture_ms": (hi - lo) / NS_PER_MS,
            "exposed_idle_ms": ((hi - lo) - span(clip(merged_all, lo, hi))) / NS_PER_MS, "ranges": out_ranges,
            "counters": counters, "metric_names": names,
            "capture_metrics": mean([[lo, hi]]) if mean is not None else None,
            "tables": sorted(tables), "columns": {t: tables[t] for t in sorted(tables)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        db = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
        result = reduce(db)
    except Missing as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except sqlite3.Error as e:
        print(f"error: cannot read {a.sqlite}: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"ranges": len(result["ranges"]), "capture_ms": result["capture_ms"],
                      "exposed_idle_ms": result["exposed_idle_ms"], "counters": result["counters"],
                      "valid": result["valid"]}))
    if not result["valid"]:
        print(f"error: broken capture: {result['reason']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
