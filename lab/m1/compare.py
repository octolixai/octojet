#!/usr/bin/env python3
"""Merge M1 JSON lines (probe, fp4_bench, tf_baseline) into a markdown table and a verdict.

  compare.py results/m1-probe.jsonl results/m1-fp4.jsonl results/m1-tf.jsonl
"""
import json
import sys

PROMPT_ROWS = 512  # cells with at least this many rows are prompt-sized (the stop rule's cells)


def load(paths):
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _times(recs, bench):
    return {r["rows"]: r["ms"] for r in recs if r.get("bench") == bench}


def table(recs):
    tf, fp4 = _times(recs, "tf"), _times(recs, "fp4")
    flops = {r["rows"]: r["flops"] for r in recs if r.get("bench") == "fp4" and "flops" in r}
    lines = ["| rows | TF ms | FP4 ms | speed-up | TF TFLOP/s | FP4 TFLOP/s |", "|---:|---:|---:|---:|---:|---:|"]
    for rows in sorted(set(tf) | set(fp4)):
        a, b, f = tf.get(rows), fp4.get(rows), flops.get(rows)
        fa = f"{a:.3f}" if a is not None else "-"
        fb = f"{b:.3f}" if b is not None else "-"
        sp = f"{a / b:.2f}x" if a is not None and b else "-"
        ta = f"{f / (a * 1e9):.1f}" if f and a else "-"   # same work on both sides: flops / (ms * 1e-3) / 1e12
        tb = f"{f / (b * 1e9):.1f}" if f and b else "-"
        lines.append(f"| {rows} | {fa} | {fb} | {sp} | {ta} | {tb} |")
    return "\n".join(lines)


def parts(recs):
    """Per-kernel median ms of the FP4 pipeline, one row per cell; empty string when not measured."""
    rs = sorted((r for r in recs if r.get("bench") == "fp4_parts"), key=lambda r: r["rows"])
    if not rs:
        return ""
    lines = ["| rows | quant | up | swiglu | down |", "|---:|---:|---:|---:|---:|"]
    for r in rs:
        lines.append(f"| {r['rows']} | {r['quant']:.3f} | {r['up']:.3f} | {r['swiglu']:.3f} | {r['down']:.3f} |")
    return "\n".join(lines)


def verdict(recs, threshold=1.5):
    failed = [r.get("check") for r in recs if "check" in r and r.get("ok") is False]
    tf, fp4 = _times(recs, "tf"), _times(recs, "fp4")
    cells = [r for r in tf if r >= PROMPT_ROWS and r in fp4]
    speed = round(min(tf[r] / fp4[r] for r in cells), 3) if cells else None
    checks_ok = not failed
    ok = checks_ok and speed is not None and speed >= threshold
    if failed:
        reason = "failed checks: " + ", ".join(failed)
    elif speed is None:
        reason = "no prompt-sized cells measured on both engines"
    else:
        reason = f"min prompt speed-up {speed}x {'>=' if ok else '<'} {threshold}x"
    # TensorFold's timings can be bimodal (median ~2x its minimum on GB10), so also judge against its fastest run.
    tf_min = {r["rows"]: r["ms_min"] for r in recs if r.get("bench") == "tf" and "ms_min" in r}
    cells_min = [r for r in tf_min if r >= PROMPT_ROWS and r in fp4]
    speed_min = round(min(tf_min[r] / fp4[r] for r in cells_min), 3) if cells_min else None
    ok_min = checks_ok and speed_min is not None and speed_min >= threshold
    return {"prompt_speedup": speed, "pass": ok, "checks_ok": checks_ok, "reason": reason,
            "prompt_speedup_vs_tf_min": speed_min, "pass_vs_tf_min": ok_min}


def main(paths):
    recs = load(paths)
    print(table(recs))
    print()
    if parts(recs):
        print(parts(recs))
        print()
    for r in recs:
        if "check" in r:
            print(json.dumps(r))
    print()
    print(json.dumps(verdict(recs)))


if __name__ == "__main__":
    main(sys.argv[1:])
