#!/usr/bin/env python3
"""F2d stage A results: prints every table of results/2026-10-01-f2d-stage-a.md.

Standard library only. Every number is read from a named file in this directory
(and, for the F2c consistency check, from ../f2c-phase1/). Run: python3 results/f2d-stage-a/compute.py
"""
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
F2C = HERE.parent / "f2c-phase1"
T = 30.0            # caller timeout, f2d-a-meta.json caller_timeout_s
WARM_TOKENS = 64    # f2d_gate.py WARM_TOKENS
TRANSPORT = 0.1     # amended allowance (spec section 8 amendment)


def rows(name):
    return [json.loads(l) for l in (HERE / name).read_text().splitlines() if l.strip()]


def f(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def table(head, body):
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for r in body:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def spread(a, b):
    """(max - min) / min, the definition in bench/f2d_gate.py."""
    return (max(a, b) - min(a, b)) / min(a, b)


# ---------------------------------------------------------------- section 2
print("## Table 2.1 container tests (f2d-exactness.out first run, f2d-exactness2.out second run, f2d-a-cuda-tests.txt)")
body = []
for fn in ("f2d-exactness.out", "f2d-exactness2.out"):
    t = (HERE / fn).read_text()
    host = re.search(r"^(?:(\d+) failed, )?(\d+) passed in ([\d.]+)s", t, re.M)
    gpu = re.search(r"^(\d+) passed, 1 deselected in ([\d.]+)s", t, re.M)
    chunk = re.search(r"^(\d+) passed, 23 deselected in ([\d.]+)s", t, re.M)
    to = len(re.findall(r"^FAILED .*", t, re.M))
    body.append([fn, f"{host.group(2)} passed, {host.group(1) or 0} failed ({to} FAILED lines)", f"{host.group(3)} s",
                 f"{gpu.group(1)} passed, {gpu.group(2)} s", f"{chunk.group(1)} passed, {chunk.group(2)} s"])
table(["file", "host-side container suite", "wall", "stage-A CUDA tests", "chunk-size test"], body)
ct = (HERE / "f2d-a-cuda-tests.txt").read_text()
print("f2d-a-cuda-tests.txt:", re.findall(r"\d+ passed[^\n]*", ct), re.findall(r"rc=\d", ct), "\n")
failed = re.findall(r"^FAILED (\S+)", (HERE / "f2d-exactness.out").read_text(), re.M)
print("first-run failed tests:", failed, "\n")

print("## Table 2.2 acceptance probes (f2d-a-acceptance.jsonl, ordered by rep)")
acc = sorted(rows("f2d-a-acceptance.jsonl"), key=lambda r: r["rep"])
table(["rep", "reuse", "reuse_miss", "cached", "TTFT s", "client_wall s", "reply_sha[:8]"],
      [[r["rep"], r["reuse"], r["reuse_miss"], r["cached"], f(r["ttft_s"], 3), f(r["client_wall_s"], 2), r["reply_sha"][:8]] for r in acc])
r1, r2 = acc[0], acc[1]
print(f"rep2 admitted_at - queued_at = {r2['admitted_at'] - r2['queued_at']:.2f} s (waited for rep1's slot); rep1 TTFT {r1['ttft_s']:.2f} s; rep2 TTFT {r2['ttft_s']:.2f} s\n")

print("## Table 2.3 bench_openai --expect-equal (f2d-a-bench-openai.json) and concurrent_equal (f2d-a-concurrent-equal.json)")
bo = json.load(open(HERE / "f2d-a-bench-openai.json"))
table(["prompt", "temperature", "runs", "draft_equal_all", "median TTFT s", "median decode tok/s"],
      [[x["prompt"], x["temperature"], len(x["draft_equal_all"]), all(x["draft_equal_all"]), f(x["ttft_s_median"], 4), f(x["decode_tps_median"], 1)] for x in bo])
print("arms:", len(bo), "x runs", {len(x["draft_equal_all"]) for x in bo}, "all true:", all(all(x["draft_equal_all"]) for x in bo))
ce = json.load(open(HERE / "f2d-a-concurrent-equal.json"))
print("concurrent_equal: all_equal", ce["all_equal"], "| prompts:", len(ce["prompts"]),
      "| every solo/concurrent/concurrent2 equal:", all(p["concurrent_equal"] and p["concurrent2_equal"] for p in ce["prompts"]),
      f"| wall solo {ce['wall_solo_s']} s, concurrent {ce['wall_concurrent_s']} s\n")

# ---------------------------------------------------------------- section 3
meta = json.load(open(HERE / "f2d-a-meta.json"))
files = {"stream-1": "f2d-a-stream-1.jsonl", "stream-2": "f2d-a-stream-2.jsonl",
         "nostream-1": "f2d-a-nostream-1.jsonl", "nostream-2": "f2d-a-nostream-2.jsonl"}
data = {k: rows(v) for k, v in files.items()}
print(f"## Table 3.1 per repetition (T = {T} s from f2d-a-meta.json caller_timeout_s = {meta['caller_timeout_s']}; PARALLEL = {meta['parallel']})")
body = []
for k, rs in data.items():
    cold = rs[0]
    ex = rs[1:]
    body.append([k, cold["client_send_utc"][11:19], f(cold["ttft_s"], 2), " / ".join(f(r["ttft_s"], 4) for r in ex),
                 "/".join(sorted({str(r["reuse"]) for r in ex})), "/".join(sorted({str(r["cached"]) for r in ex})),
                 all(r["reply_sha"] == cold["reply_sha"] for r in ex), f(cold["total_s"], 2),
                 " / ".join(f(r["total_s"], 3) for r in ex) if ex[0]["total_s"] is not None else "-"])
table(["rep", "cold sent UTC", "cold TTFT s", "exact1..4 TTFT s", "reuse", "cached", "reply == cold", "cold total_s", "exact1..4 total_s"], body)

ns_cold = [data["nostream-1"][0], data["nostream-2"][0]]
r_ref = statistics.median((r["usage"]["completion_tokens"] - 1) / r["decode_s"] for r in ns_cold)
print("r_ref per cold row:", [f((r["usage"]["completion_tokens"] - 1) / r["decode_s"], 3) for r in ns_cold], "median", f(r_ref, 3), "tok/s")
print(f"64 / r_ref = {WARM_TOKENS / r_ref:.3f} s; 0.5 T = {0.5 * T} s\n")

print("## Table 3.2 latency rule per non-streamed warm row (original bound = ttft + 64/r_ref; amended adds 0.1 s)")
body = []
nviol = {"orig": 0, "amended": 0}
for k in ("nostream-1", "nostream-2"):
    for r in data[k][1:]:
        bo_ = min(r["ttft_s"] + WARM_TOKENS / r_ref, 0.5 * T)
        ba = min(r["ttft_s"] + WARM_TOKENS / r_ref + TRANSPORT, 0.5 * T)
        ok_o, ok_a = r["total_s"] <= bo_, r["total_s"] <= ba
        nviol["orig"] += not ok_o
        nviol["amended"] += not ok_a
        body.append([k, r["step"], f(r["ttft_s"], 4), f(r["total_s"], 3), f(bo_, 3), "ok" if ok_o else "FAIL",
                     f(ba, 3), "ok" if ok_a else "FAIL", f(ba - r["total_s"], 3)])
table(["rep", "step", "TTFT s", "total_s", "orig bound", "orig", "amended bound", "amended", "amended margin s"], body)
print("violations:", nviol, "\n")

print("## Table 3.3 cold spread per mode (cold TTFT, spread = (max-min)/min, as in f2d_gate.py)")
sp = {}
body = []
for mode in ("stream", "nostream"):
    a, b = data[f"{mode}-1"][0]["ttft_s"], data[f"{mode}-2"][0]["ttft_s"]
    sp[mode] = spread(a, b)
    body.append([mode, f(a, 3), f(b, 3), f"{100 * sp[mode]:.2f} %"])
table(["mode", "rep1 cold TTFT s", "rep2 cold TTFT s", "spread"], body)
print("all four cold TTFT s:", [f(data[k][0]["ttft_s"], 3) for k in files], "| min", f(min(data[k][0]["ttft_s"] for k in files), 2),
      "max", f(max(data[k][0]["ttft_s"] for k in files), 2), "\n")
print("cold non-streamed total_s vs T:", [(k, f(data[k][0]["total_s"], 3), data[k][0]["total_s"] > T) for k in ("nostream-1", "nostream-2")], "\n")

# ---------------------------------------------------------------- section 4
print("## Table 4 gate rules: original (f2d-a-gate.json) vs amended (f2d-a-gate-amended.json)")
g0 = json.load(open(HERE / "f2d-a-gate.json"))
g1 = json.load(open(HERE / "f2d-a-gate-amended.json"))
body = []
for name in g0["rules"]:
    a, b = g0["rules"][name], g1["rules"][name]
    inf = " (informational)" if a.get("informational") else ""
    body.append([name + inf, "PASS" if a["ok"] else "FAIL", "PASS" if b["ok"] else "FAIL"])
body.append(["verdict", g0["verdict"], g1["verdict"]])
table(["rule", "original tolerances", "amended tolerances"], body)
print("original cold_spread detail:", g0["rules"]["cold_spread"]["detail"])
print("original latency detail:", [re.sub(r"^.*/", "", d) for d in g0["rules"]["latency"]["detail"]])
print("amended cold_spread detail:", g1["rules"]["cold_spread"]["detail"])
print("exact_reuse:", g1["rules"]["exact_reuse"]["detail"], "| exact_ttft max s", f(g1["rules"]["exact_ttft"]["detail"]["max_s"], 4),
      "limit", g1["rules"]["exact_ttft"]["detail"]["limit_s"], "| r_ref", f(g1["r_ref"], 3))
print("recomputed here: r_ref", f(r_ref, 3), "equals gate:", abs(r_ref - g1["r_ref"]) < 1e-9,
      "| spreads equal gate:", abs(sp["stream"] - g1["rules"]["cold_spread"]["detail"]["stream"]) < 1e-12 and abs(sp["nostream"] - g1["rules"]["cold_spread"]["detail"]["nostream"]) < 1e-12)
max_ex = max(r["ttft_s"] for k in data for r in data[k][1:])
print("max exact TTFT over 16 rows:", f(max_ex, 4), "| min", f(min(r["ttft_s"] for k in data for r in data[k][1:]), 4), "\n")

# ---------------------------------------------------------------- section 5
print("## Table 5.1 burst sequence (f2d-a-bursts.jsonl), predicted vs measured")
pred = {"b1-cold": ("none", "0"), "b1-variant": ("none", "0"), "b2-variant": ("none", "0"), "b2-extend": ("extend", "71444"), "b3-extend": ("extend", "73492")}
body = []
for r in rows("f2d-a-bursts.jsonl"):
    s = r["step"]
    p = pred.get(s, ("exact", str(r["prompt_tokens"]))) if "exact" not in s else ("exact", str(r["prompt_tokens"]))
    got_reuse = r["reuse"] or "none"
    body.append([s, r["prompt_tokens"], got_reuse, r["cached"], f(r["ttft_s"], 4), f"{p[0]} / {p[1]}",
                 "match" if (got_reuse, str(r["cached"])) == p else "DIFFERS"])
table(["step", "prompt tokens", "reuse", "cached", "TTFT s", "predicted reuse / cached", "vs prediction"], body)

print("## Table 5.2 retry probe at --parallel 3 (f2d-a-retry.jsonl, ordered by rep)")
rt = sorted(rows("f2d-a-retry.jsonl"), key=lambda r: r["rep"])
role = {1: "P cold", 2: "variant P' (71,483), first copy", 3: "variant P' identical retry", 4: "P + 2k extension (73,492)"}
table(["rep", "role", "sent UTC", "prompt tokens", "reuse", "reuse_miss", "cached", "TTFT s"],
      [[r["rep"], role[r["rep"]], r["client_send_utc"][11:23], r["prompt_tokens"], r["reuse"] or "none", r["reuse_miss"], r["cached"], f(r["ttft_s"], 3)] for r in rt])
print("server log f2d-a-retry-server.log reuse lines:", [l for l in (HERE / "f2d-a-retry-server.log").read_text().splitlines() if "prefix reuse" in l], "\n")

# ---------------------------------------------------------------- section 6
print("## Table 6 consistency with F2c (clean medians from ../f2c-phase1/f2c-phase1-S1-{32k,128k}.jsonl)")
def f2c_med(n):
    v = [json.loads(l) for l in (F2C / f"f2c-phase1-S1-{n}.jsonl").read_text().splitlines() if l.strip()]
    v = [r for r in v if r.get("arm") == "clean" and "error" not in r and r.get("ttft_s") is not None]
    return statistics.median(r["ttft_s"] for r in v), v[0]["prompt_tokens"], len(v)
m32, n32, c32 = f2c_med("32k")
m128, n128, c128 = f2c_med("128k")
n_p = 71444
interp = m32 + (n_p - n32) / (n128 - n32) * (m128 - m32)
print(f"F2c 32k: {n32} tokens, median {m32:.2f} s ({c32} rows); 128k: {n128} tokens, median {m128:.2f} s ({c128} rows)")
print(f"linear interpolation in tokens at {n_p}: {interp:.2f} s\n")
body = []
for k in files:
    c = data[k][0]["ttft_s"]
    body.append([k, f(c, 2), f"{c - interp:+.2f}", f"{100 * (c / interp - 1):+.1f} %"])
for lab, rr in (("acceptance rep1", r1), ("bursts b1-cold", rows("f2d-a-bursts.jsonl")[0]), ("retry P cold", rt[0])):
    body.append([lab, f(rr["ttft_s"], 2), f"{rr['ttft_s'] - interp:+.2f}", f"{100 * (rr['ttft_s'] / interp - 1):+.1f} %"])
table(["cold prefill", "TTFT s", "minus interpolation s", "relative"], body)
colds = [data[k][0]["ttft_s"] for k in files] + [r1["ttft_s"], rows("f2d-a-bursts.jsonl")[0]["ttft_s"], rt[0]["ttft_s"]]
print("cold TTFTs, 7 runs: min", f(min(colds), 2), "median", f(statistics.median(colds), 2), "max", f(max(colds), 2),
      "| tok/s at median:", f(n_p / statistics.median(colds), 0), "\n")

# ---------------------------------------------------------------- section 7
print("## Table 7 what the classifier would see (T = 30 s)")
cold_all = [r["ttft_s"] for r in (data[k][0] for k in files)]
exact_all = [r for k in data for r in data[k][1:]]
exact_burst = [r for r in rows("f2d-a-bursts.jsonl") if r["reuse"] == "exact"]
print(f"cold P TTFT range in the gated runs: {min(cold_all):.1f}-{max(cold_all):.1f} s (> T {T} s); non-streamed cold total_s {data['nostream-1'][0]['total_s']:.1f} / {data['nostream-2'][0]['total_s']:.1f} s")
print(f"exact-hit TTFT over the 16 gated rows: {1000 * min(r['ttft_s'] for r in exact_all):.1f}-{1000 * max(r['ttft_s'] for r in exact_all):.1f} ms; "
      f"over all {len(exact_burst)} burst exact rows: {1000 * min(r['ttft_s'] for r in exact_burst):.1f}-{1000 * max(r['ttft_s'] for r in exact_burst):.1f} ms")
ns_ex = [r["total_s"] for k in ("nostream-1", "nostream-2") for r in data[k][1:]]
print(f"non-streamed exact total_s (64 tokens): {min(ns_ex):.3f}-{max(ns_ex):.3f} s; both bounded by 0.5 T = {0.5 * T} s")
busy = [r2, [r for r in rt if r["rep"] == 3][0]]
print(f"busy identical request costs: acceptance rep2 TTFT {busy[0]['ttft_s']:.2f} s (waited {busy[0]['admitted_at'] - busy[0]['queued_at']:.2f} s for admission, then cold); retry probe rep3 TTFT {busy[1]['ttft_s']:.2f} s")
