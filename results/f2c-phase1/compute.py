#!/usr/bin/env python3
"""F2c Phase 1 results computation (standard library only).

Reads the artefacts in this directory (and, for per-span questions, the server-side dump
~/Documents/GitHub/octojet-runs/f2c-phase1/timing-records.jsonl, which is not in git) and prints every table of
results/2026-09-30-f2c-phase1.md.  `python3 compute.py` prints all tables; `python3 compute.py --json` prints the
scalar numbers.  Each function names its source files in its docstring.
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.expanduser("~/Documents/GitHub/octojet-runs/f2c-phase1/timing-records.jsonl")
P = lambda n: os.path.join(HERE, n)
SIZES = (32000, 128000, 210000)
SZ = {32000: "32k", 128000: "128k", 210000: "210k"}


def jl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def js(path):
    with open(path) as f:
        return json.load(f)


def med(v):
    return st.median(v)


def spread(v):
    return abs(v[0] - v[1]) / med(v) if len(v) == 2 else None


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f} %"


def f(x, d=1):
    return "n/a" if x is None else f"{x:,.{d}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- rows (TTFT)
EXCLUDED = []


def load_rows():
    """Source: f2c-phase1-S*-*.jsonl (rows with an `error` field are dropped: Pause A's failed S2 rep 1 and Pause B's failed S3 rep 1; listed in `excluded`)."""
    rows = []
    EXCLUDED.clear()
    for p in sorted(glob.glob(P("f2c-phase1-S*-*.jsonl"))):
        if "warmup" in p:
            continue
        for r in jl(p):
            if "error" in r:
                EXCLUDED.append([os.path.basename(p), r["label"], r["arm"], r["rep"], r["error"]])
                continue
            r["_file"] = os.path.basename(p)
            rows.append(r)
    return rows


def arm(rows, label, a):
    return sorted([r for r in rows if r["label"] == label and r["arm"] == a], key=lambda r: r["rep"])


def client_lat(r):
    return r["client_first_sse_at"] - r["client_send_at"]


def ttft_tables(rows):
    """Sources: the row files above."""
    out = {}
    spec = [("S1 mixed", "S1-32k", 32000), ("S1 mixed", "S1-128k", 128000), ("S1 mixed", "S1-210k", 210000),
            ("S4 MLX, draft on", "S4-32k", 32000), ("S4 MLX, draft on", "S4-128k", 128000),
            ("S4 MLX, draft off", "S4-128k-nd", 128000), ("S4 MLX, draft on", "S4-210k", 210000)]
    tr, nums = [], {}
    for name, lab, n in spec:
        c = arm(rows, lab, "clean")
        if not c:
            continue
        t = [r["ttft_s"] for r in c]
        cl = [client_lat(r) for r in c]
        m = med(t)
        nums[lab] = dict(reps=t, median=m, spread=spread(t), tok_s=n / m, client=cl, client_median=med(cl),
                         cached=[r["cached"] for r in c], sha=[r["prompt_sha_ok"] for r in c],
                         prompt_tokens=[r["prompt_tokens"] for r in c])
        tr.append([name, lab, f"{n:,}", " / ".join(f"{x:.2f}" for x in t), pct(spread(t), 2), f"{m:.2f}",
                   f"{n / m:,.0f}", " / ".join(f"{x:.2f}" for x in cl), f"{med(cl):.2f}",
                   "all 0" if all(x == 0 for x in nums[lab]['cached']) else str(nums[lab]['cached']),
                   "all true" if all(nums[lab]['sha']) else str(nums[lab]['sha'])])
    out["ttft"] = table(["Server", "Label", "Prompt tokens", "TTFT reps (s)", "Spread", "Median TTFT (s)",
                         "tok/s (tokens / median)", "Client first-SSE latency reps (s)", "Client median (s)",
                         "`cached`", "`prompt_sha_ok`"], tr)
    # other arms
    orows = []
    for lab in ("S1-32k", "S1-128k", "S1-210k", "S4-128k"):
        for a in ("timing",):
            for r in arm(rows, lab, a):
                orows.append([lab, a, r["rep"], f"{r['ttft_s']:.2f}", f"{client_lat(r):.2f}", r["cached"],
                              r["prompt_sha_ok"], r["prompt_tokens"]])
    for lab in ("S2-210k", "S3-128k"):
        for r in arm(rows, lab, "profile"):
            orows.append([lab, "profile (nsys)", r["rep"], f"{r['ttft_s']:.2f}", f"{client_lat(r):.2f}", r["cached"],
                          r["prompt_sha_ok"], r["prompt_tokens"]])
    for lab in ("S1-128k", "S1-210k"):
        for r in arm(rows, lab, "histogram"):
            orows.append([lab, "histogram (untimed replay, not a TTFT figure)", r["rep"], "n/a (untimed)" if r["ttft_s"] is None else f"{r['ttft_s']:.2f}",
                          "n/a" if r["client_first_sse_at"] is None else f"{client_lat(r):.2f}", r["cached"], r["prompt_sha_ok"], r["prompt_tokens"]])
    out["other_arms"] = table(["Label", "Arm", "Rep", "TTFT (s)", "Client first-SSE latency (s)", "`cached`",
                               "`prompt_sha_ok`", "Prompt tokens"], orows)
    # warm-ups
    wr = []
    for p in sorted(glob.glob(P("f2c-phase1-S*-warmup.jsonl"))):
        for r in jl(p):
            wr.append([os.path.basename(p).replace("f2c-phase1-", "").replace(".jsonl", ""), r["rep"],
                       r["prompt_tokens"], f"{r['ttft_s']:.2f}", r["cached"], r["prompt_sha_ok"]])
    out["warmup"] = table(["File", "Row", "Prompt tokens", "TTFT (s)", "`cached`", "`prompt_sha_ok`"], wr)
    # all-rows integrity
    allc = [r["cached"] for r in rows if r["cached"] is not None]
    out["integrity"] = dict(rows=len(rows), cached_all_zero=all(x == 0 for x in allc),
                            sha_all_true=all(r["prompt_sha_ok"] for r in rows),
                            sizes_ok=all(r["prompt_tokens"] == r["usage"]["prompt_tokens"] for r in rows))
    out["nums"] = nums
    return out


def overhead(rows, nums):
    """Sources: clean and timing rows (S1; S4-128k for the control)."""
    res, tr = {}, []
    for lab, tl in (("S1-32k", "S1-32k"), ("S1-128k", "S1-128k"), ("S1-210k", "S1-210k"), ("S4-128k", "S4-128k")):
        t = [r["ttft_s"] for r in arm(rows, tl, "timing")]
        c = nums[lab]["median"]
        o = med(t) / c - 1
        res[lab] = dict(timing=t, clean=c, overhead=o)
        tr.append([lab, " / ".join(f"{x:.2f}" for x in t), f"{med(t):.2f}", f"{c:.2f}", f"{100 * o:+.2f} %",
                   "within 3 %" if abs(o) <= 0.03 else "ABOVE 3 %"])
    out = table(["Arm", "Timing TTFT reps (s)", "Timing median (s)", "Clean median (s)", "Overhead", "Rule (<= 3 %)"], tr)
    ptr = []
    for lab, cl in (("S3-128k", "S1-128k"), ("S2-210k", "S1-210k")):
        r = arm(rows, lab, "profile")[-1]
        c = nums[cl]["median"]
        res[lab] = dict(ttft=r["ttft_s"], vs_clean=r["ttft_s"] / c - 1)
        ptr.append([lab, f"{r['ttft_s']:.2f}", f"{c:.2f}", f"{100 * (r['ttft_s'] / c - 1):+.1f} %"])
    out2 = table(["nsys arm", "TTFT (s), perturbation only", "S1 clean median (s)", "Difference"], ptr)
    return out, out2, res


# ---------------------------------------------------------------- summaries (attribution)
def summaries():
    """Sources: *-timing-N-timing.json (kind summary).  Two timing reps per size on S1; the mean of the two is used,
    each rep's totals are shown in the totals table."""
    S = {}
    for n in SIZES:
        files = sorted(glob.glob(P(f"f2c-phase1-S1-{SZ[n]}.jsonl-timing-*-timing.json")))
        S[n] = [js(x) for x in files]
    return S


def mean_blocks(reps, phase):
    blocks = set()
    for s in reps:
        blocks |= set(s["device_ms"].get(phase, {}))
    return {b: st.mean(s["device_ms"].get(phase, {}).get(b, 0.0) for s in reps) for b in blocks}


def attribution(S):
    out = {}
    for phase in ("main", "mtp"):
        blk = {n: mean_blocks(S[n], phase) for n in SIZES}
        tot = {n: sum(blk[n].values()) for n in SIZES}
        names = sorted(blk[210000], key=lambda b: -blk[210000][b])
        rows = []
        for b in names:
            r = [f"`{b}`"]
            for n in SIZES:
                v = blk[n].get(b, 0.0)
                r += [f(v, 0), pct(v / tot[n])]
            rows.append(r)
        rows.append(["**sum of blocks**"] + sum(([f(tot[n], 0), "100 %"] for n in SIZES), []))
        out[phase] = table(["Block (sorted by 210k)"] + sum(([f"{SZ[n]} ms", f"{SZ[n]} %"] for n in SIZES), []), rows)
        out[phase + "_blk"] = blk
        out[phase + "_tot"] = tot
    # draft
    drows = []
    for n in SIZES:
        for i, s in enumerate(S[n]):
            d = s["device_ms"].get("draft", {})
            drows.append([SZ[n], i + 1, f(s["device_total_ms"]["draft"], 3), len(d),
                          ", ".join(f"{k} {v:.3f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:4])])
    blocks = sorted({b for n in SIZES for sr in S[n] for b in sr["device_ms"].get("draft", {})},
                    key=lambda b: -max(st.mean(x["device_ms"].get("draft", {}).get(b, 0.0) for x in S[n]) for n in SIZES))
    out["draft_full"] = table(["Draft block", "32k ms (mean of 2 reps)", "128k ms", "210k ms"],
                              [[f"`{b}`"] + [f"{st.mean(x['device_ms'].get('draft', {}).get(b, 0.0) for x in S[n]):.3f}" for n in SIZES]
                               for b in blocks] +
                              [["**total**"] + [f"{st.mean(x['device_total_ms']['draft'] for x in S[n]):.3f}" for n in SIZES]])
    out["draft"] = table(["Size", "Timing rep", "draft device total (ms)", "blocks seen", "largest draft blocks (ms)"],
                         drows)
    # host
    hrows = []
    for b in ("stage_wait", "stage_tokens", "stage_copy", "stage_ple_gather"):
        hrows.append([f"`{b}`"] + [f(st.mean(s["host_ms"][b] for s in S[n]), 1) for n in SIZES])
    out["host"] = table(["Host block (ms)"] + [SZ[n] for n in SIZES], hrows)
    # totals
    trows = []
    for n in SIZES:
        for i, s in enumerate(S[n]):
            d = s["device_total_ms"]
            h = sum(s["host_ms"].values())
            trows.append([SZ[n], i + 1, f(s["wall_ms"], 0), f(d["main"], 0), f(d["mtp"], 0), f(d["draft"], 1),
                          f(h, 0), f(d["main"] + d["mtp"] + d["draft"], 0),
                          f"{s['spans']:,}", s["overflow"], s["nvtx_failures"], s["notes"]])
    out["totals"] = table(["Size", "Timing rep", "wall (ms)", "device main (ms)", "device mtp (ms)", "device draft (ms)",
                           "host sum (ms)", "device sum (ms)", "spans", "overflow", "nvtx failures", "notes"], trows)
    rrows = []
    for n in SIZES:
        for i, sr in enumerate(S[n]):
            d = sr["device_total_ms"]
            dv = d["main"] + d["mtp"] + d["draft"]
            rrows.append([SZ[n], i + 1, f(sr["wall_ms"], 1), f(dv, 1), f(sr["wall_ms"] - dv, 1),
                          pct((sr["wall_ms"] - dv) / sr["wall_ms"], 2)])
    out["remainder"] = table(["Size", "Timing rep", "wall (ms)", "device main + mtp + draft (ms)",
                              "unattributed wall time, event clocks = wall - device sum (ms)", "share of wall"], rrows)
    out["wall"] = {n: st.mean(s["wall_ms"] for s in S[n]) for n in SIZES}
    return out


def growth_tables(S):
    """Source: summaries' `growth` (first vs last quarter ms/row)."""
    out, g = {}, {}
    for phase in ("main", "mtp"):
        for n in SIZES:
            gg = {}
            for s in S[n]:
                for b, v in s["growth"].get(phase, {}).items():
                    gg.setdefault(b, []).append(v)
            g[(phase, n)] = {b: dict(first=st.mean(x["first_quarter_ms_per_row"] for x in vs),
                                     last=st.mean(x["last_quarter_ms_per_row"] for x in vs),
                                     ratio=st.mean(x["ratio"] for x in vs)) for b, vs in gg.items()}
    for phase in ("main", "mtp"):
        names = sorted(g[(phase, 210000)], key=lambda b: -g[(phase, 210000)][b]["ratio"])
        rows = []
        for b in names:
            r = [f"`{b}`"]
            for n in (128000, 210000):
                x = g[(phase, n)].get(b)
                r += ["n/a"] * 3 if not x else [f"{x['first'] * 1000:.3f}", f"{x['last'] * 1000:.3f}", f"{x['ratio']:.2f}"]
            x32 = g[(phase, 32000)].get(b)
            r.append("n/a" if not x32 else f"{x32['ratio']:.2f}")
            rows.append(r)
        out[phase] = table(["Block (sorted by 210k ratio)", "128k first q (us/row)", "128k last q (us/row)",
                            "128k ratio", "210k first q (us/row)", "210k last q (us/row)", "210k ratio", "32k ratio"],
                           rows)
    out["g"] = g
    return out


# ---------------------------------------------------------------- transition (per 256-row block, from the dump)
def stream_dump(want_requests):
    """Source: timing-records.jsonl (outside git).  Returns, per wanted request, per 256-row attention block start
    position the main-phase device ms of idx_select / idx_scores summed and counted over the 12 attention layers,
    plus the draft span census for every request."""
    per = {}
    drafts = {}
    with open(DUMP) as fh:
        for line in fh:
            if '"kind": "span"' not in line:
                continue
            r = json.loads(line)
            q = r["request"]
            if r["phase"] == "draft" and r["clock"] == "device":
                d = drafts.setdefault(q, {"ms": 0.0, "n": 0, "chunks": set(), "blocks": {}})
                d["ms"] += r["ms"]
                d["n"] += 1
                d["chunks"].add(r["chunk"])
                d["blocks"][r["block"]] = d["blocks"].get(r["block"], 0.0) + r["ms"]
            if q in want_requests and r["phase"] == "main" and r["clock"] == "device" and \
                    r["block"] in ("idx_select", "idx_scores"):
                c = per.setdefault(q, {}).setdefault(r["pos"], {"rows": r["rows"], "idx_select": [0.0, 0],
                                                                 "idx_scores": [0.0, 0]})
                c[r["block"]][0] += r["ms"]
                c[r["block"]][1] += 1
    return per, drafts


def transition(S, per):
    out = {}
    for n in (128000, 210000):
        reps = []
        for s in S[n]:
            ch = per.get(s["meta"]["request"], {})
            reps.append({p: dict(rows=c["rows"], sel=c["idx_select"][0] / c["idx_select"][1],
                                 sc=c["idx_scores"][0] / c["idx_scores"][1], calls=c["idx_select"][1])
                         for p, c in ch.items() if c["idx_select"][1]})
        out[n] = reps
    return out


def transition_tables(tr):
    """Mean ms per idx_select / idx_scores call (one 256-row attention block in one layer) by start position."""
    lines = []
    for n in (128000, 210000):
        d = tr[n][0]
        bands = [(0, 16384), (16384, 32768), (32768, 65536), (65536, 98304), (98304, 131072), (131072, 163840),
                 (163840, 196608), (196608, 262144)]
        for lo, hi in bands:
            sel = [v for p, v in d.items() if lo <= p < hi]
            if not sel:
                continue
            lines.append([SZ[n], f"{lo:,}-{hi:,}", len(sel), f"{st.mean(v['sel'] for v in sel):.3f}",
                          f"{st.mean(v['sc'] for v in sel):.3f}",
                          f"{st.mean(v['sel'] for v in sel) / st.mean(v['sc'] for v in sel):.2f}"])
    t1 = table(["Size", "Block start position band", "256-row blocks", "idx_select ms per call (mean over layers)",
                "idx_scores ms per call", "select / scores"], lines)
    # step at each power of two: mean of the 8 blocks ending at or below 2^k vs the 8 blocks starting at or above
    steps = []
    res = {}
    for n in (128000, 210000):
        d = tr[n][0]
        for k in range(12, 18):
            b = 2 ** k
            if b + 2048 > n:
                continue
            below = [v["sel"] for p, v in d.items() if b - 2048 <= p < b - 256 + 1 and p + v["rows"] <= b]
            above = [v["sel"] for p, v in d.items() if b <= p < b + 2048]
            sb = [v["sc"] for p, v in d.items() if b - 2048 <= p and p + v["rows"] <= b]
            sa = [v["sc"] for p, v in d.items() if b <= p < b + 2048]
            if below and above:
                r = st.mean(above) / st.mean(below)
                res[(n, b)] = r
                steps.append([SZ[n], f"{b:,}", f"{st.mean(below):.3f}", f"{st.mean(above):.3f}", f"{r:.2f}x",
                              f"{st.mean(sa) / st.mean(sb):.2f}x"])
    t2 = table(["Run", "Boundary (context tokens)", "idx_select ms/call, 8 blocks below", "8 blocks above",
                "select step", "scores step (same blocks)"], steps)
    # the blocks around 131,072 at 210k
    d = tr[210000][0]
    ar = []
    for p in sorted(d):
        if 130048 <= p <= 132864:
            ar.append([f"{p:,}", f"{p + d[p]['rows']:,}", f"{d[p]['sel']:.3f}", f"{d[p]['sc']:.3f}"])
    t3 = table(["Block start", "Block end", "idx_select ms/call", "idx_scores ms/call"], ar)
    # share of the 210k idx_select time spent in blocks beyond 131,072
    tot = sum(v["sel"] * v["calls"] for v in d.values())
    bey = sum(v["sel"] * v["calls"] for p, v in d.items() if p + v["rows"] > 131072)
    tail = []
    for rep in tr[210000]:
        last = max(rep)
        tail.append((last, rep[last]["rows"], rep[last]["sel"]))
    full = [v["sel"] for p, v in d.items() if p >= 131072 and v["rows"] == 256]
    tailt = table(["210k rep", "last block start", "rows", "idx_select ms/call"],
                  [[i + 1, f"{t[0]:,}", t[1], f"{t[2]:.2f}"] for i, t in enumerate(tail)] +
                  [["full 256-row blocks >= 131,072, rep 1 mean", "", 256, f"{st.mean(full):.2f}"]])
    base = [v["sel"] for p, v in d.items() if 98304 <= p < 131072]
    base_ms = st.mean(base)
    bcalls = sum(v["calls"] for p, v in d.items() if p + v["rows"] > 131072)
    cf = bey - bcalls * base_ms
    tx_tail = tailt
    return t1, t2, t3 + "\n\n" + tx_tail, dict(beyond_share=bey / tot, total_ms=tot, beyond_ms=bey, beyond_calls=bcalls,
                            baseline_ms_per_call=base_ms, saving_if_baseline_ms=cf,
                            steps={f"{k[0]}_{k[1]}": v for k, v in res.items()})


# ---------------------------------------------------------------- histogram
def histogram_tables():
    """Sources: f2c-phase1-S1-{128k,210k}.jsonl-histogram-1-timing.json (`histogram.layers[].chunks[]`:
    rows, touched, rows_per_touched_expert median/p10/p90; shared slot already dropped; no per-expert counts)."""
    out, nums = {}, {}
    for n in (128000, 210000):
        h = js(P(f"f2c-phase1-S1-{SZ[n]}.jsonl-histogram-1-timing.json"))
        L = h["histogram"]["layers"]
        per_layer = []
        for li, lay in enumerate(L):
            full = [c for c in lay["chunks"] if c["rows"] == 2048]
            if not full:
                continue
            per_layer.append(dict(layer=li, touched=st.mean(c["touched"] for c in full),
                                  med=st.mean(c["rows_per_touched_expert"]["median"] for c in full),
                                  p10=st.mean(c["rows_per_touched_expert"]["p10"] for c in full),
                                  p90=st.mean(c["rows_per_touched_expert"]["p90"] for c in full),
                                  mean_rows=st.mean(2048 * 10 / c["touched"] for c in full),
                                  chunks=len(full)))
        nums[n] = per_layer
        allfull = [c for lay in L for c in lay["chunks"] if c["rows"] == 2048]
        tails = sorted({c["rows"] for lay in L for c in lay["chunks"] if 0 < c["rows"] < 2048})
        nums[f"{n}_extra"] = dict(layers=len(L), chunks_full=len(L[0]["chunks"]) and sum(1 for c in L[0]["chunks"] if c["rows"] == 2048),
                                  tails=tails,
                                  touched_min=min(c["touched"] for c in allfull),
                                  touched_max=max(c["touched"] for c in allfull),
                                  touched_mean=st.mean(c["touched"] for c in allfull),
                                  med_all=med([c["rows_per_touched_expert"]["median"] for c in allfull]),
                                  p10_all=med([c["rows_per_touched_expert"]["p10"] for c in allfull]),
                                  p90_all=med([c["rows_per_touched_expert"]["p90"] for c in allfull]),
                                  mean_rows=st.mean(2048 * 10 / c["touched"] for c in allfull),
                                  max_med=max(c["rows_per_touched_expert"]["median"] for c in allfull),
                                  p90_over_med=st.mean(c["rows_per_touched_expert"]["p90"] / c["rows_per_touched_expert"]["median"]
                                                       for c in allfull if c["rows_per_touched_expert"]["median"]))
    rows = []
    for n in (128000, 210000):
        pl = nums[n]
        ex = nums[f"{n}_extra"]
        rows.append([SZ[n], ex["layers"], ex["chunks_full"], ",".join(map(str, ex["tails"])) or "none",
                     f"{ex['touched_mean']:.0f} ({ex['touched_min']}-{ex['touched_max']})", f"{ex['med_all']:.0f}",
                     f"{ex['p10_all']:.0f}", f"{ex['p90_all']:.0f}", f"{ex['mean_rows']:.1f}",
                     f"{ex['p90_over_med']:.2f}"])
    out["summary"] = table(["Size", "Layers", "Full 2,048-row chunks per layer", "Tail chunk rows",
                            "Touched routed experts per chunk, mean (min-max) of 512", "Median rows per touched expert",
                            "p10", "p90", "Mean rows per touched expert (20,480 picks / touched)", "p90 / median (mean)"],
                           rows)
    # by layer, 210k and 128k: min / median / max across layers
    lr = []
    for n in (128000, 210000):
        pl = nums[n]
        for key, label in (("touched", "touched"), ("med", "median rows"), ("p10", "p10 rows"), ("p90", "p90 rows")):
            v = [x[key] for x in pl]
            lr.append([SZ[n], label, f"{min(v):.1f}", f"{med(v):.1f}", f"{max(v):.1f}",
                       f"layer {pl[v.index(min(v))]['layer']}", f"layer {pl[v.index(max(v))]['layer']}"])
    out["layers"] = table(["Size", "Quantity (mean over a layer's full chunks)", "min over layers", "median over layers",
                           "max over layers", "argmin", "argmax"], lr)
    out["nums"] = nums
    for n in (128000, 210000):
        h = js(P(f"f2c-phase1-S1-{SZ[n]}.jsonl-histogram-1-timing.json"))["histogram"]["layers"]
        rows = []
        for li, lay in enumerate(h):
            full = [c for c in lay["chunks"] if c["rows"] == 2048]
            tl = [c for c in lay["chunks"] if 0 < c["rows"] < 2048]
            m = lambda k: st.mean(c["rows_per_touched_expert"][k] for c in full)
            tt = tl[0] if tl else None
            rows.append([li, f"{st.mean(c['touched'] for c in full):.1f}", f"{m('median'):.1f}", f"{m('p10'):.1f}",
                         f"{m('p90'):.1f}", f"{m('p90') / m('median'):.2f}",
                         "n/a" if not tt else f"{tt['rows']} rows: {tt['touched']} touched, median {tt['rows_per_touched_expert']['median']}, p10 {tt['rows_per_touched_expert']['p10']}, p90 {tt['rows_per_touched_expert']['p90']}"])
        out[f"layer_{SZ[n]}"] = table(["Layer", "touched (mean over full chunks)", "median rows", "p10", "p90",
                                       "p90 / median (of the means)", "tail chunk"], rows)
    return out


# ---------------------------------------------------------------- nsys ranges
def ranges(n):
    return js(P(f"f2c-{SZ[n]}-ranges.json"))


def kern(n):
    import csv
    with open(P(f"f2c-{SZ[n]}-kern.csv")) as fh:
        txt = fh.read()
    lines = txt.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("Time (%)"))
    return list(csv.DictReader(lines[i:]))


def nsys_tables(S):
    out = {}
    R = {n: ranges(n) for n in (128000, 210000)}
    for n in (128000, 210000):
        r = R[n]
        for phase in ("main", "mtp", "draft"):
            items = [(k.split(":")[1], v) for k, v in r["ranges"].items() if k.startswith(phase + ":") and v["busy_ms"] > 0]
            tot = sum(v["busy_ms"] for _, v in items)
            items.sort(key=lambda kv: -kv[1]["busy_ms"])
            rows = []
            dev = S[n][-1] if False else None
            for b, v in items:
                dm = st.mean(s["device_ms"].get(phase, {}).get(b, 0.0) for s in S[n])
                rows.append([f"`{b}`", f(v["busy_ms"], 0), pct(v["busy_ms"] / tot), f(dm, 0) if dm else "n/a",
                             f"{v['launches']:,}",
                             f(v["tensor_active_pct"], 2), f(v["sm_issue_pct"], 1), f(v["sms_active_pct"], 1),
                             f(v["gr_active_pct"], 1), f(v["warps_in_flight_pct"], 1)])
            rows.append([f"**{phase} total**", f(tot, 0), "100 %", "", f"{sum(v['launches'] for _, v in items):,}",
                         "", "", "", "", ""])
            out[(n, phase)] = table(["Range", "GPU busy (ms)", "% of phase busy", "timing device ms (S1 mean)",
                                     "Launches", "Tensor %", "SM issue %", "SMs active %", "GR active %",
                                     "Warps in flight %"], rows)
    cm = []
    for n in (128000, 210000):
        r = R[n]
        c = r["capture_metrics"]
        cm.append([SZ[n], f(r["capture_ms"], 0), f(r["exposed_idle_ms"], 1), pct(r["exposed_idle_ms"] / r["capture_ms"], 2),
                   r["capture_bounds"], r["counters"], r["valid"], f(c["tensor_active_pct"], 2), f(c["sm_issue_pct"], 2),
                   f(c["sms_active_pct"], 2), f(c["gr_active_pct"], 2), f(c["warps_in_flight_pct"], 2),
                   c["dram_read_pct"], c["dram_write_pct"]])
    sr = []
    for n in (128000, 210000):
        r = R[n]
        b = sum(v["busy_ms"] for v in r["ranges"].values())
        sr.append([SZ[n], f(r["capture_ms"], 1), f(b, 1), f(r["exposed_idle_ms"], 1),
                   f(r["capture_ms"] - b - r["exposed_idle_ms"], 2),
                   pct((r["capture_ms"] - b - r["exposed_idle_ms"]) / r["capture_ms"], 4)])
    out["spec_remainder"] = table(["Capture", "capture_ms", "sum of range busy_ms (main + mtp + draft)", "exposed idle (ms)",
                                   "capture - busy - idle (ms)", "share of capture"], sr)
    out["capture"] = table(["Capture", "capture_ms", "exposed idle (ms)", "exposed idle / capture", "bounds",
                            "counters", "valid", "Tensor %", "SM issue %", "SMs active %", "GR active %",
                            "Warps %", "dram_read", "dram_write"], cm)
    krow = []
    for n in (128000, 210000):
        for k in kern(n)[:8]:
            krow.append([SZ[n], k["Time (%)"], f"{int(k['Total Time (ns)']) / 1e9:.1f}", f"{int(k['Instances']):,}",
                         f"{float(k['Avg (ns)']) / 1e6:.3f}", k["Name"][:90].replace("|", "/")])
    out["kern"] = table(["Capture", "Time %", "Total (s)", "Instances", "Avg (ms)", "Kernel"], krow)
    out["R"] = R
    return out


# ---------------------------------------------------------------- bounds and levers
def bounds_and_levers(nums, attr, S, growth, nsysT):
    TT = {128000: nums["S1-128k"]["median"], 210000: nums["S1-210k"]["median"], 32000: nums["S1-32k"]["median"]}
    main = attr["main_blk"]
    mtp = attr["mtp_blk"]
    R = nsysT["R"]
    g = growth["g"]

    def tot(blk, names, n):
        return sum(blk[n].get(b, 0.0) for b in names)
    B = {}
    # bounds table
    GF = 13.01e9
    rows = []
    for n in (128000, 210000):
        t = TT[n]
        ach = GF * n / t / 1e12
        exp_ms = tot(main, ["expert_up", "expert_down"], n)
        exp_all = tot(main, ["expert_up", "expert_down", "router", "plan"], n)
        tab_s = 15.6 if n == 128000 else 25.6
        gbs = tab_s * 273.0 / (exp_ms / 1000)
        ti_s = (tot(main, ["idx_scores"], n) + tot(mtp, ["idx_scores"], n)) / 1000
        sel_s = tot(main, ["idx_select"], n) / 1000
        gdn_s = tot(main, ["gdn_recurrence"], n) / 1000
        flop_idx = 27.3 if n == 128000 else 73.4
        gemm_blocks = ["expert_up", "expert_down", "dense_gdn_in", "dense_gdn_out", "dense_attn_in", "dense_attn_out",
                       "dense_ple", "hc_readout", "router", "ple"]
        gemm_s = (tot(main, gemm_blocks, n) + tot(mtp, gemm_blocks + ["mtp_input"], n)) / 1000
        flop_gdn = 25.4 if n == 128000 else 41.6
        B[n] = dict(gemm_s=gemm_s, gemm_tf=12.36e9 * n / gemm_s / 1e12, ach_tok=n / t, gemm_tflops=ach, exp_ms=exp_ms, exp_all=exp_all, gbs=gbs, idx_tflops=flop_idx / ti_s,
                    gdn_tflops=flop_gdn / gdn_s, ti_s=ti_s, sel_s=sel_s, gdn_s=gdn_s)
    t = [
        ["Whole prefill vs GEMM-only bound (13.01 GFLOP/token; 118.3 TFLOPS -> about 9k tok/s)",
         f"{B[128000]['ach_tok']:,.0f} tok/s ({100 * B[128000]['ach_tok'] / 9000:.0f} % of 9k tok/s)",
         f"{B[210000]['ach_tok']:,.0f} tok/s ({100 * B[210000]['ach_tok'] / 9000:.0f} % of 9k tok/s)",
         "clean TTFT medians (S1); the 9k tok/s bound is the spec's"],
        ["GEMM blocks on their own time (main + mtp `expert_up`, `expert_down`, five `dense_*`, `hc_readout`, `router`, `ple`, mtp `mtp_input`; 12.36 GFLOP/token = 12.05 main + 0.31 MTP, which leaves out the selected-attention 0.65; FLOPs and seconds cover the same blocks)",
         f"{B[128000]['gemm_s']:.1f} s -> {B[128000]['gemm_tf']:.1f} TFLOPS ({100 * B[128000]['gemm_tf'] / 118.3:.0f} % of 118.3)",
         f"{B[210000]['gemm_s']:.1f} s -> {B[210000]['gemm_tf']:.1f} TFLOPS ({100 * B[210000]['gemm_tf'] / 118.3:.0f} % of 118.3)",
         "device ms (S1 mean); spec FLOPs; router, PLE and HC FLOPs are in the count and their time is in the seconds"],
        ["Tensor-active, whole capture (roofline signal)",
         pct(R[128000]["capture_metrics"]["tensor_active_pct"] / 100, 1), pct(R[210000]["capture_metrics"]["tensor_active_pct"] / 100, 1),
         "`f2c-*-ranges.json` capture_metrics (percent of peak, nsys arm)"],
        ["Expert GEMMs (main up + down) on the GPU",
         f"{B[128000]['exp_ms'] / 1000:.1f} s, tensor-active {R[128000]['ranges']['main:expert_up']['tensor_active_pct']:.1f} % up / {R[128000]['ranges']['main:expert_down']['tensor_active_pct']:.1f} % down",
         f"{B[210000]['exp_ms'] / 1000:.1f} s, tensor-active {R[210000]['ranges']['main:expert_up']['tensor_active_pct']:.1f} % up / {R[210000]['ranges']['main:expert_down']['tensor_active_pct']:.1f} % down",
         "timing device ms (S1 mean); tensor % from ranges"],
        ["Expert-table traffic bound (tables alone 15.6 s at 128k, 25.6 s at 210k at 273 GB/s)",
         f"bound 15.6 s; measured expert up+down {B[128000]['exp_ms'] / 1000:.1f} s (bound / measured = {100 * 15.6 / (B[128000]['exp_ms'] / 1000):.1f} %); derived {B[128000]['gbs']:.0f} GB/s if tables are read once per layer-chunk (derived; DRAM counters unavailable)",
         f"bound 25.6 s; measured {B[210000]['exp_ms'] / 1000:.1f} s (bound / measured = {100 * 25.6 / (B[210000]['exp_ms'] / 1000):.1f} %); derived {B[210000]['gbs']:.0f} GB/s (derived)",
         "derived = bound seconds x 273 GB/s / measured seconds"],
        ["Indexer scoring FP32 (27.3 / 73.4 TFLOP: 12 main layers + the MTP head, 13 executions; time = main + mtp `idx_scores`)",
         f"{B[128000]['ti_s']:.1f} s -> {B[128000]['idx_tflops']:.2f} TFLOPS achieved (derived); bound unavailable",
         f"{B[210000]['ti_s']:.1f} s -> {B[210000]['idx_tflops']:.2f} TFLOPS achieved (derived); bound unavailable",
         "idx_scores device ms (main + mtp) and spec FLOP count"],
        ["Indexer selection (`_select`)",
         f"{B[128000]['sel_s']:.1f} s; no FLOP count, bound unavailable",
         f"{B[210000]['sel_s']:.1f} s; bound unavailable; nsys: tensor {R[210000]['ranges']['main:idx_select']['tensor_active_pct']:.2f} %, SM issue {R[210000]['ranges']['main:idx_select']['sm_issue_pct']:.1f} %, SMs active {R[210000]['ranges']['main:idx_select']['sms_active_pct']:.1f} %",
         "idx_select device ms; ranges"],
        ["GDN recurrence (25.4 / 41.6 TFLOP)",
         f"{B[128000]['gdn_s']:.1f} s -> {B[128000]['gdn_tflops']:.2f} TFLOPS achieved (derived); bound unavailable",
         f"{B[210000]['gdn_s']:.1f} s -> {B[210000]['gdn_tflops']:.2f} TFLOPS achieved (derived); bound unavailable",
         "gdn_recurrence device ms x spec FLOP count"],
    ]
    bt = table(["Resource (spec section 1)", "128k measured", "210k measured", "Source"], t)

    # levers
    hist = nsysT.get("hist")
    dT = TT[210000] - TT[128000]

    def lever(name, blocks, phase_blk, cond_text, cond_ok, extra=""):
        b128 = tot(phase_blk, blocks, 128000) / 1000
        b210 = tot(phase_blk, blocks, 210000) / 1000
        sh = b128 / TT[128000]
        gs = (b210 - b128) / dT
        a = sh >= 0.10 or gs >= 0.20
        return [name, ", ".join(f"`{x}`" for x in blocks), f"{b128:.1f} s", pct(sh), f"{b210:.1f} s", pct(gs),
                "yes" if a else "no", cond_text, "yes" if cond_ok is True else ("no" if cond_ok is False else str(cond_ok)),
                "**yes**" if (a and cond_ok is True) else "no"]
    L = []
    ex_blocks = ["expert_up", "expert_down", "router", "plan"]
    sh1 = tot(main, ex_blocks, 128000) / 1000 / TT[128000]
    hh = nsysT["hist_nums"]
    med_rows = hh[128000]["med_all"]
    # L1: entry condition is 25% of TTFT and median rows per touched < 100
    L.append(lever("L1 larger chunks", ex_blocks, main,
                   f"share {pct(sh1)} vs 25 %; median rows/touched expert {med_rows:.0f} (128k), {hh[210000]['med_all']:.0f} (210k) vs < 100",
                   (sh1 >= 0.25 and med_rows < 100)))
    # L2
    sc = g[("main", 128000)]["idx_scores"]["ratio"]
    se = g[("main", 128000)]["idx_select"]["ratio"]
    sc2 = g[("main", 210000)]["idx_scores"]["ratio"]
    se2 = g[("main", 210000)]["idx_select"]["ratio"]
    L.append(lever("L2 exact indexer optimisation", ["idx_scores", "idx_select", "idx_pool"], main,
                   f"last/first quarter ms/row: scoring {sc:.1f}x (128k) / {sc2:.1f}x (210k), selection {se:.1f}x / {se2:.1f}x vs >= 1.5x",
                   (sc >= 1.5 and se >= 1.5 and sc2 >= 1.5 and se2 >= 1.5)))
    # L3
    rem = ["idx_scores", "idx_select", "attn_sparse", "attn_gate", "dense_attn_out", "router", "plan", "expert_up",
           "expert_down", "finish"]
    mtp_all = list(mtp[210000].keys())
    m128 = tot(mtp, rem, 128000) / 1000
    m_sh = m128 / TT[128000]
    mt_sh = tot(mtp, mtp_all, 128000) / 1000 / TT[128000]
    mt_gs = (tot(mtp, mtp_all, 210000) - tot(mtp, mtp_all, 128000)) / 1000 / dT
    rule_a_mtp = (mt_sh >= 0.10 or mt_gs >= 0.20)
    r3 = lever("L3 cache-only MTP absorption (mtp tree)", rem, mtp,
               f"removable part {pct(m_sh)} of TTFT vs >= 5 %; rule (a) on the whole mtp tree: share {pct(mt_sh)}, growth share {pct(mt_gs)}",
               (m_sh >= 0.05 and rule_a_mtp))
    L.append(r3)
    L.append(lever("L4 GDN recurrence scheduling", ["gdn_recurrence"], main,
                   f"share {pct(tot(main, ['gdn_recurrence'], 128000) / 1000 / TT[128000])} vs >= 15 %",
                   tot(main, ["gdn_recurrence"], 128000) / 1000 / TT[128000] >= 0.15))
    sp = tot(main, ["attn_sparse"], 128000) / 1000 / TT[128000]
    L.append(lever("L5 sparse KV read / dequant reuse", ["attn_sparse"], main,
                   f"share {pct(sp)} vs >= 10 %; DRAM throughput: counters unavailable", False))
    dn = ["dense_gdn_in", "dense_gdn_out", "dense_attn_in", "dense_attn_out", "dense_ple", "hc_readout", "writeback"]
    dsh = tot(main, dn, 128000) / 1000 / TT[128000]
    L.append(lever("L6 dense / HC fusion", dn, main, f"share {pct(dsh)} vs >= 15 %", dsh >= 0.15))
    ei = {n: R[n]["exposed_idle_ms"] / 1000 / TT[n] for n in (128000, 210000)}
    L.append(["L7 exposed launch / Python gaps", "nsys exposed idle", f"{R[128000]['exposed_idle_ms'] / 1000:.2f} s",
              pct(ei[128000], 2), f"{R[210000]['exposed_idle_ms'] / 1000:.2f} s",
              pct((R[210000]['exposed_idle_ms'] - R[128000]['exposed_idle_ms']) / 1000 / dT, 2),
              "no", f"exposed idle {pct(ei[128000], 2)} (128k) / {pct(ei[210000], 2)} (210k) of TTFT vs >= 5 %", "no", "no"])
    ap = tot(main, ["attn_prep"], 128000) / 1000 / TT[128000]
    L.append(lever("L8 int8 KV preparation", ["attn_prep"], main,
                   f"share {pct(ap)} vs >= 5 %; also needs a microbenchmark split", ap >= 0.05 and "needs microbenchmark split"))
    lt = table(["Lever", "Target buckets", "128k block", "Share of 128k TTFT", "210k block", "Growth share (210k-128k)",
                "Rule (a): >= 10 % of TTFT or >= 20 % growth", "Own entry condition", "Condition holds", "Qualifies"], L)
    return bt, lt, B, L, dict(TT=TT, mtp_rem_share=m_sh, dT=dT)


def mlx_table(nums):
    """Source: clean medians (S1 mixed vs S4 MLX)."""
    rows = []
    for name, m, x in (("32k, draft on", "S1-32k", "S4-32k"), ("128k, draft on", "S1-128k", "S4-128k"),
                       ("210k, draft on", "S1-210k", "S4-210k")):
        a, b = nums[m]["median"], nums[x]["median"]
        rows.append([name, f"{a:.2f}", f"{b:.2f}", f"{100 * (b / a - 1):+.2f} %", f"{nums[m]['tok_s']:,.0f}", f"{nums[x]['tok_s']:,.0f}"])
    on, off = nums["S4-128k"]["median"], nums["S4-128k-nd"]["median"]
    rows.append(["128k, MLX draft off vs on", "", f"{off:.2f} vs {on:.2f}", f"{100 * (off / on - 1):+.2f} % ({off - on:+.2f} s)",
                 "", f"{nums['S4-128k-nd']['tok_s']:,.0f} vs {nums['S4-128k']['tok_s']:,.0f}"])
    return table(["Arm", "Mixed median (s)", "MLX median (s)", "MLX vs mixed", "Mixed tok/s", "MLX tok/s"], rows)


def receipts():
    """Sources: f2c-phase1-S*-server.log."""
    rows = []
    for p in sorted(glob.glob(P("f2c-phase1-S*-server.log"))):
        est, strm, loaded = "", "", ""
        for l in open(p, errors="replace"):
            if "startup estimate" in l and not est:
                est = l.split("startup estimate")[1].split(";")[0].strip()
            if "streams of" in l and not strm:
                strm = l.split("drafts a round,")[1].split(",")[1].strip() if False else l[l.index("; ") + 2 if False else 0:].strip()
                strm = strm[strm.index(" a chain") :] if False else strm
                import re
                m = re.search(r"(\d+ streams of \d+ prompt/reply tokens \([\d]+ MiB a stream\)), eager; (int8 KV cache[^;]*)", l)
                strm = f"{m.group(1)}; {m.group(2)}" if m else l.strip()[:160]
            if "loaded in" in l and not loaded:
                loaded = l.split("loaded in")[1].strip(" )\n")
        rows.append([os.path.basename(p).replace("f2c-phase1-", "").replace("-server.log", ""), est, strm, loaded])
    return table(["Server", "Startup estimate", "Capacity receipt", "Loaded in"], rows)


def clean_nums_check(nums, S, attr):
    """Top-line derived numbers used in the prose."""
    out = {}
    main = attr["main_blk"]
    for n in SIZES:
        t = nums[f"S1-{SZ[n]}"]["median"]
        rows = sorted(main[n].items(), key=lambda kv: -kv[1])[:6]
        out[n] = [(b, v / 1000, v / 1000 / t) for b, v in rows]
    return out


def main():
    rows = load_rows()
    T = ttft_tables(rows)
    nums = T["nums"]
    oh, ohp, ohn = overhead(rows, nums)
    S = summaries()
    A = attribution(S)
    G = growth_tables(S)
    want = {x["meta"]["request"] for n in (128000, 210000) for x in S[n]}
    per, drafts = stream_dump(want)
    tr = transition(S, per)
    t1, t2, t3, tx = transition_tables(tr)
    H = histogram_tables()
    N = nsys_tables(S)
    N["hist"] = H
    N["hist_nums"] = {n: H["nums"][f"{n}_extra"] for n in (128000, 210000)}
    bt, lt, B, L, X = bounds_and_levers(nums, A, S, G, N)
    # draft census from the dump for the S1 timing requests
    dr = []
    for n in SIZES:
        for i, s in enumerate(S[n]):
            d = drafts.get(s["meta"]["request"], {"ms": 0, "n": 0, "chunks": set(), "blocks": {}})
            dr.append([SZ[n], i + 1, d["n"], sorted(d["chunks"]), f"{d['ms']:.3f}"])
    drafts_t = table(["Size", "Timing rep", "draft device spans in the dump", "chunk index of those spans",
                      "sum of draft device ms"], dr)
    T["_sec"] = dict(ttft=T["ttft"], other=T["other_arms"], warm=T["warmup"], oh=oh, ohp=ohp, main=A["main"], mtp=A["mtp"],
                     draft=A["draft"], drafts=drafts_t, host=A["host"], totals=A["totals"], gmain=G["main"], gmtp=G["mtp"],
                     excluded=table(['File','Label','Arm','Rep','error'], EXCLUDED), draft_full=A['draft_full'], remainder=A['remainder'], spec_remainder=N['spec_remainder'], layer128=H['layer_128k'], layer210=H['layer_210k'], trans=t1, trans2=t2, trans3=t3, hist=H["summary"], histl=H["layers"], bounds=bt, levers=lt,
                     cap=N["capture"], mlx=mlx_table(nums), receipts=receipts(), kern=N["kern"])
    for n in (128000, 210000):
        for ph in ("main", "mtp", "draft"):
            T["_sec"][f"nsys_{SZ[n]}_{ph}"] = N[(n, ph)]
    if "--json" in sys.argv:
        print(json.dumps(dict(nums=nums, overhead=ohn, integrity=T["integrity"], bounds=B, extra=X, transition=tx,
                              top=clean_nums_check(nums, S, A), wall=A["wall"]), indent=1, default=str))
        return T
    for k, v in T["_sec"].items():
        print(f"\n### {k}\n\n{v}")
    print("\n### integrity\n", T["integrity"])
    return T


if __name__ == "__main__":
    main()
