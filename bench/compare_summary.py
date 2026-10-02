#!/usr/bin/env python3
"""One table for bench/spark/compare-upstream.sh: every config's startup window, client-side TTFTs, decode, agent
steps, the live-stream gap and accuracy, read from OUT/<config>-*.

  compare_summary.py OUT CONFIG...
"""
import json, os, re, statistics, sys


def rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def last_json(path):
    r = rows(path)
    return r[-1] if r else None


def window(path):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    m = re.search(r"(\d+) streams? of ([\d,]+)", text)              # Octojet: reserved windows
    if m:
        return f"{m.group(1)} x {m.group(2)}"
    m = re.search(r"up to (\d+) streams?, each growing to ([\d,]+)", text)   # upstream 0.6: grown on demand
    return f"up to {m.group(1)} x {m.group(2)}" if m else None


def collect(out, c):
    d = {}
    d["window"] = window(os.path.join(out, f"{c}-startup.txt"))
    t = {r["label"]: r for r in rows(os.path.join(out, f"{c}-ttft.jsonl"))}
    for k in ("cold-m32k", "cold-m32k-b", "cold-m128k", "cold-m210k", "P-cold", "Pp-variant", "P-resend"):
        d[k] = (t.get(k) or {}).get("ttft_s")
    dec = load(os.path.join(out, f"{c}-decode.json")) or []
    for r in dec:
        d[f"decode {r['prompt']} t={r['temperature']:g}"] = round(r["decode_tps_median"], 1)
    ag = rows(os.path.join(out, f"{c}-agent.jsonl"))
    if ag:
        d["agent cold ttft"] = ag[0].get("ttft_s")
        fu = [r["ttft_s"] for r in ag[1:] if r.get("ttft_s") is not None]
        d["agent follow-up ttft (median)"] = statistics.median(fu) if fu else None
        tot = [r["total_s"] for r in ag[1:] if r.get("total_s") is not None]
        d["agent follow-up step total (median)"] = statistics.median(tot) if tot else None
    g = load(os.path.join(out, f"{c}-gap.json")) or {}
    d["live-stream gap during m128k"] = g.get("a_max_gap_s")
    acc = last_json(os.path.join(out, f"{c}-acc-summary.json")) or {}
    d["GSM8K"] = acc.get("gsm8k_acc")
    he = last_json(os.path.join(out, f"{c}-acc.jsonl.humaneval.txt")) or {}
    d["HumanEval pass@1"] = he.get("pass_at_1")
    return d


def main():
    out, configs = sys.argv[1], sys.argv[2:]
    data = {c: collect(out, c) for c in configs}
    keys = []
    for c in configs:
        keys += [k for k in data[c] if k not in keys]
    w = max(len(k) for k in keys) + 2
    print("metric".ljust(w) + "".join(c.rjust(18) for c in configs))
    for k in keys:
        print(k.ljust(w) + "".join(str(data[c].get(k) if data[c].get(k) is not None else "-").rjust(18) for c in configs))
    json.dump(data, open(os.path.join(out, "compare-summary.json"), "w"), indent=1)
    print("\nTTFTs and agent times in seconds (client side, request sent -> first text); decode in tok/s; window = "
          "streams x tokens at int8 KV --parallel 3. Text only; images are off for every config.")


if __name__ == "__main__":
    main()
