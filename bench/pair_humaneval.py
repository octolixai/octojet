#!/usr/bin/env python3
"""pair_humaneval.py A.jsonl B.jsonl — pair per-program HumanEval results of two runs (needs A.jsonl.humaneval.txt
and B.jsonl.humaneval.txt written by run_humaneval.sh)."""
import json
import re
import sys


def load(jsonl):
    ids = [r["id"] for r in (json.loads(l) for l in open(jsonl) if '"humaneval"' in l) if r.get("task") == "humaneval"]
    res = {}
    for line in open(f"{jsonl}.humaneval.txt"):
        m = re.fullmatch(r"p(\d+)\.py (pass|fail)", line.strip())
        if m:
            res[int(m.group(1))] = m.group(2) == "pass"
    assert len(res) == len(ids), f"{jsonl}: {len(res)} result lines vs {len(ids)} humaneval rows"
    return ids, [res[i] for i in range(len(ids))]


def pair(a, b):
    ids_a, pa = load(a)
    ids_b, pb = load(b)
    assert ids_a == ids_b, "runs have different HumanEval task ids/order"
    return {"n": len(ids_a), "pass_a": sum(pa), "pass_b": sum(pb),
            "only_a": [t for t, x, y in zip(ids_a, pa, pb) if x and not y],
            "only_b": [t for t, x, y in zip(ids_a, pa, pb) if y and not x]}


if __name__ == "__main__":
    r = pair(sys.argv[1], sys.argv[2])
    print(f"A pass {r['pass_a']}/{r['n']}  B pass {r['pass_b']}/{r['n']}")
    print(f"only A: {len(r['only_a'])} {r['only_a']}")
    print(f"only B: {len(r['only_b'])} {r['only_b']}")
