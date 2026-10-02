#!/usr/bin/env python3
"""Compare token ids across server starts: bench_openai outputs (same prompt, temperature and rep) or
concurrent_equal outputs (same prompt index, solo ids). Exit 0 when every matched cell is identical, 1 on any
difference, 2 when the files are of different kinds or nothing matches.

  compare_runs.py off-1-bench.json warm-1-bench.json warm-2-bench.json
"""
import json
import sys


def kind(doc):
    if isinstance(doc, list) and doc and "token_ids_all" in doc[0]:
        return "bench"
    if isinstance(doc, dict) and "prompts" in doc and doc["prompts"] and "solo_ids" in doc["prompts"][0]:
        return "parallel"
    return None


def valid(ids):
    """An id list is a non-empty list of real ints (no bools, floats or strings), as bench_openai and
    concurrent_equal require."""
    return isinstance(ids, list) and len(ids) > 0 and all(type(t) is int for t in ids)


def cells(doc, k):
    """Cell → value, or None for a malformed file (a bench row without reps, or a duplicate cell key, either of
    which could hide a difference). A bench cell is one rep's ids; a parallel cell is a prompt's three id lists,
    and the file is also required to say all_equal and to agree with itself, else the cell is invalid (None)."""
    out = {}
    if k == "bench":
        for r in doc:
            if not r.get("token_ids_all"):
                return None
            for i, ids in enumerate(r["token_ids_all"]):
                key = (r["prompt"], r["temperature"], i)
                if key in out:
                    return None
                out[key] = ids if valid(ids) else None
        return out
    for i, p in enumerate(doc["prompts"]):
        trio = (p.get("solo_ids"), p.get("concurrent_ids"), p.get("concurrent2_ids"))
        ok = doc.get("all_equal") is True and all(valid(t) for t in trio) and trio[0] == trio[1] == trio[2]
        out[(i, "solo")] = trio[0] if ok else None
    return out


def main(argv=None):
    paths = list(sys.argv[1:] if argv is None else argv)
    docs = []
    for p in paths:
        try:
            with open(p) as f:
                docs.append(json.load(f))
        except (OSError, ValueError) as e:
            print(f"compare_runs: cannot read {p}: {e}")
            return 2
    try:
        kinds = {kind(d) for d in docs}
    except (TypeError, KeyError, IndexError):
        kinds = {None}
    if len(paths) < 2 or len(kinds) != 1 or None in kinds:
        print("compare_runs: need two or more files of one kind (bench_openai or concurrent_equal outputs)")
        return 2
    k = kinds.pop()
    try:
        tables = [cells(d, k) for d in docs]
    except (TypeError, KeyError, AttributeError):
        tables = [None]
    if any(t is None or not t for t in tables):
        print("compare_runs: a file is malformed (a row without reps, a duplicate cell, or no cells at all)")
        return 2
    union = sorted(set.union(*(set(t) for t in tables)), key=str)
    bad = 0
    for key in union:
        vals = [t.get(key, "missing") for t in tables]
        same = all(v != "missing" and v is not None for v in vals) and all(v == vals[0] for v in vals)
        bad += not same
        shown = " | ".join("missing" if v == "missing" else "invalid" if v is None else f"{len(v)} ids" for v in vals)
        print(f"{'ok ' if same else 'DIFF'} {key} {shown}")
    print(f"{len(union) - bad} of {len(union)} cells identical across {len(paths)} files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
