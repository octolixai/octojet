#!/usr/bin/env python3
"""Exactness check: N concurrent chat streams must equal their solo runs, token for token.

Sends N distinct prompts one at a time (solo), then all at once twice, and compares the reply token ids
(the server returns them with return_token_ids). A reply without ids is an error unless --allow-text, which
falls back to comparing texts. Exit 0 if every concurrent reply equals its solo reply, 1 otherwise.

  concurrent_equal.py BASE MODEL [--prompts 4] [--max-tokens 200] [--allow-text] [--out FILE]
"""
import argparse, json, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Write a Python function that reverses a linked list and explain its complexity.",
    "Explain how a hash map handles collisions, with a short example in C.",
    "Write a JavaScript debounce function and describe when to use it.",
    "Explain the difference between processes and threads in plain English.",
]


def make_prompts(n):
    return [PROMPTS[i] if i < len(PROMPTS) else f"{PROMPTS[i % len(PROMPTS)]} (variant {i})" for i in range(n)]


def valid_ids(ids) -> bool:
    """An id list is a non-empty list of real ints (no bools, floats or strings); anything else is "no ids"."""
    return isinstance(ids, list) and len(ids) > 0 and all(type(t) is int for t in ids)


def ask(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
            "return_token_ids": True}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        reply = json.loads(r.read())
    ids = (reply.get("octojet") or reply.get("tensorfold") or {}).get("token_ids")
    ids = ids if valid_ids(ids) else None
    return reply["choices"][0]["message"]["content"] or "", ids


def try_ask(base, model, prompt, max_tokens):
    """Return ((text, ids), None) or (None, error string); a failed request never aborts the run."""
    try:
        return ask(base, model, prompt, max_tokens), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except Exception as e:  # URLError, socket timeout, bad JSON
        return None, f"{type(e).__name__}: {e}"


def first_diff(a, b):
    if a == b:
        return -1
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def concurrent(base, model, prompts, max_tokens):
    t = time.time()
    with ThreadPoolExecutor(len(prompts)) as ex:
        out = list(ex.map(lambda p: try_ask(base, model, p, max_tokens), prompts))
    return out, time.time() - t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model")
    ap.add_argument("--prompts", type=int, default=4); ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--allow-text", action="store_true", help="compare texts when a reply carries no token ids")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    prompts = make_prompts(a.prompts)
    t = time.time()
    solo = [try_ask(a.base, a.model, p, a.max_tokens) for p in prompts]
    wall_solo = time.time() - t
    c1, wall_c = concurrent(a.base, a.model, prompts, a.max_tokens)
    c2, _ = concurrent(a.base, a.model, prompts, a.max_tokens)
    rows = []
    for (s, se), (x, xe), (y, ye) in zip(solo, c1, c2):
        errs = {k: e for k, e in (("solo", se), ("concurrent", xe), ("concurrent2", ye)) if e}
        st, sid = s if s is not None else (None, None)
        xt, xid = x if x is not None else (None, None)
        yt, yid = y if y is not None else (None, None)
        row = {"solo_len_chars": len(st) if st is not None else None,
               "solo_len_tokens": len(sid) if sid is not None else None,
               "solo_ids": sid, "concurrent_ids": xid, "concurrent2_ids": yid,
               "solo_text": st, "concurrent_text": xt, "concurrent2_text": yt,
               "first_diff_index": -1}
        got = [r for r in (s, x, y) if r is not None]
        if all(r[1] is not None for r in got):
            row["compared"], pick = "tokens", 1
        elif a.allow_text:
            row["compared"], pick = "text", 0
        else:
            row["compared"], pick = None, None
            if not errs:
                row["error"] = "no token ids in the reply (pass --allow-text to compare text)"
        v = [None if r is None else r[pick] for r in (s, x, y)] if pick is not None else [None] * 3
        row["concurrent_equal"] = v[0] is not None and v[0] == v[1]
        row["concurrent2_equal"] = v[0] is not None and v[0] == v[2]
        if errs:
            row["error"] = "; ".join(f"{k}: {e}" for k, e in errs.items())
        elif v[0] is not None and v[0] != v[1]:
            row["first_diff_index"] = first_diff(v[0], v[1])
        elif v[0] is not None and v[0] != v[2]:
            row["first_diff_index"] = first_diff(v[0], v[2])
        rows.append(row)
    res = {"prompts": rows, "all_equal": all(r["concurrent_equal"] and r["concurrent2_equal"] for r in rows),
           "wall_solo_s": round(wall_solo, 2), "wall_concurrent_s": round(wall_c, 2)}
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    return 0 if res["all_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
