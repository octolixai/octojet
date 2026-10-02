#!/usr/bin/env python3
"""F2d stage-A gate reducer (spec docs/superpowers/specs/2026-09-30-f2d-prefix-reuse-design.md section 8, stage A).

Reads the replay rows of the four stage-A servers (streamed x 2 repetitions, non-streamed x 2), validates that they are
exactly the specified workload (cold + exact1..exact4 on the 71,444-token prompt, max_tokens 2 streamed / 64 non-streamed,
no error rows, finite positive measurements) and applies the gate rules:

  exact_reuse   every exact row: reuse == "exact" and cached == prompt tokens
  exact_ttft    every exact row: ttft_s <= 2.0
  reply_equal   every exact row's reply sha equals its repetition's cold reply sha
  cold_spread   per mode: (max - min) / min of the two cold ttft_s <= 5 %
  r_ref         median over the two non-streamed cold rows of (completion_tokens - 1) / decode_s, > 0
  latency       every non-streamed exact row: total_s <= min(ttft_s + 64 / r_ref + 0.1 s transport, 0.5 x T)
  cold_within_timeout   every non-streamed cold row: total_s <= T  -- informational (reported, not gating: spec section 8
                        amendment of 2026-09-30, owner to confirm)

  f2d_gate.py ROWS.jsonl [ROWS.jsonl ...] --caller-timeout T --prompt-tokens 71444 --out GATE.json
Exit 0 on PASS, 1 otherwise (FAIL, or invalid/incomplete data, listed under "problems").
"""
import argparse, json, math, statistics, sys

STEPS = ("cold", "exact1", "exact2", "exact3", "exact4")
GROUPS = ((True, 1), (True, 2), (False, 1), (False, 2))          # (stream, rep_id)
EXACT_TTFT_MAX_S = 2.0
COLD_SPREAD_MAX = 0.05
WARM_TOKENS = 64
TRANSPORT_S = 0.1
STREAM_TOKENS = 2


def num(v):
    return type(v) in (int, float) and math.isfinite(v)


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as e:
                    raise ValueError(f"{path} line {n}: {e}")
                if not isinstance(row, dict):
                    raise ValueError(f"{path} line {n}: not an object")
                row["_source"] = f"{path}:{n}"
                rows.append(row)
    return rows


def validate(rows, prompt_tokens):
    """The complete stage-A workload, every row valid. Returns (groups[(stream, rep)][step] -> row, problems)."""

    problems, groups = [], {}
    for r in rows:
        key = (r.get("stream"), r.get("rep_id"))
        if key not in GROUPS:
            problems.append(f"{r['_source']}: unexpected group stream={r.get('stream')!r} rep_id={r.get('rep_id')!r}")
            continue
        if r.get("stage") != "A":
            problems.append(f"{r['_source']}: stage {r.get('stage')!r} is not A")
        step = r.get("step")
        if step not in STEPS:
            problems.append(f"{r['_source']}: unexpected step {step!r}")
            continue
        if step in groups.setdefault(key, {}):
            problems.append(f"{r['_source']}: duplicate step {step} in group {key}")
            continue
        groups[key][step] = r
    for key in GROUPS:
        for step in STEPS:
            if step not in groups.get(key, {}):
                problems.append(f"group stream={key[0]} rep_id={key[1]}: missing step {step}")
    for key, g in groups.items():
        stream = key[0]
        for step, r in g.items():
            where = r["_source"]
            if r.get("error"):
                problems.append(f"{where}: error row: {r['error']}")
            if r.get("prompt_sha_ok") is not True:
                problems.append(f"{where}: prompt_sha_ok is not true")
            if r.get("arm") != "clean" or r.get("draft") is not True:
                problems.append(f"{where}: arm {r.get('arm')!r} draft {r.get('draft')!r}: not a clean drafting request")
            if r.get("prompt_tokens") != prompt_tokens:
                problems.append(f"{where}: prompt_tokens {r.get('prompt_tokens')!r} != {prompt_tokens}")
            want = STREAM_TOKENS if stream else WARM_TOKENS
            if r.get("max_tokens") != want:
                problems.append(f"{where}: max_tokens {r.get('max_tokens')!r} != {want}")
            if not num(r.get("ttft_s")) or r["ttft_s"] <= 0:
                problems.append(f"{where}: ttft_s {r.get('ttft_s')!r} is not a finite positive number")
            if not isinstance(r.get("reply_sha"), str) or not r["reply_sha"]:
                problems.append(f"{where}: reply_sha missing")
            for k in ("client_send_utc", "client_complete_utc", "server_label"):
                if not isinstance(r.get(k), str) or not r[k]:
                    problems.append(f"{where}: {k} missing")
            if step == "cold" and (r.get("reuse") is not None or r.get("cached") != 0):
                problems.append(f"{where}: cold row has reuse {r.get('reuse')!r} cached {r.get('cached')!r}: not a cold baseline")
            if not stream:
                if not num(r.get("total_s")) or r["total_s"] <= 0:
                    problems.append(f"{where}: total_s {r.get('total_s')!r} is not a finite positive number")
                usage = r.get("usage") if isinstance(r.get("usage"), dict) else {}
                if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < 2:
                    problems.append(f"{where}: usage.completion_tokens {usage.get('completion_tokens')!r} invalid")
                if not num(r.get("decode_s")) or r["decode_s"] <= 0:
                    problems.append(f"{where}: decode_s {r.get('decode_s')!r} is not a finite positive number")
    return groups, problems


def evaluate(paths, caller_timeout, prompt_tokens):
    out = {"caller_timeout_s": caller_timeout, "prompt_tokens": prompt_tokens, "sources": list(paths),
           "problems": [], "rules": {}, "r_ref": None, "verdict": "FAIL"}
    if not num(caller_timeout) or caller_timeout <= 0:
        out["problems"].append(f"caller timeout {caller_timeout!r} must be a finite positive number of seconds")
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        out["problems"].append(f"prompt tokens {prompt_tokens!r} must be a positive integer")
    try:
        rows = load_rows(paths)
    except (OSError, ValueError) as e:
        out["problems"].append(str(e))
        rows = []
    groups, problems = validate(rows, prompt_tokens)
    out["problems"] += problems
    if out["problems"]:
        return out
    rules = out["rules"]
    exact = [(k, s, groups[k][s]) for k in GROUPS for s in STEPS[1:]]
    bad = [f"{r['_source']}: reuse {r.get('reuse')!r} cached {r.get('cached')!r}" for _, _, r in exact
           if r.get("reuse") != "exact" or r.get("cached") != prompt_tokens]
    rules["exact_reuse"] = {"ok": not bad, "detail": bad or f"{len(exact)} exact rows, each cached {prompt_tokens}"}
    bad = [f"{r['_source']}: ttft {r['ttft_s']:.3f} s" for _, _, r in exact if r["ttft_s"] > EXACT_TTFT_MAX_S]
    rules["exact_ttft"] = {"ok": not bad, "detail": bad or {"max_s": max(r["ttft_s"] for _, _, r in exact),
                                                            "limit_s": EXACT_TTFT_MAX_S}}
    bad = [f"{r['_source']}: reply differs from the cold reply" for k, _, r in exact
           if r["reply_sha"] != groups[k]["cold"]["reply_sha"]]
    rules["reply_equal"] = {"ok": not bad, "detail": bad or "every exact reply equals its repetition's cold reply"}
    spread = {}
    for stream in (True, False):
        c = [groups[(stream, rep)]["cold"]["ttft_s"] for rep in (1, 2)]
        spread["stream" if stream else "nostream"] = (max(c) - min(c)) / min(c)
    rules["cold_spread"] = {"ok": all(v <= COLD_SPREAD_MAX for v in spread.values()),
                            "detail": {**spread, "limit": COLD_SPREAD_MAX}}
    cold_ns = [groups[(False, rep)]["cold"] for rep in (1, 2)]
    r_ref = statistics.median((r["usage"]["completion_tokens"] - 1) / r["decode_s"] for r in cold_ns)
    out["r_ref"] = r_ref
    rules["r_ref"] = {"ok": r_ref > 0, "detail": {"tokens_per_s": r_ref, "from": "two non-streamed cold rows"}}
    bad = []
    for rep in (1, 2):
        for s in STEPS[1:]:
            r = groups[(False, rep)][s]
            bound = min(r["ttft_s"] + WARM_TOKENS / r_ref + TRANSPORT_S, 0.5 * caller_timeout) if r_ref > 0 else 0.0
            if r["total_s"] > bound:
                bad.append(f"{r['_source']}: total {r['total_s']:.3f} s > bound {bound:.3f} s")
    rules["latency"] = {"ok": not bad, "detail": bad or f"every non-streamed exact total within min(ttft + {WARM_TOKENS}/r_ref + {TRANSPORT_S} s, T/2)"}
    late = [f"{r['_source']}: cold total {r['total_s']:.3f} s > T {caller_timeout} s" for r in cold_ns
            if r["total_s"] > caller_timeout]
    rules["cold_within_timeout"] = {"ok": not late, "informational": True,
                                    "detail": late or "both non-streamed cold totals within T"}
    gating = [v["ok"] for v in rules.values() if not v.get("informational")]
    out["verdict"] = "PASS" if all(gating) else "FAIL"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+", help="runner .jsonl files (the four stage-A servers)")
    ap.add_argument("--caller-timeout", type=float, required=True, metavar="T", help="the classifier caller's timeout in seconds")
    ap.add_argument("--prompt-tokens", type=int, required=True, help="the stage-A prompt length (71444)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = evaluate(a.rows, a.caller_timeout, a.prompt_tokens)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    failing = [k for k, v in res["rules"].items() if not v["ok"] and not v.get("informational")]
    info = res["rules"].get("cold_within_timeout", {})
    print(f"F2d stage A gate: {res['verdict']}"
          + (f"; failing rules: {', '.join(failing)}" if failing else "")
          + (f"; problems: {len(res['problems'])} (see {a.out})" if res["problems"] else "")
          + ("; informational: a cold non-streamed total exceeds T" if info.get("ok") is False else ""), flush=True)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
