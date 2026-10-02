#!/usr/bin/env python3
"""Engine-agnostic timing probe for the Octojet vs upstream comparison (bench/spark/compare-upstream.sh).

Upstream TensorFold's /v1/completions takes a text prompt only and its stats block lacks Octojet's fields, so every
number here is measured on the client, the same way for every engine.

  cmp_probe.py text --tokenizer TOKENIZER.json MANIFEST.json... --out-dir DIR    # ids -> DIR/<tag>.txt (container)
  cmp_probe.py ttft BASE MODEL --prompt FILE.txt --label L --out ROWS.jsonl [--max-tokens 64]
  cmp_probe.py gap  BASE MODEL --prompt FILE.txt --out GAP.json   # largest token gap of a live stream while FILE is admitted

ttft rows: label, prompt_tokens (server's usage), ttft_s (request sent -> first text), decode_tps (tokens after the
first / time after the first), total_s, and reuse / cached when the server's stats block reports them (else null).
"""
import argparse, json, os, sys, threading, time, urllib.request


def post_stream(base, body, on_text=None, timeout=3600):
    """POST a streamed completion; returns (sent, first_text_at, end, usage, stats, n_text_chunks)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base.rstrip("/") + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    sent = time.time(); first = None; usage = None; stats = None; chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("error"):
                raise RuntimeError(f"server error: {d['error']}")
            for ch in d.get("choices") or []:
                if ch.get("text"):
                    now = time.time(); chunks += 1
                    first = first or now
                    if on_text:
                        on_text(now)
            usage = d.get("usage") or usage
            stats = d.get("octojet") or d.get("tensorfold") or stats
    return sent, first, time.time(), usage or {}, stats or {}, chunks


def cmd_text(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    os.makedirs(a.out_dir, exist_ok=True)
    for path in a.manifests:
        m = json.load(open(path))
        text = tok.decode(m["ids"], skip_special_tokens=False)
        n = len(tok.encode(text, add_special_tokens=False).ids)
        tag = m.get("tag") or os.path.splitext(os.path.basename(path))[0]
        open(os.path.join(a.out_dir, tag + ".txt"), "w").write(text)
        print(json.dumps({"tag": tag, "manifest_tokens": m["tokens"], "text_tokens": n}))


def cmd_ttft(a):
    text = open(a.prompt).read()
    sent, first, end, usage, stats, _ = post_stream(a.base, {"model": a.model, "prompt": text,
                                                              "max_tokens": a.max_tokens, "temperature": 0})
    out_tok = usage.get("completion_tokens")
    row = {"label": a.label, "prompt": os.path.basename(a.prompt), "prompt_tokens": usage.get("prompt_tokens"),
           "ttft_s": round(first - sent, 3) if first else None,
           "decode_tps": round((out_tok - 1) / (end - first), 1) if first and out_tok and out_tok > 1 and end > first else None,
           "total_s": round(end - sent, 3), "completion_tokens": out_tok,
           "reuse": stats.get("reuse"), "cached": stats.get("cached"), "utc": time.strftime("%FT%TZ", time.gmtime(sent))}
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    return 0 if first else 1


def cmd_gap(a):
    """Stream A (short prompt, long reply) decodes; once it has produced 8 tokens, B (the long prompt) is sent. A's
    largest inter-token gap between B's send and B's first token is the stall a live user would see."""
    times, b_info, started = [], {}, threading.Event()

    def on_a(now):
        times.append(now)
        if len(times) == 8:
            started.set()

    def run_a():
        try:
            post_stream(a.base, {"model": a.model, "prompt": "Count slowly from 1 to 3000, one number per line:\n1\n",
                                 "max_tokens": 3000, "temperature": 0}, on_text=on_a)
        except Exception as exc:  # recorded, not raised: B's numbers still matter
            b_info["a_error"] = str(exc)
        started.set()

    ta = threading.Thread(target=run_a, daemon=True); ta.start()
    started.wait(600)
    sent, first, end, usage, _, _ = post_stream(a.base, {"model": a.model, "prompt": open(a.prompt).read(),
                                                          "max_tokens": 4, "temperature": 0})
    window = [t for t in times if sent <= t <= (first or end)]
    edges = [sent] + window + [first or end]
    gaps = [y - x for x, y in zip(edges, edges[1:])]
    res = {"b_prompt_tokens": usage.get("prompt_tokens"), "b_ttft_s": round((first or end) - sent, 3),
           "a_tokens_during_b": len(window), "a_max_gap_s": round(max(gaps), 3) if gaps else None, **b_info}
    if len(times) < 8:
        res["error"] = "stream A did not start decoding"
    print(json.dumps(res))
    json.dump(res, open(a.out, "w"), indent=1)
    return 0 if "error" not in res else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text"); t.add_argument("--tokenizer", required=True); t.add_argument("manifests", nargs="+")
    t.add_argument("--out-dir", required=True)
    for name in ("ttft", "gap"):
        p = sub.add_parser(name); p.add_argument("base"); p.add_argument("model")
        p.add_argument("--prompt", required=True); p.add_argument("--out", required=True)
        if name == "ttft":
            p.add_argument("--label", required=True); p.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    return {"text": cmd_text, "ttft": cmd_ttft, "gap": cmd_gap}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
