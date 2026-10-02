#!/usr/bin/env python3
"""Research-shaped latency bench: one long conversation, then follow-up turns that add new tokens.

Turn 0 sends ~BASE tokens (cold). Turns 1..N each append the previous reply plus ~NEW new tokens and ask for a
~600-token answer, like an agent loop reading a tool result. Streams every turn and reports, per turn:
time to first token (prompt processing, incl. cache reuse), decode tok/s, total seconds.

  agent_bench.py BASE_URL MODEL [--base 80000] [--new 5000] [--turns 3] [--label L]
"""
import argparse, json, random, time, urllib.request

WORDS = ("function value return index buffer config router session token cache memory kernel stream request "
         "result error thread worker queue schedule layer expert batch prompt decode prefill model weight").split()


def text(n_tokens, seed):
    rnd = random.Random(seed)
    lines, n = [], 0
    while n < n_tokens:  # ~1.3 tokens per word for this vocabulary, plus numbers
        k = rnd.randint(8, 16)
        lines.append(f"[{seed}.{len(lines)}] " + " ".join(rnd.choice(WORDS) for _ in range(k)) + f" = {rnd.randint(0, 99999)}.")
        n += int(k * 1.3) + 6
    return "\n".join(lines)


def stream(base, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); first = None; out = []; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    first = first or time.time()
                    out.append(piece)
    t1 = time.time()
    n_out = usage.get("completion_tokens") or max(1, len("".join(out)) // 4)
    ttft = (first or t1) - t0
    return "".join(out), {"prompt_tokens": usage.get("prompt_tokens"), "output_tokens": n_out,
                          "ttft_s": round(ttft, 2), "decode_tps": round(n_out / max(t1 - (first or t1), 1e-6), 1),
                          "total_s": round(t1 - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model")
    ap.add_argument("--base-tokens", type=int, default=80000); ap.add_argument("--new", type=int, default=5000)
    ap.add_argument("--turns", type=int, default=3); ap.add_argument("--label", default="")
    a = ap.parse_args()
    ask = ("\n\nTask: summarize the entries above that mention 'cache' and explain in about 450 words what a "
           "program using them might do. Write prose, no lists.")
    msgs = [{"role": "user", "content": text(a.base_tokens, 0) + ask}]
    for t in range(a.turns + 1):
        reply, m = stream(a.base, a.model, msgs, 700)
        m.update({"label": a.label, "turn": t, "kind": "cold" if t == 0 else "follow-up"})
        print(json.dumps(m), flush=True)
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "New tool output:\n" + text(a.new, t + 1) + ask}]


if __name__ == "__main__":
    main()
