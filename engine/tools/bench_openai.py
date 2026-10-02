"""Single-stream decode speed of an OpenAI-compatible server, measured from the stream.

Decode tok/s = (completion tokens - 1) / (last token time - first token time), the same span the
CUDA engine's bench times (the first sampled token excluded). Standard library only, so it runs on
a bare host.

  python3 tools/bench_openai.py http://127.0.0.1:8080 MODEL --tokens 64 --reps 5 --output out.json [--no-draft | --expect-equal]
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request

PROMPTS = [
    {"name": "fibonacci-raw", "kind": "completion",
     "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it."},
    {"name": "gpu-chat-no-think", "kind": "chat",
     "prompt": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."},
]


def valid_ids(ids) -> bool:
    """An id list is a non-empty list of real ints (no bools, floats or strings); anything else is "no ids"."""
    return isinstance(ids, list) and len(ids) > 0 and all(type(t) is int for t in ids)


def stream(base: str, model: str, item: dict, tokens: int, temperature: float, seed: int | None,
           draft: bool = True) -> dict:
    body = {"model": model, "max_tokens": tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True, "return_token_ids": True}
    if not draft:
        body["draft"] = False
    if seed is not None:
        body["seed"] = seed
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    if item["kind"] == "chat":
        url = base + "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": item["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = base + "/v1/completions"
        body["prompt"] = item["prompt"]
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    usage = None
    text = []
    token_ids = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            stats = chunk.get("octojet") or chunk.get("tensorfold")
            if isinstance(stats, dict) and "token_ids" in stats:
                token_ids = stats["token_ids"] if valid_ids(stats["token_ids"]) else None
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    text.append(piece)
    n = int(usage["completion_tokens"]) if usage else None
    return {"ttft_s": first - start, "decode_s": last - first, "tokens": n,
            "decode_tps": (n - 1) / (last - first) if n and last > first else None, "text": "".join(text),
            "token_ids": token_ids}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--temperatures", default="1.0,0")
    p.add_argument("--label", default="")
    p.add_argument("--output")
    p.add_argument("--seed-from-prompt", action="store_true",
                   help="send no seed: cuda_server then seeds from the prompt, as the engine benches do, "
                        "so the reply equals the bench's and the timings compare directly")
    p.add_argument("--no-draft", action="store_true",
                   help='send "draft": false in every request (TensorFold per-request serial decoding)')
    p.add_argument("--expect-equal", "--compare-draft", dest="expect_equal", action="store_true",
                   help='after every measured drafted reply, request the same prompt and seed with "draft": false; '
                        "exit 1 when any token ids differ")
    p.add_argument("--allow-missing-ids", action="store_true",
                   help="do not fail when the server sends no token ids (older servers)")
    args = p.parse_args()
    if args.expect_equal and args.no_draft:
        p.error("--expect-equal compares drafted replies with serial ones; drop --no-draft")
    if args.expect_equal and args.allow_missing_ids:
        p.error("--expect-equal needs token ids; drop --allow-missing-ids")
    draft = not args.no_draft
    results = []
    failed = False
    for temp in [float(t) for t in args.temperatures.split(",")]:
        for item in PROMPTS:
            seeds = [None] * args.reps if args.seed_from_prompt else [1234 + i for i in range(args.reps)]
            warm = stream(args.base, args.model, item, args.tokens, temp, seeds[0], draft)   # warm-up
            runs = [stream(args.base, args.model, item, args.tokens, temp, seed, draft) for seed in seeds]

            def missing(rs):
                return any(r["token_ids"] is None for r in rs)

            ids_all = [r["token_ids"] for r in runs]
            if missing(runs + [warm]) and not args.allow_missing_ids:
                print(json.dumps({"error": "no token ids in the reply; the server predates return_token_ids", "prompt": item["name"]}), flush=True)
                return 2
            tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
            row = {"label": args.label, "prompt": item["name"], "temperature": temp, "tokens": args.tokens,
                   "decode_tps_median": statistics.median(tps), "decode_tps_all": [round(x, 2) for x in tps],
                   "ttft_s_median": statistics.median(r["ttft_s"] for r in runs),
                   "sample": runs[0]["text"][:160], "token_ids_all": ids_all,
                   "token_sha_all": [hashlib.sha256(json.dumps(i).encode()).hexdigest() if i is not None else None
                                     for i in ids_all]}
            if args.expect_equal:
                serial = [stream(args.base, args.model, item, args.tokens, temp, seed, False) for seed in seeds]
                if missing(serial):
                    print(json.dumps({"error": "no token ids in a serial reply", "prompt": item["name"]}), flush=True)
                    return 2
                row["draft_equal_all"] = [s["token_ids"] == r["token_ids"] for s, r in zip(serial, runs)]
                failed |= not all(row["draft_equal_all"])
            keys = ["label", "prompt", "temperature", "decode_tps_median", "decode_tps_all", "ttft_s_median"]
            if "draft_equal_all" in row:
                keys.append("draft_equal_all")
            print(json.dumps({k: row[k] for k in keys}), flush=True)
            results.append(row)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
