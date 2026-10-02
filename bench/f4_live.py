#!/usr/bin/env python3
"""F4 live checks against a running server (stdlib only; bench/spark/f4-test.sh runs it on the host).

  f4_live.py BASE MODEL gap   --manifest m128k.json --out F.json   # a decoding stream's largest token gap while the
                                                                    # manifest's prompt is admitted (lanes)
  f4_live.py BASE MODEL twins --manifest m32k.json  --out F.json   # two identical prompts sent together: the second's
                                                                    # reuse / reuse_copy / cached and its first token
                                                                    # relative to the first's (item 2)

gap: stream A (a chat counting to 5,000, thinking off, greedy) starts; once 32 of its tokens have arrived, B (the
manifest's ids through /v1/completions, 4 tokens) is sent. A's arrivals between B's send and B's first token give
``a_max_gap_s`` (before the lanes port: about B's whole prefill; after: about one chunk and one round) and
``a_tokens_during_b``. A is closed once B is done.

twins: both requests start at once (16 tokens each). The rows are ordered by the server's admitted_at; ``b_after_a_s`` =
second first_token_at - first first_token_at (the server's monotonic clock), ``b_reuse`` / ``b_reuse_copy`` /
``b_cached`` from its stats.

Exit 0 when the check's requests completed (the summary carries the numbers; the operator's gates read them), 1 on an
error, 2 on bad input. --dry-run validates the manifest and prints the plan.
"""
import argparse, json, sys, threading, time, urllib.request

COUNT = "Count from 1 to 5000, one number per line, and nothing else."


def post(base, path, body, timeout=1800.0):
    return urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(),
                                                         {"Content-Type": "application/json"}), timeout=timeout)


def sse(resp):
    """(arrival time, chunk) for each SSE data chunk until [DONE]."""

    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield time.perf_counter(), json.loads(data)


def completion(base, model, ids, max_tokens):
    """One streamed /v1/completions request with prompt_ids: (send, first text arrival, final stats chunk)."""

    body = {"model": model, "prompt_ids": ids, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    sent, first, final = time.perf_counter(), None, {}
    with post(base, "/v1/completions", body) as r:
        for now, chunk in sse(r):
            if "error" in chunk:
                raise RuntimeError(f"server error: {chunk['error']}")
            if first is None and (chunk.get("choices") or [{}])[0].get("text"):
                first = now
            if "usage" in chunk or "octojet" in chunk or "tensorfold" in chunk:
                final = chunk
    return sent, first, final.get("octojet") or final.get("tensorfold") or {}


def window_gap(arrivals, start, end):
    """The largest gap between A's arrivals that touches [start, end] (the interval before the first arrival inside
    it counts from the last arrival before it), and the arrivals inside."""

    inside = [t for t in arrivals if start <= t <= end]
    before = [t for t in arrivals if t < start]
    after = [t for t in arrivals if t > end]
    points = before[-1:] + inside + after[:1]
    gaps = [b - a for a, b in zip(points, points[1:])]
    return (max(gaps) if gaps else None), len(inside)


def gap(base, model, ids):
    arrivals, stop, flowing, error = [], threading.Event(), threading.Event(), []

    def stream_a():
        body = {"model": model, "messages": [{"role": "user", "content": COUNT}], "max_tokens": 16000,
                "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            with post(base, "/v1/chat/completions", body) as r:
                for now, chunk in sse(r):
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        arrivals.append(now)
                        if len(arrivals) >= 32:
                            flowing.set()
                    if stop.is_set():
                        return                                # closing the connection ends A on the server
        except Exception as exc:                              # noqa: BLE001
            error.append(repr(exc))
        finally:
            flowing.set()

    a = threading.Thread(target=stream_a, daemon=True)
    a.start()
    if not flowing.wait(600) or error:
        raise RuntimeError(f"stream A did not flow: {error}")
    sent, first, stats = completion(base, model, ids, 4)
    stop.set()
    a.join(60)
    worst, inside = window_gap(arrivals, sent, first or time.perf_counter())
    return {"check": "gap", "b_prompt_tokens": len(ids), "b_ttft_s": round(first - sent, 3) if first else None,
            "b_prefill_s": stats.get("prefill_s"), "a_max_gap_s": round(worst, 3) if worst is not None else None,
            "a_tokens_during_b": inside, "a_tokens": len(arrivals), "a_error": error[0] if error else None}


def twins(base, model, ids):
    rows, errors, go = [None, None], [], threading.Barrier(2)

    def one(i):
        try:
            go.wait()
            sent, first, stats = completion(base, model, ids, 16)
            rows[i] = {"ttft_s": round(first - sent, 3) if first else None, **{k: stats.get(k) for k in (
                "reuse", "reuse_copy", "reuse_miss", "cached", "queued_at", "admitted_at", "first_token_at",
                "prefill_s")}}
        except Exception as exc:                              # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(1800)
    if errors or None in rows:
        raise RuntimeError(f"twins failed: {errors}")
    a, b = sorted(rows, key=lambda r: (r["admitted_at"] or 0, r["first_token_at"] or 0))
    after = (b["first_token_at"] - a["first_token_at"]) if a["first_token_at"] and b["first_token_at"] else None
    return {"check": "twins", "prompt_tokens": len(ids), "a": a, "b": b, "b_reuse": b["reuse"],
            "b_reuse_copy": bool(b.get("reuse_copy")), "b_cached": b["cached"],
            "b_after_a_s": round(after, 3) if after is not None else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("check", choices=("gap", "twins"))
    ap.add_argument("--manifest", required=True); ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        ids = json.load(open(a.manifest))["ids"]
        assert isinstance(ids, list) and ids and all(type(t) is int for t in ids)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        print(f"error: {a.manifest}: not a prompt manifest ({exc!r})", file=sys.stderr)
        return 2
    if a.dry_run:
        print(f"dry-run {a.check}: {len(ids)} prompt tokens from {a.manifest}")
        return 0
    if not a.out:
        ap.error("--out is required unless --dry-run")
    try:
        result = (gap if a.check == "gap" else twins)(a.base.rstrip("/"), a.model, ids)
    except Exception as exc:                                  # noqa: BLE001
        result = {"check": a.check, "error": repr(exc)}
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps(result), flush=True)
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
