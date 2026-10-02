#!/usr/bin/env python3
"""Prefill arm runner: sends exact-token prompts (prefill_prompt.py manifests) through /v1/completions, one JSON row per
request appended to --out and printed.

Single-manifest mode (F2c): arms repeat in the order given — clean:N (no instrumentation), timing:N (the recorder),
histogram:N (recorder + histogram, never reported as TTFT), profile:N (recorder + profiler range); recorder summaries go
to timing_path(out, arm, rep).

  prefill_bench.py BASE MODEL --manifest FILE --arm clean:3 --arm timing:3 --draft off --max-tokens 2 --label L --out FILE.jsonl

Replay mode (F2d): --replay FILE runs an ordered JSON list of steps {label, manifest, arm, draft, max_tokens,
expect_reuse, expect_cached, expect_reply_equal} in one invocation, so kept server state carries across steps; the
single-manifest flags are then forbidden.

Expectations (both modes): --expect-reuse exact|extend|checkpoint|none|any (default none) and --expect-cached N|any
(default 0) check the stats block's reuse / cached; --keep-reply LABEL retains the first reply's token sha of this
invocation under LABEL and --expect-reply-equal LABEL compares every reply against the sha retained under LABEL in the
same invocation (so `--arm clean:3 --keep-reply base --expect-reply-equal base` asserts three equal replies; a replay
step retains its own sha under its label and names an earlier step). --no-stream sends non-streamed completions and records total_s (send →
whole response read). --concurrent K sends the K repetitions of each request at once (rows written when the batch is
done). --server-label, --stage, --rep are copied into every row. Every row carries UTC receipts client_send_utc /
client_complete_utc taken at the transport boundaries.

Exit 0 if every request matched its expectations, 1 on any mismatch or error row, 2 on malformed input.
"""
import argparse, concurrent.futures, datetime, hashlib, json, os, sys, threading, time, urllib.error, urllib.request

ARMS = ("clean", "timing", "histogram", "profile")
REUSE_KINDS = ("exact", "extend", "checkpoint", "none", "any")
STEP_KEYS = ("label", "manifest", "arm", "draft", "max_tokens", "expect_reuse", "expect_cached", "expect_reply_equal")
ROW_KEYS = ("label", "arm", "rep", "untimed", "draft", "prompt_tokens", "prompt_sha_ok", "ttft_s", "received_at",
            "queued_at", "admitted_at", "first_token_at", "client_send_at", "client_first_sse_at", "client_wall_s",
            "cached", "profiler_rc", "notes", "timing_file", "usage", "reuse", "reuse_miss", "decode_s", "max_tokens",
            "stream", "total_s", "reply_sha", "client_send_utc", "client_complete_utc", "server_label", "stage",
            "rep_id", "step")


def timing_path(out, arm, rep):
    return f"{out}-{arm}-{rep}-timing.json"


def parse_arm(text):
    name, _, count = text.partition(":")
    if name not in ARMS or not count.isdigit() or int(count) < 1:
        raise ValueError(f"--arm must be one of {'|'.join(a + ':N' for a in ARMS)} with N >= 1, got {text!r}")
    return name, int(count)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def is_int(v):
    return type(v) is int


def is_num(v):
    return type(v) in (int, float)


def request_body(model, manifest, arm, max_tokens, draft, stream=True):
    body = {"model": model, "prompt_ids": manifest["ids"], "max_tokens": max_tokens, "temperature": 0,
            "return_token_ids": True, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if not draft:
        body["draft"] = False
    if arm != "clean":
        body["timing"] = True
    if arm == "histogram":
        body["histogram"] = True
    if arm == "profile":
        body["profile"] = True
    return body


def stream(base, body, timeout=1800.0):
    """POST and read the SSE reply: (client_send_at, client_first_sse_at, client_wall_s, client_complete_utc, final chunk).
    The completion receipt is taken when [DONE] (or the end of the body) arrives, before any validation."""

    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    first, final, error = None, None, None
    sent = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError as e:
                error = error or RuntimeError(f"malformed SSE chunk: {e}")
                continue
            if "error" in chunk:
                msg = chunk["error"].get("message") if isinstance(chunk["error"], dict) else chunk["error"]
                error = error or RuntimeError(f"server error event: {msg}")
                continue
            choices = chunk.get("choices") or []
            if first is None and choices and choices[0].get("text"):
                first = time.perf_counter()
            if "usage" in chunk or "tensorfold" in chunk or "octojet" in chunk:
                final = chunk
    wall = time.perf_counter() - sent
    complete = utc_now()
    if error is not None:
        raise error
    if final is None:
        raise RuntimeError("the stream ended without a final chunk carrying usage/stats")
    return sent, first, wall, complete, final


def fetch(base, body, timeout=1800.0):
    """POST without streaming: (client_send_at, total_s, client_complete_utc, response); total_s and the receipt end
    when the whole body has been read, before parsing."""

    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    sent = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    total = time.perf_counter() - sent
    complete = utc_now()
    response = json.loads(raw)
    if not isinstance(response, dict):
        raise RuntimeError("the response is not a JSON object")
    if "error" in response:
        msg = response["error"].get("message") if isinstance(response["error"], dict) else response["error"]
        raise RuntimeError(f"server error: {msg}")
    return sent, total, complete, response


def reply_sha(stats):
    ids = stats.get("token_ids")
    if isinstance(ids, list) and all(is_int(t) for t in ids):
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return None


def fill_row(row, final, manifest, arm, rep, out, expect_reuse="none", expect_cached=0):
    """Validate the final chunk / response against the arm's required fields and the expectations, fill the row; raises
    ValueError naming the field or the missed expectation."""

    stats = final.get("octojet") or final.get("tensorfold")
    if not isinstance(stats, dict):
        raise ValueError("stats block (octojet or tensorfold) missing or not an object")
    usage = final.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("usage missing or not an object")
    row["usage"] = usage
    row["prompt_tokens"] = usage.get("prompt_tokens")
    if not is_int(row["prompt_tokens"]):
        raise ValueError("usage.prompt_tokens missing or not an integer")
    for key in ("received_at", "queued_at", "admitted_at", "first_token_at"):
        row[key] = stats.get(key)
        if not is_num(row[key]):
            raise ValueError(f"{key} missing or not a number")
    ttft = stats.get("ttft_s")
    if not is_num(ttft):
        raise ValueError("ttft_s missing or not a number")
    row["ttft_s"] = None if row["untimed"] else ttft
    sha = stats.get("prompt_sha")
    if not isinstance(sha, str):
        raise ValueError("prompt_sha missing or not a string")
    row["prompt_sha_ok"] = sha == manifest["sha256"]
    row["cached"] = stats.get("cached")
    if not is_int(row["cached"]):
        raise ValueError("cached missing or not an integer")
    row["reuse"] = stats.get("reuse")
    if row["reuse"] is not None and not isinstance(row["reuse"], str):
        raise ValueError("reuse is neither null nor a string")
    row["reuse_miss"] = stats.get("reuse_miss")
    if row["reuse_miss"] is not None and not isinstance(row["reuse_miss"], str):
        raise ValueError("reuse_miss is neither null nor a string")
    row["decode_s"] = stats.get("decode_s") if is_num(stats.get("decode_s")) else None
    row["reply_sha"] = reply_sha(stats)
    row["notes"] = stats.get("notes")
    rc = stats.get("profiler_rc")
    if arm == "profile" and not (isinstance(rc, list) and len(rc) == 2 and all(is_int(x) for x in rc)):
        raise ValueError("profiler_rc missing or not a 2-element list of integers")
    row["profiler_rc"] = rc
    timing = stats.get("timing")
    if arm in ("timing", "histogram", "profile") and not isinstance(timing, dict):
        raise ValueError("timing summary missing or not an object")
    if isinstance(timing, dict):
        path = timing_path(out, arm, rep)
        with open(path, "x") as f:          # never overwrite an earlier run's summary
            json.dump(timing, f)
        row["timing_file"] = path
    errors = []
    if arm == "profile" and rc != [0, 0]:
        errors.append(f"invalid measurement: profiler_rc is {rc}, not [0, 0]")
    if arm in ("timing", "histogram", "profile") and isinstance(timing, dict):
        if timing.get("overflow") is not False:
            errors.append(f"invalid measurement: timing overflow is {timing.get('overflow')!r}, not False")
        if timing.get("nvtx_failures") != {"push": 0, "pop": 0}:
            errors.append(f"invalid measurement: nvtx_failures is {timing.get('nvtx_failures')!r}, not zero push/pop")
        bad = [n for n in (timing.get("notes") or []) if "left open" in str(n).lower() or "nvtx" in str(n).lower()]
        if bad:
            errors.append(f"invalid measurement: timing notes report {bad}")
    if row["prompt_tokens"] != manifest["tokens"]:
        errors.append(f"prompt_tokens {row['prompt_tokens']} != {manifest['tokens']}")
    if not row["prompt_sha_ok"]:
        errors.append("prompt_sha does not match the manifest")
    if expect_cached != "any" and row["cached"] != expect_cached:
        errors.append(f"expected cached {expect_cached}, got {row['cached']}")
    if expect_reuse != "any":
        want = None if expect_reuse == "none" else expect_reuse
        if row["reuse"] != want:
            errors.append(f"expected reuse {expect_reuse}, got {row['reuse'] or 'null'} (reuse_miss {row['reuse_miss'] or 'null'})")
    if errors:
        raise ValueError("; ".join(errors))


def run_one(base, model, manifest, arm, rep, max_tokens, draft, label, out, timeout=1800.0, *, streamed=True,
            expect_reuse="none", expect_cached=0, meta=None, replies=None, keep_reply=None, expect_reply_equal=None):
    """One request → one row. ``replies`` (a dict shared by the invocation) retains reply shas under ``keep_reply``;
    ``expect_reply_equal`` names a label retained earlier."""

    row = {k: None for k in ROW_KEYS}
    row.update(label=label, arm=arm, rep=rep, untimed=arm == "histogram", draft=draft, prompt_sha_ok=False,
               max_tokens=max_tokens, stream=streamed, **(meta or {}))
    body = request_body(model, manifest, arm, max_tokens, draft, streamed)
    row["client_send_utc"] = utc_now()
    try:
        if streamed:
            sent, first, wall, complete, final = stream(base, body, timeout)
            row.update(client_send_at=sent, client_first_sse_at=first, client_wall_s=wall, client_complete_utc=complete)
        else:
            sent, total, complete, final = fetch(base, body, timeout)
            row.update(client_send_at=sent, total_s=total, client_complete_utc=complete)
        fill_row(row, final, manifest, arm, rep, out, expect_reuse, expect_cached)
    except urllib.error.HTTPError as e:
        row["error"] = f"HTTP {e.code}: {e.reason}"
    except Exception as e:  # URLError, timeout, bad JSON, an SSE error event, a malformed block, an unwritable artifact
        row["error"] = f"{type(e).__name__}: {e}"
    if row["client_complete_utc"] is None:                 # a transport failure: the receipt is the failure time
        row["client_complete_utc"] = utc_now()
    if replies is not None and keep_reply is not None:
        replies.setdefault(keep_reply, row["reply_sha"])          # the first reply under a label is the reference
    if expect_reply_equal is not None and "error" not in row:
        ref = (replies or {}).get(expect_reply_equal)
        if row["reply_sha"] is None or ref is None:
            row["error"] = f"reply token ids missing (this step or step {expect_reply_equal})"
        elif row["reply_sha"] != ref:
            row["error"] = f"reply differs from step {expect_reply_equal}"
    return row


def load_manifest(path):
    with open(path) as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError("manifest must be a JSON object")
    ids, tokens = m.get("ids"), m.get("tokens")
    if not is_int(tokens) or tokens <= 0:
        raise ValueError("manifest tokens must be a positive integer")
    if not isinstance(ids, list) or not ids or not all(is_int(t) for t in ids):
        raise ValueError("manifest ids must be a non-empty list of integers")
    if len(ids) != tokens:
        raise ValueError(f"manifest tokens {tokens} != len(ids) {len(ids)}")
    sha = m.get("sha256", m.get("prompt_sha"))
    if sha != hashlib.sha256(json.dumps(ids).encode()).hexdigest():
        raise ValueError("manifest sha256 does not match its ids")
    m["sha256"] = sha
    return m


def parse_expect_cached(text):
    if text == "any":
        return "any"
    if isinstance(text, str) and text.isdigit():
        return int(text)
    if is_int(text) and text >= 0:
        return text
    raise ValueError(f"expect_cached must be a non-negative integer or any, got {text!r}")


def load_replay(path):
    """The validated steps of a replay file: every key present and well-formed, labels unique, references earlier
    labels, manifests loadable. Returns a list of request dicts."""

    with open(path) as f:
        steps = json.load(f)
    if not isinstance(steps, list) or not steps:
        raise ValueError("replay must be a non-empty JSON list of steps")
    seen, plan = set(), []
    for i, s in enumerate(steps, 1):
        if not isinstance(s, dict):
            raise ValueError(f"replay step {i}: not an object")
        missing = [k for k in STEP_KEYS if k not in s]
        if missing:
            raise ValueError(f"replay step {i}: missing {missing}")
        label = s["label"]
        if not isinstance(label, str) or not label or label in seen:
            raise ValueError(f"replay step {i}: label must be a unique non-empty string")
        if s["arm"] not in ARMS:
            raise ValueError(f"replay step {i} ({label}): arm must be one of {'|'.join(ARMS)}")
        if s["draft"] not in ("on", "off"):
            raise ValueError(f"replay step {i} ({label}): draft must be on or off")
        if not is_int(s["max_tokens"]) or s["max_tokens"] < 1:
            raise ValueError(f"replay step {i} ({label}): max_tokens must be a positive integer")
        if s["expect_reuse"] not in REUSE_KINDS:
            raise ValueError(f"replay step {i} ({label}): expect_reuse must be one of {'|'.join(REUSE_KINDS)}")
        expect_cached = parse_expect_cached(s["expect_cached"])
        ref = s["expect_reply_equal"]
        if ref is not None and ref not in seen:
            raise ValueError(f"replay step {i} ({label}): unknown reference step {ref!r} (must name an earlier step)")
        manifest = load_manifest(s["manifest"])
        seen.add(label)
        plan.append({"arm": s["arm"], "manifest": manifest, "draft": s["draft"] == "on", "max_tokens": s["max_tokens"],
                     "expect_reuse": s["expect_reuse"], "expect_cached": expect_cached, "keep_reply": label,
                     "expect_reply_equal": ref, "step": label})
    return plan


def previous_reps(path):
    """Highest rep already recorded per arm in an existing --out file; a malformed row names its line."""

    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                arm, rep = row["arm"], row["rep"]
                if arm not in ARMS or not is_int(rep) or rep < 1:
                    raise ValueError("bad arm or rep")
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"{path} line {n} is not a runner row ({e})")
            done[arm] = max(done.get(arm, 0), rep)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base"); ap.add_argument("model")
    ap.add_argument("--manifest")
    ap.add_argument("--arm", action="append", help="clean:N | timing:N | histogram:N | profile:N (repeatable, in order)")
    ap.add_argument("--draft", choices=("on", "off"))
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--replay", metavar="FILE", help="ordered JSON steps; excludes --manifest/--arm/--draft/--max-tokens/--keep-reply/--expect-reply-equal/--concurrent")
    ap.add_argument("--expect-reuse", choices=REUSE_KINDS, default="none")
    ap.add_argument("--expect-cached", default="0", metavar="N|any")
    ap.add_argument("--keep-reply", metavar="LABEL", help="retain the invocation's first reply sha under LABEL")
    ap.add_argument("--expect-reply-equal", metavar="LABEL", help="every reply must equal the sha retained under LABEL in this invocation")
    ap.add_argument("--no-stream", action="store_true", help="non-streamed completions; records total_s")
    ap.add_argument("--concurrent", type=int, default=1, metavar="K", help="send each request's K repetitions at once")
    ap.add_argument("--server-label"); ap.add_argument("--stage"); ap.add_argument("--rep", type=int, dest="rep_id")
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=1800.0, metavar="SECONDS",
                    help="HTTP read timeout per request in seconds (default 1800)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate arguments, manifests, replay steps and timing paths, print the planned requests, exit 0 "
                         "without connecting or writing anything (exit 2 on any validation failure)")
    a = ap.parse_args(argv)

    def fail(msg):
        print(f"error: {msg}", file=sys.stderr)
        return 2

    if a.timeout <= 0:
        return fail("--timeout must be positive")
    if a.concurrent < 1:
        return fail("--concurrent must be >= 1")
    meta = {"server_label": a.server_label, "stage": a.stage, "rep_id": a.rep_id}
    try:
        done = previous_reps(a.out)
        if a.replay is not None:
            if a.manifest or a.arm or a.draft or a.max_tokens is not None or a.keep_reply or a.expect_reply_equal \
                    or a.concurrent != 1:
                return fail("--replay excludes --manifest/--arm/--draft/--max-tokens/--keep-reply/--expect-reply-equal/--concurrent")
            requests = load_replay(a.replay)
        else:
            if not (a.manifest and a.arm and a.draft):
                return fail("--manifest, --arm and --draft are required without --replay")
            arms = [parse_arm(x) for x in a.arm]
            manifest = load_manifest(a.manifest)
            expect_cached = parse_expect_cached(a.expect_cached)
            requests = [{"arm": arm, "manifest": manifest, "draft": a.draft == "on",
                         "max_tokens": 2 if a.max_tokens is None else a.max_tokens, "expect_reuse": a.expect_reuse,
                         "expect_cached": expect_cached, "keep_reply": a.keep_reply,
                         "expect_reply_equal": a.expect_reply_equal, "step": None}
                        for arm, count in arms for _ in range(count)]
    except (ValueError, OSError) as e:      # JSONDecodeError is a ValueError
        return fail(str(e))
    if any(r["max_tokens"] < 1 for r in requests):
        return fail("--max-tokens must be positive")
    plan = dict(done)                       # per-arm counter across occurrences and invocations: timing_path never collides
    planned = []
    for req in requests:                    # the K repetitions of one request are reserved before any thread starts
        for _ in range(a.concurrent):
            plan[req["arm"]] = plan.get(req["arm"], 0) + 1
            planned.append((req, plan[req["arm"]]))
            if req["arm"] != "clean" and os.path.exists(timing_path(a.out, req["arm"], plan[req["arm"]])):
                return fail(f"{timing_path(a.out, req['arm'], plan[req['arm']])} already exists; refusing to overwrite it")
    if a.dry_run:
        for req, rep in planned:
            body = request_body(a.model, req["manifest"], req["arm"], req["max_tokens"], req["draft"], not a.no_stream)
            fields = sorted(k for k in ("timing", "histogram", "profile", "draft") if k in body)
            print(f"dry-run arm={req['arm']} rep={rep} step={req['step'] or '-'} draft={'on' if req['draft'] else 'off'} "
                  f"stream={str(not a.no_stream).lower()} max_tokens={req['max_tokens']} expect_reuse={req['expect_reuse']} "
                  f"expect_cached={req['expect_cached']} reply_equal={req['expect_reply_equal'] or '-'} "
                  f"fields={','.join(fields) or '-'} "
                  f"timing_file={timing_path(a.out, req['arm'], rep) if req['arm'] != 'clean' else '-'}")
        return 0
    failed = False
    replies = {}
    with open(a.out, "a") as out:
        def emit(row):
            line = json.dumps(row)
            out.write(line + "\n"); out.flush()
            print(line, flush=True)

        def one(item):
            req, rep = item
            return run_one(a.base, a.model, req["manifest"], req["arm"], rep, req["max_tokens"], req["draft"], a.label,
                           a.out, a.timeout, streamed=not a.no_stream, expect_reuse=req["expect_reuse"],
                           expect_cached=req["expect_cached"], meta={**meta, "step": req["step"]}, replies=replies,
                           keep_reply=req["keep_reply"], expect_reply_equal=req["expect_reply_equal"])

        for i in range(0, len(planned), a.concurrent):
            batch = planned[i:i + a.concurrent]
            if len(batch) == 1:
                rows = [one(batch[0])]
            else:
                with concurrent.futures.ThreadPoolExecutor(len(batch)) as pool:
                    rows = [f.result() for f in concurrent.futures.as_completed([pool.submit(one, item) for item in batch])]
            for row in rows:
                failed |= "error" in row
                emit(row)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
