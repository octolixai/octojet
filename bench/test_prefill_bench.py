import hashlib, itertools, json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import prefill_bench as pb

IDS = list(range(50, 70))
GOOD_TIMING = {"prefill_ms": 5, "overflow": False, "nvtx_failures": {"push": 0, "pop": 0}, "notes": []}
REPLY_SHA = hashlib.sha256(json.dumps([7, 8]).encode()).hexdigest()


def _manifest(tmp_path, raw=None, name="m.json", ids=IDS):
    m = {"tokens": len(ids), "ids": ids, "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
         "seed": 1, "tag": "t", "source": "x", "tokenizer": "t.json"}
    p = tmp_path / name
    p.write_text(json.dumps(m) if raw is None else raw)
    return str(p)


def _serve(bodies, cached=0, wrong_sha=False, status=None, mutate=None, text=True, stats=None, json_reply=False,
           delay=0.0, in_flight=None):
    """A fake completions server. ``stats``: a dict merged into the tensorfold block, or a callable body -> dict
    evaluated per request (after the defaults, so it can override token_ids). ``json_reply``: answer non-streamed
    requests (body stream false) with application/json. ``delay``: seconds before answering. ``in_flight``: a list the
    handler appends the number of concurrently open requests to, on entry."""

    active = [0]
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                bodies.append(body)
                active[0] += 1
                if in_flight is not None:
                    in_flight.append(active[0])
            try:
                if delay:
                    time.sleep(delay)
                self.answer(body)
            finally:
                with lock:
                    active[0] -= 1

        def answer(self, body):
            if status:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            ids = body["prompt_ids"]
            sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            tf = {"prompt_sha": "0" * 64 if wrong_sha else sha, "ttft_s": 1.234567891, "received_at": 1.0,
                  "queued_at": 1.1, "admitted_at": 1.2, "first_token_at": 2.4, "cached": cached, "notes": ["n"],
                  "timing": dict(GOOD_TIMING) if body.get("timing") else None, "decode_s": 0.25, "token_ids": [7, 8]}
            if body.get("profile"):
                tf["profiler_rc"] = [0, 0]
            tf.update(stats(body) if callable(stats) else (stats or {}))
            usage = {"prompt_tokens": len(ids), "completion_tokens": 2, "total_tokens": len(ids) + 2}
            if json_reply and body.get("stream") is False:
                resp = {"object": "text_completion", "choices": [{"index": 0, "text": "hi", "finish_reason": "length"}],
                        "usage": usage, "tensorfold": tf}
                if mutate:
                    mutate(tf, resp)
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            first = {"choices": [{"index": 0, "text": "hi", "finish_reason": None}]}
            last = {"choices": [{"index": 0, "text": "", "finish_reason": "length"}], "usage": usage, "tensorfold": tf}
            if mutate:
                mutate(tf, last)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"index":0,"text":"","finish_reason":null}]}\n\n')
            self.wfile.write(((f"data: {json.dumps(first)}\n\n" if text else "")
                              + f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n").encode())
            self.wfile.flush()
            self.close_connection = True

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _run(tmp_path, arms, extra=(), raw=None, **kw):
    bodies = []
    srv = _serve(bodies, **kw)
    out = tmp_path / "o.jsonl"
    try:
        argv = [f"http://127.0.0.1:{srv.server_port}", "m", "--manifest", _manifest(tmp_path, raw)]
        for a in arms:
            argv += ["--arm", a]
        code = pb.main([*argv, "--draft", "on", "--max-tokens", "2", "--label", "L", "--out", str(out), *extra])
    finally:
        srv.shutdown()
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    return code, rows, bodies, out


def _replay(tmp_path, steps, extra=(), **kw):
    """Run a replay file against a fake server: (code, rows, bodies)."""

    bodies = []
    srv = _serve(bodies, **kw)
    (tmp_path / "replay.json").write_text(json.dumps(steps))
    out = tmp_path / "rows.jsonl"
    try:
        code = pb.main([f"http://127.0.0.1:{srv.server_port}", "m", "--replay", str(tmp_path / "replay.json"),
                        "--label", "L", "--out", str(out), *extra])
    finally:
        srv.shutdown()
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    return code, rows, bodies


def _step(label, manifest, expect_reuse="none", expect_cached=0, equal=None, max_tokens=2, arm="clean", draft="on"):
    return {"label": label, "manifest": manifest, "arm": arm, "draft": draft, "max_tokens": max_tokens,
            "expect_reuse": expect_reuse, "expect_cached": expect_cached, "expect_reply_equal": equal}


KEYS = {"label", "arm", "rep", "untimed", "draft", "prompt_tokens", "prompt_sha_ok", "ttft_s", "received_at",
        "queued_at", "admitted_at", "first_token_at", "client_send_at", "client_first_sse_at", "client_wall_s",
        "cached", "profiler_rc", "notes", "timing_file", "usage", "reuse", "reuse_miss", "decode_s", "max_tokens",
        "stream", "total_s", "reply_sha", "client_send_utc", "client_complete_utc", "server_label", "stage",
        "rep_id", "step"}


def test_clean_arm(tmp_path, capsys):
    code, rows, bodies, _ = _run(tmp_path, ["clean:2"])
    assert code == 0 and len(rows) == 2 and len(bodies) == 2
    for i, (r, b) in enumerate(zip(rows, bodies), 1):
        assert set(r) == KEYS
        assert "timing" not in b and "histogram" not in b and "profile" not in b and "draft" not in b
        assert b["prompt_ids"] == IDS and b["stream"] is True and b["temperature"] == 0 and b["max_tokens"] == 2
        assert b["return_token_ids"] is True and b["stream_options"] == {"include_usage": True}
        assert (r["arm"], r["rep"], r["label"], r["draft"], r["untimed"]) == ("clean", i, "L", True, False)
        assert r["prompt_sha_ok"] is True and r["ttft_s"] == 1.234567891 and r["prompt_tokens"] == len(IDS)
        assert r["client_first_sse_at"] >= r["client_send_at"] and r["client_wall_s"] > 0
        assert r["timing_file"] is None and r["usage"]["prompt_tokens"] == len(IDS)
    assert capsys.readouterr().out.count("\n") == 2


def test_timing_arm_writes_summary(tmp_path):
    code, rows, bodies, out = _run(tmp_path, ["timing:1"])
    assert code == 0 and bodies[0]["timing"] is True and "histogram" not in bodies[0]
    path = pb.timing_path(str(out), "timing", 1)
    assert rows[0]["timing_file"] == path and json.loads(Path(path).read_text()) == GOOD_TIMING


def test_histogram_arm_is_untimed(tmp_path):
    code, rows, bodies, out = _run(tmp_path, ["histogram:1"])
    assert code == 0 and bodies[0]["timing"] is True and bodies[0]["histogram"] is True
    assert rows[0]["untimed"] is True and rows[0]["ttft_s"] is None
    assert Path(pb.timing_path(str(out), "histogram", 1)).exists()


def test_profile_arm(tmp_path):
    code, rows, bodies, _ = _run(tmp_path, ["profile:1"])
    assert code == 0 and bodies[0]["profile"] is True and bodies[0]["timing"] is True
    assert rows[0]["profiler_rc"] == [0, 0]


def test_arms_run_in_order(tmp_path):
    code, rows, bodies, _ = _run(tmp_path, ["clean:1", "timing:1"])
    assert code == 0 and [(r["arm"], r["rep"]) for r in rows] == [("clean", 1), ("timing", 1)]
    assert "timing" not in bodies[0] and bodies[1]["timing"] is True


def test_draft_off(tmp_path):
    bodies = []
    srv = _serve(bodies)
    try:
        code = pb.main([f"http://127.0.0.1:{srv.server_port}", "m", "--manifest", _manifest(tmp_path), "--arm", "clean:1",
                        "--draft", "off", "--max-tokens", "2", "--label", "L", "--out", str(tmp_path / "o.jsonl")])
    finally:
        srv.shutdown()
    assert code == 0 and bodies[0]["draft"] is False


def test_sha_mismatch(tmp_path):
    code, rows, _, _ = _run(tmp_path, ["clean:1"], wrong_sha=True)
    assert code == 1 and rows[0]["prompt_sha_ok"] is False


def test_cached_is_error(tmp_path):
    code, rows, _, _ = _run(tmp_path, ["clean:1"], cached=1)
    assert code == 1 and rows[0]["cached"] == 1 and rows[0]["error"]


def test_http_503_continues_other_arms(tmp_path):
    code, rows, _, _ = _run(tmp_path, ["clean:1"], status=503)
    assert code == 1 and "503" in rows[0]["error"]


def test_malformed_arm_exits_2(tmp_path):
    for bad in ("clean", "bogus:1", "clean:0", "clean:x"):
        code, rows, bodies, _ = _run(tmp_path, [bad])
        assert code == 2 and not bodies


def test_timing_path():
    assert pb.timing_path("o", "timing", 2) == "o-timing-2-timing.json"


def test_eos_only_reply_is_valid(tmp_path):
    code, rows, _, _ = _run(tmp_path, ["clean:1"], text=False)
    assert code == 0 and "error" not in rows[0]
    assert rows[0]["client_first_sse_at"] is None and rows[0]["ttft_s"] == 1.234567891


def _drop(key, arm="clean"):
    def f(tf, last):
        if key == "usage":
            last["usage"] = ["not", "an", "object"]
        elif key == "usage.prompt_tokens":
            last["usage"] = {}
        elif key == "tensorfold_list":
            last["tensorfold"] = [1]
        else:
            tf[key] = None
    return f


import pytest


@pytest.mark.parametrize("arm,key", [
    ("clean", "ttft_s"), ("clean", "received_at"), ("clean", "queued_at"), ("clean", "admitted_at"),
    ("clean", "first_token_at"), ("clean", "prompt_sha"), ("clean", "cached"), ("clean", "usage.prompt_tokens"),
    ("timing", "timing"), ("histogram", "timing"), ("profile", "timing"), ("profile", "profiler_rc"),
    ("clean", "usage"), ("clean", "tensorfold_list")])
def test_missing_field_is_error_row(tmp_path, arm, key):
    code, rows, _, _ = _run(tmp_path, [f"{arm}:1", "clean:1"], mutate=_drop(key))
    assert code == 1 and len(rows) == 2                       # the next arm still runs
    name = {"tensorfold_list": "stats block"}.get(key, key.split(".")[-1] if key != "usage" else "usage")
    assert name in rows[0]["error"], rows[0]["error"]


@pytest.mark.parametrize("rc", [0, [0], [0, "x"], [0, True], "ok"])
def test_profiler_rc_shape(tmp_path, rc):
    code, rows, _, _ = _run(tmp_path, ["profile:1"], mutate=lambda tf, last: tf.update(profiler_rc=rc))
    assert code == 1 and "profiler_rc" in rows[0]["error"]


def test_wrong_type_number_is_error(tmp_path):
    code, rows, _, _ = _run(tmp_path, ["clean:1"], mutate=lambda tf, last: tf.update(ttft_s="1.2"))
    assert code == 1 and "ttft_s" in rows[0]["error"]


def test_timing_write_failure_continues(tmp_path):
    def block(tf, last):                                     # the summary path becomes a directory mid-run: writing fails
        (tmp_path / "o.jsonl-timing-1-timing.json").mkdir(exist_ok=True)
    code, rows, _, _ = _run(tmp_path, ["timing:1", "clean:1"], mutate=block)
    assert code == 1 and len(rows) == 2 and "error" in rows[0] and "error" not in rows[1]


def test_repetition_counters_span_repeated_arms(tmp_path):
    code, rows, _, out = _run(tmp_path, ["timing:1", "clean:1", "timing:1"])
    assert code == 0 and [(r["arm"], r["rep"]) for r in rows] == [("timing", 1), ("clean", 1), ("timing", 2)]
    assert Path(pb.timing_path(str(out), "timing", 1)).exists() and Path(pb.timing_path(str(out), "timing", 2)).exists()
    assert rows[0]["timing_file"] != rows[2]["timing_file"]


def _bad_manifest(tmp_path, raw):
    code, rows, bodies, _ = _run(tmp_path, ["clean:1"], raw=raw)
    assert code == 2 and not bodies and not rows


def test_bad_manifests_exit_2(tmp_path):
    good = {"tokens": 2, "ids": [1, 2]}
    sha = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    for raw in ("[]", "null", "not json", json.dumps({**good, "sha256": sha([1, 2]), "ids": []}),
                json.dumps({"tokens": 2, "ids": [1, True], "sha256": sha([1, True])}),
                json.dumps({**good, "sha256": "0" * 64}), json.dumps({**good, "tokens": 3, "sha256": sha([1, 2])}),
                json.dumps({"tokens": 0, "ids": [], "sha256": sha([])})):
        _bad_manifest(tmp_path, raw)


def test_reps_continue_across_invocations(tmp_path):
    for _ in range(2):
        code, rows, _, out = _run(tmp_path, ["timing:1"])
        assert code == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["rep"] for r in rows] == [1, 2]
    assert Path(pb.timing_path(str(out), "timing", 1)).exists() and Path(pb.timing_path(str(out), "timing", 2)).exists()


def test_existing_artifact_collision_exits_2(tmp_path, capsys):
    Path(pb.timing_path(str(tmp_path / "o.jsonl"), "timing", 1)).write_text("{}")
    code, rows, bodies, _ = _run(tmp_path, ["timing:1"])
    assert code == 2 and not bodies and "o.jsonl-timing-1-timing.json" in capsys.readouterr().err
    assert Path(pb.timing_path(str(tmp_path / "o.jsonl"), "timing", 1)).read_text() == "{}"


def test_malformed_existing_out_row_exits_2(tmp_path, capsys):
    (tmp_path / "o.jsonl").write_text(json.dumps({"arm": "clean", "rep": 1}) + "\nnot json\n")
    bodies = []
    srv = _serve(bodies)
    try:
        code = pb.main([f"http://127.0.0.1:{srv.server_port}", "m", "--manifest", _manifest(tmp_path), "--arm", "clean:1",
                        "--draft", "on", "--label", "L", "--out", str(tmp_path / "o.jsonl")])
    finally:
        srv.shutdown()
    assert code == 2 and not bodies and "line 2" in capsys.readouterr().err


def _dry(tmp_path, arms, extra=(), raw=None, out="o.jsonl"):
    argv = ["http://127.0.0.1:1", "m", "--manifest", _manifest(tmp_path, raw)]
    for a in arms:
        argv += ["--arm", a]
    return pb.main([*argv, "--draft", "on", "--label", "L", "--out", str(tmp_path / out), "--dry-run", *extra])


def test_dry_run_prints_plan_and_touches_nothing(tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("connection attempted")
    monkeypatch.setattr(pb.urllib.request, "urlopen", boom)
    code = _dry(tmp_path, ["clean:1", "timing:2", "profile:1"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and len(out) == 4
    assert "arm=clean rep=1" in out[0] and "arm=timing rep=2" in out[2] and "profile" in out[3]
    assert not (tmp_path / "o.jsonl").exists() and not list(tmp_path.glob("o.jsonl-*"))


def test_dry_run_validation_failures_exit_2(tmp_path, capsys):
    assert _dry(tmp_path, ["bogus:1"]) == 2
    assert _dry(tmp_path, ["clean:1"], raw="not json") == 2
    Path(pb.timing_path(str(tmp_path / "o.jsonl"), "timing", 1)).write_text("{}")
    assert _dry(tmp_path, ["timing:1"]) == 2
    assert _dry(tmp_path, ["clean:1"], extra=["--timeout", "0"]) == 2
    (tmp_path / "bad.jsonl").write_text("not json\n")
    assert _dry(tmp_path, ["clean:1"], out="bad.jsonl") == 2


def test_dry_run_continues_rep_numbers(tmp_path, capsys):
    code, _, _, out = _run(tmp_path, ["timing:1"])
    assert code == 0
    capsys.readouterr()
    assert _dry(tmp_path, ["timing:1"]) == 0 and "rep=2" in capsys.readouterr().out


def test_timeout_is_applied(tmp_path, monkeypatch):
    seen = []
    real = pb.urllib.request.urlopen
    monkeypatch.setattr(pb.urllib.request, "urlopen", lambda req, timeout=None: seen.append(timeout) or real(req, timeout=timeout))
    code, _, _, _ = _run(tmp_path, ["clean:1"], extra=["--timeout", "7.5"])
    assert code == 0 and seen == [7.5]
    code, _, _, _ = _run(tmp_path, ["clean:1"])
    assert seen[-1] == 1800.0


@pytest.mark.parametrize("rc", [[0, 1], [1, 0], [-1, -1]])
def test_profile_nonzero_rc_is_error(tmp_path, rc):
    code, rows, _, _ = _run(tmp_path, ["profile:1"], mutate=lambda tf, last: tf.update(profiler_rc=rc))
    assert code == 1 and "profiler_rc" in rows[0]["error"] and rows[0]["profiler_rc"] == rc


@pytest.mark.parametrize("arm", ["timing", "histogram", "profile"])
@pytest.mark.parametrize("patch,word", [
    ({"overflow": True}, "overflow"), ({"overflow": None}, "overflow"),
    ({"nvtx_failures": {"push": 1, "pop": 0}}, "nvtx_failures"), ({"nvtx_failures": {"push": 0, "pop": 2}}, "nvtx_failures"),
    ({"notes": ["2 NVTX range(s) left open were popped at the terminal"]}, "notes"), ({"notes": ["NVTX push failed"]}, "notes")])
def test_invalid_timing_summary_is_error(tmp_path, arm, patch, word):
    def mut(tf, last):
        if tf["timing"]:
            tf["timing"].update(patch)
    code, rows, _, out = _run(tmp_path, [f"{arm}:1", "clean:1"], mutate=mut)
    assert code == 1 and len(rows) == 2 and word in rows[0]["error"] and "error" not in rows[1]
    assert rows[0]["timing_file"] and json.loads(Path(rows[0]["timing_file"]).read_text())      # raw values kept


def test_profile_arm_overflow_is_error_row(tmp_path):
    code, rows, _, _ = _run(tmp_path, ["profile:1"], mutate=lambda tf, last: tf["timing"].update(overflow=True))
    assert code == 1 and "overflow" in rows[0]["error"]


def test_profile_arm_clean_summary_is_normal_row(tmp_path):
    code, rows, _, _ = _run(tmp_path, ["profile:1"], mutate=lambda tf, last: tf.update(profiler_rc=[0, 0]))
    assert code == 0 and "error" not in rows[0] and rows[0]["profiler_rc"] == [0, 0]


# ---- F2d: expectations, replay, non-streamed completions, concurrency, receipts ----------------------------------

def test_rows_carry_the_new_keys_and_decode_s(tmp_path):
    code, rows, bodies, _ = _run(tmp_path, ["clean:1"])
    r = rows[0]
    assert code == 0 and set(r) == KEYS
    assert r["decode_s"] == 0.25 and r["reuse"] is None and r["reuse_miss"] is None and r["reply_sha"] == REPLY_SHA
    assert r["stream"] is True and r["max_tokens"] == 2 and r["total_s"] is None
    assert r["client_send_utc"].endswith("Z") and r["client_complete_utc"] >= r["client_send_utc"]
    assert r["server_label"] is None and r["stage"] is None and r["rep_id"] is None and r["step"] is None
    code, rows, _, _ = _run(tmp_path, ["clean:1"], extra=["--server-label", "S1", "--stage", "A", "--rep", "2"])
    assert code == 0 and (rows[-1]["server_label"], rows[-1]["stage"], rows[-1]["rep_id"]) == ("S1", "A", 2)


def test_receipts_are_taken_at_the_transport_boundaries(tmp_path, monkeypatch):
    clock = iter(["2026-10-01T00:00:00.000Z", "2026-10-01T00:00:05.000Z", "2026-10-01T00:00:09.000Z"])
    monkeypatch.setattr(pb, "utc_now", lambda: next(clock))
    real_json = pb.json

    def loads(text, *a, **k):                                       # parsing runs after the receipt: it consumes the third clock value
        pb.utc_now()
        return real_json.loads(text, *a, **k)

    monkeypatch.setattr(pb, "json", SimpleNamespace(loads=loads, load=real_json.load, dumps=real_json.dumps))   # the runner's json only
    code, rows, _, _ = _run(tmp_path, ["clean:1"], extra=["--no-stream"], json_reply=True)
    r = rows[0]
    assert code == 0 and r["client_send_utc"] == "2026-10-01T00:00:00.000Z" and r["client_complete_utc"] == "2026-10-01T00:00:05.000Z"
    monkeypatch.setattr(pb, "json", real_json)
    clock = iter(["2026-10-01T00:01:00.000Z", "2026-10-01T00:01:03.000Z"])
    code, rows, _, _ = _run(tmp_path, ["clean:1"], status=503)      # a transport failure still gets both receipts
    assert code == 1 and rows[-1]["client_send_utc"] == "2026-10-01T00:01:00.000Z" and rows[-1]["client_complete_utc"] == "2026-10-01T00:01:03.000Z"


def test_expectations_are_independent(tmp_path):
    hit = {"cached": len(IDS), "reuse": "exact"}
    code, rows, _, _ = _run(tmp_path, ["clean:1"], stats=hit)                                             # default: cold expected
    assert code == 1 and "expected cached 0, got 20" in rows[-1]["error"] and "expected reuse none, got exact" in rows[-1]["error"]
    code, rows, _, _ = _run(tmp_path, ["clean:1"], extra=["--expect-cached", "any"], stats=hit)           # reuse still expected none
    assert code == 1 and "cached" not in rows[-1]["error"] and "expected reuse none, got exact (reuse_miss null)" in rows[-1]["error"]
    code, rows, _, _ = _run(tmp_path, ["clean:1"], extra=["--expect-cached", str(len(IDS)), "--expect-reuse", "exact"], stats=hit)
    assert code == 0 and rows[-1]["reuse"] == "exact" and rows[-1]["cached"] == len(IDS)
    code, rows, _, _ = _run(tmp_path, ["clean:1"], extra=["--expect-reuse", "exact"], stats={"reuse_miss": "busy"})
    assert code == 1 and "expected reuse exact, got null (reuse_miss busy)" in rows[-1]["error"] and rows[-1]["reuse_miss"] == "busy"
    code, rows, _, _ = _run(tmp_path, ["clean:1"], extra=["--expect-reuse", "any", "--expect-cached", "any"], stats=hit)
    assert code == 0
    assert _run(tmp_path, ["clean:1"], extra=["--expect-cached", "-1"])[0] == 2
    assert _run(tmp_path, ["clean:1"], extra=["--expect-cached", "x"])[0] == 2


def test_no_stream_uses_the_json_response_and_records_total_s(tmp_path):
    code, rows, bodies, _ = _run(tmp_path, ["clean:1"], extra=["--no-stream", "--max-tokens", "64"], json_reply=True)
    r = rows[0]
    assert code == 0 and r["stream"] is False and r["total_s"] > 0 and r["client_first_sse_at"] is None and r["client_wall_s"] is None
    assert bodies[0]["stream"] is False and "stream_options" not in bodies[0] and bodies[0]["max_tokens"] == 64
    assert r["usage"]["completion_tokens"] == 2 and r["decode_s"] == 0.25 and r["max_tokens"] == 64 and r["ttft_s"] == 1.234567891
    assert r["reply_sha"] == REPLY_SHA
    code, rows, _, _ = _run(tmp_path, ["clean:1"], extra=["--no-stream"], json_reply=True,
                            mutate=lambda tf, resp: resp.__setitem__("error", {"message": "nope"}))
    assert code == 1 and "server error: nope" in rows[-1]["error"]


def test_keep_reply_and_expect_reply_equal_on_the_cli(tmp_path):
    code, rows, _, _ = _run(tmp_path, ["clean:3"], extra=["--keep-reply", "base", "--expect-reply-equal", "base"])
    assert code == 0 and len(rows) == 3 and len({r["reply_sha"] for r in rows}) == 1     # three equal replies
    calls = itertools.count(1)
    code, rows, _, _ = _run(tmp_path, ["clean:3"], extra=["--keep-reply", "base", "--expect-reply-equal", "base"],
                            stats=lambda body: {"token_ids": [9] if next(calls) == 2 else [7, 8]})
    assert code == 1 and "reply differs from step base" in rows[-2]["error"] and "error" not in rows[-1] and "error" not in rows[-3]
    code, rows, _, _ = _run(tmp_path, ["clean:2"], extra=["--expect-reply-equal", "never"])
    assert code == 1 and all("reply token ids missing" in r["error"] for r in rows[-2:])   # nothing retained under "never"
    code, rows, _, _ = _run(tmp_path, ["clean:1"], extra=["--expect-reply-equal", "x"], stats={"token_ids": None})
    assert code == 1 and "reply token ids missing" in rows[-1]["error"] and rows[-1]["reply_sha"] is None


def test_replay_runs_steps_in_order_with_expectations_and_reply_equality(tmp_path):
    m = _manifest(tmp_path)
    hits = {1: {"cached": 0, "reuse": None}, 2: {"cached": len(IDS), "reuse": "exact"}}
    calls = itertools.count(1)
    steps = [_step("cold", m), _step("again", m, "exact", len(IDS), equal="cold")]
    code, rows, bodies = _replay(tmp_path, steps, extra=["--server-label", "A1", "--stage", "A", "--rep", "1"],
                                 stats=lambda body: hits[next(calls)], json_reply=True)
    assert code == 0 and [r["step"] for r in rows] == ["cold", "again"] and rows[1]["reuse"] == "exact" and rows[0]["reuse"] is None
    assert rows[0]["reply_sha"] == rows[1]["reply_sha"] == REPLY_SHA
    assert rows[0]["server_label"] == "A1" and rows[0]["stage"] == "A" and rows[0]["rep_id"] == 1
    assert [r["rep"] for r in rows] == [1, 2] and len(bodies) == 2


def test_replay_reply_mismatch_is_an_error_row(tmp_path):
    m = _manifest(tmp_path)
    calls = itertools.count(1)
    steps = [_step("cold", m, "exact", len(IDS)), _step("again", m, "exact", len(IDS), equal="cold"), _step("third", m, "exact", len(IDS), equal="cold")]
    code, rows, bodies = _replay(tmp_path, steps, stats=lambda body: {"reuse": "exact", "cached": len(IDS),
                                                                        "token_ids": [1] if next(calls) == 2 else [7, 8]})
    assert code == 1 and len(rows) == 3 and "reply differs from step cold" in rows[1]["error"] and "error" not in rows[2]


def test_replay_prevalidates_every_step_before_any_request(tmp_path, capsys):
    m = _manifest(tmp_path)
    bad = [([_step("a", m), _step("b", m, equal="nope")], "unknown reference step 'nope'"),
           ([_step("a", m), _step("b", m, equal="b")], "unknown reference step 'b'"),
           ([_step("a", m, "maybe")], "expect_reuse"),
           ([_step("a", m, expect_cached=-1)], "expect_cached"),
           ([_step("a", m, max_tokens=0)], "max_tokens"),
           ([_step("a", m, arm="bogus")], "arm"),
           ([_step("a", m, draft="maybe")], "draft"),
           ([_step("a", m), _step("a", m)], "unique"),
           ([{"label": "a"}], "missing"),
           ([_step("a", str(tmp_path / "missing.json"))], "missing.json"),
           ([], "non-empty"), ("nope", "non-empty")]
    for steps, needle in bad:
        code, rows, bodies = _replay(tmp_path, steps)
        assert code == 2 and not rows and not bodies, (steps, code)
        assert needle in capsys.readouterr().err, (steps, needle)
    assert pb.main(["http://127.0.0.1:1", "m", "--replay", str(tmp_path / "replay.json"), "--manifest", m, "--label", "L",
                    "--out", str(tmp_path / "o.jsonl")]) == 2                                   # single-run flags excluded
    (tmp_path / "replay.json").write_text(json.dumps([_step("a", m)]))
    for flag in (["--manifest", m], ["--arm", "clean:1"], ["--draft", "on"], ["--max-tokens", "2"], ["--keep-reply", "x"],
                 ["--expect-reply-equal", "x"], ["--concurrent", "2"]):
        assert pb.main(["http://127.0.0.1:1", "m", "--replay", str(tmp_path / "replay.json"), *flag, "--label", "L",
                        "--out", str(tmp_path / "o.jsonl")]) == 2, flag
    assert pb.main(["http://127.0.0.1:1", "m", "--replay", str(tmp_path / "replay.json"), "--label", "L",
                    "--out", str(tmp_path / "o.jsonl"), "--dry-run"]) == 0                     # a valid replay dry-runs


def test_replay_dry_run_lists_every_step(tmp_path, capsys):
    m = _manifest(tmp_path)
    (tmp_path / "replay.json").write_text(json.dumps([_step("cold", m), _step("again", m, "exact", len(IDS), equal="cold", max_tokens=64)]))
    assert pb.main(["http://127.0.0.1:1", "m", "--replay", str(tmp_path / "replay.json"), "--label", "L", "--no-stream",
                    "--out", str(tmp_path / "o.jsonl"), "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and "step=cold" in out[0] and "step=again" in out[1] and "reply_equal=cold" in out[1]
    assert "stream=false" in out[1] and "max_tokens=64" in out[1] and "expect_cached=20" in out[1]
    assert not (tmp_path / "o.jsonl").exists()


def test_concurrent_sends_k_requests_with_unique_reps(tmp_path):
    bodies, peak = [], []
    srv = _serve(bodies, delay=0.3, in_flight=peak, stats={"reuse": None, "cached": 0})
    out = tmp_path / "k.jsonl"
    try:
        code = pb.main([f"http://127.0.0.1:{srv.server_port}", "m", "--manifest", _manifest(tmp_path), "--arm", "clean:1",
                        "--draft", "on", "--concurrent", "3", "--label", "L", "--out", str(out)])
    finally:
        srv.shutdown()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert code == 0 and len(rows) == 3 and sorted(r["rep"] for r in rows) == [1, 2, 3] and max(peak) >= 2
    code2 = pb.main(["http://127.0.0.1:1", "m", "--manifest", _manifest(tmp_path), "--arm", "clean:1", "--draft", "on",
                     "--concurrent", "2", "--label", "L", "--out", str(out), "--dry-run"])
    assert code2 == 0                                                                # the dry run plans reps 4 and 5
    assert pb.main(["http://127.0.0.1:1", "m", "--manifest", _manifest(tmp_path), "--arm", "clean:1", "--draft", "on",
                    "--concurrent", "0", "--label", "L", "--out", str(out)]) == 2
