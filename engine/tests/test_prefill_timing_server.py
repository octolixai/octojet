"""The CUDA server's Phase 1 request fields: timing/profile/histogram are validated before any header is sent,
prompt_ids give an exact prompt, prompt_sha and the timestamps land in the stats block, busy timing is a 400."""

import hashlib
import http.client
import inspect
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tensorfold.cuda import prefill_timing as pt
from tensorfold.cuda import server as cuda_server


def sha(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


class FakeTok:
    """Whitespace tokens -> ids by hash; decode joins. Enough for prompt handling and StreamDecoder."""
    def encode(self, text, add_special_tokens=False):
        ids = [1000 + (abs(hash(w)) % 5000) for w in text.split()]
        return type("E", (), {"ids": ids})()
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"t{i}" for i in ids)


class FakeEngine:
    eos = [2]
    concurrent = True
    vocab_size = 6000

    def __init__(self):
        self.seen = {}
        self.busy = False
        self.profiler_rc = None
        self.stats_override = {}                      # merged into every generate() result (reuse tests)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, timing=False, profile=False,
                 histogram=False, received_at=0.0):
        self.seen = dict(prompt=list(prompt), timing=timing, profile=profile, histogram=histogram, received_at=received_at)
        if self.busy and (timing or histogram):
            raise pt.TimingBusy("prefill timing is busy with another request")
        on_tokens([7])
        stats = {"prefill_s": 0.01, "decode_s": 0.0, "rounds": 1, "drafts": draft, "cached": 0, "min_rows": 1,
                 "received_at": received_at, "queued_at": received_at + 0.001, "admitted_at": received_at + 0.002,
                 "first_token_at": received_at + 0.5, "ttft_s": 0.5, "profiler_rc": self.profiler_rc if profile else None,
                 "timing": {"spans": 3} if timing else None}
        return {**stats, **self.stats_override}


@pytest.fixture
def served(monkeypatch, tmp_path):
    monkeypatch.delenv(pt.ENV, raising=False); monkeypatch.delenv(pt.ENV_NSYS, raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    app = cuda_server.App.__new__(cuda_server.App)
    # the attributes run()/prepare()/the handler read; keep in step with App.__init__ (read it first)
    app.engine = FakeEngine(); app.tok = FakeTok(); app.served = "f1"; app.default_thinking = False
    app.sampling = {"temperature": 0.0}; app.max_tokens = 64; app.context_window = 4096
    app.native_context_window = 4096; app.template = None; app.lock = threading.Lock()
    # App.__init__ sets: engine, served, tok, template, default_thinking, sampling, max_tokens, native_context_window,
    # context_window, lock — re-read __init__ before editing and add any attribute it gains
    server = ThreadingHTTPServer(("127.0.0.1", 0), cuda_server.make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield app, server
    server.shutdown()


def post(server, path, payload):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    conn.request("POST", path, body=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read().decode()
    ctype = resp.getheader("Content-Type", "")
    return resp.status, ctype, raw


def final_stats(raw_sse):
    for line in raw_sse.splitlines():
        if line.startswith("data:") and '"tensorfold"' in line or '"octojet"' in line:
            d = json.loads(line[5:])
            return d.get("tensorfold") or d.get("octojet")
    return None


def test_fields_need_env_before_headers(served):
    app, server = served
    for field, env in (("timing", pt.ENV), ("histogram", pt.ENV), ("profile", pt.ENV_NSYS)):
        for stream in (False, True):
            status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt": "hi there", "max_tokens": 1,
                                                                   field: True, "stream": stream})
            assert status == 400 and "application/json" in ctype, (field, stream, status, ctype)
            assert env in json.loads(raw)["error"]["message"]


def test_fields_are_forwarded_when_enabled(served, monkeypatch):
    app, server = served
    monkeypatch.setenv(pt.ENV, "1"); monkeypatch.setenv(pt.ENV_NSYS, "1")
    app.engine.profiler_rc = [0, 0]
    status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt": "hi there", "max_tokens": 1,
                                                           "timing": True, "profile": True, "histogram": True})
    assert status == 200
    body = json.loads(raw)
    assert app.engine.seen["timing"] and app.engine.seen["profile"] and app.engine.seen["histogram"]
    assert app.engine.seen["received_at"] > 0
    stats = body["octojet"]
    assert stats["timing"] == {"spans": 3} and stats["profiler_rc"] == [0, 0]
    for k in ("received_at", "queued_at", "admitted_at", "first_token_at", "ttft_s", "prompt_sha"):
        assert k in stats


def test_non_boolean_fields_are_400(served, monkeypatch):
    app, server = served
    monkeypatch.setenv(pt.ENV, "1")
    status, _, raw = post(server, "/v1/completions", {"model": "f1", "prompt": "hi", "max_tokens": 1, "timing": "yes"})
    assert status == 400 and "boolean" in json.loads(raw)["error"]["message"]


def test_prompt_ids_validation_and_sha(served):
    app, server = served
    ids = [11, 22, 33, 44]
    status, _, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": ids, "max_tokens": 1})
    assert status == 200
    body = json.loads(raw)
    assert body["usage"]["prompt_tokens"] == 4 and app.engine.seen["prompt"] == ids
    assert body["octojet"]["prompt_sha"] == sha(ids)
    for bad in ([], [1, "x"], [1.5, 2], [True, 2], "1,2"):
        status, _, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": bad, "max_tokens": 1})
        assert status == 400, bad
    status, _, raw = post(server, "/v1/completions", {"model": "f1", "prompt": "hi", "prompt_ids": ids, "max_tokens": 1})
    assert status == 400 and "one of" in json.loads(raw)["error"]["message"]
    status, _, raw = post(server, "/v1/chat/completions", {"model": "f1", "prompt_ids": ids, "max_tokens": 1,
                                                            "messages": [{"role": "user", "content": "x"}]})
    assert status == 400                                                       # completions only


def test_text_prompt_gets_prompt_sha(served):
    app, server = served
    status, _, raw = post(server, "/v1/completions", {"model": "f1", "prompt": "hi there", "max_tokens": 1})
    assert status == 200
    assert json.loads(raw)["octojet"]["prompt_sha"] == sha(app.engine.seen["prompt"])


def test_armed_recorder_is_a_400_before_headers(served, monkeypatch):
    """The primary busy path: a recorder already armed refuses timing/histogram in validation, before any header, on
    both endpoints and both modes (the same text as the scheduler's TimingBusy)."""
    app, server = served
    monkeypatch.setenv(pt.ENV, "1")
    monkeypatch.setattr(pt.TIMER, "armed", True)
    for field in ("timing", "histogram"):
        for stream in (True, False):
            status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt": "hi", "max_tokens": 1,
                                                                   field: True, "stream": stream})
            assert status == 400 and "application/json" in ctype, (field, stream, status, ctype)
            assert json.loads(raw)["error"]["message"] == "prefill timing is busy with another request"
            status, ctype, raw = post(server, "/v1/chat/completions", {"model": "f1", "max_tokens": 1, field: True,
                                                                        "stream": stream,
                                                                        "messages": [{"role": "user", "content": "x"}]})
            assert status == 400 and "application/json" in ctype, (field, stream, "chat", status, ctype)
            assert "busy" in json.loads(raw)["error"]["message"]
    app.engine.seen = {}
    status, _, _ = post(server, "/v1/completions", {"model": "f1", "prompt": "hi", "max_tokens": 1, "stream": True})
    assert status == 200 and app.engine.seen                                   # untimed requests are unaffected


def test_busy_during_admission_race_fallback(served, monkeypatch):
    """The race fallback: the recorder is unarmed at validation (the pre-check passes) and only the scheduler finds it
    busy at admission. Non-streaming is still a 400; a stream has sent its headers, so the refusal is an SSE error
    event followed by [DONE]."""
    app, server = served
    monkeypatch.setenv(pt.ENV, "1")
    monkeypatch.setattr(pt.TIMER, "armed", False)
    app.engine.busy = True
    for stream in (False, True):
        status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt": "hi", "max_tokens": 1,
                                                               "timing": True, "stream": stream})
        if stream:
            assert status == 200 and "text/event-stream" in ctype
            assert "busy" in raw and raw.rstrip().endswith("data: [DONE]")
        else:
            assert status == 400 and "busy" in json.loads(raw)["error"]["message"]


def test_unsupported_engine_is_400(served, monkeypatch):
    app, server = served
    monkeypatch.setenv(pt.ENV, "1")
    class Old:
        eos = [2]
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            on_tokens([7]); return {}
    app.engine = Old()
    status, _, raw = post(server, "/v1/completions", {"model": "f1", "prompt": "hi", "max_tokens": 1, "timing": True})
    assert status == 400 and "not supported" in json.loads(raw)["error"]["message"]
    status, _, raw = post(server, "/v1/completions", {"model": "f1", "prompt": "hi", "max_tokens": 1})
    assert status == 200                                                       # plain requests still work


def test_prompt_ids_are_range_checked_before_headers(served):
    app, server = served
    for stream in (False, True):
        for bad in ([2147483648], [6000], [-1], [5, -3]):
            status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": bad, "max_tokens": 1,
                                                                   "stream": stream})
            assert status == 400 and "application/json" in ctype, (bad, stream)
            assert "prompt_ids" in json.loads(raw)["error"]["message"]
    status, _, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": [0, 5999], "max_tokens": 1})
    assert status == 200 and app.engine.seen["prompt"] == [0, 5999]


def test_prompt_ids_bound_falls_back_to_the_tokenizer(served):
    app, server = served
    class Plain(FakeEngine):
        vocab_size = None
    class Tok(FakeTok):
        def get_vocab_size(self, with_added_tokens=True): return 100
    app.engine, app.tok = Plain(), Tok()
    status, _, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": [100], "max_tokens": 1})
    assert status == 400 and "100" in json.loads(raw)["error"]["message"]
    status, _, _ = post(server, "/v1/completions", {"model": "f1", "prompt_ids": [99], "max_tokens": 1})
    assert status == 200
