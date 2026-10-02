"""The server passes cached/reuse/reuse_miss through from the first run of a request, and logs every admission's reuse
and busy miss (both endpoints, both response modes, gated continuations included)."""

import http.client
import json

from tensorfold.cuda import server as cuda_server
from tests.test_cuda_tool_choice import Engine as GatedEngine, app_for, ask
from tests.test_prefill_timing_server import final_stats, post, served  # noqa: F401


def test_stats_carry_reuse_and_the_log_line(served, capsys, tmp_path):  # noqa: F811
    app, server = served
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}"}))
    app.template = cuda_server.ChatTemplate(tmp_path)                      # the served fixture has none: the chat leg needs one
    app.engine.stats_override = {"cached": 4, "reuse": "exact", "reuse_miss": None}
    status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": [1, 2, 3, 4], "max_tokens": 1})
    stats = json.loads(raw)["octojet"]
    assert status == 200 and stats["reuse"] == "exact" and stats["cached"] == 4 and stats["reuse_miss"] is None
    assert capsys.readouterr().err.count("[octojet] prefix reuse exact 4/4 tokens") == 1
    status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": [1, 2, 3, 4], "max_tokens": 1, "stream": True})
    assert status == 200 and final_stats(raw)["reuse"] == "exact"
    assert capsys.readouterr().err.count("[octojet] prefix reuse exact 4/4 tokens") == 1
    status, ctype, raw = post(server, "/v1/chat/completions", {"model": "f1", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
    assert status == 200 and json.loads(raw)["octojet"]["reuse"] == "exact"
    err = capsys.readouterr().err
    assert err.count("[octojet] prefix reuse exact 4/") == 1 and "reuse miss" not in err


def test_busy_miss_is_reported_and_logged(served, capsys):  # noqa: F811
    app, server = served
    app.engine.stats_override = {"cached": 0, "reuse": None, "reuse_miss": "busy"}
    status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": [1, 2, 3, 4], "max_tokens": 1})
    stats = json.loads(raw)["octojet"]
    assert status == 200 and stats["reuse_miss"] == "busy" and stats["reuse"] is None
    err = capsys.readouterr().err
    assert err.count("[octojet] prefix reuse miss (busy) 4 tokens") == 1 and "prefix reuse exact" not in err


def test_a_hit_beside_a_busy_miss_logs_both(served, capsys):  # noqa: F811
    app, server = served
    app.engine.stats_override = {"cached": 3, "reuse": "extend", "reuse_miss": "busy"}     # the exact entry was busy, an extend was idle
    status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": [1, 2, 3, 4], "max_tokens": 1})
    err = capsys.readouterr().err
    assert status == 200 and "[octojet] prefix reuse extend 3/4 tokens" in err and "[octojet] prefix reuse miss (busy) 4 tokens" in err


def test_a_failed_admission_yields_no_success_response_and_no_reuse_line(served, capsys):  # noqa: F811
    """The handler does not catch an engine exception: a non-streamed request sees a dropped connection, a streamed
    one a 200 whose body carries nothing. Either way the client gets no stats block and no reuse line is logged."""

    app, server = served

    def boom(*a, **k):
        raise RuntimeError("injected failure")

    app.engine.generate = boom
    try:
        status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": [1, 2, 3, 4], "max_tokens": 1})
    except (http.client.HTTPException, OSError):
        status, raw = None, ""
    assert status != 200 and "octojet" not in raw and "reuse" not in raw and "cached" not in raw
    try:                                                   # streamed: the headers may already be out (200), the body carries no stats
        status, ctype, raw = post(server, "/v1/completions", {"model": "f1", "prompt_ids": [1, 2, 3, 4], "max_tokens": 1, "stream": True})
    except (http.client.HTTPException, OSError):
        raw = ""
    assert "octojet" not in raw and "reuse" not in raw and "cached" not in raw and final_stats(raw) is None
    assert "[octojet] prefix reuse" not in capsys.readouterr().err


class ReuseGatedEngine(GatedEngine):
    """The tool-choice harness's engine, reporting a different reuse per generate call: 4 exact, then 7 fresh."""

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        stats = super().generate(prompt, max_tokens, sampling, on_tokens, draft)
        n = len(self.calls)
        return {**stats, "cached": 4 if n == 1 else 7, "reuse": "exact" if n == 1 else None, "reuse_miss": None}


def test_gated_continuation_reports_the_first_runs_reuse_and_logs_each_admission(tmp_path, capsys):
    engine = ReuseGatedEngine()
    status, body = ask(app_for(tmp_path, engine), draft=True)          # tool_choice "required": two generate calls
    payload = json.loads(body)
    assert status == 200 and len(engine.calls) == 2
    stats = payload["octojet"]
    assert (stats["cached"], stats["reuse"], stats["reuse_miss"]) == (4, "exact", None)      # not 4 + 7, not the second run's
    err = capsys.readouterr().err
    first_len = len(engine.calls[0][0])
    assert err.count("[octojet] prefix reuse") == 1 and f"[octojet] prefix reuse exact 4/{first_len} tokens" in err
