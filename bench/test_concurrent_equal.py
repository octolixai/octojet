import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import concurrent_equal as ce


def _serve(diverge, fail=None, barrier_n=0, ids=True, token_diverge=False, ids_value=None):
    """fail=(prompt_index_text, nth_request_for_it) -> 503. barrier_n: 2nd+ request per prompt waits on a Barrier."""
    barrier = threading.Barrier(barrier_n, timeout=10) if barrier_n else None
    state = {"seen": {}, "inflight": 0, "lock": threading.Lock()}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["temperature"] == 0 and body["stream"] is False
            assert body.get("return_token_ids") is True
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            prompt = body["messages"][0]["content"]
            with state["lock"]:
                n = state["seen"][prompt] = state["seen"].get(prompt, 0) + 1
            if fail and prompt == fail[0] and n == fail[1]:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if barrier and n > 1:
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    self.send_response(500)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            with state["lock"]:
                state["inflight"] += 1
                busy = state["inflight"] > 1
            import time
            time.sleep(0.05)
            text = "reply:" + prompt + ("!DIFF" if diverge and busy else "")
            with state["lock"]:
                state["inflight"] -= 1
            reply = {"choices": [{"message": {"content": text}}]}
            if ids:
                reply["tensorfold"] = {"token_ids": ids_value if ids_value is not None else
                                       [len(prompt), 1, 2] + ([9] if (diverge or token_diverge) and busy else [3])}
            out = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _run(diverge, tmp_path, extra=(), **kw):
    srv = _serve(diverge, **kw)
    try:
        out = tmp_path / "o.json"
        code = ce.main([f"http://127.0.0.1:{srv.server_port}", "m", "--out", str(out), *extra])
    finally:
        srv.shutdown()
    return code, json.loads(out.read_text())


def test_equal(tmp_path, capsys):
    code, res = _run(False, tmp_path)
    assert code == 0 and res["all_equal"] is True
    assert len(res["prompts"]) == 4
    for p in res["prompts"]:
        assert p["concurrent_equal"] and p["concurrent2_equal"] and p["first_diff_index"] == -1
        assert p["solo_len_chars"] > 0
    assert "wall_solo_s" in res and "wall_concurrent_s" in res
    assert json.loads(capsys.readouterr().out)["all_equal"] is True


def test_not_equal(tmp_path):
    code, res = _run(True, tmp_path)
    assert code == 1 and res["all_equal"] is False
    assert any(not p["concurrent_equal"] and p["first_diff_index"] >= 0 for p in res["prompts"])


def test_eight_prompts_all_in_flight(tmp_path):
    code, res = _run(False, tmp_path, ["--prompts", "8"], barrier_n=8)
    assert code == 0 and len(res["prompts"]) == 8 and res["all_equal"]
    assert res["wall_concurrent_s"] < res["wall_solo_s"]


def test_four_prompts_all_in_flight(tmp_path):
    code, res = _run(False, tmp_path, barrier_n=4)
    assert code == 0 and res["wall_concurrent_s"] < res["wall_solo_s"]


def test_503_in_solo_phase(tmp_path):
    code, res = _run(False, tmp_path, fail=(ce.PROMPTS[1], 1))
    assert code == 1 and len(res["prompts"]) == 4
    bad = res["prompts"][1]
    assert "503" in bad["error"] and bad["concurrent_equal"] is False
    assert all("error" not in p for i, p in enumerate(res["prompts"]) if i != 1)


def test_503_in_concurrent_phase(tmp_path):
    code, res = _run(False, tmp_path, fail=(ce.PROMPTS[2], 2))
    assert code == 1 and len(res["prompts"]) == 4
    bad = res["prompts"][2]
    assert "503" in bad["error"] and bad["concurrent_equal"] is False and bad["concurrent2_equal"] is True
    assert res["all_equal"] is False


def test_token_ids_are_compared_and_persisted(tmp_path):
    code, res = _run(False, tmp_path, ids=True)
    assert code == 0 and all(p["compared"] == "tokens" for p in res["prompts"])
    for p in res["prompts"]:
        assert p["solo_ids"] == p["concurrent_ids"] == p["concurrent2_ids"] and len(p["solo_ids"]) == 4
        assert p["solo_text"] == p["concurrent_text"] == p["concurrent2_text"]


def test_same_text_different_tokens_is_not_equal(tmp_path):
    code, res = _run(False, tmp_path, ids=True, token_diverge=True)
    assert code == 1 and res["all_equal"] is False
    assert any(p["compared"] == "tokens" and not p["concurrent_equal"] and p["first_diff_index"] >= 0 for p in res["prompts"])


def test_missing_ids_is_an_error_unless_text_allowed(tmp_path):
    code, res = _run(False, tmp_path, ids=False)
    assert code == 1 and all("error" in p and "token ids" in p["error"] for p in res["prompts"])
    code, res = _run(False, tmp_path, ["--allow-text"], ids=False)
    assert code == 0 and all(p["compared"] == "text" for p in res["prompts"])


def test_malformed_ids_are_the_error_row(tmp_path):
    """A fractional id is not truncated, a boolean is not an id, an empty list is no ids: each is the error row."""
    for bad in ([1.5, 2], [True, False], []):
        code, res = _run(False, tmp_path, ids_value=bad)
        assert code == 1 and res["all_equal"] is False, bad
        assert all("token ids" in p["error"] and p["solo_ids"] is None for p in res["prompts"]), bad
        code, res = _run(False, tmp_path, ["--allow-text"], ids_value=bad)
        assert code == 0 and all(p["compared"] == "text" for p in res["prompts"]), bad
