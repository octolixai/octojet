"""bench/f4_live.py without a GPU: the gap window arithmetic, and both checks end to end against a stub server whose
chat stream stalls while a completion "prefills" (gap) and whose second identical completion reports a copied exact
hit (twins)."""

import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import f4_live  # noqa: E402

STATE = {"prefilling": threading.Event(), "admitted": 0, "lock": threading.Lock()}


class Stub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _sse(self, chunk):
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        try:
            if self.path == "/v1/chat/completions":            # stream A: a token every 5 ms, none while B prefills
                for i in range(4000):
                    while STATE["prefilling"].is_set():
                        time.sleep(0.005)
                    self._sse({"choices": [{"delta": {"content": f"{i}\n"}}]})
                    time.sleep(0.005)
                return
            with STATE["lock"]:
                STATE["admitted"] += 1
                n = STATE["admitted"]
            admitted = time.perf_counter()
            if len(body["prompt_ids"]) > 100:                # gap: B's prefill stalls A for 0.3 s
                STATE["prefilling"].set(); time.sleep(0.3); STATE["prefilling"].clear()
            else:
                time.sleep(0.05 * n)
            stats = {"reuse": "exact" if n == 2 else None, "cached": len(body["prompt_ids"]) if n == 2 else 0,
                     "queued_at": admitted, "admitted_at": admitted, "first_token_at": time.perf_counter(),
                     "prefill_s": 0.3, **({"reuse_copy": True} if n == 2 else {})}
            self._sse({"choices": [{"text": "x"}]})
            self._sse({"choices": [], "usage": {"prompt_tokens": 1}, "octojet": stats})
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_window_gap_counts_the_gap_into_and_out_of_the_window():
    arrivals = [0.0, 0.1, 0.2, 1.5, 1.6]
    assert f4_live.window_gap(arrivals, 0.25, 1.0) == (1.3, 0)          # no arrival inside: before -> after
    worst, inside = f4_live.window_gap(arrivals, 0.15, 1.55)
    assert abs(worst - 1.3) < 1e-9 and inside == 2


def test_gap_and_twins_against_a_stub(tmp_path):
    srv, base = serve()
    try:
        big, small = tmp_path / "big.json", tmp_path / "small.json"
        big.write_text(json.dumps({"ids": list(range(500))}))
        small.write_text(json.dumps({"ids": list(range(20))}))
        assert f4_live.main([base, "f1", "gap", "--manifest", str(big), "--out", str(tmp_path / "gap.json")]) == 0
        gap = json.loads((tmp_path / "gap.json").read_text())
        assert gap["a_max_gap_s"] >= 0.25 and gap["b_prompt_tokens"] == 500 and gap["a_error"] is None
        STATE["admitted"] = 0
        assert f4_live.main([base, "f1", "twins", "--manifest", str(small), "--out", str(tmp_path / "tw.json")]) == 0
        tw = json.loads((tmp_path / "tw.json").read_text())
        assert tw["b_reuse"] == "exact" and tw["b_reuse_copy"] is True and tw["b_cached"] == 20
        assert tw["b_after_a_s"] is not None
    finally:
        srv.shutdown()


def test_bad_manifest_and_dry_run(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert f4_live.main(["http://x", "f1", "gap", "--manifest", str(bad)]) == 2
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"ids": [1, 2, 3]}))
    assert f4_live.main(["http://x", "f1", "twins", "--manifest", str(good), "--dry-run"]) == 0
    assert "3 prompt tokens" in capsys.readouterr().out
