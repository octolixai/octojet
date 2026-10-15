"""bench/needle.py without a server: placement, sizing against the refusal receipt, answer matching, the run loop
against a fake chat endpoint (one thread, real HTTP), the dry-run and the report."""

import json, random, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import needle  # noqa: E402

FILLER = "".join(f"def f{i}(x):\n    return x + {i}  # line {i} <|im_end|>\n" for i in range(4000))


def test_filler_drops_special_tokens_and_wraps(tmp_path):
    p = tmp_path / "f.txt"; p.write_text(FILLER)
    text = needle.load_filler([str(p)])
    assert "<|" not in text and "return x + 7" in text
    w = needle.window(text, len(text) - 10, 50)
    assert len(w) > 50 and "(the document continues)" in w
    with pytest.raises(ValueError, match="too short"):
        (tmp_path / "s.txt").write_text("tiny")
        needle.load_filler([str(tmp_path / "s.txt")])


@pytest.mark.parametrize("depth", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_the_fact_sits_at_its_depth_on_its_own_line(depth):
    n = needle.make_needle(random.Random(1))
    body, at = needle.place(FILLER, depth, n["fact"])
    assert body.count(n["fact"]) == 1 and abs(at - depth * len(FILLER)) < 200
    assert body[body.index(n["fact"]) - 1] == "\n"
    text = needle.build(FILLER, 20000, depth, n, "run-0", 123)
    assert text.startswith("Document run-0.") and text.endswith(n["question"]) and n["fact"] in text


def test_needles_are_random_per_run_and_answers_match_loosely():
    a, b = needle.make_needle(random.Random(1)), needle.make_needle(random.Random(2))
    assert a != b and a["answer"] in a["fact"] and needle.make_needle(random.Random(1)) == a
    assert needle.found("The passcode is **7391-helix**.", "7391-HELIX")
    assert needle.found("7391 HELIX", "7391-HELIX") and not needle.found("7391-ORBIT", "7391-HELIX")


def test_fit_converges_on_the_counted_tokens():
    counts = []

    def counter(base, model, text):            # a tokenizer of 4 characters a token, plus 30 template tokens
        counts.append(len(text) // 4 + 30)
        return counts[-1]

    text, n = needle.fit("b", "m", lambda chars: "x" * chars, 100000, counter=counter)
    assert abs(n - 100000) <= 500 and len(counts) <= 4
    text, n = needle.fit("b", "m", lambda chars: "x" * chars, 100000, counter=lambda *a: None)
    assert n is None and len(text) == int(100000 * needle.CHARS_PER_TOKEN)


class Fake(BaseHTTPRequestHandler):
    """Octojet's chat surface as needle.py uses it: the refusal receipt, a 32-token window, SSE replies."""

    window = 1_000_000
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][0]["content"]
        tokens = len(text) // 4 + 20
        type(self).seen.append((tokens, body["max_tokens"], body["chat_template_kwargs"]))
        if tokens + body["max_tokens"] > self.window:
            msg = (f"the rendered prompt has {tokens} tokens and requests {body['max_tokens']} reply tokens, "
                   "exceeding the server's window")
            raw = json.dumps({"error": {"message": msg}}).encode()
            self.send_response(400); self.send_header("Content-Length", str(len(raw))); self.end_headers()
            self.wfile.write(raw); return
        answer = text.split("is ")[1].split(".")[0] if "secret passcode for the" in text else "?"
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for piece in ("The passcode: ", answer):
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
        usage = {"prompt_tokens": tokens, "completion_tokens": 5}
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\ndata: [DONE]\n\n".encode())


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    Fake.seen = []
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_run_sizes_each_case_and_records_the_answer(server, tmp_path):
    filler = tmp_path / "f.txt"; filler.write_text(FILLER * 3)
    out = tmp_path / "rows.jsonl"
    Fake.window = 60000
    rc = needle.main(["run", server, "f1", "--out", str(out), "--sizes", "20000,50000,90000", "--depths", "0.1,0.9",
                      "--filler", str(filler), "--seed", "3", "--label", "yarn"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rc == 0 and len(rows) == 6
    for r in rows[:4]:
        assert r["found"] and abs(r["prompt_tokens"] - r["target_tokens"]) <= 0.005 * r["target_tokens"] + 1
        assert r["ttft_s"] is not None and r["error"] is None
    assert all(r["error"] and "rendered prompt has" in r["error"] and not r["found"] for r in rows[4:])   # past the window
    assert len({r["answer"] for r in rows}) > 1
    assert all(kw == {"enable_thinking": False} for _, _, kw in Fake.seen)
    report = needle.report([str(out)])
    assert "yarn 20000: 2/2 found" in report and "yarn 90000: 0/2 found" in report and "refused" in report


def test_dry_run_sends_nothing(tmp_path, capsys):
    filler = tmp_path / "f.txt"; filler.write_text(FILLER)
    assert needle.main(["run", "http://127.0.0.1:9", "f1", "--out", str(tmp_path / "r.jsonl"), "--filler",
                        str(filler), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["cases"] == 9 and plan["sizes"] == [128000, 256000, 450000] and not (tmp_path / "r.jsonl").exists()
    with pytest.raises(SystemExit):
        needle.main(["run", "b", "m", "--out", "x", "--depths", "1.5"])


def test_the_default_filler_is_the_stdlib_corpus():
    text = needle.load_filler(None)
    assert len(text) > 450000 * 4 and "def " in text
