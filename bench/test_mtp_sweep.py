import gzip
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import mtp_sweep as ms

HE = [{"task_id": f"HumanEval/{i}", "prompt": f"def f{i}(x):\n    \"\"\"doc {i}\"\"\"\n"} for i in range(30)]
GSM = [{"question": f"What is {i} + {i}?", "answer": f"#### {2 * i}"} for i in range(30)]


@pytest.fixture
def data(tmp_path, monkeypatch):
    he = tmp_path / "HumanEval.jsonl.gz"
    with gzip.open(he, "wt") as f:
        f.write("\n".join(json.dumps(x) for x in HE))
    gsm = tmp_path / "gsm8k_test.jsonl"
    gsm.write_text("\n".join(json.dumps(x) for x in GSM))
    monkeypatch.setattr(ms, "fetch", lambda url, name: str(tmp_path / name))


def serve(tokens_per_round, seen):
    """A fake streaming server: ids depend on the prompt and seed only (as exact sampling), rounds on the setting."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            n = body["max_tokens"]
            key = body.get("prompt") or body["messages"][0]["content"]
            ids = [hash((key, body.get("seed"))) % 1000 + i for i in range(n)]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(n):
                piece = {"text": "x"} if self.path == "/v1/completions" else {"delta": {"content": "x"}}
                self.wfile.write(f"data: {json.dumps({'choices': [piece]})}\n\n".encode())
            end = {"choices": [], "usage": {"completion_tokens": n},
                   "octojet": {"rounds": max(1, round((n - 1) / tokens_per_round)), "decode_s": 0.5, "token_ids": ids}}
            self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_the_prompt_set(data):
    items = ms.build_prompts(20, 20)
    code = [i for i in items if i["category"] == "code"]
    chat = [i for i in items if i["category"] == "chat"]
    assert len(code) == 20 and all(i["kind"] == "completion" for i in code) and code[0]["prompt"] == HE[0]["prompt"]
    assert len(chat) == 20 and all(i["kind"] == "chat" for i in chat)
    assert sum(i["id"].startswith("prose/") for i in chat) == 10 and chat[0]["id"].startswith("gsm8k/")
    assert chat[0]["prompt"].endswith(ms.GSM_SUFFIX) and len({i["id"] for i in items}) == 40


def test_request_bodies():
    path, b = ms.body_for("f1", {"kind": "chat", "prompt": "hi"}, "greedy", 200, 7)
    assert path == "/v1/chat/completions" and b["temperature"] == 0 and "seed" not in b
    assert b["chat_template_kwargs"] == {"enable_thinking": False} and b["ignore_eos"] and b["return_token_ids"]
    path, b = ms.body_for("f1", {"kind": "completion", "prompt": "def f():"}, "sampled", 200, 7)
    assert path == "/v1/completions" and b["prompt"] == "def f():" and b["seed"] == 7
    assert (b["temperature"], b["top_k"], b["top_p"], b["max_tokens"]) == (1.0, 20, 0.95, 200)


def test_run_and_summary_against_a_fake_server(data, tmp_path, capsys):
    seen = []
    rows = {}
    for label, conf, tpr in (("c0.3", "0.3", 3.0), ("c0.7", "0.7", 2.0)):
        srv = serve(tpr, seen)
        out = tmp_path / f"{label}.jsonl"
        assert ms.main(["run", f"http://127.0.0.1:{srv.server_address[1]}", "f1", "--label", label, "--out", str(out),
                        "--tokens", "21", "--code", "4", "--chat", "4", "--setting", f"confidence={conf}"]) == 0
        srv.shutdown()
        rows[label] = [json.loads(line) for line in out.read_text().splitlines()]
    r = rows["c0.3"]
    assert len(r) == 16 and {x["sampling"] for x in r} == {"greedy", "sampled"}            # warm-up excluded
    assert all(x["completion_tokens"] == 21 and x["decode_tps"] > 0 for x in r)
    assert r[0]["setting_confidence"] == 0.3 and r[0]["tokens_per_round"] == pytest.approx(20 / 7)
    assert r[0]["accepted_per_round"] == pytest.approx(20 / 7 - 1) and r[0]["server_decode_tps"] == 40.0
    assert seen[0][1]["max_tokens"] <= 21 and len(seen) == 34                               # 2 x (warm-up + 16)
    s = ms.summarise(ms.load([tmp_path / "c0.3.jsonl", tmp_path / "c0.7.jsonl"]))
    assert s["default"] == "c0.3" and s["mismatches"] == []
    assert set(s["settings"]["c0.3"]["cells"]) == {"code/greedy", "code/sampled", "chat/greedy", "chat/sampled"}
    assert s["settings"]["c0.3"]["cells"]["code/greedy"]["n"] == 4
    text = ms.format_summary(s)
    assert "geomean" in text and "recommendation:" in text and "same replies" in text


def rows_for(label, conf, meds):
    out = []
    for (cat, samp), m in meds.items():
        for d in (-1.0, 0.0, 1.0):
            out.append({"label": label, "setting_confidence": conf, "id": f"{cat}{d}", "category": cat,
                        "sampling": samp, "decode_tps": m + d, "accepted_per_round": 2.0, "token_sha": f"{cat}{samp}{d}"})
    return out


CELLS = [("code", "greedy"), ("code", "sampled"), ("chat", "greedy"), ("chat", "sampled")]


def test_the_rule_picks_the_largest_geometric_mean_and_flags_a_losing_cell():
    rows = (rows_for("c0.3", 0.3, dict(zip(CELLS, (54, 64, 88, 62))))
            + rows_for("c0.7", 0.7, dict(zip(CELLS, (64, 70, 84, 61)))))
    s = ms.summarise(rows, default="c0.3")
    assert s["recommendation"] == "c0.7"
    assert s["recommendation_losses"] == ["chat/greedy 84.0 vs 88.0"]
    assert s["settings"]["c0.3"]["cells"]["code/greedy"]["q1"] == 53.5


def test_the_default_wins_within_one_percent_then_the_higher_confidence():
    base = dict(zip(CELLS, (60, 60, 60, 60)))
    rows = rows_for("c0.3", 0.3, base) + rows_for("c0.5", 0.5, dict(zip(CELLS, (60.3, 60.3, 60.3, 60.3))))
    assert ms.summarise(rows, default="c0.3")["recommendation"] == "c0.3"
    rows = rows_for("c0.2", 0.2, base) + rows_for("c0.5", 0.5, base) + rows_for("c0.3", 0.3, dict(zip(CELLS, (50,) * 4)))
    assert ms.summarise(rows, default="c0.3")["recommendation"] == "c0.5"


def test_a_token_mismatch_is_named_and_fails_the_summary(tmp_path):
    a = rows_for("c0.3", 0.3, dict(zip(CELLS, (60,) * 4)))
    b = rows_for("c0.7", 0.7, dict(zip(CELLS, (60,) * 4)))
    b[0]["token_sha"] = "different"
    s = ms.summarise(a + b)
    assert len(s["mismatches"]) == 1 and "c0.3" in s["mismatches"][0] and "c0.7" in s["mismatches"][0]
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in a + b))
    assert ms.main(["summary", str(p)]) == 1


def test_dry_run_sends_nothing(data, capsys):
    assert ms.main(["run", "http://127.0.0.1:9", "f1", "--label", "x", "--out", "/nonexistent/x", "--dry-run"]) == 0
    assert "40 prompts (20 code, 20 chat) x 2 samplings = 80 requests of 200 tokens" in capsys.readouterr().out
