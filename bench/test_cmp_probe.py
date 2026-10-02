"""cmp_probe.py queue (F7's measure) and its row in compare_summary.py, against a stand-in for the server."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_queue_sends_repeat_long_short_repeat_and_reports_each_ttft(tmp_path, monkeypatch):
    probe = load("cmp_probe")
    for name, text in (("long", "L" * 50), ("short", "S"), ("repeat", "R")):
        (tmp_path / f"{name}.txt").write_text(text)
    sent = []

    def post_stream(base, body, on_text=None, timeout=3600):
        sent.append(body["prompt"][0])
        ttft = {"L": 60.0, "S": 1.5, "R": 0.2}[body["prompt"][0]]
        stats = {"reuse": "exact", "cached": 1} if body["prompt"] == "R" and len(sent) > 1 else {}
        return 100.0, 100.0 + ttft, 101.0 + ttft, {"prompt_tokens": len(body["prompt"])}, stats, 3

    monkeypatch.setattr(probe, "post_stream", post_stream)
    monkeypatch.setattr(probe.time, "sleep", lambda s: None)
    a = SimpleNamespace(base="http://x", model="f1", prompt=str(tmp_path / "long.txt"), short=str(tmp_path / "short.txt"),
                        repeat=str(tmp_path / "repeat.txt"), delay=5.0, out=str(tmp_path / "q.json"))
    assert probe.cmd_queue(a) == 0
    assert sent[0] == "R" and sorted(sent[1:]) == ["L", "R", "S"]
    res = json.loads((tmp_path / "q.json").read_text())
    assert (res["long_ttft_s"], res["short_ttft_s"], res["repeat_ttft_s"]) == (60.0, 1.5, 0.2)
    assert res["repeat_reuse"] == "exact" and res["delay_s"] == 5.0

    summary = load("compare_summary")
    (tmp_path / "ours-queue.json").write_text(json.dumps(res))
    d = summary.collect(str(tmp_path), "ours")
    assert d["queue: cold short ttft during m128k"] == 1.5 and d["queue: resend ttft during m128k"] == 0.2
    assert d["queue: m128k ttft"] == 60.0


def test_queue_reports_a_failed_request(tmp_path, monkeypatch):
    probe = load("cmp_probe")
    for name in ("long", "short", "repeat"):
        (tmp_path / f"{name}.txt").write_text(name)
    calls = []

    def post_stream(base, body, on_text=None, timeout=3600):
        calls.append(1)
        if body["prompt"] == "short":
            raise RuntimeError("server error")
        return 0.0, 1.0, 2.0, {}, {}, 1

    monkeypatch.setattr(probe, "post_stream", post_stream)
    monkeypatch.setattr(probe.time, "sleep", lambda s: None)
    a = SimpleNamespace(base="b", model="m", prompt=str(tmp_path / "long.txt"), short=str(tmp_path / "short.txt"),
                        repeat=str(tmp_path / "repeat.txt"), delay=1.0, out=str(tmp_path / "q.json"))
    assert probe.cmd_queue(a) == 1
    assert json.loads((tmp_path / "q.json").read_text())["short_error"] == "server error"


if __name__ == "__main__":
    sys.exit(0)
