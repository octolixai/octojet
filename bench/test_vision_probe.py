"""bench/vision_probe.py without a GPU: the request plan, the SSE reader and the verdicts, against a stub server."""

import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import vision_probe as vp  # noqa: E402

pytest.importorskip("PIL")


def media(tmp_path):
    from PIL import Image

    for name in ("robots-sim-observation.png", "towel-crumpled.jpg", "towel-flat.jpg", "towel-crumpled-1280.jpg",
                 "towel-flat-1280.jpg"):
        Image.new("RGB", (8, 8), "gray").save(tmp_path / name)
    return tmp_path


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        # the color is encoded in the image bytes: tell the two generated squares apart by their data URL
        urls = [p["image_url"]["url"] for m in body["messages"] for p in m["content"] if p.get("type") == "image_url"]
        word = "Red." if urls and urls[0] == vp.png("red") else ("Blue." if urls and urls[0] == vp.png("blue") else "Right")
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': 'thinking'}}]})}\n\n".encode())
        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': word}}]})}\n\n".encode())
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 1}, 'octojet': {'reuse': None}})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def test_probe_end_to_end_against_a_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(vp, "moving_square_mp4", lambda: "data:video/mp4;base64,AA")
    monkeypatch.setattr(vp.time, "sleep", lambda s: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (tmp_path / "m").mkdir()
        d = media(tmp_path / "m")
        assert vp.main([f"http://127.0.0.1:{server.server_port}", "f1", "--media", str(d), "--dry-run"]) == 0
        rc = vp.main([f"http://127.0.0.1:{server.server_port}", "f1", "--media", str(d), "--out", str(tmp_path / "p")])
    finally:
        server.shutdown()
    rows = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    by = {r["case"]: r for r in rows}
    assert by["red-square"]["pass"] and by["isolation-red"]["pass"] and by["isolation-blue"]["pass"]
    assert by["isolation"]["pass"] and by["video-direction"]["pass"] and rc == 0
    assert {"towel-plan-original", "towel-plan-1280", "towel-describe-towel-flat-1280", "robots", "concurrent"} <= set(by)
    assert by["robots"]["reasoning"] == "thinking" and by["robots"]["ttft_s"] is not None
    md = (tmp_path / "p.md").read_text()
    assert "## towel-plan-original" in md and "thinking" in md


def test_missing_media_is_refused(tmp_path):
    assert vp.main(["http://x", "f1", "--media", str(tmp_path), "--out", str(tmp_path / "p")]) == 2
