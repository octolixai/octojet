import importlib.util
import io
import json
import sys
import urllib.request
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "tools" / "bench_openai.py"


def _load():
    spec = importlib.util.spec_from_file_location("bench_openai_under_test", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _run(monkeypatch, tmp_path, extra):
    bodies = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        chunks = [{"choices": [{"delta": {"content": "x"}}]},
                  {"choices": [{"text": "y"}], "usage": {"completion_tokens": 2}}]
        return _Resp("".join("data: " + json.dumps(c) + "\n\n" for c in chunks).encode() + b"data: [DONE]\n\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    mod = _load()
    monkeypatch.setattr(sys, "argv", ["bench_openai.py", "http://x", "m", "--reps", "1", "--temperatures", "0",
                                      "--allow-missing-ids", "--output", str(tmp_path / "o.json")] + extra)
    mod.main()
    return bodies


def test_no_draft_only_with_flag(monkeypatch, tmp_path):
    off = _run(monkeypatch, tmp_path, [])
    assert off and all("draft" not in b for b in off)
    on = _run(monkeypatch, tmp_path, ["--no-draft"])
    assert on and all(b.get("draft") is False for b in on)


def _run_ids(monkeypatch, tmp_path, extra, ids_for, reps="2"):
    """ids_for(body) -> the token ids the fake server reports for that request (None: no stats block)."""
    bodies = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        bodies.append(body)
        ids = ids_for(body)
        chunks = [{"choices": [{"delta": {"content": "x"}}]},
                  {"choices": [{"text": "y"}], "usage": {"completion_tokens": 2}}]
        if ids is not None:
            chunks.append({"choices": [], "tensorfold": {"token_ids": ids}})
        return _Resp("".join("data: " + json.dumps(c) + "\n\n" for c in chunks).encode() + b"data: [DONE]\n\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    mod = _load()
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["bench_openai.py", "http://x", "m", "--reps", reps, "--temperatures", "0",
                                      "--output", str(out)] + extra)
    code = mod.main()
    rows = json.loads(out.read_text()) if out.exists() else None
    return code, bodies, rows


def test_requests_ask_for_token_ids_and_rows_record_every_rep(monkeypatch, tmp_path):
    code, bodies, rows = _run_ids(monkeypatch, tmp_path, [], lambda b: [1, 2, 3])
    assert code == 0 and all(b.get("return_token_ids") is True for b in bodies)
    assert all(r["token_ids_all"] == [[1, 2, 3], [1, 2, 3]] and len(r["token_sha_all"]) == 2
               and all(len(s) == 64 for s in r["token_sha_all"]) for r in rows)


def test_missing_ids_is_an_error_unless_allowed(monkeypatch, tmp_path):
    code, _, _ = _run_ids(monkeypatch, tmp_path, [], lambda b: None)
    assert code == 2
    code, _, rows = _run_ids(monkeypatch, tmp_path, ["--allow-missing-ids"], lambda b: None)
    assert code == 0 and all(r["token_ids_all"] == [None, None] for r in rows)


def test_expect_equal_runs_serial_twin_per_rep(monkeypatch, tmp_path):
    code, bodies, rows = _run_ids(monkeypatch, tmp_path, ["--expect-equal"], lambda b: [b["seed"] % 1000, 1])
    assert code == 0 and all(r["draft_equal_all"] == [True, True] for r in rows)
    per_prompt = 1 + 2 + 2                                            # warm-up, two drafted reps, two serial twins
    assert len(bodies) == per_prompt * len(rows)
    for k in range(len(rows)):
        chunk = bodies[k * per_prompt:(k + 1) * per_prompt][1:]       # drop the warm-up
        drafted, serial = chunk[:2], chunk[2:]
        assert all("draft" not in b for b in drafted) and all(b.get("draft") is False for b in serial)
        assert [b["seed"] for b in drafted] == [b["seed"] for b in serial] == [1234, 1235]


def test_warmup_without_ids_is_2(monkeypatch, tmp_path):
    n = {"i": 0}

    def ids_for(body):
        n["i"] += 1
        return None if n["i"] == 1 else [1, 2]

    code, _, _ = _run_ids(monkeypatch, tmp_path, [], ids_for)
    assert code == 2


def test_expect_equal_mismatch_exits_1(monkeypatch, tmp_path):
    code, _, rows = _run_ids(monkeypatch, tmp_path, ["--compare-draft"],
                             lambda b: [5, 7] if b.get("draft") is False else [5, 6])
    assert code == 1 and all(r["draft_equal_all"] == [False, False] for r in rows)


def test_expect_equal_conflicts_with_no_draft_and_allow_missing(monkeypatch, tmp_path, capsys):
    import pytest
    for flag, text in (("--no-draft", "drop --no-draft"), ("--allow-missing-ids", "drop --allow-missing-ids")):
        with pytest.raises(SystemExit) as e:
            _run_ids(monkeypatch, tmp_path, ["--expect-equal", flag], lambda b: [1])
        assert e.value.code == 2 and text in capsys.readouterr().err


def test_expect_equal_with_missing_serial_ids_is_2(monkeypatch, tmp_path):
    code, _, _ = _run_ids(monkeypatch, tmp_path, ["--expect-equal"],
                          lambda b: None if b.get("draft") is False else [5, 6])
    assert code == 2


def test_malformed_ids_count_as_missing(monkeypatch, tmp_path):
    """Only a non-empty list of real ints is an id list: fractions are not truncated, booleans are not ids."""
    for bad in ([1.5, 2], [True, False], [], ["1", "2"]):
        code, _, _ = _run_ids(monkeypatch, tmp_path, [], lambda b, bad=bad: bad)
        assert code == 2, bad
        code, _, rows = _run_ids(monkeypatch, tmp_path, ["--allow-missing-ids"], lambda b, bad=bad: bad)
        assert code == 0 and all(r["token_ids_all"] == [None, None] for r in rows), bad
    assert _run_ids(monkeypatch, tmp_path, [], lambda b: [1, 0])[0] == 0         # real ints still pass
