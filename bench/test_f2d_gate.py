"""The stage-A gate reducer: PASS only when the rows are exactly the stage-A workload and every gating rule of spec
section 8 (stage A) holds; the cold-within-timeout rule is reported, not gating (spec section 8 amendment, 2026-09-30)."""

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import f2d_gate as fg

N = 71_444


def rows(mode, rep, cold_ttft=37.6, hit_ttft=1.2, total=None, decode_s=0.9, sha="s", reuse="exact", n=5):
    out = []
    for i in range(n):
        cold = i == 0
        r = {"stage": "A", "rep_id": rep, "stream": mode == "stream", "step": "cold" if cold else f"exact{i}",
             "server_label": f"A-{mode}-{rep}", "arm": "clean", "draft": True, "client_send_utc": "2026-10-01T10:00:00.000Z",
             "client_complete_utc": "2026-10-01T10:00:40.000Z",
             "ttft_s": cold_ttft if cold else hit_ttft, "reuse": None if cold else reuse, "cached": 0 if cold else N,
             "reply_sha": sha, "prompt_sha_ok": True, "prompt_tokens": N, "max_tokens": 64 if mode == "nostream" else 2,
             "usage": {"prompt_tokens": N, "completion_tokens": 64 if mode == "nostream" else 2},
             "decode_s": decode_s,
             "total_s": ((total if total is not None else (cold_ttft + 0.7 if cold else hit_ttft + 0.7))
                         if mode == "nostream" else None)}
        out.append(r)
    return out


def write(tmp_path, *groups, name="rows.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for g in groups for r in g) + "\n")
    return str(p)


def base():
    return [rows("stream", 1), rows("stream", 2), rows("nostream", 1), rows("nostream", 2)]


def test_pass(tmp_path):
    files = [write(tmp_path, g, name=f"{i}.jsonl") for i, g in enumerate(base())]     # one file per server, as the replay writes them
    res = fg.evaluate(files, caller_timeout=60.0, prompt_tokens=N)
    assert res["verdict"] == "PASS" and res["r_ref"] == 70.0 and res["problems"] == []
    assert all(v["ok"] for v in res["rules"].values()) and res["sources"] == files
    assert fg.main([*files, "--caller-timeout", "60", "--prompt-tokens", str(N), "--out", str(tmp_path / "g.json")]) == 0
    assert json.loads((tmp_path / "g.json").read_text())["verdict"] == "PASS"


def test_each_gating_rule_fails_alone(tmp_path):
    b = base()
    slow = rows("stream", 1, hit_ttft=2.5)
    res = fg.evaluate([write(tmp_path, slow, *b[1:])], 60.0, N)
    assert res["verdict"] == "FAIL" and res["rules"]["exact_ttft"]["ok"] is False and res["rules"]["exact_reuse"]["ok"]
    miss = rows("stream", 1, reuse=None)
    assert fg.evaluate([write(tmp_path, miss, *b[1:])], 60.0, N)["rules"]["exact_reuse"]["ok"] is False
    short = rows("stream", 1); short[2]["cached"] = N - 1
    assert fg.evaluate([write(tmp_path, short, *b[1:])], 60.0, N)["rules"]["exact_reuse"]["ok"] is False
    diff = rows("stream", 1); diff[2]["reply_sha"] = "other"
    assert fg.evaluate([write(tmp_path, diff, *b[1:])], 60.0, N)["rules"]["reply_equal"]["ok"] is False
    spread = rows("stream", 2, cold_ttft=40.0)                                        # (40 - 37.6) / 37.6 = 6.4 % > 5 %
    assert fg.evaluate([write(tmp_path, b[0], spread, *b[2:])], 60.0, N)["rules"]["cold_spread"]["ok"] is False
    near = rows("stream", 2, cold_ttft=39.0)                                          # 3.7 % <= 5 %
    assert fg.evaluate([write(tmp_path, b[0], near, *b[2:])], 60.0, N)["rules"]["cold_spread"]["ok"] is True
    late = rows("nostream", 1); late[3]["total_s"] = 2.3                             # 2.3 > min(1.2 + 64/70 + 0.1, 30) = 2.214
    res = fg.evaluate([write(tmp_path, *b[:2], late, b[3])], 60.0, N)
    assert res["rules"]["latency"]["ok"] is False and res["rules"]["cold_within_timeout"]["ok"] is True
    ok = rows("nostream", 1); ok[3]["total_s"] = 2.2                                  # 2.2 <= 2.214 with the transport allowance
    assert fg.evaluate([write(tmp_path, *b[:2], ok, b[3])], 60.0, N)["rules"]["latency"]["ok"] is True
    tight = fg.evaluate([write(tmp_path, *b)], 4.0, N)                                # bound min(2.214, 2.0) = 2.0; 1.9 <= 2.0 passes
    assert tight["rules"]["latency"]["ok"] is True
    tighter = fg.evaluate([write(tmp_path, *b)], 3.0, N)                              # 0.5 T = 1.5 < 1.9
    assert tighter["rules"]["latency"]["ok"] is False
    assert fg.main([write(tmp_path, slow, *b[1:]), "--caller-timeout", "60", "--prompt-tokens", str(N),
                    "--out", str(tmp_path / "g.json")]) == 1


def test_cold_within_timeout_is_reported_not_gating(tmp_path):
    res = fg.evaluate([write(tmp_path, *base())], 30.0, N)                           # cold total 38.3 s > T = 30 s
    assert res["rules"]["cold_within_timeout"] == {"ok": False, "informational": True,
                                                   "detail": res["rules"]["cold_within_timeout"]["detail"]}
    assert len(res["rules"]["cold_within_timeout"]["detail"]) == 2 and res["verdict"] == "PASS"
    assert fg.evaluate([write(tmp_path, *base())], 70.0, N)["rules"]["cold_within_timeout"]["ok"] is True


def _fails_with(tmp_path, groups, needle, T=60.0, tokens=N):
    res = fg.evaluate([write(tmp_path, *groups)], T, tokens)
    assert res["verdict"] == "FAIL" and res["rules"] == {}, res
    assert any(needle in p for p in res["problems"]), (needle, res["problems"])


def test_invalid_or_incomplete_data_fails_before_any_rule(tmp_path):
    b = base()
    _fails_with(tmp_path, b[:3], "missing step")                                      # a repetition missing
    err = rows("stream", 1); err[3]["error"] = "boom"; err[3]["ttft_s"] = None
    _fails_with(tmp_path, [err, *b[1:]], "error row")
    dup = rows("stream", 1, n=6); dup[5]["step"] = "exact4"
    _fails_with(tmp_path, [dup, *b[1:]], "duplicate step")
    wrong = [rows("stream", 1), rows("stream", 2), rows("nostream", 1), rows("nostream", 2)]
    for g in wrong:
        for r in g:
            r["prompt_tokens"] = r["usage"]["prompt_tokens"] = 70_000                 # a different prompt than the stage-A one
    _fails_with(tmp_path, wrong, "prompt_tokens 70000 != 71444")
    warm = rows("stream", 1); warm[0]["reuse"], warm[0]["cached"] = "exact", N         # a "cold" row that was a hit
    _fails_with(tmp_path, [warm, *b[1:]], "not a cold baseline")
    tokens = rows("nostream", 1); tokens[2]["max_tokens"] = 2                          # a warm step with the wrong budget
    _fails_with(tmp_path, [*b[:2], tokens, b[3]], "max_tokens 2 != 64")
    inf = rows("stream", 1); inf[1]["ttft_s"] = float("inf")
    _fails_with(tmp_path, [inf, *b[1:]], "ttft_s inf is not a finite positive number")
    nan = rows("nostream", 1); nan[0]["decode_s"] = float("nan")
    _fails_with(tmp_path, [*b[:2], nan, b[3]], "decode_s nan is not a finite positive number")
    stage = rows("stream", 1); stage[0]["stage"] = "B"
    _fails_with(tmp_path, [stage, *b[1:]], "stage 'B' is not A")
    sha = rows("stream", 1); sha[2]["prompt_sha_ok"] = False
    _fails_with(tmp_path, [sha, *b[1:]], "prompt_sha_ok")
    label = rows("stream", 1); label[1]["server_label"] = None
    _fails_with(tmp_path, [label, *b[1:]], "server_label missing")
    armed = rows("stream", 1); armed[2]["arm"] = "timing"
    _fails_with(tmp_path, [armed, *b[1:]], "not a clean drafting request")
    serial = rows("nostream", 2); serial[0]["draft"] = False
    _fails_with(tmp_path, [*b[:3], serial], "not a clean drafting request")
    stray = rows("stream", 3)                                                          # a fifth server
    _fails_with(tmp_path, [*b, stray], "unexpected group")
    _fails_with(tmp_path, b, "caller timeout 0.0 must be", T=0.0)
    _fails_with(tmp_path, b, "caller timeout inf must be", T=float("inf"))
    res = fg.evaluate([str(tmp_path / "missing.jsonl")], 60.0, N)
    assert res["verdict"] == "FAIL" and any("missing.jsonl" in p for p in res["problems"])
    (tmp_path / "bad.jsonl").write_text("not json\n")
    res = fg.evaluate([str(tmp_path / "bad.jsonl")], 60.0, N)
    assert res["verdict"] == "FAIL" and any("line 1" in p for p in res["problems"])
