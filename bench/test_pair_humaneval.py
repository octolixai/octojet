import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pair_humaneval import pair


def _make(tmp_path, name, statuses):
    j = tmp_path / f"{name}.jsonl"
    rows = [{"task": "gsm8k", "id": 0}]
    rows += [{"task": "humaneval", "id": f"HumanEval/{i}", "program": "pass"} for i in range(3)]
    j.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    lines = [f"p{i:03d}.py {s}" for i, s in enumerate(statuses)]
    lines.append(json.dumps({"humaneval_pass": statuses.count("pass"), "humaneval_n": 3, "pass_at_1": 0.0}))
    Path(str(j) + ".humaneval.txt").write_text("\n".join(lines) + "\n")
    return j


def test_pair(tmp_path):
    a = _make(tmp_path, "a", ["pass", "pass", "fail"])
    b = _make(tmp_path, "b", ["fail", "pass", "pass"])
    r = pair(a, b)
    assert (r["pass_a"], r["pass_b"], r["n"]) == (2, 2, 3)
    assert r["only_a"] == ["HumanEval/0"]
    assert r["only_b"] == ["HumanEval/2"]
