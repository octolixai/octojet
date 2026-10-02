import hashlib, json, sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prefill_prompt as pp


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class _Tok:
    """One id per character: exact counts are easy to reason about."""

    @classmethod
    def from_file(cls, path):
        return cls()

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return _Enc([ord(c) % 1000 for c in text])


def _fake(monkeypatch):
    mod = types.ModuleType("tokenizers")
    mod.Tokenizer = _Tok
    monkeypatch.setitem(sys.modules, "tokenizers", mod)
    pc = types.ModuleType("prefill_cold")
    pc.corpus = lambda: "abcdefghij\n" * 20
    monkeypatch.setitem(sys.modules, "prefill_cold", pc)


def test_manifest_exact_count_and_sha(tmp_path, monkeypatch, capsys):
    _fake(monkeypatch)
    out = tmp_path / "m.json"
    assert pp.main(["--tokenizer", "t.json", "--tokens", "1000", "--seed", "3", "--tag", "mx", "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    assert m["tokens"] == 1000 and len(m["ids"]) == 1000
    assert m["sha256"] == hashlib.sha256(json.dumps(m["ids"]).encode()).hexdigest()
    assert (m["seed"], m["tag"], m["source"], m["tokenizer"]) == (3, "mx", "prefill_cold.corpus", "t.json")
    prefix = "[mx 1000 seed 3]\n"
    assert m["ids"][:len(prefix)] == [ord(c) % 1000 for c in prefix]
    assert json.loads(capsys.readouterr().out)["sha256"] == m["sha256"]


def test_manifests_share_no_prefix(tmp_path, monkeypatch):
    _fake(monkeypatch)
    ids = []
    for tag, tokens in (("a", 500), ("b", 500), ("a", 900)):
        out = tmp_path / f"{tag}{tokens}.json"
        pp.main(["--tokenizer", "t.json", "--tokens", str(tokens), "--tag", tag, "--out", str(out)])
        ids.append(json.loads(out.read_text())["ids"])
    for i in range(3):
        for j in range(i + 1, 3):
            assert ids[i][:12] != ids[j][:12]


def test_non_positive_tokens_exit_2(tmp_path, monkeypatch, capsys):
    _fake(monkeypatch)
    for n in ("0", "-5"):
        out = tmp_path / "m.json"
        assert pp.main(["--tokenizer", "t.json", "--tokens", n, "--tag", "z", "--out", str(out)]) == 2
        assert not out.exists() and "positive" in capsys.readouterr().err


def _source(tmp_path, monkeypatch, tokens=300):
    _fake(monkeypatch)
    src = tmp_path / "p.json"
    assert pp.main(["--tokenizer", "t.json", "--tokens", str(tokens), "--seed", "1", "--tag", "p", "--out", str(src)]) == 0
    return src, json.loads(src.read_text())


def test_from_take_tail_derives_a_prompt_sharing_exactly_k_tokens(tmp_path, monkeypatch, capsys):
    src, p = _source(tmp_path, monkeypatch)
    out = tmp_path / "pp.json"
    assert pp.main(["--from", str(src), "--take", "200", "--tail", "50", "--seed", "9", "--tag", "pp", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["tokens"] == 250 and len(d["ids"]) == 250 and d["ids"][:200] == p["ids"][:200] and d["ids"][200] != p["ids"][200]
    assert d["shared"] == 200 and d["derived_from"] == p["sha256"] and d["tokenizer"] == "t.json" and d["seed"] == 9
    assert d["sha256"] == hashlib.sha256(json.dumps(d["ids"]).encode()).hexdigest()
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["tokens"] == 250


def test_take_at_a_source_bracket_still_differs_at_the_cut(tmp_path, monkeypatch):
    src, p = _source(tmp_path, monkeypatch)
    k = p["ids"].index(ord("["), 1)                                   # a later "[" in the source (a separator line)
    assert p["ids"][k] == ord("[")
    out = tmp_path / "q.json"
    assert pp.main(["--from", str(src), "--take", str(k), "--tail", "5", "--tag", "q", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["ids"][:k] == p["ids"][:k] and d["ids"][k] != ord("[") and d["shared"] == k and d["tokens"] == k + 5


def test_long_tails_repeat_the_corpus(tmp_path, monkeypatch):
    src, p = _source(tmp_path, monkeypatch)
    out = tmp_path / "long.json"
    assert pp.main(["--from", str(src), "--take", "10", "--tail", "500", "--tag", "long", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["tokens"] == 510
    out2 = tmp_path / "ext.json"
    assert pp.main(["--from", str(src), "--extend", "700", "--tag", "ext", "--out", str(out2)]) == 0
    assert json.loads(out2.read_text())["tokens"] == 1000


def test_from_extend_appends_fresh_tokens(tmp_path, monkeypatch):
    src, p = _source(tmp_path, monkeypatch)
    out = tmp_path / "px.json"
    assert pp.main(["--from", str(src), "--extend", "40", "--seed", "3", "--tag", "px", "--out", str(out)]) == 0
    x = json.loads(out.read_text())
    assert x["tokens"] == 340 and x["ids"][:300] == p["ids"] and x["shared"] == 300 and x["derived_from"] == p["sha256"]


def test_derivation_rejects_bad_arguments(tmp_path, monkeypatch, capsys):
    src, p = _source(tmp_path, monkeypatch)
    q = str(tmp_path / "q.json")
    cases = [(["--from", str(src), "--take", "300", "--tail", "5"], "--take must be in 1..299"),
             (["--from", str(src), "--take", "0", "--tail", "5"], "--take must be in"),
             (["--from", str(src), "--take", "10", "--tail", "0"], "--tail must be positive"),
             (["--from", str(src), "--extend", "0"], "--extend must be positive"),
             (["--from", str(src), "--take", "10"], "either --take K --tail T or --extend T"),
             (["--from", str(src), "--extend", "5", "--take", "3", "--tail", "2"], "--extend excludes"),
             (["--from", str(src), "--tokens", "5"], "exclusive"),
             (["--from", str(tmp_path / "missing.json"), "--extend", "5"], "missing.json"),
             (["--tokenizer", "t.json", "--tokens", "5", "--extend", "5"], "need --from"),
             (["--tokens", "5"], "--tokenizer and --tokens are required")]
    for argv, needle in cases:
        assert pp.main([*argv, "--tag", "q", "--out", q]) == 2, argv
        assert needle in capsys.readouterr().err, (argv, needle)
        assert not Path(q).exists()
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps({"tokens": 3, "ids": [1, 2, 3], "sha256": "0" * 64}))
    assert pp.main(["--from", str(bad), "--extend", "5", "--tag", "q", "--out", q]) == 2
    assert "not a valid manifest" in capsys.readouterr().err
