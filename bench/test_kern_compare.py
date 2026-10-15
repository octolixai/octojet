import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import kern_compare as kc  # noqa: E402

HEAD = "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"


def _csv(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("NOTICE: Existing SQLite export found\n\nProcessing [x] with [y]...\n" + HEAD +
                 "".join(f'1.0,{ns},{n},1,1,1,1,0,"{k}"\n' for k, ns, n in rows))
    return str(p)


def test_short_names():
    assert kc.short_name("void <unnamed>::prefill_kernel<(int)32, (int)128, (bool)0>(const __nv_bfloat16 *, int)") \
        == "<unnamed>::prefill_kernel<(int)32, (int)128, (bool)0>"
    assert kc.short_name("_hc_mix") == "_hc_mix"
    assert kc.short_name("<unnamed>::plan_rank(const int *, int, int, int *, int *)") == "<unnamed>::plan_rank"


def test_compare_and_render(tmp_path, capsys):
    a = _csv(tmp_path, "a.csv", [("_hc_mix", 2_000_000, 10), ("void k<(int)1>(int)", 5_000_000, 3),
                                 ("void k<(int)1>(float)", 1_000_000, 1)])
    b = _csv(tmp_path, "b.csv", [("void k<(int)1>(int)", 4_000_000, 3), ("upmix_kernel", 500_000, 10)])
    out = tmp_path / "r.json"
    assert kc.main([f"main={a}", f"f12={b}", "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["totals_ms"] == {"main": 8.0, "f12": 4.5}
    row = next(r for r in res["table"] if r["kernel"] == "k<(int)1>")
    assert row["main"] == {"ms": 6.0, "n": 4} and row["f12"] == {"ms": 4.0, "n": 3}
    d = res["diffs"]["f12 - main"]
    assert d["more"][0][0] == "upmix_kernel" and [n for n, _ in d["less"]] == ["k<(int)1>", "_hc_mix"]
    text = capsys.readouterr().out
    assert "f12 - main: kernels with the most LESS time" in text and "upmix_kernel" in text


def test_a_file_without_rows_is_refused(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("nothing here\n")
    try:
        kc.read_kern(str(p))
    except ValueError as e:
        assert "Time (%)" in str(e)
    else:
        raise AssertionError("expected ValueError")
