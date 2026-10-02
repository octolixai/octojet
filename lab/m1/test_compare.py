import json

import compare


def recs(tf, fp4, checks=None):
    out = [{"bench": "tf", "rows": r, "ms": ms} for r, ms in tf.items()]
    out += [{"bench": "fp4", "rows": r, "ms": ms} for r, ms in fp4.items()]
    out += checks or []
    return out


def test_load_skips_non_json(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('noise\n{"bench":"tf","rows":1,"ms":2.0}\n\n{"check":"layout","ok":true}\n')
    assert compare.load([str(p)]) == [{"bench": "tf", "rows": 1, "ms": 2.0}, {"check": "layout", "ok": True}]


def test_table_has_speedup():
    t = compare.table(recs({512: 3.0}, {512: 1.5}))
    assert "| 512 | 3.000 | 1.500 | 2.00x |" in t


def test_table_missing_side_shows_dash():
    t = compare.table(recs({1: 0.1}, {}))
    assert "| 1 | 0.100 | - | - |" in t


def test_verdict_pass():
    v = compare.verdict(recs({512: 3.0, 2048: 9.0}, {512: 1.5, 2048: 5.0}))
    assert v["pass"] is True
    assert v["prompt_speedup"] == 1.8


def test_verdict_uses_prompt_cells_only():
    # decode cells are 10x faster on FP4 but prompt cells are not: still a fail
    v = compare.verdict(recs({1: 1.0, 512: 3.0}, {1: 0.1, 512: 2.5}))
    assert v["pass"] is False
    assert v["prompt_speedup"] == 1.2


def test_verdict_failed_check_blocks_pass():
    v = compare.verdict(recs({512: 3.0}, {512: 1.0}, [{"check": "layout", "ok": False}]))
    assert v["checks_ok"] is False
    assert v["pass"] is False
    assert "layout" in v["reason"]


def test_verdict_no_prompt_cells():
    v = compare.verdict(recs({1: 1.0}, {1: 0.5}))
    assert v["pass"] is False
    assert v["prompt_speedup"] is None


def test_table_reports_tflops_from_fp4_flops():
    r = [{"bench": "tf", "rows": 512, "ms": 4.0}, {"bench": "fp4", "rows": 512, "ms": 2.0, "flops": 4e12}]
    t = compare.table(r)
    assert "| rows | TF ms | FP4 ms | speed-up | TF TFLOP/s | FP4 TFLOP/s |" in t
    assert "| 512 | 4.000 | 2.000 | 2.00x | 1000.0 | 2000.0 |" in t


def test_parts_table():
    r = [{"bench": "fp4_parts", "rows": 512, "quant": 0.1, "up": 0.5, "swiglu": 0.05, "down": 0.25}]
    t = compare.parts(r)
    assert "| rows | quant | up | swiglu | down |" in t
    assert "| 512 | 0.100 | 0.500 | 0.050 | 0.250 |" in t


def test_parts_empty():
    assert compare.parts([]) == ""


def test_verdict_reports_speedup_against_tf_min():
    # TF is bimodal: median 3.0 but its fastest run 1.5; FP4 1.5 -> 2.0x by median, 1.0x against TF's best
    r = [{"bench": "tf", "rows": 512, "ms": 3.0, "ms_min": 1.5}, {"bench": "fp4", "rows": 512, "ms": 1.5}]
    v = compare.verdict(r)
    assert v["prompt_speedup"] == 2.0 and v["pass"] is True
    assert v["prompt_speedup_vs_tf_min"] == 1.0
    assert v["pass_vs_tf_min"] is False


def test_verdict_tf_min_missing():
    v = compare.verdict(recs({512: 3.0}, {512: 1.5}))
    assert v["prompt_speedup_vs_tf_min"] is None and v["pass_vs_tf_min"] is False
