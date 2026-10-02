import json, sqlite3, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import nsys_ranges as nr

MS = 1_000_000
TID = 7


REAL_NAMES = ['Async Copy Engine Active 0 [Cycles Active]', 'Async Copy Engine Active 0 [Throughput %]',
              'Compute Warps in Flight [Avg Warps per Cycle]', 'Compute Warps in Flight [Avg]',
              'Compute Warps in Flight [Throughput %]', 'GPC Clock Frequency [MHz]', 'GR Active [Throughput %]',
              'Pixel Warps in Flight [Avg Warps per Cycle]', 'Pixel Warps in Flight [Avg]',
              'Pixel Warps in Flight [Throughput %]', 'SM Issue [Throughput %]', 'SMs Active [Throughput %]',
              'SYS Clock Frequency [MHz]', 'Sync Copy Engine Active [Cycles Active]',
              'Sync Copy Engine Active [Throughput %]', 'Tensor Active [Throughput %]',
              'Vertex/Tess/Geometry Warps in Flight [Avg Warps per Cycle]', 'Vertex/Tess/Geometry Warps in Flight [Avg]',
              'Vertex/Tess/Geometry Warps in Flight [Throughput %]']


def _build(path, metrics=True, memset=True, admission=True, drop=None, real=False):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)")
    for t in ("KERNEL", "MEMCPY", "MEMSET"):
        if t != "MEMSET" or memset:
            db.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_{t}(start INTEGER, end INTEGER, correlationId INTEGER)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, correlationId INTEGER, globalTid INTEGER)")
    nv = [(10 * MS, 20 * MS, "main:router", TID), (20 * MS, 40 * MS, "main:expert_up", TID)]
    if admission:
        nv.append((0, 100 * MS, "admission", TID))
    db.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)", nv)
    # (corr, launch time, tid); the runtime call starts inside the range, the GPU work may run later
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)", [
        (11 * MS, 11 * MS + 1000, 1, TID), (12 * MS, 12 * MS + 1000, 2, TID), (25 * MS, 25 * MS + 1000, 3, TID),
        (26 * MS, 26 * MS + 1000, 4, TID), (50 * MS, 50 * MS + 1000, 5, TID), (13 * MS, 13 * MS + 1000, 6, TID + 1)])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)", [
        (11 * MS, 14 * MS, 1),                    # router, 3 ms
        (30 * MS, 32 * MS, 3), (31 * MS, 33 * MS, 4),   # expert_up: union 3 ms
        (60 * MS, 65 * MS, 5)])                   # outside block ranges, inside admission
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?)", [(70 * MS, 71 * MS, 99)])
    if memset:
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(?,?,?)", [(15 * MS, 16 * MS, 2)])
    if metrics and real:          # the GB10 metric set: no DRAM bandwidth
        db.execute("CREATE TABLE GPU_METRICS(rawTimestamp INTEGER, timestamp INTEGER, typeId INTEGER, metricId INTEGER, value REAL)")
        db.execute("CREATE TABLE TARGET_INFO_GPU_METRICS(typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER, metricName TEXT)")
        db.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES(?,?,?,?,?)",
                       [(1, 0, "GPU", i, n) for i, n in enumerate(REAL_NAMES)])
        base = {"SMs Active [Throughput %]": 30.0, "Tensor Active [Throughput %]": 10.0, "SM Issue [Throughput %]": 20.0,
                "GR Active [Throughput %]": 90.0, "Compute Warps in Flight [Throughput %]": 5.0}
        rows = []
        for ts, bump in ((12 * MS, 0.0), (13 * MS, 10.0), (31 * MS, 40.0), (62 * MS, 1.0)):
            rows += [(ts, ts, 1, i, base.get(n, 77.0) + bump) for i, n in enumerate(REAL_NAMES)]
        db.executemany("INSERT INTO GPU_METRICS VALUES(?,?,?,?,?)", rows)
    elif metrics:
        db.execute("CREATE TABLE GPU_METRICS(rawTimestamp INTEGER, timestamp INTEGER, typeId INTEGER, metricId INTEGER, value REAL)")
        db.execute("CREATE TABLE TARGET_INFO_GPU_METRICS(typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER, metricName TEXT)")
        db.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES(?,?,?,?,?)", [
            (1, 0, "GPU", 10, "DRAM Read Bandwidth [Throughput %]"), (1, 0, "GPU", 11, "DRAM Write Bandwidth [Throughput %]"),
            (1, 0, "GPU", 12, "SM Active [Throughput %]"),
            (2, 0, "GPU", 10, "Unrelated")])            # same metricId under another typeId: the join is on both
        rows = []
        for ts, r, w in ((12 * MS, 40.0, 10.0), (13 * MS, 60.0, 30.0), (31 * MS, 80.0, 20.0), (62 * MS, 5.0, 5.0)):
            rows += [(ts, ts, 1, 10, r), (ts, ts, 1, 11, w), (ts, ts, 1, 12, 99.0), (ts, ts, 2, 10, 1000.0)]
        db.executemany("INSERT INTO GPU_METRICS VALUES(?,?,?,?,?)", rows)
    if drop:
        db.execute(f"DROP TABLE {drop}")
    db.commit()
    db.close()


def _run(tmp_path, **kw):
    db = tmp_path / "x.sqlite"
    _build(db, **kw)
    out = tmp_path / "r.json"
    code = nr.main([str(db), "--out", str(out)])
    return code, (json.loads(out.read_text()) if out.exists() else None)


def test_union_attribution_and_launches(tmp_path):
    code, r = _run(tmp_path)
    assert code == 0
    router, up = r["ranges"]["main:router"], r["ranges"]["main:expert_up"]
    assert router["busy_ms"] == pytest.approx(4.0) and router["launches"] == 2     # 3 ms kernel + 1 ms memset
    assert up["busy_ms"] == pytest.approx(3.0) and up["launches"] == 2
    assert "admission" not in r["ranges"]


def test_other_thread_launch_is_not_attributed(tmp_path):
    code, r = _run(tmp_path)
    assert r["ranges"]["main:router"]["launches"] == 2      # the tid+1 call at 13 ms had no GPU work anyway
    db = sqlite3.connect(tmp_path / "x.sqlite")
    db.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid = 99 WHERE correlationId = 1")
    db.commit(); db.close()
    out = tmp_path / "r2.json"
    assert nr.main([str(tmp_path / "x.sqlite"), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["ranges"]["main:router"]["busy_ms"] == pytest.approx(1.0)


def test_capture_bounds_and_exposed_idle(tmp_path):
    code, r = _run(tmp_path)
    assert r["capture_bounds"] == "admission" and r["capture_ms"] == pytest.approx(100.0)
    # GPU activity: 3 + 1 (memset, 15-16) + 3 (30-33) + 5 + 1 (memcpy) = 13 ms
    assert r["exposed_idle_ms"] == pytest.approx(87.0)


def test_activity_bounds_without_admission(tmp_path):
    code, r = _run(tmp_path, admission=False)
    assert r["capture_bounds"] == "activity" and r["capture_ms"] == pytest.approx(60.0)   # 11 ms .. 71 ms
    assert r["exposed_idle_ms"] == pytest.approx(47.0)


def test_dram_percent_of_peak(tmp_path):
    code, r = _run(tmp_path)
    assert r["counters"] == "available"
    router, up = r["ranges"]["main:router"], r["ranges"]["main:expert_up"]
    assert router["dram_read_pct"] == pytest.approx(50.0) and router["dram_write_pct"] == pytest.approx(20.0)
    assert up["dram_read_pct"] == pytest.approx(80.0) and up["dram_write_pct"] == pytest.approx(20.0)
    assert {"DRAM Read Bandwidth [Throughput %]", "DRAM Write Bandwidth [Throughput %]"} <= set(r["metric_names"])
    assert router["sms_active_pct"] is None and r["capture_metrics"]["dram_read_pct"] == pytest.approx((40 + 60 + 80 + 5) / 4)


def test_real_gb10_metric_set(tmp_path):
    code, r = _run(tmp_path, real=True)
    assert code == 0 and r["counters"] == "available" and r["metric_names"] == sorted(REAL_NAMES)
    router, up = r["ranges"]["main:router"], r["ranges"]["main:expert_up"]      # router samples: bump 0 and 10
    assert router["sms_active_pct"] == pytest.approx(35.0) and router["tensor_active_pct"] == pytest.approx(15.0)
    assert router["sm_issue_pct"] == pytest.approx(25.0) and router["gr_active_pct"] == pytest.approx(95.0)
    assert router["warps_in_flight_pct"] == pytest.approx(10.0)                 # not the Avg / Avg-per-cycle variants
    assert router["dram_read_pct"] is None and router["dram_write_pct"] is None
    assert up["sms_active_pct"] == pytest.approx(70.0)
    cap = r["capture_metrics"]                                                   # admission 0-100 ms: all four samples
    assert cap["sms_active_pct"] == pytest.approx((30 + 40 + 70 + 31) / 4) and cap["dram_read_pct"] is None


def test_counters_unavailable_without_tables(tmp_path):
    code, r = _run(tmp_path, metrics=False)
    assert code == 0 and r["counters"] == "unavailable"
    assert "dram_read_pct" not in r["ranges"]["main:router"]
    assert "GPU_METRICS" not in r["tables"] and "NVTX_EVENTS" in r["tables"]
    assert r["columns"]["NVTX_EVENTS"] == ["start", "end", "text", "globalTid"]


def test_missing_memset_is_tolerated(tmp_path):
    code, r = _run(tmp_path, memset=False)
    assert code == 0 and r["ranges"]["main:router"]["busy_ms"] == pytest.approx(3.0)


def test_missing_required_table_exits_2(tmp_path, capsys):
    code, r = _run(tmp_path, drop="CUPTI_ACTIVITY_KIND_RUNTIME")
    assert code == 2 and r is None
    assert "CUPTI_ACTIVITY_KIND_RUNTIME" in capsys.readouterr().err


def test_missing_column_exits_2(tmp_path, capsys):
    db = tmp_path / "x.sqlite"
    _build(db)
    c = sqlite3.connect(db)
    c.execute("ALTER TABLE NVTX_EVENTS RENAME COLUMN globalTid TO tid")
    c.commit(); c.close()
    assert nr.main([str(db), "--out", str(tmp_path / "r.json")]) == 2
    err = capsys.readouterr().err
    assert "NVTX_EVENTS" in err and "globalTid" in err


@pytest.mark.parametrize("table,col", [("GPU_METRICS", "timestamp"), ("GPU_METRICS", "value"), ("GPU_METRICS", "typeId"),
                                       ("GPU_METRICS", "metricId"), ("TARGET_INFO_GPU_METRICS", "metricName"),
                                       ("TARGET_INFO_GPU_METRICS", "typeId"), ("TARGET_INFO_GPU_METRICS", "metricId"),
                                       ("CUPTI_ACTIVITY_KIND_MEMSET", "correlationId"), ("CUPTI_ACTIVITY_KIND_KERNEL", "end")])
def test_present_table_missing_column_exits_2(tmp_path, capsys, table, col):
    db = tmp_path / "x.sqlite"
    _build(db)
    c = sqlite3.connect(db)
    c.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
    c.commit(); c.close()
    assert nr.main([str(db), "--out", str(tmp_path / "r.json")]) == 2
    err = capsys.readouterr().err
    assert table in err and col in err


def test_nested_ranges_attribute_to_innermost(tmp_path):
    db = tmp_path / "n.sqlite"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)")
    c.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, correlationId INTEGER)")
    c.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, correlationId INTEGER)")
    c.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, correlationId INTEGER, globalTid INTEGER)")
    c.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)", [(0, 50 * MS, "main:outer", TID), (10 * MS, 20 * MS, "main:inner", TID),
                                                   (0, 60 * MS, "admission", TID)])
    c.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)",
                  [(12 * MS, 12 * MS + 1, 1, TID), (30 * MS, 30 * MS + 1, 2, TID)])
    c.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)", [(12 * MS, 14 * MS, 1), (30 * MS, 35 * MS, 2)])
    c.commit(); c.close()
    out = tmp_path / "r.json"
    assert nr.main([str(db), "--out", str(out)]) == 0
    r = json.loads(out.read_text())["ranges"]
    assert r["main:inner"]["busy_ms"] == pytest.approx(2.0) and r["main:inner"]["launches"] == 1
    assert r["main:outer"]["busy_ms"] == pytest.approx(5.0) and r["main:outer"]["launches"] == 1


def test_no_metric_match_is_distinct_from_unavailable(tmp_path):
    db = tmp_path / "x.sqlite"
    _build(db)
    c = sqlite3.connect(db)
    c.execute("DELETE FROM TARGET_INFO_GPU_METRICS WHERE metricName LIKE '%DRAM%'")   # SM Active (singular) is no match
    c.commit(); c.close()
    out = tmp_path / "r.json"
    assert nr.main([str(db), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["counters"] == "no_metric_match"
    assert set(r["metric_names"]) == {"SM Active [Throughput %]", "Unrelated"}
    assert "dram_read_pct" not in r["ranges"]["main:router"]


def test_normal_capture_is_valid(tmp_path):
    code, r = _run(tmp_path)
    assert code == 0 and r["valid"] is True and r["reason"] is None


def test_no_admission_is_invalid_exit_3(tmp_path, capsys):
    code, r = _run(tmp_path, admission=False)
    assert code == 3 and r["valid"] is False and "admission" in r["reason"] and r["capture_bounds"] == "activity"
    assert "admission" in capsys.readouterr().err


def test_no_attributed_launches_is_invalid_exit_3(tmp_path, capsys):
    db = tmp_path / "x.sqlite"
    _build(db)
    c = sqlite3.connect(db)
    c.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start = start + 500 * 1000000")   # every launch after every range
    c.commit(); c.close()
    out = tmp_path / "r.json"
    assert nr.main([str(db), "--out", str(out)]) == 3
    r = json.loads(out.read_text())
    assert r["valid"] is False and r["reason"] == "no launches attributed to any phase:block range"
    assert "no launches attributed" in capsys.readouterr().err


def test_no_block_range_is_invalid_exit_3(tmp_path, capsys):
    db = tmp_path / "x.sqlite"
    _build(db)
    c = sqlite3.connect(db)
    c.execute("DELETE FROM NVTX_EVENTS WHERE text != 'admission'")
    c.commit(); c.close()
    out = tmp_path / "r.json"
    assert nr.main([str(db), "--out", str(out)]) == 3
    r = json.loads(out.read_text())
    assert r["valid"] is False and "phase:block" in r["reason"] and r["ranges"] == {}
    assert "phase:block" in capsys.readouterr().err


def _brute(blocks, calls):
    """Reference: innermost = shortest containing range on the same tid, first match in length order."""
    by = {}
    for b in sorted(blocks, key=lambda b: b[1] - b[0]):
        by.setdefault(b[3], []).append(b)
    out = []
    for tid, t in calls:
        hit = None
        for bs, be, name, _ in by.get(tid, ()):
            if bs <= t <= be:
                hit = name
                break
        out.append(hit)
    return out


def test_index_matches_bruteforce_on_nested_ranges():
    import random
    rnd = random.Random(3)
    blocks = []
    for tid in (1, 2):
        pos = 0
        for i in range(300):                      # outer ranges holding sequential inner ranges holding a leaf
            o = pos + rnd.randint(1, 5)
            e = o + 400
            blocks.append((o, e, f"o:{tid}_{i}", tid))
            q = o + 5
            for j in range(20):
                blocks.append((q, q + 10, f"i:{tid}_{i}_{j}", tid))
                blocks.append((q + 2, q + 6, f"l:{tid}_{i}_{j}", tid))
                q += 15
            pos = e + 1
    calls = [(rnd.choice((1, 2, 3)), rnd.randint(-5, 300 * 420)) for _ in range(3000)]
    idx = nr.RangeIndex(blocks)
    assert [idx.innermost(t, x) for t, x in calls] == _brute(blocks, calls)


def test_large_export_is_fast_and_matches_reference(tmp_path):
    import time
    n = 20000
    db = tmp_path / "big.sqlite"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)")
    for t in ("KERNEL", "MEMCPY"):
        c.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_{t}(start INTEGER, end INTEGER, correlationId INTEGER)")
    c.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, correlationId INTEGER, globalTid INTEGER)")
    nv = [(0, n * 100 + 100, "admission", TID)]
    nv += [(i * 100, i * 100 + 90, f"main:b{i % 50}", TID) for i in range(n)]
    c.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)", nv)
    c.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)",
                  [(i * 100 + 10, i * 100 + 11, i, TID) for i in range(n)])
    c.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)", [(i * 100 + 20, i * 100 + 30, i) for i in range(n)])
    c.commit(); c.close()
    out = tmp_path / "r.json"
    t0 = time.perf_counter()
    assert nr.main([str(db), "--out", str(out)]) == 0
    assert time.perf_counter() - t0 < 2.0
    r = json.loads(out.read_text())["ranges"]
    assert len(r) == 50 and all(v["launches"] == n // 50 for v in r.values())
    blocks = [(s, e, t, g) for s, e, t, g in nv if t != "admission"][:2000]
    ref = _brute(blocks, [(TID, i * 100 + 10) for i in range(2000)])
    assert ref == [f"main:b{i % 50}" for i in range(2000)]
