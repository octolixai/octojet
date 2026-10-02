import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compare_runs as cr


def bench(path, ids, prompts=("fibonacci-raw", "gpu-chat-no-think")):
    path.write_text(json.dumps([{"prompt": p, "temperature": 0, "token_ids_all": ids} for p in prompts]))
    return str(path)


def parallel(path, ids, all_equal=True, concurrent=None):
    concurrent = ids if concurrent is None else concurrent
    path.write_text(json.dumps({"prompts": [{"solo_ids": s, "concurrent_ids": c, "concurrent2_ids": c}
                                            for s, c in zip(ids, concurrent)], "all_equal": all_equal}))
    return str(path)


def test_bench_files_equal_and_unequal(tmp_path, capsys):
    a = bench(tmp_path / "a.json", [[1, 2], [1, 2]])
    b = bench(tmp_path / "b.json", [[1, 2], [1, 2]])
    assert cr.main([a, b]) == 0
    c = bench(tmp_path / "c.json", [[1, 2], [1, 3]])
    assert cr.main([a, c]) == 1
    assert "gpu-chat-no-think" in capsys.readouterr().out


def test_bench_missing_cells_or_none_ids_fail(tmp_path):
    a = bench(tmp_path / "a.json", [[1, 2], [1, 2]])
    fewer = bench(tmp_path / "f.json", [[1, 2], [1, 2]], prompts=("fibonacci-raw",))
    assert cr.main([a, fewer]) == 1                              # cell sets differ
    shorter = bench(tmp_path / "s.json", [[1, 2]])                # one rep instead of two
    assert cr.main([a, shorter]) == 1
    none_a = bench(tmp_path / "na.json", [None, None])
    none_b = bench(tmp_path / "nb.json", [None, None])
    assert cr.main([none_a, none_b]) == 1                        # None == None is not a match


def test_parallel_files(tmp_path):
    a = parallel(tmp_path / "p1.json", [[1, 2, 3], [4, 5]])
    b = parallel(tmp_path / "p2.json", [[1, 2, 3], [4, 5]])
    c = parallel(tmp_path / "p3.json", [[1, 2, 3], [4, 6]])
    assert cr.main([a, b]) == 0 and cr.main([a, c]) == 1
    d = parallel(tmp_path / "p4.json", [[1, 2, 3], [4, 5]], concurrent=[[1, 2, 3], [4, 9]])
    assert cr.main([a, d]) == 1                                  # concurrent ids differ within the start
    e = parallel(tmp_path / "p5.json", [[1, 2, 3], [4, 5]], all_equal=False)
    assert cr.main([a, e]) == 1                                  # the tool itself reported inequality


def test_mixed_kinds_too_few_files_or_malformed_is_2(tmp_path):
    a = bench(tmp_path / "a.json", [[1]])
    p = parallel(tmp_path / "p.json", [[1], [2]])
    assert cr.main([a, p]) == 2 and cr.main([a]) == 2
    empty = bench(tmp_path / "e.json", [])                        # a row without reps: no cells to compare
    assert cr.main([empty, empty]) == 2
    dup = tmp_path / "d.json"
    dup.write_text(json.dumps([{"prompt": "x", "temperature": 0, "token_ids_all": [[1]]},
                               {"prompt": "x", "temperature": 0, "token_ids_all": [[2]]}]))
    assert cr.main([str(dup), str(dup)]) == 2                     # a duplicate cell key would hide a difference


def test_unreadable_or_wrongly_shaped_files_are_2(tmp_path):
    a = bench(tmp_path / "a.json", [[1]])
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert cr.main([a, str(bad)]) == 2
    noprompt = tmp_path / "np.json"
    noprompt.write_text(json.dumps([{"temperature": 0, "token_ids_all": [[1]]}]))
    assert cr.main([str(noprompt), str(noprompt)]) == 2
    notlist = tmp_path / "nl.json"
    notlist.write_text(json.dumps({"token_ids_all": [[1]]}))
    assert cr.main([str(notlist), str(notlist)]) == 2
    nonobj = tmp_path / "no.json"
    nonobj.write_text(json.dumps([5]))
    assert cr.main([str(nonobj), str(nonobj)]) == 2
    assert cr.main([a, str(tmp_path / "missing.json")]) == 2


def test_malformed_ids_are_invalid_cells(tmp_path):
    good = bench(tmp_path / "g.json", [[1, 0], [1, 0]])
    for i, bad in enumerate(([1.5, 2], [True, False], [])):
        a = bench(tmp_path / f"a{i}.json", [bad, bad])
        b = bench(tmp_path / f"b{i}.json", [bad, bad])
        assert cr.main([a, b]) == 1, bad                          # invalid on both sides is not a match
        assert cr.main([good, a]) == 1, bad                       # [True, False] must not equal [1, 0]
    pa = parallel(tmp_path / "pa.json", [[True, False], [4, 5]])
    pb = parallel(tmp_path / "pb.json", [[1, 0], [4, 5]])
    assert cr.main([pa, pb]) == 1
