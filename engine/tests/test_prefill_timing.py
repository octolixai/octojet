"""The prefill recorder on the CPU: inert when off, pool sizing, arming, spans, overflow, summary, histogram math.
CUDA events are replaced by a fake with the same interface (record / synchronize / elapsed_time)."""

import json

import pytest
import torch

from tensorfold.cuda import prefill_timing as pt


class FakeEvent:
    clock = 0.0                                   # advanced by the test between records

    def __init__(self, enable_timing=True):
        self.t = None

    def record(self, stream=None):
        self.t = FakeEvent.clock

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return other.t - self.t


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(pt.torch.cuda, "Event", FakeEvent)
    pushed = []
    monkeypatch.setattr(pt, "_nvtx_push", lambda name: pushed.append(name) or True)
    monkeypatch.setattr(pt, "_nvtx_pop", lambda: pushed.append("<pop>") or True)
    FakeEvent.clock = 0.0
    FakeEvent.pushed = pushed
    return FakeEvent


def configured(monkeypatch, fake_cuda, **kw):
    monkeypatch.setenv(pt.ENV, "1")
    r = pt.Recorder()
    args = dict(layers=4, attention_layers=1, prefill_rows=64, capacity=256, experts=8, top_k=2, slots=3, device="cpu")
    args.update(kw)
    assert r.configure(**args) is True
    return r


def test_off_is_inert(monkeypatch):
    monkeypatch.delenv(pt.ENV, raising=False)
    r = pt.Recorder()
    assert r.configure(layers=4, attention_layers=1, prefill_rows=64, capacity=256, experts=8, top_k=2, slots=3, device="cpu") is False
    assert r.enabled is False and r.armed is False and r.arm({"request": 1}) is False
    assert r.begin("router") == -1
    r.end(-1)
    assert r.host_begin("stage_wait") == 0.0
    r.host_end("stage_wait", 0.0)
    assert r.terminal() is None and r.resolve() is None
    r.histogram_add(torch.zeros((2, 3), dtype=torch.int32), 2)


def test_pool_size_formula():
    assert pt.pool_events(48, 12, 2048, 210_000) == 2 * 103 * (48 * 16 + 13 * 8 * 3 + 200) + 128
    assert pt.pool_events(4, 1, 64, 256) == 2 * 4 * (4 * 16 + 2 * 1 * 3 + 200) + 128


def test_configure_preallocates(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    n = pt.pool_events(4, 1, 64, 256)
    assert r.enabled and len(r.events) == n and len(r.span_meta) == n // 2
    assert r.hist.shape == (4, 4, 9) and r.hist.dtype == torch.int32          # experts + 1 bins (shared last)
    assert r.idx_scratch.shape == (64 * 3,) and r.idx_scratch.dtype == torch.int64
    assert r.ones.shape == (64 * 3,) and r.terminal_event is not None
    assert all(e.t is not None for e in r.events)                              # pre-recorded once at configure


def test_spans_resolve_into_records_summary_and_nvtx_names(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    assert r.arm({"request": 7, "prompt_tokens": 128}) is True
    r.phase, r.chunk, r.rows, r.pos, r.layer = "main", 0, 64, 0, 0
    i = r.begin("router"); fake_cuda.clock = 1.5; r.end(i)
    i = r.begin("expert_up"); fake_cuda.clock = 4.0; r.end(i)
    r.layer = 1
    i = r.begin("router"); fake_cuda.clock = 5.0; r.end(i)
    t0 = r.host_begin("stage_wait"); r.host_end("stage_wait", t0 - 0.010)
    r.phase = "mtp"; r.layer = -1
    i = r.begin("expert_up"); fake_cuda.clock = 5.5; r.end(i)
    r.phase = "draft"
    i = r.begin("mtp_input"); fake_cuda.clock = 5.7; r.end(i)
    fake_cuda.clock = 6.0
    assert r.terminal() is not None
    s = r.resolve()
    assert r.armed is False
    by = {(d["phase"], d["layer"], d["block"]): d["ms"] for d in r.records if d["clock"] == "device"}
    assert by[("main", 0, "router")] == pytest.approx(1.5)
    assert by[("main", 0, "expert_up")] == pytest.approx(2.5)
    assert by[("main", 1, "router")] == pytest.approx(1.0)
    assert by[("mtp", -1, "expert_up")] == pytest.approx(0.5)
    assert by[("draft", -1, "mtp_input")] == pytest.approx(0.2)
    assert s["device_ms"]["main"]["router"] == pytest.approx(2.5)
    assert s["device_ms"]["mtp"]["expert_up"] == pytest.approx(0.5)
    assert s["device_ms"]["draft"]["mtp_input"] == pytest.approx(0.2)
    assert s["host_ms"]["stage_wait"] == pytest.approx(10.0, abs=0.5)
    assert s["device_total_ms"] == {"main": pytest.approx(5.0), "mtp": pytest.approx(0.5), "draft": pytest.approx(0.2)}
    assert s["spans"] == 5 and s["overflow"] is False and s["meta"] == {"request": 7, "prompt_tokens": 128}
    assert s["wall_ms"] >= 0
    assert fake_cuda.pushed[:2] == ["main:router", "<pop>"] and "mtp:expert_up" in fake_cuda.pushed
    assert "main:stage_wait" in fake_cuda.pushed                               # host cuts push ranges too


def test_cuts_allocate_no_tensors_and_grow_no_containers(monkeypatch, fake_cuda):
    """The timed region may churn Python scalars, but it must not allocate tensors or grow lists/dicts."""
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1}, histogram=True); r.phase, r.layer, r.chunk, r.rows, r.pos = "main", 0, 0, 64, 0
    r.chunk_rows[0] = 64
    sizes = (len(r.events), len(r.span_meta), len(r.host_meta), len(r.records), len(r.notes), len(r.chunk_rows))
    pick = torch.tensor([[0, 3, 8]] * 64, dtype=torch.int32)
    def boom(*a, **k):
        raise AssertionError("tensor allocation in the timed region")
    for name in ("empty", "zeros", "ones", "tensor", "arange", "full", "empty_like", "zeros_like", "cat", "stack"):
        monkeypatch.setattr(pt.torch, name, boom)
    for name in ("clone", "to", "cpu", "item", "tolist", "new_empty", "new_zeros", "__deepcopy__", "contiguous"):
        monkeypatch.setattr(pt.torch.Tensor, name, boom)          # a `.clone()` / `.item()` in a cut fails here too
    for _ in range(50):
        i = r.begin("router"); fake_cuda.clock += 1.0; r.end(i)
        t0 = r.host_begin("stage_wait"); r.host_end("stage_wait", t0)
        r.histogram_add(pick, 64)
    assert (len(r.events), len(r.span_meta), len(r.host_meta), len(r.records), len(r.notes), len(r.chunk_rows)) == sizes
    assert r.n == 100 and r.nh == 50


def test_growth_normalises_per_row_with_subblock_rows(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda, capacity=64 * 8)
    r.arm({"request": 1}); r.phase, r.layer = "main", 0
    clock = 0.0
    for chunk in range(8):
        r.chunk, r.rows, r.pos = chunk, 64, chunk * 64
        for sub in range(2):                       # two 32-row sub-blocks, constant cost per row
            i = r.begin("attn_sparse", rows=32, pos=chunk * 64 + sub * 32); clock += 32.0; fake_cuda.clock = clock; r.end(i)
        i = r.begin("idx_scores"); clock += 64.0 * (1 + chunk / 7); fake_cuda.clock = clock; r.end(i)
    r.terminal(); s = r.resolve()
    g = s["growth"]["main"]
    assert g["attn_sparse"]["ratio"] == pytest.approx(1.0)                      # sub-block rows: no false growth
    assert g["idx_scores"]["first_quarter_ms_per_row"] == pytest.approx((64 * 1 + 64 * (1 + 1 / 7)) / 128)
    assert g["idx_scores"]["last_quarter_ms_per_row"] == pytest.approx((64 * (1 + 6 / 7) + 64 * 2) / 128)


def test_overflow_stops_recording_and_notes(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1})
    n = len(r.events) // 2
    for _ in range(n + 5):
        i = r.begin("router"); fake_cuda.clock += 1.0; r.end(i)
    assert r.overflow is True
    r.terminal(); s = r.resolve()
    assert s["overflow"] is True and s["spans"] == n and any("pool" in note for note in s["notes"])


def test_arm_twice_is_refused_until_resolved(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    assert r.arm({"request": 1}) is True
    assert r.arm({"request": 2}) is False
    r.terminal(); r.resolve()
    assert r.arm({"request": 3}) is True


def test_abort_disarms_without_resolving(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1}); i = r.begin("router"); r.end(i)
    r.abort("terminal wait failed")
    assert r.armed is False and r.records == [] and r.last_abort == "terminal wait failed"
    assert r.arm({"request": 2}) is True


def test_open_cut_is_closed_at_terminal_and_ranges_balanced(monkeypatch, fake_cuda):
    """An exception inside the timed region skips end()/host_end(): the terminal bounds the span and pops the ranges."""
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1}); r.phase, r.layer = "main", 0
    r.begin("router"); fake_cuda.clock = 2.0
    r.host_begin("stage_wait")
    assert r.open_ranges == 2 and r.open_span == 0
    fake_cuda.clock = 3.0
    assert r.terminal() is not None and r.open_ranges == 0 and r.open_span == -1
    assert fake_cuda.pushed.count("<pop>") == 2
    s = r.resolve()
    assert s["device_ms"]["main"]["router"] == pytest.approx(3.0) and any("left open" in n for n in s["notes"])


def test_abort_never_raises_even_when_nvtx_pop_does(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1}); r.begin("router")
    def bad(): raise RuntimeError("nvtx down")
    monkeypatch.setattr(pt, "_nvtx_pop", bad)
    r.abort("terminal wait failed")
    assert r.armed is False and r.open_ranges == 0 and r.open_span == -1 and r.records == []


def test_abort_after_resolve_drops_records_and_is_unconditional(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1}); i = r.begin("router"); r.end(i); r.terminal()
    assert r.resolve() is not None and r.records
    r.abort("dump failed")
    assert r.records == [] and r.last_abort == "dump failed" and r.armed is False and r.open_ranges == 0


def test_histogram_counts_main_routed_picks_only(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1}, histogram=True)
    r.phase, r.layer, r.chunk = "main", 2, 1
    r.chunk_rows[1] = 3
    pick = torch.tensor([[0, 3, 8], [3, 3, 8], [7, 0, 8]], dtype=torch.int32)   # slot 2 = the shared expert (id 8)
    r.histogram_add(pick, 3)
    assert r.hist[2, 1].tolist() == [2, 0, 0, 3, 0, 0, 0, 1, 3]
    r.phase, r.layer = "mtp", -1
    r.histogram_add(pick, 3)                                                    # the MTP head never enters
    assert int(r.hist.sum()) == 9
    r.terminal(); s = r.resolve()
    c = s["histogram"]["layers"][2]["chunks"][1]
    assert c["touched"] == 3 and c["rows"] == 3
    assert c["rows_per_touched_expert"] == {"median": 2.0, "p10": 1, "p90": 3}   # nearest-rank on [1, 2, 3]
    r2 = configured(monkeypatch, fake_cuda)
    r2.arm({"request": 1}, histogram=False)
    r2.phase, r2.layer, r2.chunk = "main", 0, 0
    r2.histogram_add(pick, 3)
    assert int(r2.hist.sum()) == 0


def test_histogram_overflow_is_noted(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)                                      # 4 chunk slots
    r.arm({"request": 1}, histogram=True)
    r.phase, r.layer, r.chunk = "main", 0, 4
    r.histogram_add(torch.zeros((2, 3), dtype=torch.int32), 2)                  # beyond the last slot: skipped
    r.terminal(); s = r.resolve()
    assert int(r.hist.sum()) == 0 and any("histogram" in n for n in s["notes"])


def test_nearest_rank_percentiles():
    assert pt.percentile([1, 2, 3], 0.10) == 1 and pt.percentile([1, 2, 3], 0.90) == 3
    assert pt.percentile([5], 0.5) == 5 and pt.percentile([], 0.5) is None
    assert pt.percentile(list(range(1, 11)), 0.90) == 9


def test_context_reset_between_admissions(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1}); r.phase, r.layer, r.chunk = "mtp", 3, 2; r.chunk_rows[2] = 64
    i = r.begin("embed"); fake_cuda.clock = 1.0; r.end(i); r.terminal(); r.resolve()
    r.arm({"request": 2})
    assert (r.phase, r.layer, r.chunk, r.rows, r.pos) == ("main", -1, -1, 0, 0)
    assert r.records == [] and int(r.hist.sum()) == 0 and r.chunk_rows == [0, 0, 0, 0]


def test_wall_is_sampled_after_the_terminal_wait(monkeypatch, fake_cuda):
    import time
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1})
    slow = type("E", (), {"record": lambda self, stream=None: None,
                          "synchronize": lambda self: time.sleep(0.02)})()
    r.terminal_event = slow
    r.terminal(); s = r.resolve()
    assert s["wall_ms"] >= 20.0


def test_dump_writes_jsonl_records_and_summary(monkeypatch, fake_cuda, tmp_path):
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 9}); r.phase, r.layer, r.chunk, r.rows, r.pos = "main", 0, 0, 64, 0
    i = r.begin("commit"); fake_cuda.clock = 2.0; r.end(i); r.terminal(); s = r.resolve()
    out = tmp_path / "t.jsonl"
    r.dump(s, str(out))
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert lines[-1]["kind"] == "summary" and lines[-1]["meta"] == {"request": 9}
    assert [x for x in lines if x["kind"] == "span"] == [
        {"kind": "span", "request": 9, "phase": "main", "chunk": 0, "layer": 0, "block": "commit", "rows": 64, "pos": 0,
         "ms": 2.0, "clock": "device"}]
    r.dump(s, None)


def test_unknown_block_or_phase_is_rejected(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1})
    with pytest.raises(ValueError):
        r.begin("not_a_block")
    with pytest.raises(ValueError):
        r.host_end("router", 0.0)
    with pytest.raises(ValueError, match="unknown phase/block"):
        r.host_begin("not_a_block")
    with pytest.raises(ValueError, match="is a device block"):
        r.host_begin("router")
    assert r.open_ranges == 0
    r.phase = "elsewhere"
    with pytest.raises(ValueError):
        r.begin("router")
    with pytest.raises(ValueError, match="unknown phase/block"):
        r.host_begin("stage_wait")
    assert pt.BLOCKS == pt.HOST_BLOCKS | pt.DEVICE_BLOCKS and "attn_other" in pt.DEVICE_BLOCKS and "prefill_other" in pt.DEVICE_BLOCKS
    assert "draft" not in pt.BLOCKS and "draft" in pt.PHASES


@pytest.fixture
def real_nvtx(monkeypatch):
    """Fake events, but the real _nvtx_push/_nvtx_pop helpers over a patched torch.cuda.nvtx."""
    monkeypatch.setattr(pt.torch.cuda, "Event", FakeEvent)
    FakeEvent.clock = 0.0
    calls = {"push": 0, "pop": 0}
    def push(name):
        calls["push"] += 1
    def pop():
        calls["pop"] += 1
    monkeypatch.setattr(pt.torch.cuda.nvtx, "range_push", push)
    monkeypatch.setattr(pt.torch.cuda.nvtx, "range_pop", pop)
    return calls


def test_failed_nvtx_push_is_not_popped_and_is_noted(monkeypatch, real_nvtx):
    r = configured(monkeypatch, FakeEvent)
    r.arm({"request": 1}); r.phase, r.layer = "main", 0
    def bad_push(name):
        raise RuntimeError("nvtx down")
    monkeypatch.setattr(pt.torch.cuda.nvtx, "range_push", bad_push)
    i = r.begin("router")
    assert i == 0 and r.open_ranges == 0
    FakeEvent.clock = 1.0
    r.end(i)
    assert real_nvtx["pop"] == 0 and r.open_ranges == 0 and r.open_span == -1
    r.terminal(); s = r.resolve()
    assert real_nvtx["pop"] == 0
    assert s["device_ms"]["main"]["router"] == pytest.approx(1.0)
    assert s["nvtx_failures"] == {"push": 1, "pop": 0} and any("nvtx" in n for n in s["notes"])


def test_failed_nvtx_pop_keeps_the_range_accounted_and_terminal_retries_once(monkeypatch, real_nvtx):
    r = configured(monkeypatch, FakeEvent)
    r.arm({"request": 1}); r.phase, r.layer = "main", 0
    pops = {"n": 0}
    def bad_pop():
        pops["n"] += 1
        raise RuntimeError("nvtx down")
    monkeypatch.setattr(pt.torch.cuda.nvtx, "range_pop", bad_pop)
    i = r.begin("router"); FakeEvent.clock = 1.0; r.end(i)
    assert real_nvtx["push"] == 1 and pops["n"] == 1 and r.open_ranges == 1
    assert r.terminal() is not None
    assert pops["n"] == 2 and r.open_ranges == 1                                # one retry, no spin
    s = r.resolve()
    assert s["nvtx_failures"] == {"push": 0, "pop": 2} and any("nvtx" in n for n in s["notes"])


def test_host_cut_nvtx_failures_are_accounted(monkeypatch, real_nvtx):
    r = configured(monkeypatch, FakeEvent)
    r.arm({"request": 1}); r.phase = "main"
    def bad_push(name):
        raise RuntimeError("nvtx down")
    monkeypatch.setattr(pt.torch.cuda.nvtx, "range_push", bad_push)
    t0 = r.host_begin("stage_wait"); r.host_end("stage_wait", t0)
    assert real_nvtx["pop"] == 0 and r.open_ranges == 0
    r.terminal(); s = r.resolve()
    assert s["nvtx_failures"] == {"push": 1, "pop": 0} and s["host_ms"]["stage_wait"] >= 0


def test_end_after_abort_is_a_no_op(monkeypatch, fake_cuda):
    r = configured(monkeypatch, fake_cuda)
    r.arm({"request": 1}); i = r.begin("router")
    r.abort("admission failed")
    pops = fake_cuda.pushed.count("<pop>")
    r.end(i)
    assert r.open_ranges == 0 and r.open_span == -1 and fake_cuda.pushed.count("<pop>") == pops
