"""F8: shorter prompt chunks while other streams decode. The chunking rules (CPU): with the full chunk size the new
per-chunk planner chunks exactly as before; a smaller live size ends a chunk on every full-size multiple and every cut;
the live size is used only when it is a multiple of 256 that divides the full size."""

import importlib
import random

import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)


@pytest.fixture
def mods(allocations):  # noqa: F811
    decode = importlib.import_module("tensorfold.families.qwen4_exp.cuda.decode")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    return decode, multi


def plan(decode, begin, total, pick, cuts=()):
    """Chunk starts of the per-chunk planner when ``pick(start)`` gives each chunk's rows."""

    starts, at = [], begin
    while at < total:
        starts.append(at)
        at = decode.next_chunk_end(at, total, pick(at), cuts)
    return starts


def test_full_size_planning_matches_chunk_starts(mods):
    decode, _ = mods
    rnd = random.Random(7)
    for _ in range(500):
        rows = rnd.choice([16, 256, 2048, 4096])
        total = rnd.randint(1, 20 * rows)
        begin = rnd.randint(0, total - 1)
        cuts = {rnd.randint(1, total) for _ in range(rnd.randint(0, 3))}
        assert plan(decode, begin, total, lambda at: rows, cuts) == decode.chunk_starts(begin, total, rows, cuts)


def test_live_chunks_keep_every_full_size_end_and_cut(mods):
    decode, _ = mods
    rnd = random.Random(11)
    full, small = 2048, 1024
    for _ in range(300):
        total = rnd.randint(1, 40_000)
        begin = rnd.randint(0, total - 1)
        cuts = {rnd.randint(1, total) for _ in range(rnd.randint(0, 3))}
        live = {s: rnd.random() < 0.5 for s in range(0, total + 1)}       # a stream decodes before some chunks only
        starts = plan(decode, begin, total, lambda at: small if live[at] else full, cuts)
        ends = set(starts[1:]) | {total}
        must = {m for m in range(full, total, full) if m > begin} | {c for c in cuts if begin < c < total}
        assert must <= ends                                              # checkpoints and turn starts stay chunk ends
        sizes = [b - a for a, b in zip(starts, starts[1:] + [total])]
        assert max(sizes) <= full and all(s > 0 for s in sizes)


def test_live_rows_only_when_valid(mods, monkeypatch):
    _, multi = mods
    monkeypatch.delenv("OCTOJET_LIVE_PREFILL_ROWS", raising=False)
    assert multi.live_prefill_rows(2048) == 1024 and multi.live_prefill_rows(4096) == 1024
    assert multi.live_prefill_rows(1024) is None                         # not smaller than the full size
    for value, rows, want in [("512", 2048, 512), ("0", 2048, None), ("768", 2048, None), ("300", 2048, None),
                              ("2048", 2048, None), ("1024", 2560, None), ("1024", 3072, 1024)]:
        monkeypatch.setenv("OCTOJET_LIVE_PREFILL_ROWS", value)
        assert multi.live_prefill_rows(rows) == want, (value, rows)
