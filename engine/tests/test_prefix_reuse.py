"""Prefix reuse bookkeeping on the CPU: the matching rule, checkpoint arithmetic and the exact-hit helper."""

from types import SimpleNamespace

from tensorfold.families.qwen4_exp.cuda import prefix as px


def kept(ids, serial=0, checkpoints=(), slot=None):
    return px.Kept(list(ids), {"pos": len(ids)}, tail=f"tail{serial}", logits=f"logits{serial}",
                   checkpoints=list(checkpoints), serial=serial, slot=slot)


def ckpt(pos):
    return px.Checkpoint(pos, {"pos": pos}, tail=f"ctail{pos}")


IDLE = lambda e: True      # noqa: E731
BUSY = lambda e: False     # noqa: E731


def test_common_prefix():
    assert px.common_prefix([], [1]) == 0
    assert px.common_prefix([1, 2, 3], [1, 2, 3]) == 3
    assert px.common_prefix([1, 2, 3], [1, 2, 4, 5]) == 2
    assert px.common_prefix([1, 2], [1, 2, 3]) == 2
    assert px.common_prefix(list(range(70_000)), list(range(69_013)) + [7] * 2_470) == 69_013


def test_exact_extend_checkpoint_kinds_and_cached():
    p = list(range(100))
    m, busy = px.match(p, [kept(p, 1)], IDLE)
    assert m.kind == "exact" and m.cached == 100 and m.resume is None and busy is None
    m, _ = px.match(p, [kept(p[:40], 2)], IDLE)
    assert m.kind == "extend" and m.cached == 40 and m.resume == {"state": {"pos": 40}, "tail": "tail2"}
    e = kept(p[:60] + [999] * 40, 3, checkpoints=[ckpt(16), ckpt(48), ckpt(64)])
    m, _ = px.match(p, [e], IDLE)
    assert m.kind == "checkpoint" and m.cached == 48 and m.resume == {"state": {"pos": 48}, "tail": "ctail48"}


def test_largest_cached_wins_across_kinds_then_kind_then_recency():
    p = list(range(100_000))
    ext = kept(p[:2_048], 1)
    div = kept(p[:70_000] + [5] * 1_444, 2, checkpoints=[ckpt(61_440)])
    m, _ = px.match(p, [ext, div], IDLE)
    assert m.entry is div and m.kind == "checkpoint" and m.cached == 61_440       # 61,440 > 2,048
    a, b = kept(p[:50], 1), kept(p[:50], 2)
    m, _ = px.match(p, [a, b], IDLE)
    assert m.entry is b                                                           # same cached and kind: recency
    ck = kept(p[:50] + [9], 3, checkpoints=[ckpt(50)])                            # diverges at 50: checkpoint at 50
    m, _ = px.match(p, [a, ck], IDLE)
    assert m.entry is a and m.kind == "extend"                                    # same cached: extend beats checkpoint


def test_busy_entries_are_never_matched_and_reported_only_when_they_would_have_won():
    p = [1, 2, 3]
    m, busy = px.match(p, [kept(p, 1)], BUSY)
    assert m is None and busy == "exact"
    m, busy = px.match(p, [kept(p, 1), kept(p[:2], 2)], lambda e: e.serial == 2)
    assert m.kind == "extend" and busy == "exact"                    # the busy exact would have served more: a miss
    m, busy = px.match(p, [kept(p, 1), kept(p, 2)], lambda e: e.serial == 2)
    assert m.kind == "exact" and busy is None                        # an idle duplicate served the same: no miss
    m, busy = px.match(p, [kept(p[:2], 1), kept(p[:1], 2)], lambda e: e.serial == 1)
    assert m.kind == "extend" and m.cached == 2 and busy is None     # the busy entry would have served less


def test_pos_equal_to_prompt_length_and_no_checkpoint_below_divergence_do_not_match():
    p = list(range(64))
    e = kept(p[:64] + [1], 1, checkpoints=[ckpt(64)])          # the prompt is a prefix of ids: no extend; pos == len
    assert px.match(p, [e], IDLE) == (None, None)
    e = kept(p[:32] + [7] * 32, 2, checkpoints=[ckpt(48)])     # d = 32 < 48
    assert px.match(p, [e], IDLE) == (None, None)


def test_boundaries():
    assert px.boundaries(9_000, 2_048, 0) == []
    assert px.boundaries(2_048, 2_048, 8) == [] and px.boundaries(100, 2_048, 8) == []
    assert px.boundaries(9_000, 2_048, 1) == [8_192]
    assert px.boundaries(4_096, 2_048, 8) == [2_048]                              # exactly two chunks: one interior end
    # F4: half of N on the last interior ends (real traffic diverges near the end), the rest spread before them
    assert px.boundaries(71_444, 2_048, 4) == [32_768, 65_536, 67_584, 69_632]   # P' at 69,013 resumes at 67,584
    assert px.boundaries(71_444, 2_048, 8) == [14_336, 30_720, 45_056, 61_440, 63_488, 65_536, 67_584, 69_632]
    assert px.boundaries(71_444, 2_048, 16)[-8:] == [2_048 * k for k in range(27, 35)]
    assert len(px.boundaries(71_444, 2_048, 16)) == 16
    assert px.boundaries(210_000, 2_048, 8) == [49_152, 100_352, 149_504, 200_704, 202_752, 204_800, 206_848, 208_896]
    assert px.boundaries(9_000, 512, 4) == [3_584, 7_680, 8_192, 8_704]          # 17 interior ends: 2 spread, the last 2
    assert px.boundaries(16_384, 2_048, 8) == [2_048 * k for k in range(1, 8)]    # at most N interior ends: all of them
    assert px.tail_count(1) == 1 and px.tail_count(4) == 2 and px.tail_count(8) == 4 and px.tail_count(0) == 0


def test_thin_keeps_spacing_and_is_deterministic():
    assert px.thin([2_048, 4_096, 6_144, 8_192, 20_480, 30_720, 40_960, 51_200, 61_440, 71_680], 8) == \
        [4_096, 8_192, 20_480, 30_720, 40_960, 51_200, 61_440, 71_680]
    assert px.thin([10, 20, 30], 5) == [10, 20, 30]
    assert px.thin([10, 20, 30], 1) == [20]                                         # gaps 10/10/10: drop 10, then 30 (gap 10 < 20)


def test_plan_checkpoints_inherits_below_cached_and_bounds_the_peak():
    inherited = [ckpt(p) for p in (2_048, 4_096, 6_144, 8_192, 10_240, 12_288)]
    take, keep = px.plan_checkpoints(73_728, 2_048, 8, inherited, cached=12_288)
    assert [c.pos for c in keep] == [8_192] and take == [30_720, 47_104, 63_488, 65_536, 67_584, 69_632, 71_680]
    positions = sorted([c.pos for c in keep] + take)
    gaps = [b - a for a, b in zip([0] + positions, positions + [73_728])]
    assert max(gaps) == 22_528 <= 2 * 16_384                # the chained bound: 2 x the spread stride (8 ends x 2,048)
    assert {67_584, 69_632, 71_680} <= set(take)            # the new prompt's last interior ends are never thinned away
    p8 = [ckpt(p) for p in px.boundaries(71_444, 2_048, 8)]   # extended to 75,540: its tail is 67,584 .. 73,728, and the
    take, keep = px.plan_checkpoints(75_540, 2_048, 8, p8, cached=71_444)   # inherited 67,584 / 69,632 stay in it
    assert {67_584, 69_632} <= {c.pos for c in keep} and {71_680, 73_728} <= set(take) and len(keep) + len(take) <= 8
    assert px.plan_checkpoints(9_000, 2_048, 0, inherited, cached=0) == ([], [])
    four = [ckpt(p) for p in (2_048, 4_096, 6_144, 8_192)]
    take, keep = px.plan_checkpoints(16_384, 2_048, 8, four, cached=8_193)
    assert take == [10_240, 12_288, 14_336] and [c.pos for c in keep] == [2_048, 4_096, 6_144, 8_192]
    take, keep = px.plan_checkpoints(16_384, 2_048, 8, four, cached=10_240)
    assert take == [12_288, 14_336] and [c.pos for c in keep] == [2_048, 4_096, 6_144, 8_192]   # the resume position is not retaken


def test_snapshot_bytes_for_the_flash_next_geometry():
    cfg = SimpleNamespace(layer_types=["linear"] * 36 + ["attention"] * 12, nv=48, dv=128, dk=128, conv_kernel=4,
                          conv_dim=10_240, ple_kernel=4, ngram_size=3, streams=4, hidden=2_048)
    assert px.snapshot_bytes(cfg) == 115_605_504                                    # 110.25 MiB: rec 108 + conv 2.1 + PLE 0.14


def test_exact_hit_restores_and_samples_with_the_request_parameters():
    calls = []
    st = SimpleNamespace(restore=lambda snap: calls.append(("restore", snap)))
    e = SimpleNamespace(st=st, sample=lambda logits, positions, sampling: calls.append(("sample", logits, positions, sampling)) or [17])
    entry = kept([5, 6, 7], serial=4)
    first = px.exact_hit(e, entry, "S2")
    assert first == 17 and e.first == 17 and e.last_streams == "tail4" and e.last_logits == "logits4"
    assert calls == [("restore", {"pos": 3}), ("sample", "logits4", [3], "S2")]
