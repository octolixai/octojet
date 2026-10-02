"""Concurrent Flash Next keeps every stream slot: replacing a kept prompt never loses the displaced state."""

import importlib
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import prefix as px
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch


def decoder(module, free, kept, keep=8):
    dec = module.MultiDecoder.__new__(module.MultiDecoder)
    dec.streams, dec.free, dec.keep, dec.next_serial = {}, list(free), keep, len(kept)
    dec.filling, dec.fills = [], {}
    dec.kept = [px.Kept(list(ids), {}, None, None, [], serial=i, slot=slot) for i, (ids, slot) in enumerate(kept)]
    return dec


def test_the_same_prompt_twice_keeps_both_slots(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    old, fresh = object(), object()
    prompt = [1, 2, 3]
    dec = decoder(multi, [fresh], [(prompt, old)])
    dec.streams = {0: SimpleNamespace(st=old)}                       # the kept slot is busy: the prompt cannot hit it
    chosen, m, miss = dec._slot_for(prompt, True)
    assert chosen is fresh and m is None and miss == "exact"
    dec._remember(prompt, chosen, {}, None)                          # a duplicate; old's entry is busy and stays
    assert {k.slot for k in dec.kept} == {old, fresh}
    dec.streams = {0: SimpleNamespace(st=fresh)}
    other, m, _ = dec._slot_for([9, 9], True)                        # old is idle and unmatched: it is the eviction
    assert other is old and m is None


def test_a_displaced_state_shared_or_busy_stays_out_of_the_free_list(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    old, fresh, busy = object(), object(), object()
    dec = decoder(multi, [], [([1], old), ([2], old), ([3], busy)], keep=8)
    dec.streams = {0: SimpleNamespace(st=busy)}
    dec._remember([1], fresh, {}, None)                              # old still backs [2]: not free
    assert old not in dec.free
    dec._remember([3], fresh, {}, None)                              # busy is a live stream's: its entry stays, not free
    assert busy not in dec.free and any(k.slot is busy for k in dec.kept)
