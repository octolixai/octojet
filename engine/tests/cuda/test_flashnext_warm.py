"""F10 on the NVFP4 Flash Next cut (OCTOJET_NVFP4_FLASHNEXT, 4 layers, CAP 20,000: QSA on): the concurrent decoder's
warm-up runs the sparse path and launches every block-selector specialisation (``attention.LAUNCHED``) that a long
prompt later uses, so that prompt launches none the server has not compiled; with OCTOJET_WARM_LONG=0 the same long
prompt does launch new ones (the check can fail); the warm-up keeps nothing and changes no reply.

The first-long-prompt time itself is printed (-s), not asserted: in one pytest process Triton's in-memory cache
already holds whatever an earlier test compiled."""

import gc
import time

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_prefill_timing_flashnext import CAP, cut, needs_model  # noqa: E402,F401

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attention as attn_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import warmplan  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

pytestmark = needs_model
COUNT = 8
LONG = 9_000                     # contexts past the warm prompt's 2,066 rows: selector widths 4,096 (and 2,048)


def ids(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=n)]


def fresh(w):
    return MultiDecoder(w, slots=1, capacity=CAP, depth=3, confidence=0.3, stop_eos=False, kv_dtype="int8")


def run(dec, prompt):
    s = Stream(list(prompt), COUNT, None)
    t0 = time.perf_counter()
    dec.admit(s)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    if s.done:
        dec.finish([s])
    while dec.live():
        dec.finish(dec.round())
    return s.out, ttft


def spy(monkeypatch):
    """A fresh LAUNCHED set, and every selector launch's specialisation and sparse-row count, in order."""

    monkeypatch.setattr(attn_mod, "LAUNCHED", set())
    seen = {"specs": [], "sparse": 0}
    real = attn_mod._launch_select

    def wrapped(scratch, pos0, rows, blocks):
        real(scratch, pos0, rows, blocks)
        seen["specs"].append(warmplan.specialisation(rows, blocks, attn_mod.SELECT_REGS))
        seen["sparse"] += int(scratch.sparse[:rows].sum())

    monkeypatch.setattr(attn_mod, "_launch_select", wrapped)
    return seen


def test_the_warm_up_launches_every_selector_a_long_prompt_uses(cut, monkeypatch):
    monkeypatch.delenv("OCTOJET_WARM_LONG", raising=False)
    seen = spy(monkeypatch)
    dec = fresh(cut)
    info = dec.warm()
    warmed = set(attn_mod.LAUNCHED)
    assert seen["sparse"] > 0                                         # the warm prompt passes the indexer budget
    plan = warmplan.select_launches(dec.pbuf.attn.nb * dec.pbuf.attn.ratio, dec.pbuf.attn.budget,
                                    dec.pbuf.attn.ratio, attn_mod.SELECT_REGS)
    assert info["select_launches"] == len(plan) and not info["select_skipped"]
    assert {warmplan.specialisation(r, b, attn_mod.SELECT_REGS) for r, b in plan} <= warmed
    assert dec.kept == [] and len(dec.free) == 1
    seen["specs"].clear()
    _, first = run(dec, ids(1, LONG))
    _, second = run(dec, ids(2, LONG))
    used = set(seen["specs"])
    assert ("select", 4096) in used                                   # the long prompt needs a width the prompt warm-up lacks
    assert attn_mod.LAUNCHED == warmed                                # ... and the warm-up launched it already
    print(f"\nwarm-up {info}; first {LONG}-token prompt {first:.2f} s, second {second:.2f} s")
    del dec; gc.collect(); torch.cuda.empty_cache()


def test_without_the_selector_warm_up_a_long_prompt_launches_new_ones(cut, monkeypatch):
    monkeypatch.setenv("OCTOJET_WARM_LONG", "0")
    spy(monkeypatch)
    dec = fresh(cut)
    info = dec.warm()
    assert info["select_skipped"] and info["select_launches"] == 0
    warmed = set(attn_mod.LAUNCHED)
    run(dec, ids(3, LONG))
    assert attn_mod.LAUNCHED - warmed >= {("select", 4096)}
    del dec; gc.collect(); torch.cuda.empty_cache()


def test_the_warm_up_changes_no_reply(cut, monkeypatch):
    monkeypatch.delenv("OCTOJET_WARM_LONG", raising=False)
    prompt = ids(4, 3_001)
    cold = fresh(cut)
    ref, _ = run(cold, prompt)
    del cold; gc.collect(); torch.cuda.empty_cache()
    warm = fresh(cut)
    warm.warm()
    got, _ = run(warm, prompt)
    assert got == ref
    del warm; gc.collect(); torch.cuda.empty_cache()
