"""F7 on the NVFP4 Flash Next cut (OCTOJET_NVFP4_FLASHNEXT; the fixtures of test_prefix_reuse_flashnext): every new
scheduling path keeps each stream's solo bits.

- Interleaved fills: a short prompt admitted while a long one fills takes the next chunks (fewest rows left) and both
  replies equal their fresh solo runs and the serial reference.
- Forks: a variant of a prompt whose stream is still decoding resumes beside it from a checkpoint (or extends it), its
  rows copied from the busy slot, and equals a fresh run; the source's own reply is unchanged.
- Turn start: a checkpoint at the prompt's last message start (an unaligned chunk cut) leaves the source's reply
  unchanged, and a follow-up resumes there and equals a fresh run.

Ideas from TensorFold 0.6.0/0.6.1 (short prompts admitted while a long one fills, forks that resume from a shared
prefix, the next turn resuming); the implementation and these tests are Octojet's own."""

import gc

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_prefix_reuse_flashnext import (COUNT, GREEDY, SEEDED, PB, admit_and_run, b_decoder, b_fresh,  # noqa: E402
                                         features, ids, same_state, serial_oracle, state_view)
from test_prefill_timing_flashnext import cut, needs_model  # noqa: E402,F401

from tensorfold.cuda.streams import Stream  # noqa: E402

pytestmark = needs_model
MARK = 1_001                                # a token the random prompts (ids below 1,000) never hold: the turn marker


def drive(dec, streams):
    while any(not s.done for s in streams):
        dec.finish(dec.round())


@pytest.mark.parametrize("kv_dtype", ["int8", "bf16"])
@pytest.mark.parametrize("sampling", [GREEDY, SEEDED])
def test_a_short_prompt_fills_inside_a_long_ones_fill_and_both_keep_their_solo_bits(cut, kv_dtype, sampling):
    features(cut)
    LONG, SHORT = ids(31, 3_300), ids(32, 700)           # 7 chunks and 2 chunks of 512 rows
    dec = b_decoder(cut, kv_dtype, slots=3, n=0)
    dec.arrived = lambda: True                           # stop after each chunk, as a waiting request makes it
    long_ = Stream(list(LONG), COUNT, sampling)
    dec.admit(long_, defer=True)
    dec.finish(dec.round())
    dec.finish(dec.round())                              # two of the long prompt's chunks
    short = Stream(list(SHORT), COUNT, sampling)
    dec.admit(short, defer=True)
    drive(dec, [long_, short])
    assert short.started < long_.started                 # the short prompt joined first
    for s, p in ((long_, LONG), (short, SHORT)):
        want, view = b_fresh(cut, p, sampling, kv_dtype)
        assert s.out == want[2] and len(s.out) == COUNT
        assert same_state(state_view(s.st, cut), view)
        assert s.out == serial_oracle(cut, p, sampling, kv_dtype, stop_eos=False)


@pytest.mark.parametrize("kv_dtype", ["int8", "bf16"])
@pytest.mark.parametrize("kind", ["checkpoint", "extend"])
def test_a_variant_of_a_decoding_prompt_forks_beside_it_and_equals_a_fresh_run(cut, kv_dtype, kind):
    features(cut)
    dec = b_decoder(cut, kv_dtype, slots=3)
    a = Stream(list(PB), COUNT, GREEDY)
    dec.admit(a, defer=True)
    while a.sid not in dec.streams:                      # A joins the rounds and is still decoding
        dec.finish(dec.round())
    assert not a.done
    Q = PB[:3_700] + ids(41, 300) if kind == "checkpoint" else PB + ids(42, 300)
    q = Stream(list(Q), COUNT, SEEDED)
    dec.admit(q, defer=True)
    assert q.reuse == kind and q.reuse_copy and q.reuse_miss is None and q.st is not a.st
    assert q.cached == (3_584 if kind == "checkpoint" else len(PB))
    drive(dec, [a, q])
    want, view = b_fresh(cut, Q, SEEDED, kv_dtype)
    assert q.out == want[2] and same_state(state_view(q.st, cut), view)
    assert a.out == b_fresh(cut, PB, GREEDY, kv_dtype)[0][2]          # the source decoded through the copy unchanged
    again, _, _ = admit_and_run(dec, PB, GREEDY)                      # and its entry is still an exact hit
    assert again.reuse == "exact" and again.out == a.out


@pytest.mark.parametrize("kv_dtype", ["int8", "bf16"])
def test_a_turn_start_checkpoint_changes_no_bits_and_a_follow_up_resumes_there(cut, kv_dtype):
    features(cut)
    T = ids(51, 2_700) + [MARK] + ids(52, 300)           # the last message starts at 2,700 (not a 512-row chunk end)
    dec = b_decoder(cut, kv_dtype, slots=2)
    dec.turn_marker = MARK
    t, _, _ = admit_and_run(dec, T, GREEDY)
    assert 2_700 in [c.pos for c in dec.kept[0].checkpoints] and len(dec.kept[0].checkpoints) <= 4
    assert t.out == b_fresh(cut, T, GREEDY, kv_dtype)[0][2]          # the extra chunk cut moved no bits
    F = T[:2_703] + ids(53, 400)                         # the next turn: the same history up to the message start
    f, first, drafts = admit_and_run(dec, F, SEEDED)
    assert f.reuse == "checkpoint" and f.cached == 2_700
    want, view = b_fresh(cut, F, SEEDED, kv_dtype)
    assert (first, drafts, f.out) == want and same_state(state_view(f.st, cut), view)
    gc.collect(); torch.cuda.empty_cache()
