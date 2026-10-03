"""F8: a prompt's next chunk's n-gram rows are read ahead (``forward.prestage``) while the GPU runs the current chunk.
``stage`` uses them only for exactly that chunk (same state object, position, n-gram history and tokens), stages the
same bytes either way, and never keeps a prefetch for later (CPU fakes)."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

HEADS, DH = 2, 32


class Table:
    def __init__(self):
        self.calls = 0

    def gather(self, ids):
        self.calls += 1
        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        words = (flat[:, None] * 7 + np.arange(DH // 8)).astype(np.int32)
        scales = (flat[:, None] + np.arange(DH // 32)).astype(np.int16)
        return words, scales, scales + 1


class Ngram:
    def ids(self, history, toks):
        h = int(np.asarray(history).sum())
        return np.stack([np.asarray(toks) * 3 + h, np.asarray(toks) * 5 - h], axis=1)     # [rows, heads]


class Event:
    def synchronize(self):
        pass

    def record(self):
        pass


def buffers(rows=64):
    n = rows * HEADS
    b = SimpleNamespace(rows=rows, staged=Event(), prefetched=None,
                        ids_host=torch.zeros(rows, dtype=torch.int32), ids=torch.zeros(rows, dtype=torch.int32),
                        ple_hw=torch.zeros((n, DH // 8), dtype=torch.int32),
                        ple_hs=torch.zeros((n, DH // 32), dtype=torch.int16),
                        ple_hb=torch.zeros((n, DH // 32), dtype=torch.int16),
                        ple_w=torch.zeros((n, DH // 8), dtype=torch.int32),
                        ple_s=torch.zeros((n, DH // 32), dtype=torch.bfloat16),
                        ple_b=torch.zeros((n, DH // 32), dtype=torch.bfloat16))
    return b


def state(pos=0, history=(1, 2)):
    return SimpleNamespace(pos=pos, capacity=4096, ple_history=np.asarray(history, dtype=np.int64), ple_last=None)


@pytest.fixture
def fwd(allocations):  # noqa: F811
    return importlib.import_module("tensorfold.families.qwen4_exp.cuda.forward")


def weights(table):
    return SimpleNamespace(x3=None, layers=[SimpleNamespace(ple=None),
                                            SimpleNamespace(ple=SimpleNamespace(ngram=Ngram(), table=table))])


def staged_bytes(b):
    return [t.clone() for t in (b.ids, b.ple_w, b.ple_s, b.ple_b)]


def test_a_matching_prestage_is_used_and_stages_the_same_bytes(fwd):
    toks = list(range(10, 40))
    direct_t, ahead_t = Table(), Table()
    bd, ba = buffers(), buffers()
    fwd.stage(weights(direct_t), bd, [(state(pos=128), toks)])
    st = state(pos=128)
    fwd.prestage(weights(ahead_t), ba, st, toks)
    assert ahead_t.calls == 1
    fwd.stage(weights(ahead_t), ba, [(st, toks)])
    assert ahead_t.calls == 1                                          # the table was read once, ahead
    assert all(torch.equal(x, y) for x, y in zip(staged_bytes(bd), staged_bytes(ba)))
    assert ba.prefetched is None                                       # used once, then gone


@pytest.mark.parametrize("change", ["state", "pos", "history", "tokens", "many windows"])
def test_a_stale_prestage_is_never_used(fwd, change):
    toks = list(range(10, 40))
    t = Table()
    b = buffers()
    st = state(pos=128)
    fwd.prestage(weights(t), b, st, toks)
    other, use = st, toks
    if change == "state":
        other = state(pos=128)
    elif change == "pos":
        st.pos = 256
    elif change == "history":
        st.ple_history = np.asarray([9, 9], dtype=np.int64)
    elif change == "tokens":
        use = list(range(11, 41))
    windows = [(other, use)] if change != "many windows" else [(other, use[:15]), (state(pos=4), use[15:])]
    reference = buffers()
    fwd.stage(weights(Table()), reference, [(SimpleNamespace(**vars(w)), list(x)) for w, x in windows])
    calls = t.calls
    fwd.stage(weights(t), b, windows)
    assert t.calls > calls                                             # read again: the prefetch did not apply
    assert all(torch.equal(x, y) for x, y in zip(staged_bytes(reference), staged_bytes(b)))
    assert b.prefetched is None
