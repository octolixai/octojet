"""The packed-table cache through the real loader on the cut mixed model: a warm load equals a cold load and a
cache-off load bit for bit, in tables and in logits; verify mode passes; a corrupted file is rebuilt.
Set ``OCTOJET_NVFP4_FLASHNEXT`` to the mixed served directory (as ``test_qwen4_exp_nvfp4.py``). Each cache directory
holds about 5.3 GiB for the 4-layer cut; ``tmp_path`` must have room for two."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

DEV = "cuda"
MODEL = os.environ.get("OCTOJET_NVFP4_FLASHNEXT", "")
LAYERS = int(os.environ.get("OCTOJET_NVFP4_FLASHNEXT_LAYERS", "4"))
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(),
                                 reason="set OCTOJET_NVFP4_FLASHNEXT to a mixed NVFP4 Flash Next directory")


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-for-bit equality (torch.equal calls -0.0 and 0.0 equal)."""

    itype = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[a.dtype]
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().view(itype), b.contiguous().view(itype)))


def load_cut(packed_cache):
    from tensorfold.families.qwen4_exp.cuda import weights as W

    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = LAYERS
        c.ple_layers = [i for i in c.ple_layers if i < LAYERS]
        return c

    W.Config.read = staticmethod(cut)
    try:
        return W.load(MODEL, DEV, mtp=True, draft_vocab="default", packed_cache=packed_cache)
    finally:
        W.Config.read = real


def tables(w):
    return [(l.moe.experts.up, l.moe.experts.down, l.moe.experts.gscale_up, l.moe.experts.gscale_down) for l in w.layers]


def same_tables(a, b):
    return all(torch.equal(x, y) for ta, tb in zip(tables(a), tables(b)) for x, y in zip(ta, tb))


def logits_of(w, toks):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward

    e = Engine(w, capacity=256, max_rows=8, prefill_rows=64, graphs=False)
    e.reset()
    out = []
    for t in toks:
        out.append(forward(w, e.st, e.buf, [t])[:1].clone())
        commit(w, e.st, e.buf, 1, 1)
    return torch.cat(out)


@needs_model
def test_cold_warm_and_off_loads_are_identical(tmp_path):
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, size=24)]
    off = load_cut("off")
    assert "packed_cache" not in off.meta
    ref_logits = logits_of(off, toks)
    cold = load_cut(str(tmp_path))
    pcm = cold.meta["packed_cache"]
    assert pcm["builds"] == LAYERS and pcm["saved"] == LAYERS and pcm["hits"] == 0 and pcm["disabled"] is False
    files = sorted(p.name for p in Path(pcm["dir"]).iterdir())
    assert "meta.json" in files and len([f for f in files if f.endswith(".safetensors")]) == LAYERS
    assert not [f for f in files if ".tmp-" in f]
    assert same_tables(off, cold)
    del off
    torch.cuda.empty_cache()
    assert bits_equal(logits_of(cold, toks), ref_logits)
    warm = load_cut(str(tmp_path))
    assert warm.meta["packed_cache"]["hits"] == LAYERS and warm.meta["packed_cache"]["builds"] == 0
    assert same_tables(cold, warm)
    assert bits_equal(logits_of(warm, toks), ref_logits)
    assert warm.mtp is not None and warm.mtp.layer.moe.experts.fmt == "affine"      # the MTP head is not cached
    del cold, warm
    torch.cuda.empty_cache()


@needs_model
def test_verify_mode_passes_and_corruption_is_rebuilt(tmp_path, monkeypatch):
    """``verify`` reads the default root, so the test points ``OCTOJET_CACHE_DIR`` at tmp_path first."""
    monkeypatch.setenv("OCTOJET_CACHE_DIR", str(tmp_path))
    first = load_cut(None)                                    # default root = $OCTOJET_CACHE_DIR/packed, mode on
    d = Path(first.meta["packed_cache"]["dir"])
    assert d.parent == tmp_path / "packed" and first.meta["packed_cache"]["builds"] == LAYERS
    del first
    torch.cuda.empty_cache()
    ok = load_cut("verify")
    assert ok.meta["packed_cache"] == {"hits": LAYERS, "builds": 0, "saved": 0, "dir": str(d), "mode": "verify",
                                       "disabled": False}
    del ok
    torch.cuda.empty_cache()
    victim = sorted(d.glob("*.safetensors"))[0]
    with open(victim, "r+b") as f:                            # flip one payload bit in place (no full-file copy)
        f.seek(-1, os.SEEK_END)
        b = f.read(1)[0] ^ 0x01
        f.seek(-1, os.SEEK_END)
        f.write(bytes([b]))
    again = load_cut("verify")
    pc = again.meta["packed_cache"]
    assert pc["builds"] == 1 and pc["hits"] == LAYERS - 1 and pc["saved"] == 1
    del again
    torch.cuda.empty_cache()
    healed = load_cut("verify")                               # payload hashes checked: the rebuilt file is valid
    assert healed.meta["packed_cache"]["hits"] == LAYERS and healed.meta["packed_cache"]["builds"] == 0
    del healed
    torch.cuda.empty_cache()
