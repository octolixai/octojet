"""Octojet's fused hyper-connection read-out for prompt chunks (F12), the same bits as the separate kernels.

The plain prompt read-out is 5 kernels after the write-back (norm, down, SiLU/inject, up, mix) with two [rows, S*D]
bf16 round trips through memory besides the streams. The fused one is 4: the write-back with the norm
(``glue.hc_writeback_normed``: the streams are re-read from cache, not memory), the down projection and the
activation as before, then the up projection with the mix in its epilogue (``upmix``: no [rows, S*D] up tensor) and
the mix's group sums (``glue.hc_mix_sums``). Idea from TensorFold 0.6.1's GB10 prompt kernels; Octojet's own
implementation.

``OCTOJET_PREFILL_HC_FUSED``: unset or 1 both fusions; ``norm`` or ``up`` one of them; 0 the plain kernels.
``OCTOJET_PREFILL_HC_TILE``: the up+mix kernel's block shape (0-3; speed only)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

ENV = "OCTOJET_PREFILL_HC_FUSED"
ENV_TILE = "OCTOJET_PREFILL_HC_TILE"
STREAMS = 4               # the up+mix kernel's stream count (Flash Next's)


class Parts(NamedTuple):
    norm: bool            # the write-back with the norm in one kernel
    up: bool              # the up projection with the mix in its epilogue


PLAIN = Parts(False, False)


def parts() -> Parts:
    """The fusions ``OCTOJET_PREFILL_HC_FUSED`` asks for (read at each call, so a test or an operator can switch)."""

    value = os.environ.get(ENV, "1").strip().lower()
    if value in ("0", "off", "plain", "false", "no"):
        return PLAIN
    if value == "norm":
        return Parts(True, False)
    if value == "up":
        return Parts(False, True)
    if value in ("", "1", "on", "true", "yes", "all"):
        return Parts(True, True)
    raise ValueError(f"{ENV}={value!r}: expected 1, 0, norm or up")


def eligible(hc, b, streams: int) -> bool:
    """A prompt chunk's read-out of an MLX 4-bit hyper-connection (tiled down, group-32 prefill up) the kernels fit."""

    if not getattr(b, "prefill", False) or streams != STREAMS:
        return False
    if getattr(hc.down, "layout", None) != "tiled":             # an EXL3 pack's matrices keep the plain path
        return False
    up = getattr(hc, "prefill_up", None)
    if up is None or getattr(up, "layout", None) != "frag" or getattr(up, "gs", None) != 32:
        return False
    n, k = up.n, up.k
    d = n // streams
    return n % streams == 0 and d % 256 == 0 and k % 32 == 0


def choose(hc, b, streams: int) -> Parts:
    """Which fusions this read-out takes: ``parts()`` when it is eligible, else none."""

    return parts() if eligible(hc, b, streams) else PLAIN


def tile() -> int:
    value = os.environ.get(ENV_TILE, "0").strip() or "0"
    t = int(value)
    if not 0 <= t <= 3:
        raise ValueError(f"{ENV_TILE}={value!r}: expected 0-3")
    return t


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    kernels = here.parents[2] / "cuda" / "kernels"               # qmm_frag.cuh: the shared fragment helpers
    return load(name="octojet_qwen4_exp_hc_prefill", sources=[str(here / "hc_prefill.cpp"), str(here / "hc_prefill.cu")],
                extra_include_paths=[str(kernels)], extra_cuda_cflags=["-O3"], verbose=False)


def upmix(act, q, normed, mixed, streams: int, tile_: int | None = None) -> None:
    """act (R, K) bf16 times ``q`` (the up projection packed for the prefill matmul) then glue.hc_mix's mix with
    ``normed`` (R, S*D): ``mixed`` (R, D) bf16, the bits of ``qmm.prefill_matmul`` then ``glue.hc_mix``."""

    _ext().upmix(act, q.weight, q.scales, q.biases, normed, mixed, q.n, streams, tile() if tile_ is None else tile_)
