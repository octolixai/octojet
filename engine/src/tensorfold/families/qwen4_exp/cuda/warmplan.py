"""F10: which block-selector specialisations a server compiles at startup (no torch, no triton: tested on CPU).

``attention._launch_select`` compiles ``_select`` once per power-of-two width of a row's block scores
(``BLOCK = next_power_of_2(context / ratio)``) and ``_select_tiles`` once per tile shape (a prompt's row blocks, a
decode window). The startup prompt (``decode.warm``: one chunk and a partial one) reaches only the widths of its own
length, so before F10 the first long prompt after a start compiled the rest on the request (Triton JIT; seconds each
when the on-disk cache lacks them). The plan below lists every launch a context of up to the capacity can make."""

from __future__ import annotations

import os

TILE_ROWS = 64          # ``_launch_select``: _select_tiles takes 4,096-block tiles from this many rows, 8,192 below


def warm_long() -> bool:
    """OCTOJET_WARM_LONG=0 skips the selector warm-up (the startup prompt still runs)."""

    return os.environ.get("OCTOJET_WARM_LONG", "1").strip() != "0"


def select_launches(capacity: int, budget: int, ratio: int, regs: int) -> list[tuple[int, int]]:
    """(rows, blocks) for ``_launch_select`` calls that between them compile every specialisation contexts up to
    ``capacity`` keys reach: ``_select`` at each power-of-two width up to ``regs``, then ``_select_tiles`` in both tile
    shapes once a row can hold more than ``regs`` blocks. Empty when the capacity never passes the budget (no QSA)."""

    if capacity <= budget:
        return []
    nb = -(-capacity // ratio)
    top = 1 << (nb - 1).bit_length()                  # next_power_of_2(nb): the widest width a row reaches
    out = []
    width = 1
    while width <= min(top, regs):
        out.append((1, width))
        width *= 2
    if top > regs:
        out += [(TILE_ROWS, nb), (1, nb)]              # a prompt's row blocks, then a decode window
    return out


def specialisation(rows: int, blocks: int, regs: int) -> tuple:
    """The kernel and constexprs ``_launch_select(rows, blocks)`` compiles: ("select", width) or ("tiles", TB, warps)."""

    width = 1 << (max(1, blocks) - 1).bit_length()
    if width <= regs:
        return ("select", width)
    return ("tiles", 4096, 8) if rows >= TILE_ROWS else ("tiles", 8192, 16)
