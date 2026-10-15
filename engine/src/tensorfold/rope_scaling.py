"""YaRN rotary scaling (``--rope-yarn``) for Flash Next on CUDA: the parameters, frequencies and window.

The math ports transformers 5.18's ``modeling_rope_utils._compute_yarn_parameters`` (the reference for every
formula here), as ``Qwen4ExpTextRotaryEmbedding`` consumes it: YaRN changes the inverse frequencies and multiplies
cos and sin by ``attention_factor``; positions, the partial rotary width and the M-RoPE interleave are unchanged.
YaRN is static: the same tables at every length (the reference's ``dynamic_rope_update`` touches only the
"dynamic" and "longrope" types). No torch import at module load: the CLI's checks run before any weights.
"""

from __future__ import annotations

import math
from typing import Any


def rope_parameters(text: dict[str, Any]) -> dict[str, Any]:
    """The text config's rope dict (``rope_parameters``, or the older ``rope_scaling``)."""

    return dict(text.get("rope_parameters") or text.get("rope_scaling") or {})


def rope_kind(text: dict[str, Any]) -> str:
    rope = rope_parameters(text)
    return str(rope.get("rope_type", rope.get("type", "default")))


def yarn_parameters(text: dict[str, Any], factor: float | None = None) -> dict[str, Any] | None:
    """YaRN's parameters for this text config, or None (plain RoPE: today's tables, bit for bit).

    ``factor`` (``--rope-yarn``) turns YaRN on over a default-rope checkpoint, or replaces the factor of a
    checkpoint whose config already says ``rope_type: yarn``; None keeps the config as it is. The pretraining
    window is ``original_max_position_embeddings`` (the rope dict's, else the config's), else
    ``max_position_embeddings``, as the reference's ``standardize_rope_params`` fills it in."""

    kind = rope_kind(text)
    if factor is None and kind != "yarn":
        return None
    if kind not in ("default", "yarn"):
        raise ValueError(f"--rope-yarn applies YaRN over plain RoPE; this checkpoint's rope_type is {kind!r}")
    rope = rope_parameters(text) if kind == "yarn" else {}
    original = int(rope.get("original_max_position_embeddings") or text.get("original_max_position_embeddings")
                   or text.get("max_position_embeddings") or 0)
    if original <= 0:
        raise ValueError("YaRN needs the checkpoint's pretraining window (original_max_position_embeddings or "
                         "max_position_embeddings in config.json)")
    if factor is not None:
        factor = float(factor)
        if not (math.isfinite(factor) and factor > 1.0):
            raise ValueError(f"--rope-yarn takes a factor above 1 (2 doubles the window), not {factor}")
        chosen = factor
    elif rope.get("factor") is not None:
        chosen = float(rope["factor"])
    else:   # the reference: DeepSeek-style configs derive the factor from the two windows
        chosen = float(text.get("max_position_embeddings") or 0) / original
    return {"factor": chosen, "original_max_position_embeddings": original,
            "attention_factor": rope.get("attention_factor"), "mscale": rope.get("mscale"),
            "mscale_all_dim": rope.get("mscale_all_dim"), "beta_fast": rope.get("beta_fast"),
            "beta_slow": rope.get("beta_slow"), "truncate": rope.get("truncate", True)}


def window(text: dict[str, Any], factor: float | None = None) -> int:
    """The native prompt-plus-reply window: original x factor under YaRN, else ``max_position_embeddings``."""

    yarn = yarn_parameters(text, factor)
    if yarn is None:
        limit = text.get("max_position_embeddings")
        return int(limit) if isinstance(limit, int) and limit > 0 else 0
    return int(yarn["original_max_position_embeddings"] * yarn["factor"])


def attention_factor(yarn: dict[str, Any]) -> float:
    """The cos/sin multiplier (``_compute_yarn_parameters``' ``attention_factor``)."""

    def get_mscale(scale: float, mscale: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    factor = yarn["factor"]
    if yarn.get("attention_factor") is not None:
        return float(yarn["attention_factor"])
    if yarn.get("mscale") and yarn.get("mscale_all_dim"):
        return float(get_mscale(factor, yarn["mscale"]) / get_mscale(factor, yarn["mscale_all_dim"]))
    return float(get_mscale(factor))


def correction_range(yarn: dict[str, Any], dim: int, base: float) -> tuple[float, float]:
    """The ramp's (low, high) rotary pair indices (``find_correction_range``; beta_fast 32 and beta_slow 1 by default)."""

    beta_fast = yarn.get("beta_fast") or 32
    beta_slow = yarn.get("beta_slow") or 1
    original = yarn["original_max_position_embeddings"]

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = find_correction_dim(beta_fast), find_correction_dim(beta_slow)
    if yarn.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def yarn_inv_freq(base: float, dim: int, yarn: dict[str, Any]):
    """float64 inverse frequencies [dim // 2] of ``_compute_yarn_parameters`` (``dim``: the rotary width).

    The reference evaluates in float32; this keeps the engine's plain tables' float64-then-float32 rounding (the
    two agree to float32 rounding, which the tests bound)."""

    import torch

    factor = yarn["factor"]
    pos_freqs = torch.tensor(float(base), dtype=torch.float64) ** (
        torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)
    low, high = correction_range(yarn, dim, base)
    if low == high:
        high += 0.001          # the reference's guard against a zero-width ramp
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low), 0, 1)
    keep = 1 - ramp            # inv_freq_extrapolation_factor
    return interpolation * (1 - keep) + extrapolation * keep


def rope_tables(cfg) -> tuple:
    """The fp32 inverse frequencies [rotary_dim // 2] (host) and the cos/sin multiplier for a Flash Next ``Config``:
    plain RoPE's float64 powers rounded to fp32 (the loaders' expression, unchanged), or YaRN's with its
    attention factor."""

    import torch

    yarn = getattr(cfg, "yarn", None)
    if yarn is None:
        inv = torch.tensor(cfg.rope_theta, dtype=torch.float64) ** (
            -torch.arange(0, cfg.rotary_dim // 2, dtype=torch.float64) / (cfg.rotary_dim // 2))
        return inv.to(torch.float32), 1.0
    return yarn_inv_freq(cfg.rope_theta, cfg.rotary_dim, yarn).to(torch.float32), attention_factor(yarn)


def describe(yarn: dict[str, Any] | None, native: int) -> str:
    if yarn is None:
        return f"[octojet] rope: plain, window {native}"
    return f"[octojet] rope: YaRN factor {float(yarn['factor'])}, window {native}"


__all__ = ["attention_factor", "correction_range", "describe", "rope_kind", "rope_parameters", "rope_tables", "window",
           "yarn_inv_freq", "yarn_parameters"]
