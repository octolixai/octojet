#!/usr/bin/env python3
"""A model cut to a few layers, loaded from the mixed NVFP4 directory and from its plain MLX base: logits compared.

  python tools/diag_cut_compare.py MIXED_DIR [--layers 4] [--tokens 96] [--capacity 512] [--capacity2 262151]
  python tools/diag_cut_compare.py MIXED_DIR --self-check

Reports JSON on stdout and always exits 0 (a diagnostic; errors are captured per stage into "errors").

The cut mirrors ``cut_model`` in tests/cuda/test_qwen4_exp_nvfp4.py: ``weights.Config.read`` patched to keep
``--layers`` decoder layers (and the PLE layers below that), then ``weights.load(dir, "cuda", mtp=True,
draft_vocab="default")`` -- the whole loader path, through the symlink directory for the mixed model (n-gram tables,
head, embeddings, MTP head, octojet.json). Two models, loaded one at a time (the first freed before the second):
(1) MIXED_DIR (decoder experts NVFP4), (2) ``nvfp4.sources(MIXED_DIR).base`` (the MLX checkpoint, all affine).

For each model and each capacity (``--capacity``, then ``--capacity2``, the served model's slot count), an
``Engine(w, capacity=C, max_rows=8, prefill_rows=64, graphs=False)`` and the same seeded random tokens fed one row at a
time exactly as ``test_cut_model_windows_equal_one_row_steps`` does (``forward(w, e.st, e.buf, [t])`` then
``commit(w, e.st, e.buf, 1, 1)``); logits [T, V] kept as fp32 on the CPU. Per capacity: per-position cosine of the
mixed and MLX logits, top-1 agreement, mean |logit|, non-finite counts, and a verdict:
"tables_and_loader_ok" (mean cosine >= 0.99 and top-1 >= 0.8 at both capacities), "mixed_cut_wrong" (disagrees at
both), "capacity_dependent" (agrees at --capacity, not at --capacity2), else "inconclusive".
Two 4-bit quantizations of the same weights through a 4-layer cut should agree closely (cosine ~0.99+); garbage
from token 1 in the served model would show here as cosine far below that if the cut reproduces it.

``--prefill-rows N`` (default 0 = off): per model and capacity, a second pass over the same tokens in chunks of N rows
through the engine's PREFILL buffers (``e.pbuf``, as ``decode.prefill`` does: ``forward(w, e.st, e.pbuf, chunk)`` then
``commit(w, e.st, e.pbuf, R, R)``; Engine built with ``prefill_rows=max(64, N)``). Reported as ``compare_prefill`` /
``prefill`` (mixed vs MLX, same metrics) and ``prefill_vs_onerow`` (each model's prefill logits against its own one-row
logits). The prefill buffers run the head on a chunk's LAST row only, so a pass yields one logits row per chunk (rows
N-1, 2N-1, ... of the one-row pass), not one per token. Verdict "mixed_prefill_path_wrong": the one-row comparison passes at both capacities but the prefill
comparison fails (top-1 < 0.8 or mean cosine < 0.99) at some capacity.

``--self-check``: the mixed model only; forward windows (1, 8) over 40 tokens against one-row steps, bitwise (the
engine path the cut tests exercised).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback

import numpy as np
import torch

DEV = "cuda"
COS_OK = 0.99
TOP1_OK = 0.8
TRIM = 16


def rnd(x, n=5):
    x = float(x)
    return x if not math.isfinite(x) else round(x, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mixed")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--capacity", type=int, default=512)
    ap.add_argument("--capacity2", type=int, default=262151)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--prefill-rows", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    report: dict = {"mixed": a.mixed, "layers": a.layers, "tokens": a.tokens, "capacities": [a.capacity, a.capacity2],
                    "seed": a.seed, "prefill_rows": a.prefill_rows, "self_check_mode": a.self_check, "errors": [], "seconds": {}}
    try:
        run(a, report)
    except BaseException as exc:                      # noqa: BLE001 - a diagnostic reports everything
        report["errors"].append({"step": "top", "error": repr(exc), "traceback": traceback.format_exc()[-4000:]})
    print(json.dumps(report, indent=1, default=str))
    raise SystemExit(0)


def step(report: dict, name: str):
    """Context manager: time a step and capture its exception into the report (the tool keeps going)."""

    class _Step:
        ok = False

        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, kind, exc, tb):
            report["seconds"][name] = round(time.time() - self.t0, 1)
            if exc is not None:
                report["errors"].append({"step": name, "error": repr(exc),
                                         "traceback": "".join(traceback.format_exception(kind, exc, tb))[-4000:]})
                return not isinstance(exc, KeyboardInterrupt)
            self.ok = True
            return False

    return _Step()


def load_cut(model_dir, layers: int):
    """``cut_model`` of tests/cuda/test_qwen4_exp_nvfp4.py, without its NVFP4 assertions (also loads the MLX base)."""

    from tensorfold.families.qwen4_exp.cuda import weights as W

    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = layers
        c.ple_layers = [i for i in c.ple_layers if i < layers]
        return c

    W.Config.read = staticmethod(cut)
    try:
        return W.load(model_dir, DEV, mtp=True, draft_vocab="default")
    finally:
        W.Config.read = real


def meta(w) -> dict:
    return {"cfg_layers": w.cfg.layers, "cfg_vocab": w.cfg.vocab, "layers_loaded": len(w.layers),
            "nvfp4_layers": sum(1 for l in w.layers if l.moe.experts.fmt == "nvfp4"),
            "expert_fmts": [l.moe.experts.fmt for l in w.layers],
            "mtp_present": w.mtp is not None,
            "mtp_experts_fmt": w.mtp.layer.moe.experts.fmt if w.mtp is not None else None,
            "ple_layers": list(getattr(w.cfg, "ple_layers", []) or []),
            "draft_vocab": int(w.draft_ids.numel()) if w.draft_ids is not None else None}


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def one_row_logits(w, capacity: int, toks: list[int], prefill_rows: int = 0, out: dict | None = None):
    """One-row pass; with ``prefill_rows`` > 0 also a chunked pass on the prefill buffers.

    Returns (one_row [T, V], info) or, when ``out`` is a dict, the prefill logits are stored in ``out["prefill"]`` (so a
    prefill failure keeps the one-row result)."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward

    e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=max(64, prefill_rows) if prefill_rows > 0 else 64,
               graphs=False)
    info = {"engine_capacity": e.capacity, "state_capacity": e.st.capacity,
            "cuda_allocated_gb": rnd(torch.cuda.memory_allocated() / 2**30, 3)}
    try:
        e.reset()
        res = []
        for t in toks:
            res.append(forward(w, e.st, e.buf, [t])[:1].float().cpu())
            commit(w, e.st, e.buf, 1, 1)
        torch.cuda.synchronize()
        info["final_pos"] = int(e.st.pos)
        res = torch.cat(res)
        if prefill_rows > 0 and out is not None:
            out["onerow"] = res
            e.reset()
            pre = []
            for at in range(0, len(toks), prefill_rows):
                chunk = toks[at:at + prefill_rows]
                # the prefill buffers run the head on a chunk's LAST row only (forward.finish): [1, V] a chunk
                pre.append(forward(w, e.st, e.pbuf, chunk)[-1:].float().cpu())
                commit(w, e.st, e.pbuf, len(chunk), len(chunk))
            torch.cuda.synchronize()
            info["prefill_final_pos"] = int(e.st.pos)
            out["prefill"] = torch.cat(pre)
        return res, info
    finally:
        del e
        free()


def prefill_ends(n_tokens: int, prefill_rows: int) -> list[int]:
    """One-row indices of each prefill chunk's last token: the only rows the prefill buffers return logits for."""

    return [min(at + prefill_rows, n_tokens) - 1 for at in range(0, n_tokens, prefill_rows)]


def prefill_vs_onerow(pre: torch.Tensor, one: torch.Tensor, n_tokens: int, prefill_rows: int) -> dict:
    """Prefill logits (one row a chunk) against the same positions of the one-row pass."""

    ends = prefill_ends(n_tokens, prefill_rows)
    return {**within(pre, one[ends]), "rows": len(ends), "positions": ends}


def within(pre: torch.Tensor, one: torch.Tensor) -> dict:
    """A model's prefill-pass logits against its own one-row logits."""

    a, b = pre.float(), one.float()
    a0, b0 = a.nan_to_num(0.0, 0.0, 0.0), b.nan_to_num(0.0, 0.0, 0.0)
    cos = torch.nn.functional.cosine_similarity(a0.double(), b0.double(), dim=-1).float()
    agree = a0.argmax(-1) == b0.argmax(-1)
    return {"cos_mean": rnd(cos.mean(), 5), "cos_min": rnd(cos.min(), 5), "cos_argmin": int(cos.argmin()),
            "top1_agree": rnd(agree.float().mean(), 4),
            "first_disagree": int((~agree).nonzero()[0]) if bool((~agree).any()) else None,
            "nonfinite_prefill": int((~torch.isfinite(a)).sum()), "nonfinite_onerow": int((~torch.isfinite(b)).sum()),
            "max_abs_diff": rnd((a0 - b0).abs().max(), 5)}


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    itype = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[a.dtype]
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().view(itype), b.contiguous().view(itype)))


def self_check(w, toks: list[int]) -> dict:
    """``test_cut_model_windows_equal_one_row_steps`` on ``toks`` with windows (1, 8)."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward

    windows = (1, 8)
    e = Engine(w, capacity=512, max_rows=max(windows), prefill_rows=64, graphs=False)
    e.reset()
    ref = []
    for t in toks:
        ref.append(forward(w, e.st, e.buf, [t])[:1].clone())
        commit(w, e.st, e.buf, 1, 1)
    ref = torch.cat(ref)
    out = {"tokens": len(toks), "ref_nonfinite": int((~torch.isfinite(ref.float())).sum())}
    for rows in windows:
        e.reset()
        got = []
        for at in range(0, len(toks), rows):
            chunk = toks[at:at + rows]
            got.append(forward(w, e.st, e.buf, chunk)[:len(chunk)].clone())
            commit(w, e.st, e.buf, len(chunk), len(chunk))
        got = torch.cat(got)
        out[f"window_{rows}_bitwise_equal"] = bits_equal(got, ref)
        out[f"window_{rows}_torch_equal"] = bool(torch.equal(got, ref))
        out[f"window_{rows}_max_abs_diff"] = rnd((got.float() - ref.float()).abs().max(), 6)
    top = ref.float().argmax(-1).cpu()
    out["greedy_head"] = top[:TRIM].tolist()
    del e
    free()
    return out


def compare(mx: torch.Tensor, ml: torch.Tensor) -> dict:
    V = min(mx.shape[1], ml.shape[1])
    out: dict = {"vocab_cols": [int(mx.shape[1]), int(ml.shape[1])]}
    a, b = mx[:, :V], ml[:, :V]
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    out["nonfinite_mixed"] = int((~fa).sum())
    out["nonfinite_mlx"] = int((~fb).sum())
    out["nan_mixed"], out["nan_mlx"] = int(torch.isnan(a).sum()), int(torch.isnan(b).sum())
    out["inf_mixed"], out["inf_mlx"] = int(torch.isinf(a).sum()), int(torch.isinf(b).sum())
    a0, b0 = torch.where(fa, a, torch.zeros_like(a)), torch.where(fb, b, torch.zeros_like(b))
    cos = torch.nn.functional.cosine_similarity(a0.double(), b0.double(), dim=-1).float()        # [T]
    ta, tb = a0.argmax(-1), b0.argmax(-1)
    agree = (ta == tb)
    out["cos_mean"] = rnd(cos.mean(), 5)
    out["cos_min"] = rnd(cos.min(), 5)
    out["cos_argmin"] = int(cos.argmin())
    out["top1_agree"] = rnd(agree.float().mean(), 4)
    out["first_disagree"] = int((~agree).nonzero()[0]) if bool((~agree).any()) else None
    out["mean_abs_logit_mixed"] = rnd(a0.abs().mean(), 5)
    out["mean_abs_logit_mlx"] = rnd(b0.abs().mean(), 5)
    out["std_logit_mixed"] = rnd(a0.std(), 5)
    out["std_logit_mlx"] = rnd(b0.std(), 5)
    rel = (a0 - b0).norm(dim=-1) / b0.norm(dim=-1).clamp_min(1e-30)
    out["rel_err_mean"] = rnd(rel.mean(), 5)
    # top-1 of MLX: its rank in the mixed logits (0 = mixed agrees)
    rank = (a0 > a0.gather(1, tb[:, None])).sum(-1)
    out["mlx_top1_rank_in_mixed_median"] = int(rank.median())
    out["per_position"] = {"cos": [rnd(v, 4) for v in cos[:TRIM].tolist()],
                           "argmax_mixed": ta[:TRIM].tolist(), "argmax_mlx": tb[:TRIM].tolist(),
                           "rel_err": [rnd(v, 4) for v in rel[:TRIM].tolist()]}
    out["ok"] = bool(out["cos_mean"] >= COS_OK and out["top1_agree"] >= TOP1_OK)
    return out


def run(a, report: dict) -> None:
    from tensorfold.families.qwen4_exp.cuda import nvfp4

    report["torch"] = torch.__version__
    report["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    src = nvfp4.sources(a.mixed)
    report["sources"] = {"experts": str(src.experts), "base": str(src.base)}
    report["is_mixed"] = bool(nvfp4.is_mixed(a.mixed))

    if a.self_check:
        w = None
        with step(report, "load_mixed"):
            w = load_cut(a.mixed, a.layers)
            report["meta_mixed"] = meta(w)
        if w is not None:
            with step(report, "self_check"):
                toks = [int(t) for t in np.random.default_rng(a.seed).integers(0, w.cfg.vocab, size=40)]
                report["self_check"] = self_check(w, toks)
        del w
        free()
        return

    caps = [a.capacity, a.capacity2]
    logits: dict = {"mixed": {}, "mlx": {}}
    plogits: dict = {"mixed": {}, "mlx": {}}
    toks = None
    for name, d in (("mixed", a.mixed), ("mlx", str(src.base))):
        w = None
        with step(report, f"load_{name}"):
            w = load_cut(d, a.layers)
            report[f"meta_{name}"] = meta(w)
        if w is None:
            continue
        if toks is None:
            toks = [int(t) for t in np.random.default_rng(a.seed).integers(0, w.cfg.vocab, size=a.tokens)]
            report["tokens_head"] = toks[:TRIM]
        for cap in caps:
            with step(report, f"run_{name}_{cap}"):
                got: dict = {}
                try:
                    lg, info = one_row_logits(w, cap, toks, a.prefill_rows, got)
                finally:                       # a failed prefill pass keeps the one-row result
                    if "onerow" in got:
                        logits[name][cap] = got["onerow"]
                    if "prefill" in got:
                        plogits[name][cap] = got["prefill"]
                logits[name][cap] = lg
                report.setdefault(f"engine_{name}", {})[str(cap)] = info
        del w
        free()

    report["compare"] = {}
    report["prefill"] = {}
    report["prefill_vs_onerow"] = {}
    for cap in caps:
        with step(report, f"compare_{cap}"):
            if cap in logits["mixed"] and cap in logits["mlx"]:
                report["compare"][str(cap)] = compare(logits["mixed"][cap], logits["mlx"][cap])
            else:
                report["compare"][str(cap)] = {"missing": [n for n in ("mixed", "mlx") if cap not in logits[n]]}
    if a.prefill_rows > 0:
        for cap in caps:
            with step(report, f"compare_prefill_{cap}"):
                if cap in plogits["mixed"] and cap in plogits["mlx"]:
                    report["prefill"][str(cap)] = compare(plogits["mixed"][cap], plogits["mlx"][cap])
                else:
                    report["prefill"][str(cap)] = {"missing": [n for n in ("mixed", "mlx") if cap not in plogits[n]]}
            with step(report, f"prefill_vs_onerow_{cap}"):
                report["prefill_vs_onerow"][str(cap)] = {
                    n: prefill_vs_onerow(plogits[n][cap], logits[n][cap], len(toks), a.prefill_rows)
                    if cap in plogits[n] and cap in logits[n]
                    else {"missing": True} for n in ("mixed", "mlx")}
    # each model against itself across capacities (a capacity effect inside one model)
    report["capacity_self"] = {}
    for name in ("mixed", "mlx"):
        if all(c in logits[name] for c in caps):
            x, y = logits[name][caps[0]], logits[name][caps[1]]
            report["capacity_self"][name] = {"bitwise_equal": bool(torch.equal(x, y)),
                                             "max_abs_diff": rnd((x - y).abs().nan_to_num(float("inf")).max(), 6)}

    c1, c2 = (report["compare"].get(str(c), {}) for c in caps)
    ok1, ok2 = c1.get("ok"), c2.get("ok")
    if ok1 and ok2:
        v = "tables_and_loader_ok"
    elif ok1 is False and ok2 is False:
        v = "mixed_cut_wrong"
    elif ok1 and ok2 is False:
        v = "capacity_dependent"
    elif ok1 and ok2 is None:
        v = "ok_at_capacity_only_capacity2_missing"
    elif ok1 is False and ok2 is None:
        v = "mixed_cut_wrong_capacity2_missing"
    else:
        v = "inconclusive"
    if a.prefill_rows > 0 and v in ("tables_and_loader_ok", "ok_at_capacity_only_capacity2_missing"):
        pc = [report["prefill"].get(str(c), {}) for c in caps]
        bad = [p for p in pc if p.get("ok") is False]
        if bad:
            v = "mixed_prefill_path_wrong"
    report["verdict"] = v


if __name__ == "__main__":
    main()
