"""F14: time the prompt DeltaNet kernels alone at a Flash Next chunk's shapes (CUDA; run in the serving container).

Prints ms a launch for the chain (staged and double-buffered, OCTOJET_GDN_CHAIN_AHEAD) and for the front and back
kernels, whose code is the same as upstream TensorFold 0.6.2's. Compare with the per-launch averages of an nsys capture
(kern_compare's totals / launches): a kernel that runs faster here than inside the server is slowed by its
surroundings (clocks, memory traffic beside it), not by its code.

    PYTHONPATH=engine/src python bench/f14_gdn_bench.py [--rows 4096] [--iters 50]
"""

from __future__ import annotations

import argparse


def _time(fn, iters: int) -> float:
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=4096, help="rows a chunk (Flash Next prompts: 4096)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    import torch

    from tensorfold.cuda.kernels import gdn as shared
    from tensorfold.families.qwen4_exp.cuda import gdn, gdn_io

    dev, rows = "cuda", args.rows
    gen = torch.Generator(device=dev).manual_seed(14)
    p = (torch.randn((rows, gdn.PW), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    cs = (torch.randn((3, gdn.CONV), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    cw = (torch.randn((gdn.CONV, 4), generator=gen, device=dev) * 0.3).to(torch.bfloat16)
    state = torch.randn((gdn.NV, gdn.DV, gdn.DK), generator=gen, device=dev) * 0.05
    a_log = torch.randn((gdn.NV,), generator=gen, device=dev) * 0.5
    dt = torch.randn((gdn.NV,), generator=gen, device=dev) * 0.5
    nw = (1 + 0.1 * torch.randn((gdn.DV,), generator=gen, device=dev)).to(torch.bfloat16)
    ptrs = torch.tensor([cs.data_ptr()], dtype=torch.int64, device=dev)
    win = torch.tensor([[j + t for t in range(4)] for j in range(rows)], dtype=torch.int32,   # taps < 3: conv state
                       device=dev)
    sid = torch.zeros((rows,), dtype=torch.int32, device=dev)
    q, k, v, g, beta = gdn_io.front(p, ptrs, sid, win, cw, a_log, dt, gdn.NK)
    last = torch.empty_like(state)
    out = torch.empty((rows, gdn.NV * gdn.DV), dtype=torch.bfloat16, device=dev)
    xs = torch.empty((rows, gdn.NV * gdn.DV // 32), dtype=torch.float32, device=dev)
    y0, ran0 = shared.chain_form(q, k, v, g, beta, state, last, ahead=False)
    l1 = torch.empty_like(state)
    y1, ran1 = shared.chain_form(q, k, v, g, beta, state, l1, ahead=True)
    same = torch.equal(y0.view(torch.int16), y1.view(torch.int16)) and torch.equal(last.view(torch.int32),
                                                                                  l1.view(torch.int32))
    rows_ms = {
        "chain staged": _time(lambda: shared.chain_form(q, k, v, g, beta, state, last, ahead=False), args.iters),
        "chain ahead": _time(lambda: shared.chain_form(q, k, v, g, beta, state, last, ahead=True), args.iters),
        "front": _time(lambda: gdn_io.front(p, ptrs, sid, win, cw, a_log, dt, gdn.NK), args.iters),
        "back": _time(lambda: gdn_io.back(y0, p, nw, 1e-6, out, xs), args.iters),
    }
    print(f"rows {rows}; {torch.cuda.get_device_name(0)}; ahead ran: {ran1} (staged: {not ran0}); same bits: {same}")
    for name, ms in rows_ms.items():
        print(f"{name:14s} {ms:8.3f} ms a launch")
    staged, ahead = rows_ms["chain staged"], rows_ms["chain ahead"]
    print(f"chain ahead / staged: {ahead / staged:.3f}")
    return 0 if same and ran1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
