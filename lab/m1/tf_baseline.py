#!/usr/bin/env python3
"""M1 baseline: TensorFold 0.3.6.2's grouped expert kernels (MLX affine 4-bit, groups of 32) at Flash Next shapes.

Times gate_up (fused SwiGLU) + down per cell, routing planned outside the timed region, CUDA events, 5 warm-ups,
median of 20. Run inside the tensorfold:0.3.6.2 image. TensorFold is MIT (github.com/ashhart/TensorFold).
"""
import argparse
import json

import torch
from tensorfold.cuda import experts as tx


def mlx_matrix(e, n, k, gs, gen):
    """Random MLX 4-bit arrays: words int32 [E, N, K/8], scales and biases bf16 [E, N, K/gs]."""
    words = torch.randint(-2**31, 2**31 - 1, (e, n, k // 8), dtype=torch.int64, generator=gen).to(torch.int32)
    scales = (torch.rand((e, n, k // gs), generator=gen) * 0.01 + 0.005).to(torch.bfloat16)
    biases = (torch.rand((e, n, k // gs), generator=gen) * -0.04).to(torch.bfloat16)
    return words.cuda(), scales.cuda(), biases.cuda()


def picks_for(rows, topk, n_experts, gen):
    """Distinct experts per row, uniform."""
    return torch.stack([torch.randperm(n_experts, generator=gen)[:topk] for _ in range(rows)]).to(torch.int32).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--rows", default="1,2,4,8,16,32,64,128,512,2048,8192")
    a = ap.parse_args()
    d, i, e, k, gs = a.hidden, a.inter, a.experts, a.topk, 32
    gen = torch.Generator().manual_seed(42)
    ex = tx.make([mlx_matrix(e, i, d, gs, gen), mlx_matrix(e, i, d, gs, gen)], mlx_matrix(e, d, i, gs, gen), gs)
    for rows in [int(r) for r in a.rows.split(",")]:
        prefill = rows >= 512
        plan = tx.Plan(rows, k, e, "cuda", prefill=prefill)
        tx.route(picks_for(rows, k, e, torch.Generator().manual_seed(1234)).contiguous(), plan)
        x = torch.randn((rows, d), generator=torch.Generator().manual_seed(1235)).to(torch.bfloat16).cuda()
        act = torch.empty((rows * k, i), dtype=torch.bfloat16, device="cuda")
        y = torch.empty((rows * k, d), dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")

        def step():
            tx.gate_up(x, ex, plan, act, rows)
            tx.down(act, ex, plan, y, rows)

        for _ in range(10):  # its plans may build per shape on first use; timings were bimodal with 5
            step()
        torch.cuda.synchronize()
        ms = []
        for _ in range(20):
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            step()
            t1.record()
            t1.synchronize()
            ms.append(t0.elapsed_time(t1))
        ms.sort()
        print(json.dumps({"bench": "tf", "rows": rows, "ms": round(ms[10], 4), "ms_min": round(ms[0], 4),
                          "ms_max": round(ms[19], 4), "p10": round(ms[2], 4), "p90": round(ms[17], 4),
                          "samples": [round(m, 4) for m in ms], "plan": "prefill" if prefill else "decode"}),
              flush=True)


if __name__ == "__main__":
    main()
