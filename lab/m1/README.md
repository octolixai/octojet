# M1 kernel lab

Answers two questions on one GB10 (spec: `docs/superpowers/specs/2026-09-28-octojet-engine-design.md`, 6.1):

1. Is a W4A4 NVFP4 expert pipeline (block-scaled `mma.sync`) at least ~1.5x faster than TensorFold 0.3.6.2's
   affine 4-bit expert kernels for prompt-sized batches (R ≥ 512 rows) at Flash Next's shapes?
2. Do conditional CUDA graphs (device-side while loops) work on sm_121a in this toolkit?

| Program | Measures |
|---|---|
| `probe.cu` | toolkit and device; peak `mma.sync` TFLOP/s for bf16, fp8 e4m3, block-scaled fp4; discovery of the block-scale lane mapping (`fp4mma.cuh`); a 1,000-iteration conditional-graph while loop |
| `fp4_bench.cu` | quantize+gather, gate/up GEMM, SwiGLU+quantize, down GEMM, total and per kernel; discovers the scale mapping itself at startup; `--check` adds CPU checks of both GEMMs (R = 1, 3, 64, 2048, 8192) and a bitwise row-invariance check with the row moved through every position of a 128-row item and into the next item (windows 1-200) |
| `tf_baseline.py` | TensorFold's `gate_up` + `down` on random MLX 4-bit weights, same shapes and cells (TensorFold, MIT) |
| `compare.py` | table, peak numbers, verdict |

Run (operator, on the GB10): see the header of `run.sh`. About 10 minutes of GPU time; peak GPU memory about 5 GB
(fp4_bench ~3 GB incl. a 1.4 GB host copy of the packed weights on unified memory, plus ~1.3 GB during the
R = 8192 check for h and host copies; the TensorFold baseline ~4-5 GB).

CPU tests (anywhere): `clang++ -std=c++17 -O1 -Wall -Wextra -o /tmp/test_ref test_ref.cpp && /tmp/test_ref` and
`python3 -m pytest test_compare.py -q`.

Also written: `m1-timeline.txt` (UTC start of each stage) and `m1-fp4-repeat.jsonl` (FP4 timings again after the
TensorFold baseline). A large gap between the two FP4 passes means the GPU was shared during the run.

Reading the result:
- `layout`: `regs_ok:false` → the A/B/C register layout itself is wrong (fix `pack_*` in `ref.h`); `a_found` or
  `b_found` false → none of the candidate scale-lane mappings in `ref.h` matched (add candidates). Either way no
  fp4 correctness result is trustworthy until it reads `ok:true`; timing is unaffected.
- `fp4_parts` and the TFLOP/s columns show where FP4 time goes and how far it is from the `peak` numbers.
- Verdict pass → M2 uses NVFP4. Fail with the fp4 pipeline far below the fp4 peak (see `peak`) while fp4 peak ≥ 3x
  bf16 peak → report as kernel-limited; the owner decides whether to tune further. Fail otherwise → MLX 4-bit.
