# M1b phase B, 2026-09-28: contaminated, not a verdict

Code d8acffe. Run by the operator next to tf-serve (not paused; the operator's permissions don't allow stopping it).
A research request (55.7 s, finished 23:33:07 UTC) overlapped almost the whole 64 s run. The probe's first
`cudaMalloc` failed with out-of-memory (unified memory full), so `checks_ok` is false and the verdict line is void.
Median/min spreads of about 2x on both engines (FP4 R=8192: 43.6 / 26.7 ms; TensorFold: 26.4 / 14.5 ms).

What still holds: every correctness check that ran passed (layout; both GEMMs at R = 1, 3, 64, 2048, 8192;
bitwise row invariance over windows 1-200), and quantize+gather at R = 1 fell from 0.58 ms (M1) to 0.021 ms.
Open question for the clean rerun: FP4's R=8192 minimum (26.7 ms) is above M1's median (21.8 ms), so either
contention or a large-R regression in the fused up GEMM.
