| rows | TF ms | FP4 ms | speed-up | TF TFLOP/s | FP4 TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.235 | 0.851 | 0.28x | 0.4 | 0.1 |
| 2 | 0.385 | 1.015 | 0.38x | 0.5 | 0.2 |
| 4 | 0.683 | 1.313 | 0.52x | 0.6 | 0.3 |
| 8 | 1.199 | 1.907 | 0.63x | 0.7 | 0.4 |
| 16 | 4.612 | 2.830 | 1.63x | 0.3 | 0.6 |
| 32 | 6.434 | 4.397 | 1.46x | 0.5 | 0.7 |
| 64 | 11.248 | 6.347 | 1.77x | 0.6 | 1.0 |
| 128 | 15.493 | 7.803 | 1.99x | 0.8 | 1.6 |
| 512 | 14.152 | 8.646 | 1.64x | 3.6 | 5.8 |
| 2048 | 15.149 | 10.338 | 1.47x | 13.3 | 19.5 |
| 8192 | 26.931 | 21.791 | 1.24x | 29.9 | 37.0 |

| rows | quant | up | swiglu | down |
|---:|---:|---:|---:|---:|
| 1 | 0.580 | 0.069 | 0.164 | 0.039 |
| 2 | 0.563 | 0.193 | 0.166 | 0.096 |
| 4 | 0.560 | 0.384 | 0.184 | 0.188 |
| 8 | 0.638 | 0.728 | 0.193 | 0.355 |
| 16 | 0.640 | 1.331 | 0.213 | 0.648 |
| 32 | 0.746 | 2.286 | 0.244 | 1.125 |
| 64 | 0.796 | 3.554 | 0.266 | 1.732 |
| 128 | 0.834 | 4.510 | 0.281 | 2.183 |
| 512 | 0.887 | 4.996 | 0.328 | 2.436 |
| 2048 | 1.053 | 5.427 | 1.052 | 2.800 |
| 8192 | 3.183 | 8.016 | 4.202 | 4.907 |

{"check": "toolkit", "device": "NVIDIA GB10", "cc": "12.1", "sms": 48, "runtime": 13030, "driver": 13000, "fp4": true, "fp8": true, "cond": true}
{"check": "peak", "format": "bf16", "tflops": 117.5, "ms": 1.754}
{"check": "peak", "format": "fp8", "tflops": 235.0, "ms": 1.755}
{"check": "peak", "format": "fp4bs", "tflops": 469.6, "ms": 1.756}
{"check": "layout", "ok": true, "regs_ok": true, "a_found": true, "b_found": true, "combined_ok": true, "a_sel": 0, "a_pat": 0, "b_sel": 0}
{"check": "cond_graph", "ok": true, "iters": 1000, "us_per_iter": 5.89}
{"check": "layout", "ok": true, "regs_ok": true, "a_found": true, "b_found": true, "combined_ok": true, "a_sel": 0, "a_pat": 0, "b_sel": 0}
{"check": "gemm", "rows": 1, "ok": true, "max_rel_up": 0, "max_rel_down": 0.000907}
{"check": "gemm", "rows": 3, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00114}
{"check": "gemm", "rows": 64, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00184}
{"check": "gemm", "rows": 2048, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00166}
{"check": "row_invariance", "ok": true, "windows": "1..128", "first_bad_window": 0}

{"prompt_speedup": 1.236, "pass": false, "checks_ok": true, "reason": "min prompt speed-up 1.236x < 1.5x"}
