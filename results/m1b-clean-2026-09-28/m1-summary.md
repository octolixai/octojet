| rows | TF ms | FP4 ms | speed-up | TF TFLOP/s | FP4 TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.148 | 0.144 | 1.03x | 0.7 | 0.7 |
| 2 | 0.301 | 0.339 | 0.89x | 0.7 | 0.6 |
| 4 | 0.524 | 0.623 | 0.84x | 0.7 | 0.6 |
| 8 | 0.984 | 1.136 | 0.87x | 0.8 | 0.7 |
| 16 | 1.806 | 2.050 | 0.88x | 0.9 | 0.8 |
| 32 | 3.108 | 3.486 | 0.89x | 1.0 | 0.9 |
| 64 | 4.851 | 5.359 | 0.91x | 1.3 | 1.2 |
| 128 | 6.260 | 6.742 | 0.93x | 2.0 | 1.9 |
| 512 | 6.920 | 7.375 | 0.94x | 7.3 | 6.8 |
| 2048 | 7.888 | 9.499 | 0.83x | 25.5 | 21.2 |
| 8192 | 13.911 | 22.416 | 0.62x | 57.9 | 35.9 |

| rows | quant | up | swiglu | down |
|---:|---:|---:|---:|---:|
| 1 | 0.021 | 0.092 | 0.000 | 0.035 |
| 2 | 0.025 | 0.225 | 0.000 | 0.094 |
| 4 | 0.033 | 0.416 | 0.000 | 0.181 |
| 8 | 0.048 | 0.749 | 0.000 | 0.347 |
| 16 | 0.070 | 1.348 | 0.000 | 0.637 |
| 32 | 0.112 | 2.287 | 0.000 | 1.094 |
| 64 | 0.163 | 3.517 | 0.000 | 1.690 |
| 128 | 0.206 | 4.406 | 0.000 | 2.136 |
| 512 | 0.232 | 4.748 | 0.000 | 2.412 |
| 2048 | 0.678 | 5.641 | 0.000 | 3.188 |
| 8192 | 2.837 | 12.482 | 0.000 | 7.126 |

{"check": "toolkit", "device": "NVIDIA GB10", "cc": "12.1", "sms": 48, "runtime": 13030, "driver": 13030, "fp4": true, "fp8": true, "cond": true}
{"check": "peak", "format": "bf16", "tflops": 118.3, "ms": 1.743}
{"check": "peak", "format": "fp8", "tflops": 236.6, "ms": 1.743}
{"check": "peak", "format": "fp4bs", "tflops": 473.0, "ms": 1.743}
{"check": "layout", "ok": true, "regs_ok": true, "a_found": true, "b_found": true, "combined_ok": true, "a_sel": 0, "a_pat": 0, "b_sel": 0}
{"check": "cond_graph", "ok": true, "iters": 1000, "us_per_iter": 3.54}
{"check": "layout", "ok": true, "regs_ok": true, "a_found": true, "b_found": true, "combined_ok": true, "a_sel": 0, "a_pat": 0, "b_sel": 0}
{"check": "gemm", "rows": 1, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00133}
{"check": "gemm", "rows": 3, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00144}
{"check": "gemm", "rows": 64, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00144}
{"check": "gemm", "rows": 2048, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00179}
{"check": "gemm", "rows": 8192, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00167}
{"check": "row_invariance", "ok": true, "windows": "1..200", "first_bad_window": 0}

{"prompt_speedup": 0.621, "pass": false, "checks_ok": true, "reason": "min prompt speed-up 0.621x < 1.5x", "prompt_speedup_vs_tf_min": 0.607, "pass_vs_tf_min": false}
