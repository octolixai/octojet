| rows | TF ms | FP4 ms | speed-up | TF TFLOP/s | FP4 TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.245 | 0.229 | 1.07x | 0.4 | 0.4 |
| 2 | 0.388 | 0.364 | 1.06x | 0.5 | 0.5 |
| 4 | 0.684 | 0.772 | 0.89x | 0.6 | 0.5 |
| 8 | 1.183 | 1.378 | 0.86x | 0.7 | 0.6 |
| 16 | 4.761 | 4.956 | 0.96x | 0.3 | 0.3 |
| 32 | 6.160 | 7.447 | 0.83x | 0.5 | 0.4 |
| 64 | 10.920 | 13.723 | 0.80x | 0.6 | 0.5 |
| 128 | 14.792 | 13.854 | 1.07x | 0.9 | 0.9 |
| 512 | 14.095 | 12.919 | 1.09x | 3.6 | 3.9 |
| 2048 | 14.992 | 16.905 | 0.89x | 13.4 | 11.9 |
| 8192 | 26.438 | 43.636 | 0.61x | 30.5 | 18.5 |

| rows | quant | up | swiglu | down |
|---:|---:|---:|---:|---:|
| 1 | 0.021 | 0.078 | 0.000 | 0.035 |
| 2 | 0.031 | 0.245 | 0.000 | 0.110 |
| 4 | 0.052 | 0.490 | 0.000 | 0.229 |
| 8 | 0.064 | 0.859 | 0.000 | 0.407 |
| 16 | 0.108 | 1.532 | 0.000 | 2.845 |
| 32 | 0.175 | 5.066 | 0.000 | 1.333 |
| 64 | 0.233 | 7.114 | 0.000 | 4.313 |
| 128 | 0.201 | 9.270 | 0.000 | 4.453 |
| 512 | 0.238 | 9.579 | 0.000 | 4.688 |
| 2048 | 0.675 | 12.846 | 0.000 | 5.530 |
| 8192 | 5.379 | 25.401 | 0.000 | 14.470 |

{"check": "toolkit", "device": "NVIDIA GB10", "cc": "12.1", "sms": 48, "runtime": 13030, "driver": 13030, "fp4": true, "fp8": true, "cond": true}
{"check": "error", "ok": false, "where": "cudaMalloc(&sink, 4)", "msg": "out of memory"}
{"check": "layout", "ok": true, "regs_ok": true, "a_found": true, "b_found": true, "combined_ok": true, "a_sel": 0, "a_pat": 0, "b_sel": 0}
{"check": "gemm", "rows": 1, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00133}
{"check": "gemm", "rows": 3, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00144}
{"check": "gemm", "rows": 64, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00144}
{"check": "gemm", "rows": 2048, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00179}
{"check": "gemm", "rows": 8192, "ok": true, "max_rel_up": 0, "max_rel_down": 0.00167}
{"check": "row_invariance", "ok": true, "windows": "1..200", "first_bad_window": 0}

{"prompt_speedup": 0.606, "pass": false, "checks_ok": false, "reason": "failed checks: error", "prompt_speedup_vs_tf_min": 0.332, "pass_vs_tf_min": false}
