# Octojet design (draft, 2026-09-28)

> Historical: the plan written before any engine code. Its numbers describe the starting point, not Octojet.
> Current results, with exact versions: [`benchmarks.md`](benchmarks.md).

## Goal

Serve Qwen3.8 Flash Next on one GB10 system (128 GB unified memory) with native NVFP4 weights and exact
speculative decoding, beating the best existing engine on **seconds per agent step** at equal accuracy.

## Baseline to beat

TensorFold 0.3.6.2 with its MLX 4-bit checkpoint (`results/2026-09-28-gb10.md`): 12-15 s per agent step at
117k-132k context, 71 s cold for 110k, decode 57-79 tok/s. Its limit on GB10: the MLX checkpoint's weights leave
room for only a 154,690-token window.

## Hypothesis

Two parts of an agent step can get faster with NVFP4:

1. **Prompt processing.** It is compute-bound. GB10's tensor cores multiply FP4 directly. TensorFold's MLX path
   dequantizes affine 4-bit weights to 16-bit first, and its EXL3 path re-decodes trellis tiles every 16 rows
   (930-970 tok/s cold). A native FP4 GEMM could beat both. *Unproven*: measure vLLM's NVFP4 prefill on the same
   box before writing kernels.
2. **Decode.** It is memory-bound. NVFP4 is ~4.5 bits per weight (4-bit E2M1 values, one FP8 scale per 16)
   against MLX affine 4-bit's ~5 (FP16 scale and bias per 32), so about 10% fewer bytes per token. Small gain.

Memory is not the argument: a fully NVFP4 checkpoint (experts, n-gram table, dense layers) would be ~95-100 GB,
about 10% smaller than MLX 4-bit and larger than EXL3 3 bpw (80 GB). The mixed NVFP4/FP8 checkpoint we run on
vLLM today is 139 GB.

## Approach (if the measurement supports it)

> Superseded 2026-09-28 by [`superpowers/specs/2026-09-28-octojet-engine-design.md`](superpowers/specs/2026-09-28-octojet-engine-design.md):
> Octojet is now a from-scratch engine aimed at agent-step time; NVFP4 is one lever that M1 decides.

Start as an NVFP4 weight backend in the TensorFold structure (the way its EXL3 backend was added), so the
exactness tests apply unchanged:

1. Loader for NVFP4 safetensors (packed E2M1 pairs, FP8 E4M3 block scales, per-tensor FP32 scale), including
   FP8-kept layers.
2. Row-invariant dense linear and grouped-expert kernels on FP4 tensor-core MMA, with a fixed reduction order
   (a 1-128 row window must reproduce one-row logits bit for bit).
3. A prompt-chunk GEMM for prefill.
4. The n-gram table quantized to NVFP4 or kept host-mapped.

Decide later whether Octojet stays a TensorFold backend (and goes upstream as a PR) or becomes its own engine.

## Success criteria

- Exactness: drafted replies equal `draft: false` replies; the TensorFold CUDA suite passes.
- Accuracy: GSM8K and HumanEval within noise of vLLM NVFP4 on the same questions (paired comparison).
- Speed: agent step at least 20% faster than TensorFold + MLX 4-bit on `bench/agent_bench.py`, same box.
- Context: at least a 200k window on one GB10.

## Open questions

- vLLM NVFP4 prefill speed on GB10 (the first measurement to take).
- Does the FP8 KV cache (`--kv-dtype int8`) change accuracy measurably?
- Upstream-first or separate engine?
- License when the repository goes public.
