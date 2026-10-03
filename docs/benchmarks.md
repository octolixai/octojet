# Octojet benchmarks

The single source for every number quoted in the GitHub README, the Hugging Face model card and launch posts. Each
table names the exact version of every engine and checkpoint it compares. Raw run files live in `results/`.

Where a download revision was not recorded at the time, the table gives the revision whose `config.json` matches the
files used and says so ("config-matched").

## Test machine and software

| Item | Value |
|---|---|
| Machine | NVIDIA DGX Spark (GB10, compute capability 12.1, 128 GB unified memory), one box |
| OS / driver | Ubuntu 24.04.5 LTS, kernel 6.17.0-1032-nvidia, NVIDIA driver 580.173.02 (CUDA 13.0 driver; containers use CUDA 13.3 forward compatibility) |
| Octojet container | `tensorfold:0.3.6.2` image with Octojet's engine on `PYTHONPATH`: torch 2.13.0a0+9186a08b2c.nv26.07, triton 3.7.1, CUDA 13.3 |
| Upstream TensorFold container | `nvcr.io/nvidia/pytorch:26.07-py3`, as upstream's README instructs: the same torch, triton and CUDA versions |
| Model | Qwen/Qwen3.8-Flash-Next (Qwen Community License 1.0) |

## Versions compared

| Name in tables | Engine | Checkpoint |
|---|---|---|
| Octojet | Octojet at e53e17d (run 1), 78c1215 (run 2) and 542715f (run 3; engine identical to the published release candidate 230c695) | Octojet mixed NVFP4 as served in production: routed experts from `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ 7b719225242aacd3dbd3f9407468c2ee9a9d2594 (read from a locally derived `-fp8hybrid` copy of that revision, made by blazux/qwen3.8-Flash-DGX @ bd60fcb, whose shared experts are FP8), everything else from `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` main @ 2b170fa6309d5d1ee380b35636075fac7945f286 (config-matched; download revision not recorded). The published checkpoint takes its shared experts from RadixArk's bf16 copy instead and is re-verified separately |
| TF 0.6.2 + lil, TF 0.6.2 + RadixArk (run 3) | TensorFold v0.6.2 (tag commit 56e2e3ec), the latest release on 2 Oct 2026; default mode and `--precision full` | the two checkpoints below |
| TF 0.6 + lil | TensorFold v0.6.0 (tag commit c4646171139ee8a3c38103eaa1699dad226ec12b), the latest release on 1 Oct 2026 | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision 7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd (99 GB on disk) |
| TF 0.6 + RadixArk | TensorFold v0.6.0 (same commit) | `RadixArk/Qwen3.8-Flash-Next-NVFP4` revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594, as published (126 GB on disk) |

Common settings for every server: `--kv-dtype int8 --parallel 3`, text only, drafting on (each engine's default), the
GPU exclusive (no other process), page cache dropped before each start. Prompts are the same text for every engine.
Times are measured on the client, from sending the request to the first streamed text.

How to reproduce: `bench/spark/compare-upstream.sh` (run 1: full mode, accuracy off; run 2: `MODE=short`,
`CONFIGS="ours up-lil"`), timing via `bench/cmp_probe.py`, decode via `engine/tools/bench_openai.py`, agent steps via
`bench/agent_bench.py`, accuracy via `bench/acc_eval.py` + `bench/run_humaneval.sh`. Details and notes:
`results/2026-10-01-compare-upstream.md`.

## Head-to-head against TensorFold v0.6.2 on all three published checkpoints (run 4, 3 Oct 2026, ~17:52-18:45 UTC)

Octojet 3f49925 (F8: 1,024-row prompt chunks while other streams decode, prompt chunks on their own CUDA stream; every
token identical to earlier builds) against TensorFold v0.6.2 (56e2e3ec) on `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` @
2b170fa6 (the checkpoint of MiaAI-Lab's single-Spark TensorFold recipe, whose page reports 63.6 tok/s prose decode on
TensorFold v0.6.1) and, with `--precision full`, on the two NVFP4 checkpoints. Same settings as the other runs (int8 KV,
`--parallel 3`, text only, GPU exclusive, client-side timing); accuracy not rerun.

| Name in tables | Engine | Checkpoint |
|---|---|---|
| Octojet | Octojet at 3f49925 (run 4) | Octojet mixed NVFP4 as served in production (routed experts from `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ 7b719225, everything else from `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` @ 2b170fa6) |
| TF 0.6.2 + MLX 4-bit | TensorFold v0.6.2 (56e2e3ec), the latest release on 3 Oct 2026 | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` @ 2b170fa6, the checkpoint of MiaAI-Lab's single-Spark TensorFold recipe |
| TF 0.6.2 + lil | TensorFold v0.6.2, `--precision full` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ 7c4f1bc1 |
| TF 0.6.2 + RadixArk | TensorFold v0.6.2, `--precision full` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ 7b719225, as published |

| Measure | Octojet | TF 0.6.2 + MLX 4-bit | TF 0.6.2 + lil | TF 0.6.2 + RadixArk |
|---|---:|---:|---:|---:|
| Cold 32k-token prompt, first token | 28.3 * | 12.5 | 19.6 | 22.5 |
| Cold 128k-token prompt, first token | 67.6 | 57.1 | 84.7 | 95.5 |
| Cold 210k-token prompt, first token | 112.7 | 105.6 | 146.0 | 164.5 |
| Cold 71k-token prompt, first token | 35.3 | 30.8 | 44.9 | 51.1 |
| Prompt sharing 69k of those 71k tokens | 3.3 | 29.5 | 45.0 | 49.3 |
| The 71k prompt resent | 0.13 | 0.19 | 0.20 | 0.21 |
| Decode, code prompt, temperature 1 (tok/s) | 64.3 | 70.5 | 49.8 | 39.3 |
| Decode, chat prompt, temperature 1 (tok/s) | 62.0 | 56.3 | 41.4 | 31.2 |
| Decode, code prompt, greedy (tok/s) | 54.5 | 63.8 | 50.9 | 42.9 |
| Decode, chat prompt, greedy (tok/s) | 88.2 | 60.2 | 39.8 | 34.7 |
| Agent: 80k-token first turn, first token | 52.0 | 47.0 | 67.4 | 74.8 |
| Agent: +5k-token follow-up, first token (median of 3) | 4.46 | 4.12 | 5.27 | 6.80 |
| Agent: +5k-token follow-up, whole step (median of 3) | 12.1 | 12.5 | 14.3 | 23.3 |
| A short prompt sent while a cold 128k prompt fills, first token | 3.27 | 2.85 | 3.66 | 4.38 |
| A resend sent while a cold 128k prompt fills, first token | 0.77 | 3.83 | 2.66 | 3.47 |
| Longest pause of a live reply while a 128k prompt arrives | 1.35 | 1.29 | 1.61 | 1.79 |
| Typical pause of a live reply while a 128k prompt arrives (median gap) | 0.70 | 1.09 | 1.43 | 1.61 |
| That 128k prompt's first token while the reply streams | 90.2 | 70.7 | 91.8 | 102.4 |
| Startup memory estimate (GiB) | 87.2 | 84.3 | 81.2 | 97.4 |
| Context windows | 3 × 262,144 reserved | up to 3 × 262,144 | up to 3 × 262,144 | up to 3 × 262,144 |

\* Octojet's first long prompt after a server start pays a one-time cost (13-15 s here); its cold 32k prompt took
15.6-18.2 s in every other run. TensorFold warms its prompt kernels at startup.

Reading: TensorFold on the MLX 4-bit checkpoint is the fastest upstream setup, ahead of Octojet on cold prompts
(7-18%), code decode (9-15%) and memory (3 GiB); Octojet leads on prompts that repeat or share a prefix (9x on the
variant, 3.5x on a resend during a fill), on chat decode, and on how smoothly live replies stream during a fill.

## Head-to-head against TensorFold v0.6.2 (run 3, 2 Oct 2026, ~17:10-18:25 UTC)

Octojet 542715f (branch f7: requests admitted while a prompt fills, forks beside a decoding stream, turn-start
checkpoints; every token identical to the previous build) against TensorFold v0.6.2 (tag commit 56e2e3ec), the
latest release on 2 Oct 2026, on both checkpoints, in its default mode and with `--precision full` (16-bit
activations against 4-bit weights, as Octojet runs). All five servers logged bf16 prompt activations on this machine,
and default and full agree within 3%. Same settings as above; accuracy not rerun (ACC=0). Octojet's load time is with
its packed-table cache already built.

| Measure | Octojet | TF 0.6.2 + lil | TF 0.6.2 + lil, full | TF 0.6.2 + RadixArk | TF 0.6.2 + RadixArk, full |
|---|---:|---:|---:|---:|---:|
| Cold 32k-token prompt, first token | 15.6 | 19.9 | 19.6 | 22.4 | 22.3 |
| Cold 128k-token prompt, first token | 68.8 | 87.2 | 84.7 | 96.2 | 95.0 |
| Cold 210k-token prompt, first token | 124.0 | 150.1 | 146.2 | 165.3 | 164.4 |
| Cold 71k-token prompt, first token | 32.2 | 45.0 | 45.0 | 51.8 | 51.1 |
| Prompt sharing 69k of those 71k tokens | 3.2 | 45.0 | 45.1 | 49.8 | 49.2 |
| The 71k prompt resent | 0.13 | 0.32 | 0.19 | 0.21 | 0.19 |
| Decode, code prompt, temperature 1 (tok/s) | 64.3 | 50.1 | 50.1 | 39.9 | 39.7 |
| Decode, chat prompt, temperature 1 (tok/s) | 62.9 | 41.8 | 41.7 | 31.4 | 31.4 |
| Decode, code prompt, greedy (tok/s) | 54.9 | 51.1 | 51.0 | 42.9 | 42.9 |
| Decode, chat prompt, greedy (tok/s) | 88.6 | 40.1 | 39.9 | 34.7 | 34.7 |
| Agent: 80k-token first turn, first token | 53.2 | 67.5 | 67.4 | 75.0 | 74.4 |
| Agent: +5k-token follow-up, first token (median of 3) | 4.39 | 5.27 | 5.27 | 6.46 | 6.67 |
| Agent: +5k-token follow-up, whole step (median of 3) | 11.9 | 14.3 | 14.3 | 22.8 | 23.5 |
| A short prompt sent while a cold 128k prompt fills, first token | 3.16 | 2.95 | 3.62 | 4.07 | 4.52 |
| A resend sent while a cold 128k prompt fills, first token | 1.12 | 1.94 | 2.62 | 3.15 | 3.60 |
| That 128k prompt's own first token | 63.5 | 89.6 | 86.5 | 96.9 | 94.5 |
| Longest pause of a live reply while a 128k prompt arrives | 2.52 | 1.61 | 1.61 | 1.79 | 1.78 |
| Startup memory estimate | 87.2 GiB | 81.2 GiB | 81.2 GiB | 97.4 GiB | 97.4 GiB |
| Load time | 66 s | 201 s | 82 s | 92 s | 91 s |

## Head-to-head: speed (run 1, 1 Oct 2026 19:06-19:46 UTC)

Seconds unless stated; lower is better for times, higher for decode.

| Measure | Octojet | TF 0.6 + lil | TF 0.6 + RadixArk |
|---|---:|---:|---:|
| Cold 128k-token prompt, first token | 68.9 | 89.5 | 96.0 |
| Cold 210k-token prompt, first token | 117.9 | 152.3 | 166.4 |
| Cold 71k-token prompt, first token | 32.4 | 46.1 | 52.6 |
| Variant sharing 69k of those 71k tokens | 3.2 | 47.4 | 50.0 |
| Decode, code prompt, temperature 1 (tok/s) | 64.0 | 50.5 | 40.4 |
| Decode, chat prompt, temperature 1 (tok/s) | 62.0 | 42.0 | 31.4 |
| Decode, code prompt, greedy (tok/s) | 55.4 | 51.2 | 43.4 |
| Decode, chat prompt, greedy (tok/s) | 89.6 | 40.2 | 35.1 |
| Agent: 80k-token first turn, first token | 53.5 | 68.3 | 77.0 |
| Agent: +5k-token follow-up, first token (median of 3) | 4.29 | 5.40 | 6.29 |
| Agent: +5k-token follow-up, whole step (median of 3) | 11.8 | 14.9 | 22.0 |
| Longest pause of a live reply while a 128k prompt arrives | 2.34 | 1.64 | 2.26 |
| Startup memory estimate | 87.2 GiB | 81.1 GiB | 97.4 GiB |
| Context windows | 3 × 262,144 reserved | up to 3 × 262,144 | up to 3 × 262,144 |
| Load time | 150 s | 205 s | 87 s |

Run 1's cold 32k result (Octojet 27.7 s, lil 21.4 s, RadixArk 22.5 s) is superseded by run 2: Octojet's first long
prompt after start paid a one-time cost. Run 1's identical-resend result (Octojet 3.2 s) exposed a bug fixed in run 2.

## Head-to-head: follow-up and accuracy (run 2, 1 Oct 23:45 - 2 Oct 00:17 UTC)

Octojet 78c1215 against TF 0.6 + lil; greedy, thinking off, the same requests for both.

| Measure | Octojet | TF 0.6 + lil |
|---|---:|---:|
| Cold 32k-token prompt, first token | 18.2 s | 19.8 s |
| A second, different cold 32k prompt | 16.9 s | 19.8 s |
| Cold 71k prompt / variant sharing 69k / identical resend | 37.8 / 4.9 / 0.12 s | 45.6 / 45.6 / 0.18 s |
| GSM8K, first 250 test problems | 0.980 (245) | 0.984 (246) |
| HumanEval pass@1, 164 programs | 0.945 (155) | 0.963 (158) |

## The published checkpoint's own checks (2 Oct 2026 01:16-02:07 UTC)

The Hugging Face checkpoint, built by `bench/spark/release-build.sh` from RadixArk @ 7b719225 and Vontra @ 2b170fa6,
served by Octojet with the same settings (int8 KV, `--parallel 3`, text only). It differs from the production build in
the tables above only in its shared experts (RadixArk's bf16 copy instead of an FP8-derived one).

| Check | Result |
|---|---|
| Every tensor against its source | 224,211 tensors, byte for byte identical; nothing dropped that is needed, nothing extra |
| Routers, routed and shared experts against the sources | routers equal; 48 of 48 layers agree for both |
| Drafted vs serial, concurrent vs solo | identical token ids |
| Startup | 87.2 GiB estimate, 3 × 262,144 windows |
| GSM8K (250) | 242 (production build 245; 4 right only in production, 1 only here) |
| HumanEval pass@1 (164) | 154 (production build 155; 2 only in production, 1 only here) |
| Size | 106 GB (34.98 GiB base tensors, 63.72 GiB experts) |

## Octojet against the MLX 4-bit checkpoint (29 Sep 2026)

Same engine family (TensorFold kernels), int8 KV, measured in Octojet's F1/F1.2 runs (`results/2026-09-29-f1.md`).

| Measure | Octojet mixed NVFP4 | TensorFold + Vontra MLX 4-bit |
|---|---:|---:|
| Windows at `--parallel 3` | 3 × 262,144 | 3 × 174,961 |
| Windows at `--parallel 4` | 4 × 227,687 | 4 × 120,136 |
| Resident weights (startup estimate, same caches) | 80.16 GiB | 87.19 GiB |
| GSM8K (250) | 244 | 244 |
| HumanEval (164) | 155 | 154 |

## Before Octojet: the starting point (28 Sep 2026)

**None of these rows is Octojet.** This is the survey that started the project: the engines available on one Spark
before any Octojet code existed, including TensorFold 0.3.6.2, the version Octojet forked. From
`results/2026-09-28-gb10.md`. Its prompts are a different size from the tables above (a 110k prompt, +7k-token agent
steps), so compare within this table only. For Octojet's current speed see the head-to-head tables; for how far
Octojet moved from its own starting point see "Octojet production history" below. vLLM settings: `--gpu-memory-utilization 0.80 --max-model-len 200000 --max-num-seqs 8`,
prefix caching and chunked prefill on, piecewise CUDA graphs; one live request shared the box during its decode cells.

| Engine + checkpoint | Cold 110k prompt | Agent step (+7k tokens, ~600 out) | GSM8K / HumanEval |
|---|---:|---:|---|
| vLLM 0.1.dev20073+g8e685d198 (image built from blazux/qwen3.8-Flash-DGX @ bd60fcb1b492ca920f74df7462f05da7b6d98f73; torch 2.13.0+cu130) + an NVFP4/FP8 checkpoint derived from `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ 7b719225 by that repo's tooling, MTP 2 | 74 s | 19-28 s | 0.968 / 0.945 |
| TensorFold 0.3.6.2 + `turboderp/Qwen3.8-Flash-Next-exl3` branch 3.05bpw_h5_ng5 @ 69e33439ae950f17bcbe95c98f117d80f759ab6d (config-matched) | 162 s | 18-25 s | 0.972 / 0.957 |
| TensorFold 0.3.6.2 + `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` main @ 2b170fa6 (config-matched) | 71 s | 12-15 s | 0.976 / 0.939 |

## Octojet production history (server-side times)

| Change | Before | After |
|---|---|---|
| Warm start (packed-table cache) | ~355 s | 138 s |
| Cold 32k / 128k / 210k prompt | 15.2 / 68.5 / 227 s | 13.8 / 62.5 / 111 s |
| Identical prompt resent | 36.7 s | ~5 ms |
| Live reply's pause while a 128k prompt arrives | ~68 s | ~2.5 s |
| Variant of a 71k prompt (shares 69k) | 35.7 s | 2.2 s |

## References

Everything Octojet is built on, compared against or measured with. The model card and the README carry this list.

**Model**
- Qwen3.8 Flash Next: https://huggingface.co/Qwen/Qwen3.8-Flash-Next (Qwen Community License 1.0)

**Checkpoints Octojet's checkpoint is built from**
- RadixArk NVFP4 export (NVIDIA Model Optimizer): https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4 @ 7b719225242aacd3dbd3f9407468c2ee9a9d2594
- Vontra MLX 4-bit with MTP: https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP @ 2b170fa6 (config-matched)

**Checkpoints compared against**
- local-inference-lab NVFP4: https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4 @ 7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
- RadixArk NVFP4, as published (above)
- turboderp EXL3: https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3, branch 3.05bpw_h5_ng5 @ 69e33439 (config-matched)
- NVIDIA NVFP4 (cited for context, not benchmarked): https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4

**Engines**
- TensorFold, which Octojet forks (v0.3.6.2, commit 71377a5; MIT up to v0.5, Apache-2.0 from v0.6.0): https://github.com/ashhart/TensorFold. Compared against at v0.6.0 (c4646171). Ported from upstream: vision (v0.3.6.3), tiled sparse-attention block selection above 131,072 keys (a2dba9c), prompts inside decode rounds (d23087c, 68c6e35).
- vLLM: https://github.com/vllm-project/vllm, as built by https://github.com/blazux/qwen3.8-Flash-DGX @ bd60fcb1b492ca920f74df7462f05da7b6d98f73
- ExLlamaV3 (EXL3 format; int8/int4 KV cache scheme): https://github.com/turboderp-org/exllamav3

**Patches and code Octojet uses**
- MiaAI-Lab's TensorFold patches for Flash Next vision on one Spark (patches 0008, 0009 @ a3aa898): https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
- Hugging Face Transformers (Qwen3-VL image and video preprocessing, Apache-2.0): https://github.com/huggingface/transformers
- PyTorch and Triton, from NVIDIA's container: https://github.com/pytorch/pytorch, https://github.com/triton-lang/triton
- Inherited from TensorFold (see `engine/THIRD_PARTY_NOTICES.md`): MLX and mlx-lm, mlx-vlm, oMLX, z-lab/dflash

**Benchmarks**
- GSM8K: https://github.com/openai/grade-school-math (first 250 test problems)
- HumanEval: https://github.com/openai/human-eval (all 164 programs, run in a container with no network)
