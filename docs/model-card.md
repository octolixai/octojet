---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - dgx-spark
  - gb10
  - moe
  - speculative-decoding
  - octojet
---

<!-- Owner-approved draft (2026-10-02): repo octolix/Qwen3.8-Flash-Next-Octojet-NVFP4; numbers as measured
(docs/benchmarks.md). The `octojet` command ships with Octojet 0.1.0. Upload as README.md, with docs/images/*.png as assets/, docs/release-checkpoint/NOTICE, and LICENSE copied verbatim from the Vontra source. -->

# Qwen3.8 Flash Next, Octojet NVFP4

Qwen3.8 Flash Next for one NVIDIA DGX Spark (GB10), packaged for the [Octojet](https://github.com/octolixai/octojet)
inference engine. Routed experts are native FP4 (NVFP4), which GB10's tensor cores compute in directly; everything
else is 4-bit affine. On one Spark it serves three conversations of up to 262,144 tokens each, with image input,
and it is the fastest setup we measured for agent-style work: long prompts, follow-up turns and drafted decoding.

**This checkpoint runs on Octojet only.** It combines two formats that vLLM, SGLang and Transformers do not load
together. If you need one of those engines, use the checkpoints listed under References.

## Quick start

```bash
# NVIDIA's PyTorch container supplies CUDA, PyTorch and Triton
docker run -it --gpus all --ipc=host --network host nvcr.io/nvidia/pytorch:26.07-py3
python -m pip install "git+https://github.com/octolixai/octojet.git#subdirectory=engine"
octojet serve octolix/Qwen3.8-Flash-Next-Octojet-NVFP4 --kv-dtype int8 --parallel 3 --vision
```

The server speaks the OpenAI API at `http://127.0.0.1:8080/v1`. The first start packs the expert tables (a few
minutes) and caches them; later starts take about 150 s.

## Measured results

One DGX Spark, the same prompts and settings for every engine: int8 KV cache, three streams, text only, the GPU to
itself, times measured on the client. Full tables, settings and reproduction scripts:
[benchmarks](https://github.com/octolixai/octojet/blob/main/docs/benchmarks.md).

Compared on 4 October 2026 against TensorFold v0.6.5 (609ca419), its latest release, on the three published Flash
Next checkpoints it serves: `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` @ 2b170fa6 (the checkpoint of MiaAI-Lab's single-Spark
recipe), and with `--precision full` `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ 7c4f1bc1 and
`RadixArk/Qwen3.8-Flash-Next-NVFP4` @ 7b719225.

In short: Octojet is far faster when prompts repeat or share a long prefix, keeps live replies smoother while a long
prompt arrives, finishes agent follow-up steps sooner and decodes chat faster; TensorFold on the MLX 4-bit checkpoint
reads cold prompts sooner (by 3 s at 32k, 1-3% at 71k-210k), decodes code 7-9% faster and uses about 2 GiB less memory. With 1 to 8 requests at once (run 6), TensorFold + MLX decodes more in total on most rows (up to 14% at eight requests, 34% on sampled code alone), and Octojet leads greedy chat at one to four.

![Octojet against TensorFold 0.6.5 on the three published Flash Next checkpoints, one DGX Spark](assets/speed-vs-upstream.png)

**Run 5** (4 Oct 2026, all four setups, one session; Octojet 6db4754):

| Measure | Octojet | TF 0.6.5 + MLX 4-bit | TF 0.6.5 + lil | TF 0.6.5 + RadixArk |
|---|---:|---:|---:|---:|
| Cold 32k-token prompt, first token | 15.8 | 12.8 | 19.1 | 21.4 |
| Cold 128k-token prompt, first token | 58.6 | 57.3 | 83.4 | 92.8 |
| Cold 210k-token prompt, first token | 107.2 | 104.2 | 143.7 | 162.1 |
| Cold 71k-token prompt, first token | 30.5 | 30.3 | 44.1 | 49.6 |
| Prompt sharing 69k of those 71k tokens | 3.1 | 29.2 | 44.1 | 49.6 |
| The 71k prompt resent | 0.13 | 0.17 | 0.20 | 0.19 |
| Decode, code prompt, temperature 1 (tok/s) | 64.1 | 68.3 | 52.2 | 41.4 |
| Decode, chat prompt, temperature 1 (tok/s) | 62.1 | 56.4 | 44.2 | 32.5 |
| Decode, code prompt, greedy (tok/s) | 61.5 | 66.9 | 55.5 | 46.2 |
| Decode, chat prompt, greedy (tok/s) | 86.0 | 63.2 | 43.2 | 37.4 |
| Agent: 80k-token first turn, first token | 49.6 | 46.6 | 66.4 | 74.1 |
| Agent: +5k-token follow-up, first token (median of 3) | 4.18 | 5.49 | 5.26 | 6.58 |
| Agent: +5k-token follow-up, whole step (median of 3) | 11.5 | 13.8 | 15.7 | 21.8 |
| A short prompt sent while a cold 128k prompt fills, first token | 4.54 | 2.84 | 3.47 | 3.15 |
| A resend sent while a cold 128k prompt fills, first token | 1.90 | 3.83 | 2.46 | 2.25 |
| Longest pause of a live reply while a 128k prompt arrives | 1.32 | 1.33 | 1.62 | 1.80 |
| Typical pause of a live reply while a 128k prompt arrives (median gap) | 0.66 | 1.19 | 1.45 | 1.62 |
| That 128k prompt's first token while the reply streams | 84.3 | 74.9 | 92.8 | 102.7 |
| Startup memory estimate (GiB) | 86.2 | 84.3 | 81.5 | 97.4 |
| Context windows | 3 × 243,197 reserved | up to 3 × 262,144 | up to 3 × 262,144 | up to 3 × 262,144 |

Octojet's window is what its memory plan reserves for three text-only streams after keeping 2 GiB beside its mapped n-gram tables (0.1.1); TensorFold grows each stream's cache while memory lasts.

**Run 6** (multi-stream decode):

Run 6, 15 Oct 2026, one session: total decode tok/s with N requests at once, **Octojet 6db4754 / TensorFold v0.6.5 + MLX 4-bit**, both `--parallel 8`, int8 KV, text only (`engine/tools/bench_concurrent.py`: 256 tokens a request, median of 3; higher is better, the higher of each pair in bold). Octojet at `--parallel 3` (production) measured the same at 1-3 requests. At `--parallel 8` Octojet reserves 8 × 63,198 tokens; TensorFold grows each stream's cache while memory lasts (up to 262,144).

| Requests at once | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
|---|---:|---:|---:|---:|
| 1 | 68.9 / **92.0** | 61.0 / **62.4** | 69.9 / **71.4** | **69.5** / 65.5 |
| 2 | 94.9 / **108.8** | 82.2 / **88.0** | 127.5 / **127.9** | **127.1** / 117.6 |
| 3 | 117.5 / **128.8** | 101.8 / **106.4** | 175.9 / **181.6** | **175.2** / 167.2 |
| 4 | 135.2 / **145.8** | 117.4 / **122.3** | 216.6 / **229.8** | **215.8** / 211.4 |
| 8 | 180.5 / **188.3** | 158.9 / **167.5** | 332.5 / **380.5** | 331.4 / **349.4** |

**Run 2** (1-2 Oct 2026, accuracy, Octojet against TensorFold v0.6.0 + local-inference-lab, one session; Octojet's later
changes keep every token identical):

| Measure | Octojet | TF 0.6.0 + local-inference-lab |
|---|---:|---:|
| Cold 32k-token prompt, first token | 18.2 s | 19.8 s |
| Cold 71k prompt, then a prompt sharing 69k of it, then the 71k prompt resent | 37.8 / 4.9 / 0.12 s | 45.6 / 45.6 / 0.18 s |
| GSM8K, 250 problems | 245 | 246 |
| HumanEval pass@1, 164 programs | 155 | 158 |

![Accuracy: Octojet against TensorFold 0.6 + local-inference-lab](assets/accuracy.png)

Where it does not lead: TensorFold on the MLX 4-bit checkpoint reads cold prompts sooner and decodes code faster;
local-inference-lab's checkpoint needs about 6 GiB less memory, and its quantization-aware
distillation scores 3 more HumanEval programs (within noise at 164). Rows measured before a fix in the same week
are marked in the benchmarks file.

Speed and accuracy were measured on Octojet's production build of these weights, whose shared experts came from an
FP8-derived copy of the same RadixArk revision; this repository carries RadixArk's bf16 shared experts, and its own
correctness and accuracy checks are in the benchmarks file.

Every drafted reply equals what serial decoding produces with the same settings (`"draft": false`), and concurrent
replies equal solo ones; both are checked on every release.

## What is in this repository

| Part | Format | Source |
|---|---|---|
| Routed experts, 48 layers × 512 | NVFP4 (E2M1 values, FP8 block scales per 16, FP32 global scale) | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ 7b719225, copied byte for byte |
| Shared experts | bf16 in the files, packed to NVFP4 at load | the same RadixArk revision |
| Attention, DeltaNet, routers, embeddings, n-gram tables, head, MTP head, vision tower | MLX affine 4-bit, groups of 32 (the vision tower stays floating point) | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` @ 2b170fa6, copied byte for byte |

Built by `engine/tools/build_release_checkpoint.py`; `octojet.json` records both sources and revisions. Every
tensor was checked against its source byte for byte. No weights were retrained or requantized for this release.

## Limitations

- Octojet only; one GPU (no tensor parallelism for this checkpoint).
- Measured on GB10 only. Other Blackwell GPUs with enough memory should work but are untested.
- The MTP draft head stays 4-bit affine; drafted speed depends on how often drafts are accepted.
- Memory use is higher than local-inference-lab's checkpoint.

## License

Qwen Community License 1.0, inherited from Qwen3.8 Flash Next. Clause 2 requires a separate license from Qwen for
anyone offering the model as a service ("Model as a Service") or running an AI assistant business on it; using it
on your own hardware for your own work does not. Read [LICENSE](LICENSE) (the Qwen Community License 1.0, included) and
[NOTICE](NOTICE) (attribution) before commercial use. The Octojet engine is MIT licensed.

## References

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Weight sources: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) @ 7b719225 (NVIDIA Model Optimizer export), [Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) @ 2b170fa6
- Compared against: [local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) @ 7c4f1bc1, [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) (3.05bpw_h5_ng5 @ 69e33439); for context [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
- Engines: [TensorFold](https://github.com/ashhart/TensorFold) (Octojet forks v0.3.6.2; compared at v0.6.0), [vLLM](https://github.com/vllm-project/vllm) via [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) @ bd60fcb, [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
- Vision patches: [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) @ a3aa898
- Benchmarks: [GSM8K](https://github.com/openai/grade-school-math), [HumanEval](https://github.com/openai/human-eval)

Thanks to the Qwen team, RadixArk, Vontra, the TensorFold contributors, MiaAI-Lab and turboderp, whose work this
release builds on.
