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

Compared against the fastest published setup on 2 October 2026: TensorFold v0.6.2 (56e2e3ec, `--precision full`) with
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ 7c4f1bc1 and with `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ 7b719225.

![Octojet against TensorFold 0.6.2 on the published NVFP4 checkpoints, one DGX Spark](assets/speed-vs-upstream.png)

**Run 3** (2 Oct 2026, all three setups, one session):

| Measure | Octojet | TF 0.6.2 + local-inference-lab | TF 0.6.2 + RadixArk |
|---|---:|---:|---:|
| Cold 32k-token prompt, first token | 15.6 | 19.6 | 22.3 |
| Cold 128k-token prompt, first token | 68.8 | 84.7 | 95.0 |
| Cold 210k-token prompt, first token | 124.0 | 146.2 | 164.4 |
| Cold 71k-token prompt, first token | 32.2 | 45.0 | 51.1 |
| Prompt sharing 69k of those 71k tokens | 3.2 | 45.1 | 49.2 |
| The 71k prompt resent | 0.13 | 0.19 | 0.19 |
| Decode, code prompt, temperature 1 (tok/s) | 64.3 | 50.1 | 39.7 |
| Decode, chat prompt, temperature 1 (tok/s) | 62.9 | 41.7 | 31.4 |
| Decode, code prompt, greedy (tok/s) | 54.9 | 51.0 | 42.9 |
| Decode, chat prompt, greedy (tok/s) | 88.6 | 39.9 | 34.7 |
| Agent: 80k-token first turn, first token | 53.2 | 67.4 | 74.4 |
| Agent: +5k-token follow-up, first token (median of 3) | 4.39 | 5.27 | 6.67 |
| Agent: +5k-token follow-up, whole step (median of 3) | 11.9 | 14.3 | 23.5 |
| A short prompt sent while a cold 128k prompt fills, first token | 3.16 | 3.62 | 4.52 |
| A resend sent while a cold 128k prompt fills, first token | 1.12 | 2.62 | 3.60 |
| Longest pause of a live reply while a 128k prompt arrives | 2.52 | 1.61 | 1.78 |
| Startup memory estimate | 87.2 GiB | 81.2 GiB | 97.4 GiB |
| Context windows | 3 × 262,144 reserved | up to 3 × 262,144 | up to 3 × 262,144 |
| Load time (Octojet's packed-table cache warm) | 66 s | 82 s | 91 s |

`--precision full` runs 16-bit activations against the 4-bit weights, as Octojet does. TensorFold's default mode was
measured too: on this machine it also logged bf16 prompt activations, and its results were within 3% of these
(every column is in the [benchmarks](https://github.com/octolixai/octojet/blob/main/docs/benchmarks.md)).

**Run 2** (1-2 Oct 2026, accuracy, Octojet against TensorFold v0.6.0 + local-inference-lab, one session; Octojet's later
changes keep every token identical):

| Measure | Octojet | TF 0.6.0 + local-inference-lab |
|---|---:|---:|
| Cold 32k-token prompt, first token | 18.2 s | 19.8 s |
| Cold 71k prompt, then a prompt sharing 69k of it, then the 71k prompt resent | 37.8 / 4.9 / 0.12 s | 45.6 / 45.6 / 0.18 s |
| GSM8K, 250 problems | 245 | 246 |
| HumanEval pass@1, 164 programs | 155 | 158 |

![Accuracy: Octojet against TensorFold 0.6 + local-inference-lab](assets/accuracy.png)

Where it does not lead: local-inference-lab's checkpoint needs about 6 GiB less memory, a live reply pauses longer while a
long prompt arrives (2.5 s against 1.6 s), and its quantization-aware
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
