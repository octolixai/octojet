# octojet

The Octojet inference engine: Qwen3.8 Flash Next with NVFP4 routed experts on one NVIDIA GB10 (DGX Spark), behind
an OpenAI-compatible API, with drafted decoding that stays exact (every drafted reply equals serial decoding).
Octojet is a fork of [TensorFold](https://github.com/ashhart/TensorFold) 0.3.6.2 by Octolix. Project page,
benchmarks and the full documentation: https://github.com/octolixai/octojet.

## Install

Inside NVIDIA's PyTorch container, which supplies CUDA, PyTorch and Triton:

```bash
docker run -it --gpus all --ipc=host --network host nvcr.io/nvidia/pytorch:26.07-py3
python -m pip install "git+https://github.com/octolixai/octojet.git#subdirectory=engine"
```

Python 3.11 or newer.

## Serve

```bash
octojet serve octolix/Qwen3.8-Flash-Next-Octojet-NVFP4 --kv-dtype int8 --parallel 3 --vision
```

The endpoint is `http://127.0.0.1:8080/v1`. The checkpoint downloads on first use (`octojet pull <repo>` downloads
it ahead of time). The first start packs the expert tables and caches them under `~/.cache/octojet/packed/` (about
64 GB; `--packed-cache DIR|off|verify`); later starts take about 150 s. `octojet serve --help` lists every option.

## Scope

Supported and measured: NVIDIA GB10 with Qwen3.8 Flash Next NVFP4, one GPU. Everything else in this package (the
Apple Silicon/MLX engines, other model families, two-machine tensor parallelism, EXL3 packs, other GPUs) is
inherited from TensorFold and untested here.

The Python package keeps TensorFold's name, `tensorfold` (`python -m tensorfold.cli` is the same as `octojet`), and
environment variables keep their `TENSORFOLD_*` names. `octojet update` is disabled: reinstall to update.

## Upstream

This package is a fork of TensorFold 0.3.6.2 (commit 71377a5), MIT License, Copyright (c) 2026 TensorFold
contributors. TensorFold's README is kept in [`docs/upstream-README.md`](docs/upstream-README.md); what changed since
0.3.6.2 is in [`CHANGELOG.md`](CHANGELOG.md). License: MIT, see [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Octojet is not affiliated with TensorFold.
