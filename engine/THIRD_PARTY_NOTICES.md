# Third-party notices

## TensorFold

Octojet's engine (this directory) is a fork of [TensorFold](https://github.com/ashhart/TensorFold) 0.3.6.2,
upstream commit `71377a5`, MIT License, Copyright (c) 2026 TensorFold contributors (the full text is in
[LICENSE](LICENSE), which keeps the upstream copyright line). The later upstream code ported into this fork (the
vision package from v0.3.6.3, the tiled block selection of `a2dba9c` from 0.5.0, the prompts inside rounds of
`d23087c` as run by `68c6e35`; see the sections below) comes from commits under the same MIT License; upstream moved
to Apache-2.0 only from v0.6.0, and nothing from v0.6.0 or later is included. The
notices below are TensorFold's own, kept as they were, plus the sections marked "Octojet port". "TensorFold" in them
refers to the code this fork inherited.

## MLX and mlx-lm

TensorFold uses [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
They are installed as dependencies.

## MLX and mlx-lm adaptations

The DeltaNet implementations in `src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py` and
`lane_tree.py` adapt mlx-lm's `qwen3_5` and `gated_delta` model math and kernels under its MIT License.

## Qwen Flash Next

The n-gram ID helpers in `src/tensorfold/families/qwen4_exp/model.py` and `cuda/ngram.py` translate
Hugging Face transformers' `models/qwen4_exp/modeling_qwen4_exp.py` into MLX and NumPy with renamed
identifiers. Copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
See [the license text](LICENSES/Apache-2.0.txt). The same helpers appear in mlx-vlm's
`models/qwen4_exp/language.py`, MIT License, Copyright © 2025 Prince Canuma.

## GLM-5.3-Flash on Apple Silicon

The MLX engine of `glm5_next` (`src/tensorfold/families/glm5_next/`: the forward pass in `model.py`, `kda.py`,
`mla.py` and `mlp.py`, the draft head in `mtp.py`, `runtime.py`) and its Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/`) are written for TensorFold. What they follow or port:

- The forward pass follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`, `moe.py` and `hc.py`.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `hc.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py`, `moe.py` and `hc.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits.

## CUDA

CUDA backends use [PyTorch](https://github.com/pytorch/pytorch), BSD-3-Clause, and
[Triton](https://github.com/triton-lang/triton), MIT, supplied by NVIDIA's container rather than bundled.
Dense Qwen implements mlx-lm's model math. CUDA DFlash2 implementations port z-lab's architecture under
the MIT License, Copyright © 2026 Z Lab.

Flash Next implements transformers' model math under the attribution above. Its new DeltaNet kernel
follows flash-linear-attention's numerics, MIT; its NCCL wrapper follows vLLM's stream convention,
Apache-2.0, without copying either implementation.

GLM's CUDA engine implements transformers' `models/glm5_next/modular_glm5_next.py` math, Apache-2.0,
without including that source. Its draft inputs and thinking-off rendering follow the public GLM recipe
from Mia-AiLab without including recipe code.

GLM EXL3 (`families/glm5_next/cuda/exl3.py`, `exl3.cu`, `exl3_mm.py`), the shared EXL3 module
(`src/tensorfold/cuda/exl3/`) and the EXL3 loaders of Qwen3.8-27B and Qwen3.8 Flash Next
(`families/qwen3_5/cuda/exl3_load.py`, `families/qwen4_exp/cuda/exl3.py`) read
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 format: its trellis layout and bitstream, its
"3inst", "mcg" and "mul1" codebooks, its half-integer bit widths and its tensor-core fragment order. Flash
Next's packs also carry ExLlamaV3's n-gram row codec, read as its `ngram_dequant` reads it. ExLlamaV3 uses the
MIT License, Copyright © 2025 Turboderp. TensorFold's decoders and kernels are separate implementations,
checked bit for bit against ExLlamaV3's dequantization.

Flash Next's optional int8 and int4 KV caches (`families/qwen4_exp/cuda/kvcache.py`) follow the cache quantization scheme of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) `-cq 8` and `-cq 4` (MIT License, Copyright (c) 2025 Turboderp, text below): groups of 32, one fp16 absmax scale per group, the group rotated by a 32-point Hadamard, midpoint-grid codes, `compand_a == 0`. 8-bit stores each code as a signed int8 (`q - 128`). 4-bit stores two unsigned codes per byte, low nibble first (the same bits as ExLlamaV3's little-endian packing, a uint8 tensor rather than their uint32 words). Their dequantizer folds another `1/sqrt(32)` into the scale and applies the unnormalized butterfly on the way out; this cache applies the normalized H32 to the query and to the merged output instead, and leaves the stored codes rotated. Scales match their quantizer bit for bit. Reconstructed values agree within fp16/bf16 rounding (under 0.01 on random groups), not bit for bit. The quantizer and the attention dequant are written for TensorFold and checked against an independent reference of that arithmetic.

## Vendored code and weights

`src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is the unmodified `dflash/model_mlx.py` from
[z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab.

TensorFold ships no model weights. The `z-lab/Qwen3.8-27B-DFlash2` model card states Apache-2.0.
The optional `incoai/GLM-5.3-Flash-DFlash2` model card states CC BY-NC-ND 4.0, for non-commercial use
without derivatives. Each checkpoint keeps its own license.

## MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Image and video input (Octojet port)

`src/tensorfold/vision/` (except `videos.py`) and `src/tensorfold/server/prompts.py` are taken from upstream
[TensorFold](https://github.com/ashhart/TensorFold) v0.3.6.3, MIT License, Copyright (c) 2026 TensorFold contributors
(the same license as the rest of this fork), with the server and option hooks they need adapted to Octojet.

Flash Next image and video input on CUDA (`src/tensorfold/vision/videos.py`; the vision changes in
`vision/*.py`, `server/prompts.py`, `server/messages.py`, `serve_options.py`, `cuda/server.py` and
`families/qwen4_exp/` — interleaved 3-D rotary positions in `cuda/glue.py` `_attn_prep` and `cuda/attention.py`
`_pool`, image features in the prefill, many images a request) is ported from MiaAI-Lab's patches
`0008-flash-next-vision.patch` and `0009-flash-next-many-images.patch` in
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
(commit a3aa898), resolved by hand against Octojet's fork. MIT License:

Copyright (c) 2026 MiaAI-Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Those patches follow, as their authors note, Hugging Face transformers' Qwen3.5 rotary index (`get_rope_index`) and
Qwen3-VL's video processing (Apache License 2.0, see [the license text](LICENSES/Apache-2.0.txt)); at runtime the
vision tower is transformers' own `Qwen3_5VisionModel`.

## Prompts inside rounds (Octojet port)

The concurrent Flash Next decoder's prompts inside rounds (`families/qwen4_exp/cuda/multi.py`: queued admissions,
`_fill` / `_chunk`, filling prompts counted as live and busy; `families/qwen4_exp/cuda/decode.py` `prefill_steps`;
`cuda/scheduler.py` `defer`) is ported from upstream [TensorFold](https://github.com/ashhart/TensorFold) commit
`d23087c` ("CUDA lanes (prompts inside rounds, caches that grow, one launch a kernel a round)"), whose tree is under
the MIT License, Copyright (c) 2026 TensorFold contributors (the same license as the rest of this fork; upstream moved
to Apache-2.0 only from v0.6.0). Only the one-prompt-a-pass subset is taken, with a pass between rounds as upstream
runs it where a round and a prompt pass don't share an expert launch (`68c6e35`); the growing caches, the shared
forward of a round and a pass, the one-launch DeltaNet and attention steps, `--decode-share` and packed passes are
not. Its CUDA test `test_prompts_fill_between_rounds_while_streams_decode` is ported into
`tests/cuda/test_flashnext_multi.py`. Upstream's `603d3bc` (prompts filling between Mac rounds) and the kept-one-token-
early change of `d8057db` / `0dcc1ff` were read and not taken: the first is Mac-only, and Octojet's F2d exact hits
already resume identical resends.

## Tiled sparse-attention block selection (Octojet port)

`_select_tiles` and `_launch_select` in `src/tensorfold/families/qwen4_exp/cuda/attention.py` (block selection for
rows past 131,072 keys) and their CUDA test in `tests/cuda/test_flashnext_kernels.py` are ported from upstream
[TensorFold](https://github.com/ashhart/TensorFold) commit `a2dba9c` (by MovieMaker93, released in 0.5.0), MIT
License, Copyright (c) 2026 TensorFold contributors. MiaAI-Lab's patch 0004 proposed the same idea; its code is not
used.
