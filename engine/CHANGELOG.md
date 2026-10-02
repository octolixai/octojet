# What's new in Octojet

Octojet is a fork of TensorFold 0.3.6.2 (commit 71377a5). The sections from 0.3.6.2 down are upstream TensorFold's
release notes, kept as they were; their commands read `tensorfold ...`, which is `octojet ...` here.

## 0.1.0 (October 2026)

The first Octojet release. Measured numbers are in `docs/benchmarks.md` at the repository root. Supported and
measured: NVIDIA GB10 with Qwen3.8 Flash Next NVFP4; the rest of the tree is inherited from TensorFold and untested.

- **NVFP4 routed experts for Qwen3.8 Flash Next on CUDA.** The grouped-expert kernels (decode and prefill) read
  NVFP4 weights (E2M1 values, FP8 block scales per 16, FP32 global scale), so GB10's tensor cores compute in FP4. A
  mixed checkpoint loader takes NVFP4 experts from an NVIDIA Model Optimizer export and everything else from the MLX
  4-bit checkpoint; FP8 or bf16 shared experts are dequantized and packed to NVFP4 at load, with range and
  finiteness guards. At `--kv-dtype int8 --parallel 3` it holds 3 x 262,144-token windows (3 x 174,961 with the MLX
  4-bit checkpoint). `--tp 2` is refused on this checkpoint.
- **Packed-table cache.** `--packed-cache DIR|off|verify` keeps the packed expert tables on disk (default
  `$OCTOJET_CACHE_DIR/packed` or `~/.cache/octojet/packed`), bound to the checkpoint's contents with atomic writes,
  validated headers and sequential reads; a damaged entry is rebuilt and a cache failure serves uncached. Warm starts
  went from about 355 s to 138 s in production. `OCTOJET_LOAD_TIMING=1` logs the load phases.
- **Faster long prompts.** Sparse-attention block selection above 131,072 keys runs in tiles (a port of upstream
  TensorFold a2dba9c), and `TENSORFOLD_PREFILL_ROWS` sets Flash Next's prompt chunk (a port of TensorFold #40): a
  cold 210k-token prompt went from 227 s to 111 s in production.
- **Prefix reuse.** An identical prompt replays the kept first-token logits instead of prefilling (`reuse: "exact"`;
  about 5 ms instead of 36.7 s in production); chunk-boundary checkpoints (`--prefix-checkpoints N`, default 4) let a
  prompt that shares a long prefix with an earlier one resume from the nearest checkpoint (a 71k prompt sharing 69k:
  2.2 s instead of 35.7 s in production). Replies equal fresh runs. The stats block reports `cached`, `reuse` and
  `reuse_miss`.
- **Prompts inside rounds.** A new prompt prefills one chunk a pass between decode rounds instead of stopping the
  live replies (a port of upstream TensorFold d23087c, one prompt a pass): a live reply's longest pause while a
  128k prompt arrives went from about 68 s to about 2.5 s.
- **Twin copies.** An identical prompt that arrives while its twin is still filling waits for the fill and then
  exact-hits a copy of its state, instead of prefilling a second time.
- **Image and video input for Flash Next on CUDA** (`--vision`, `--vision-urls`): upstream TensorFold v0.3.6.3's
  vision package with MiaAI-Lab's patches 0008 and 0009 (several images a request).
- **Release checkpoint tools.** `tools/build_release_checkpoint.py` builds the published mixed checkpoint (byte-for-byte
  copies of both sources, `octojet.json` records sources and revisions), `tools/check_mixed_checkpoint.py` checks
  it against its sources, and the hub treats a mixed checkpoint as complete only with its marker, experts index and
  shards.
- **Measurement hooks.** CUDA requests take `prompt_ids`, `return_token_ids` and opt-in prefill timing, profile and
  routing-histogram fields (`docs/api.md`); bench tools compare token ids across drafted, serial, solo and concurrent
  runs.
- **Rebrand.** The command is `octojet` (the `tensorfold` script is gone; `python -m tensorfold.cli` still works),
  the package is published as `octojet` 0.1.0, log lines start with `[octojet]`, the per-reply stats block is
  `"octojet"` (was `"tensorfold"`), `/v1/models` reports `owned_by: "octojet"`, and the MLX prefix-snapshot default
  moved to `~/.cache/octojet/prefix-snapshots`. The Python package `tensorfold` and the `TENSORFOLD_*` environment
  variables keep their names. The self-updater is disabled.

## 0.3.6.2 (28 Sep 2026)

- **EXL3 replies stop at the end of the turn.** Flash Next and 27B EXL3 packs list `<|im_end|>` only in
  `generation_config.json`, so replies ran past their turn and leaked tool calls and think tags. The CUDA engine now
  reads that file too. Thanks to @vcruz305 (#69).
- **pip installs serve 27B EXL3 packs.** The package was missing the 27B's CUDA sources; a test now checks that every
  CUDA source ships. Thanks to @taussoe (#66).
- **A quantized KV cache for Flash Next on CUDA.** `--kv-dtype int8` or `int4` holds about 1.7x or 2.6x the default
  window in the same memory, and drafted replies still equal serial ones. `--mtp-confidence` sets where MTP chains
  stop. Thanks to @vcruz305 (#47).
- **Flash Next on CUDA reaches the first token sooner.** The head runs on a prompt's final chunk only, and the prompt
  kernels load at startup, so the first 2k prompt takes 1.26 s instead of 1.79 s. Thanks to @MovieMaker93 (#40).
- **GLM on two Sparks holds 256k tokens** with a latent attention cache; its next-token loss is within 0.001 nats of
  the per-head cache. Thanks to @taussoe (#54).
- **Mixed-bit Qwen checkpoints** (4-bit with some 5- and 6-bit layers, such as oQ4) load on every lane backend,
  exact, with new row kernels for 2- to 8-bit weights. On an M3 Ultra, oQ4 27B decodes 114-120 tok/s on code and
  59-61 on chat, against mlx_lm's 34-36.
- **Concurrent 27B on CUDA:** `--parallel 16` serves 161.7 tok/s on one Spark in 25.4 GiB, each reply equal to its
  solo run (#38). **Qwen3.6-35B-A3B on CUDA,** exact: decode 1.36-1.49x vLLM with MTP, prompts 1.21-1.37x (#45).
- **Gemma 4 drafts (opt-in):** `--drafter z-lab/gemma-4-26B-A4B-it-DFlash` decodes 1.3-2.1x mlx_lm on an M3 Ultra,
  exact.
- **Tool calls:** `tool_choice: "required"` and a named tool are enforced on both servers (#52), and a complete tool
  call inside an unclosed think block comes back as a tool call (#60).
- **Conversations come back warm on Macs.** `--spill-gib N` writes a conversation pushed out of the prompt cache
  to disk, up to N GiB, and reads it back when the conversation returns. On a 48 GB budget a 35k-token
  conversation came back in 0.27 s on an M5 Max instead of 75 s, with the same reply. Off by default;
  `--checkpoint-slots` sets how many conversations stay in memory. Thanks to @gilby (#68, #55).
- **Memory:** `TENSORFOLD_MEMORY_LIMIT_GB` raises the budget above the default share, and Flash Next's memory check
  counts its host-mapped n-gram tables, so it starts on a 128 GB Mac. Thanks to @Chedrian07 (#49, #50).
- **CUDA server fixes from @nood-co1:** an abandoned request stops within a round (#57), keys and values stay within
  the admitted window (#58), and a failed admission no longer stops the scheduler (#59).
- **M1-M4:** a prompt split into parts attends exactly as it does in one piece at every length; two 8,192-key cases
  rounded differently in 0.3.6.

## 0.3.6.1 (28 Sep 2026)

- **CUDA builds inside NVIDIA's containers again.** Their `TORCH_CUDA_ARCH_LIST` names every architecture back to
  sm_80, so the kernels' thread-block clusters and FP8 MMA failed to compile for GPUs that lack them. Every extension
  now builds for the GPU that is present, and a GPU older than compute capability 9.0 gets a clear message. Thanks to
  @ss-cong for the report and the exact errors (#56).

## 0.3.6 (28 Sep 2026)

- **GLM-5.3-Flash on Macs** with 256 GB, drafted replies equal to serial ones. Prompts process at or above mlx-vlm
  from 2k to 32k tokens on an M3 Ultra, and tool calls parse in both servers. Thanks to @chadhurley25075-png (#9,
  #39) and @jeidbugs404 (#35).
- **Gemma 4 26B-A4B on the lanes,** exact at every width, with prompts at or above mlx_lm from 2k to 64k tokens on
  an M3 Ultra. Thanks to @cshintov (#10).
- **Bigger models on smaller Macs.** `--ple-on-ssd` reads Flash Next's n-gram tables from disk, so a 128 GB Mac holds
  it (#16). `--ssd-experts GIB` streams routed experts from the checkpoint into a GPU pool of that size, so Flash Next
  fits a 64 GB Mac and GLM a 128 GB one. Replies are the resident model's tokens; decode runs at 0.31-0.39x resident
  speed for Flash Next and 0.13-0.17x for GLM (measured on an M3 Ultra; `pip install "tensorfold[ssd]"` first) (#17).
- **EXL3 checkpoints on CUDA (experimental):** Qwen3.8-27B and Flash Next packs from turboderp, exact on the lanes.
  Decode runs 1.6-3.6x vLLM with MTP; prompt processing is about half the MLX checkpoints' speed for now, and the
  fix is next. Thanks to @vcruz305 (#42).
- **Faster prompts.** Flash Next sizes its prompt chunks to the memory it has, Nemotron takes up to 8,192 tokens a
  chunk on M5 GPUs, and the weights stay wired while a server runs.
- **Fixes:** GLM on two Sparks answered "!" past about 2,000 prompt tokens, and EXL3 GLM prompts past 128 tokens
  failed (#53). A reply that isn't a tool call comes back as content, not an HTTP 500 (#51). The server hands MLX's
  freed buffers back when it goes idle (@kingjamez, #44).
- **MLX 0.32.2 or newer** is required on Macs.
- **What's new after an update:** `tensorfold update` prints these notes when it finishes.
- **Known:** replies to prompts longer than one prompt chunk can differ between machines with different memory,
  because the chunk size follows the memory budget. Within one server, drafted replies always equal serial ones and
  resumed prompts equal fresh ones.

## 0.3.5.1 (28 Sep 2026)

- Qwen3.8-27B loads on M1 and M2 Macs again. Kernels there fit Metal's per-kernel thread limit, with the same sums in
  the same order, so drafted output still equals serial output.
- M3, M4 and M5 run 0.3.5's machine code unchanged.
- Thanks to @hichaiuse, @simonmd, @gcarusso, @tonydehnke, @Cyb3r-Monk and @tinyapps for the reports and the repro.

## 0.3.5 (27 Sep 2026)

- **Concurrent requests share each verification round.** `--parallel auto` is on by default, and every stream's reply
  equals the same request served alone, on Metal and on CUDA.
- **Follow-up turns resume at the start of their newest messages,** with output identical to a fresh prompt. A 12-turn
  agent session with the 27B spent 14.9 s on first tokens instead of 31.9 s.
- **Memory that fits.** The whole process stays inside 70% of RAM, an omitted `--context` defaults to the window the
  machine can hold, and a prompt past it gets a clear 400.
- **Flash Next prefill** with sparse prompt attention, 1.1-1.4x faster than 0.3.4.1 on an M3 Ultra (@quigles1977, #29).
- **2- to 8-bit weights** on the lanes, so mixed-precision 27B checkpoints decode fully (@jasontitus, #34).
- **CUDA:** FP8 prefill and shared expert kernels, concurrent streams for the 27B and Flash Next, and admission from
  available memory before loading.
- **API:** raw `/v1/completions` prompts, `ignore_eos` and `stop`, request `reasoning_effort` and typed tool arguments
  (@chris247474, #28), `developer` messages, `parallel_tool_calls: false`, and cancellation when a client disconnects.

## 0.3.4.1 (27 Sep 2026)

- Prompt processing is back to MLX's speed on every model. Prompts prefill through MLX's own forward on a fixed
  2,048-token grid, so a resumed conversation still equals a fresh one byte for byte.
- Flash Next's peak memory stays within 20 GB of its weights up to a 196k-token prompt.

## 0.3.4 (26 Sep 2026, pre-release)

- Every model runs on the lane engine, and the serial engine is gone. Nemotron drafts on M1 to M4 with row-exact
  kernels.
- Qwen3.8-27B on M1 to M4 decodes at 1.9 to 4x serial, through a new 4-bit matmul on the simdgroup matrix units.
- CUDA: every default path verifies at least two rows a round.

## 0.3.3 (26 Sep 2026)

- Qwen3.8-27B verifies drafted tokens together on every M1 to M5 GPU, with output byte-identical to serial decoding.
- Streamed `/v1/completions` send plain text.

## 0.3.2 (26 Sep 2026)

- `tensorfold update` installs the newest release.
- GLM-5.3-Flash reads Mia-AiLab's EXL3 weights on two DGX Sparks (experimental).

## 0.3.1 (26 Sep 2026)

- `tensorfold info` shows how a checkpoint stores its weights and which backends read them. `serve` and `pull` refuse
  checkpoints no engine reads yet, before anything downloads.

## 0.3.0 (26 Sep 2026)

- NVIDIA GPUs: `tensorfold serve` picks CUDA on Linux and runs on one GPU or two (one rank per DGX Spark).
- A new family, GLM-5.3-Flash, on two Sparks.

## 0.2.0 (25 Sep 2026)

- A rewrite: `tensorfold serve`, `pull`, `models` and `info` for Nemotron 3.5 Lightning, Qwen3.8-27B and Qwen3.8
  Flash Next on Apple Silicon, with drafts that never change the output.
