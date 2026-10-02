# Spark operator scripts for F2c Phase 1

Rendered from `docs/superpowers/plans/2026-09-30-f2c-phase1-profile.md` (Task 5 Step 4 and Task 6); the plan is the source of
truth, these files save the operator re-typing blockquoted scripts. Order: `f2c-phase1-preflight.sh` (nsys + reducer check,
beside production, ~3 min), `f2c-phase1-manifests.sh` (exact-count prompts, ~2 min), `f2c-phase1-exactness.sh` (the container
exactness test on the 2-layer cut model, CORRECTNESS, beside production, ~25 min), then, with the owner's approval, the two
pauses `f2c-phase1-pause-a.sh` (~45 min) and `f2c-phase1-pause-b.sh` (~25 min); `f2c-phase1-reduce.sh <report>` re-runs the host-side reduction of an existing capture without a pause. The pause scripts carry operator placeholders
(`<your oj-serve stop command>`, `<your oj-serve start command>`, `<your health check>`): fill them, then `bash -n` the file.
`PARALLEL` (default 3) must equal production's `--parallel` at pause time; `Q27` handles the 27B neighbour (absent
today); the scripts refuse to start a profile server while a research GPU job (`llm-research.gpu-job=true`) or any other GPU
process is present; `SHARED_GPU=1` overrides that refusal when the owner accepts a shared GPU (numbers labelled).

## F2d stage A

Operator scripts for identical-prompt reuse (stage A), run in this order:

1. `f2d-a-manifests.sh` (~2 min, no GPU): builds the six prompt manifests (P, the stage-B variants, the burst sequence).
2. `f2d-a-exactness.sh` (CORRECTNESS, beside research, ~30 min): the container tests on the 4-layer NVFP4 cut, then the
   acceptance receipts against a test server with production's flags (`bench_openai --expect-equal`, `concurrent_equal`,
   and the exact-hit probe).
3. With the owner's approval, `f2d-a-replay.sh --caller-timeout 30` (TIMING, GPU exclusive, ~35 min): four fresh servers
   for the gate, then the informational bursts and retry-probe servers.

Environment knobs: `PARALLEL` must equal the `--parallel` that `oj-serve` will run with (production runs 3 as of 2026-09-30 19:25 UTC; the
default stays 2; the retry probe's expectation follows the value: with 2 slots the retry evicts P, with 3 it does not). `SHARED_GPU=1` is accepted by the
exactness script for a labelled run and refused by the replay script. `REPO`, `OUT` and `M` override the repository,
run-directory and manifest locations.

Standby pause switch: `~/.spark-standby.pause` is created before each server start and removed by every exit path, unless it
already existed (then it is left in place and the timeline says "not ours").

`test_f2d_a_scripts.py` runs the scripts under stubs (docker, nvidia-smi, curl, ss, free, python3); no GPU or network.

## F3 (image/video input, tiled QSA select, prompt chunk rows)

`f3-test.sh [--dry-run] [--steps 1,2,3,4,5]` is one combined window (sources `f2d-a-lib.sh` with `PREFIX=f3` and the pause
held across servers). Step 1 runs the container tests beside production. The script then pauses the standby watcher and stops
production oj-serve (`PROD_STOP`, owner's standing approval). Steps 2-5 run a vision server (production flags + `--vision`),
then a text server (no `--vision`): startup lines, `bench_openai --expect-equal` on both with the token_sha compared,
prefill timing (m32k/m128k/m210k, two cold reps each), and `bench/vision_probe.py`. Removing the pause on exit lets the
operator's watcher restart oj-serve. The vision dependencies (transformers 5.17.0, av 19.0.0) come from
`$HOME/tensorfold/octojet-pydeps` mounted at `/deps`. `test_f3_script.py` runs it under the F2d stubs.

## F6 / A3 (self-contained public checkpoint)

`release-build.sh [--dry-run] [--steps 1,2,3,4]` builds the release directory (`OUT`, default
`$HOME/tensorfold/octojet-release`) from the MLX checkpoint (`MLX`) and RadixArk's HF main snapshot (`NVFP4`) with
`engine/tools/build_release_checkpoint.py`, then verifies it: (2) `check_mixed_checkpoint.py --layers all --source-bytes`
against both sources, (3) the 4-layer cut-model CUDA tests on it beside production, (4) a paused window (production
stopped via `PROD_STOP`, standby pause held, removed on every exit path) serving it text-only with production's flags and
its own packed cache (`PACKED`): startup estimate and window, `bench_openai --expect-equal`, `concurrent_equal`, GSM8K +
HumanEval with gates against production's 245/250 and 155/164. Steps 1-3 run beside production. `MLX_REVISION` records the
MLX source's commit (default `unknown`). Logs and `f6-summary.txt` go to `RUNS` (default `$HOME/octojet-runs/f6`).
`test_release_build_script.py` runs it under the F2d stubs.
