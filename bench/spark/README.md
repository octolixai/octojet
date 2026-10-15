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

## F10 (MTP decode tuning)

`f10-mtp.sh [--dry-run]` serves one test server per `--mtp-confidence` in `CONFS` (default `0.2 0.3 0.5 0.7`) and
`--mtp-drafts` in `DEPTHS` (default `6`), production's flags otherwise (text only), and runs `bench/mtp_sweep.py` against
each: 20 HumanEval completions and 20 chat/prose prompts (GSM8K answered briefly, short prose), greedy and sampled,
200-token replies, thinking off. `f10-summary.txt` has the median and IQR of decode tok/s per (category, sampling), the
median accepted drafts per round, and the recommendation (largest geometric mean of the four medians; within 1% the
production setting 0.3 / 6, then the higher confidence). Production is stopped for the window (`PROD_STOP`, standby pause
held, removed on every exit path); about 8 min a setting. `test_f10_script.py` runs it under the F2d stubs.
## F9 (YaRN: 3 x 524,288)

`f9-yarn.sh [--dry-run | --tests-only]` (about 2 h; `ACC=0` saves about 1 h) runs the F9 CUDA and CPU tests beside
production, then one paused window (production stopped via `PROD_STOP`, standby pause held, removed on every exit
path): server A with production's flags, text only, plus `--rope-yarn 2` (the startup lines answer whether 3 x 524,288
fits), a cold 450k prompt, `bench/needle.py` at 128k/256k/450k x depths 0.1/0.5/0.9, GSM8K 250 + HumanEval 164; then
server B without YaRN (needle at 128k/256k, GSM8K + HumanEval) as the baseline. `f9-summary.txt` holds all of it
(`OUT`, default `$HOME/octojet-runs/f9`). The needle probe needs only the server (chat completions); its filler is the
host's Python standard library sources. `test_f9_script.py` runs the script under the F2d stubs.
## F12 (fused prompt hyper-connection read-out)

`f12-profile.sh [--dry-run] [--steps tests,profile,ttft]` (about 75 min): `tests` runs the F12 CUDA tests (synthetic
shapes, the 4-layer NVFP4 cut, the MLX checkpoint's first 4 layers when `PROF_CKPT` exists) and the forward /
prefill-timing suites beside production; a failure stops the run (`FORCE=1` goes on). Then production is stopped
(`PROD_STOP`, standby pause held, removed on every exit path). `profile`: three servers under `nsys launch` (CUDA trace,
black box), each warmed with a short request and a throwaway long prompt, then `nsys start` / `stop` around one cold
32k and one cold 128k prompt: Octojet main (`REPO_MAIN`, a checkout of main before F12), Octojet F12 (`REPO`) and
upstream TensorFold `UP_TAG` (`UP_SRC`, `UP_IMAGE`; install with `compare-upstream.sh --prepare`), all on `PROF_CKPT`
(default upstream's MLX 4-bit checkpoint, so the weights are the same). The nsys CLI is copied once from `IMAGE` to
`NSYS_HOST` and mounted into every container. `ttft`: cold 32k / 128k / 210k on main vs F12 with production's flags
and checkpoint. After the restore, `nsys stats` and `bench/kern_compare.py` write `f12-kernels-m32k.txt` and
`f12-kernels-m128k.txt` (per-kernel GPU time side by side, then the kernels where F12 and upstream spend more / less
than main) and `f12-ttft.txt` to `OUT`. `test_f12_script.py` runs it under the F2d stubs.

## F14 (the DeltaNet prompt chain loads ahead)

`f14-test.sh [--dry-run] [--steps tests,ttft]` (about 35 min): `tests` runs, beside production, the F14 CUDA tests
(both chain forms byte-equal on synthetic chains; whole prefills of the 4-layer NVFP4 cut and, when `PROF_CKPT`
exists, the MLX checkpoint's first 4 layers with `OCTOJET_GDN_CHAIN_AHEAD` on and off), the GDN and Flash Next suites
they touch, and `bench/f14_gdn_bench.py` (ms a launch of the chain's two forms and of the front / back kernels alone,
to compare with an nsys capture's per-launch averages); a failure stops the run (`FORCE=1` goes on). Then production is
stopped (`PROD_STOP`, standby pause held, removed on every exit path) and `ttft` measures cold 32k / 128k / 210k on
main (`REPO_MAIN`, a checkout of main before F14) vs F14 (`REPO`) with production's flags and checkpoint, a throwaway
long prompt first after each start: `f14-ttft.txt` in `OUT`. `test_f14_script.py` runs it under the F2d stubs.
