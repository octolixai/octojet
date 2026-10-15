#!/usr/bin/env bash
# F9: Flash Next with YaRN (--rope-yarn FACTOR, default 2: a 524,288-token window). Does 3 x 524,288 with the int8 KV
# cache fit on one Spark, does recall hold past 262,144, and what does static YaRN cost on short prompts?
#
# Estimated wall time: about 2 h (tests ~10 min beside production; then the paused window ~1 h 50 min: server A start
# ~4 min, a cold 450k prompt ~5 min, the needle 9 cases ~25 min, GSM8K 250 + HumanEval 164 ~30 min; server B start
# ~4 min, the needle 6 cases ~10 min, GSM8K + HumanEval ~30 min). ACC=0 skips both accuracy runs (~1 h less).
#
# 1. tests (TESTS=1, beside production): the F9 CUDA tests (cut model: YaRN kernels against the reference, drafted ==
#    serial and concurrent == solo with YaRN on) and the suites that hold YaRN-off's bits (forward, multi, kernels,
#    kvquant, vision, the 4-layer NVFP4 cut), plus the CPU tests -> f9-tests.txt. A failure stops the run (FORCE=1 goes on).
# 2. window: production oj-serve STOPPED (owner's standing test approval; PROD_STOP), standby pause held for the whole
#    window, page cache dropped before each start; every exit path removes the pause and the watcher restarts oj-serve.
#    server A  production's flags, text only (--kv-dtype int8 --parallel 3 --packed-cache) + --rope-yarn FACTOR:
#              startup lines (rope, estimate, window, streams: the key result is whether 3 x 524,288 fits)
#              -> f9-yarn-startup.txt; warm-up; a cold COLD-token prompt (needle at depth 0.5) -> f9-yarn-cold.jsonl;
#              needle YARN_SIZES x DEPTHS -> f9-needle-yarn.jsonl; GSM8K 250 + HumanEval 164 -> f9-yarn-acc.jsonl
#    server B  the same without YaRN (baseline): startup -> f9-base-startup.txt; needle BASE_SIZES x DEPTHS
#              -> f9-needle-base.jsonl; GSM8K + HumanEval -> f9-base-acc.jsonl
# 3. f9-summary.txt: all of it in one file.
#
# Usage: bash bench/spark/f9-yarn.sh [--dry-run | --tests-only]
#   --dry-run validates (prompts, the needle plan) and stops nothing; --tests-only runs step 1 alone.
# Knobs: FACTOR (2), YARN_SIZES (128000,256000,450000), BASE_SIZES (128000,256000), DEPTHS (0.1,0.5,0.9), COLD (450000),
#        ACC (1), TESTS (1), FORCE (0), REPO ($HOME/octojet-f9), OUT ($HOME/octojet-runs/f9).
set -euo pipefail
REPO=${REPO:-$HOME/octojet-f9}; OUT=${OUT:-$HOME/octojet-runs/f9}; NAME=${NAME:-oj-f9}; PARALLEL=${PARALLEL:-3}   # PARALLEL=2: what 2 x 524,288 gives
PREFIX=f9; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
FACTOR=${FACTOR:-2}; ACC=${ACC:-1}; TESTS=${TESTS:-1}; FORCE=${FORCE:-0}
YARN_SIZES=${YARN_SIZES:-128000,256000,450000}; BASE_SIZES=${BASE_SIZES:-128000,256000}
DEPTHS=${DEPTHS:-0.1,0.5,0.9}; COLD=${COLD:-450000}
DRYRUN=0; TESTS_ONLY=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRYRUN=1; shift;; --tests-only) TESTS_ONLY=1; TESTS=1; shift;;
  *) echo "usage: $0 [--dry-run | --tests-only]" >&2; exit 2;; esac; done
case "$FACTOR" in ''|*[!0-9.]*) echo "error: FACTOR must be a number above 1, not '$FACTOR'" >&2; exit 2;; esac
awk "BEGIN{exit !($FACTOR > 1)}" || { echo "error: FACTOR must be above 1, not $FACTOR" >&2; exit 2; }

cd "$REPO"
URL=http://127.0.0.1:8080
T=$OUT/prompts; mkdir -p "$T"
printf "Say hello in five words." > "$T/warm.txt"
needle(){ python3 bench/needle.py run "$URL" f1 "$@"; }
log "plan: FACTOR $FACTOR; YaRN needle $YARN_SIZES x $DEPTHS, cold $COLD; baseline needle $BASE_SIZES x $DEPTHS; ACC=$ACC"
needle --sizes "$YARN_SIZES,$COLD" --depths "$DEPTHS" --label yarn --out "$OUT/f9-dry.jsonl" --dry-run | tee -a "$TL"
needle --sizes "$BASE_SIZES" --depths "$DEPTHS" --label base --out "$OUT/f9-dry.jsonl" --dry-run | tee -a "$TL"
if [ "$DRYRUN" = 1 ]; then echo "dry-run OK: FACTOR $FACTOR, TESTS=$TESTS, ACC=$ACC, prompts in $T"; exit 0; fi
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true

if [ "$TESTS" = 1 ]; then
  SCRATCH=$HOME/tensorfold/octojet-scratch; mkdir -p "$SCRATCH"
  DEPS=${DEPS:-$HOME/tensorfold/octojet-pydeps}; TMOUNTS=(); TENVS=("${ENVS[@]}")
  if [ -d "$DEPS" ]; then TMOUNTS=(-v "$DEPS:/deps:ro"); TENVS=(-e PYTHONPATH=/octojet/engine/src:/deps -e PYTHONDONTWRITEBYTECODE=1); fi
  CUDA="tests/cuda/test_flashnext_yarn.py tests/cuda/test_flashnext_forward.py tests/cuda/test_flashnext_multi.py tests/cuda/test_flashnext_kernels.py tests/cuda/test_flashnext_kvquant.py tests/cuda/test_flashnext_vision.py tests/cuda/test_qwen4_exp_nvfp4.py"
  CPU="tests/test_rope_yarn.py tests/test_cuda_capacity.py tests/test_flash_capacity_dispatch.py"
  log "tests: F9 CUDA + CPU tests (OCTOJET_NVFP4_FLASHNEXT_LAYERS=4), beside production"
  docker run --rm --gpus all --ipc=host "${MOUNTS[@]}" ${TMOUNTS[@]+"${TMOUNTS[@]}"} -v "$SCRATCH:/scratch" "${TENVS[@]}" -e TMPDIR=/scratch \
    -e OCTOJET_CACHE_DIR=/scratch/default -e OCTOJET_NVFP4_FLASHNEXT=/tf/flashnext-nvfp4-mixed \
    -e OCTOJET_NVFP4_FLASHNEXT_LAYERS=4 -w /octojet/engine --entrypoint bash "$IMAGE" -c "
      set -uo pipefail; pip install -q 'pytest>=8,<10'; rc=0
      python -m pytest $CUDA -q -p no:cacheprovider --basetemp=/scratch/p1 || rc=1
      python -m pytest $CPU -q -p no:cacheprovider --basetemp=/scratch/p2 || rc=1
      echo \"F9 TESTS rc=\$rc\"; exit \$rc" 2>&1 | tee "$OUT/f9-tests.txt" | tee -a "$TL" || TESTS_RC=1
  docker run --rm -v "$SCRATCH:/scratch" --entrypoint sh "$IMAGE" -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null; true' || true
  rmdir "$SCRATCH" 2>/dev/null || true
  if [ "${TESTS_RC:-0}" = 1 ]; then
    log "tests: FAILED (see $OUT/f9-tests.txt)"; [ "$FORCE" = 1 ] || { log "not measuring a build that fails its tests (FORCE=1 goes on)"; exit 1; }
    FAIL=1
  else log "tests: passed"; fi
fi
if [ "$TESTS_ONLY" = 1 ]; then exit "${TESTS_RC:-0}"; fi

step(){ local what=$1; shift; log "$C: $what"; "$@" 2>&1 | tee -a "$TL" || { log "$C: $what FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }; }
startup(){ grep -h "rope:\|startup estimate\|streams of\|loaded in\|serving f1" "$OUT/$PREFIX-$1-server.log" \
             > "$OUT/f9-$1-startup.txt" 2>/dev/null || true; log "$1 startup:"; tee -a "$TL" < "$OUT/f9-$1-startup.txt"; }
accuracy(){ [ "$ACC" = 1 ] || return 0
  step "GSM8K 250 + HumanEval 164" bash -c "python3 bench/acc_eval.py $URL f1 '$OUT/f9-$1-acc' > '$OUT/f9-$1-acc-summary.json'"
  step humaneval bash bench/run_humaneval.sh "$OUT/f9-$1-acc.jsonl"; }

BASE_SERVE=("${SERVE[@]}")
pause_on
log "WINDOW: stopping production oj-serve ($PROD_STOP)"
bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
port_free || { log "WINDOW: port 8080 still listening"; exit 1; }

C=yarn; SERVE=("${BASE_SERVE[@]}" --rope-yarn "$FACTOR")
start yarn; startup yarn
step warm-up python3 bench/cmp_probe.py ttft "$URL" f1 --prompt "$T/warm.txt" --label warm-yarn --out "$OUT/f9-warm.jsonl"
step "cold $COLD prompt" needle --sizes "$COLD" --depths 0.5 --label yarn-cold --out "$OUT/f9-yarn-cold.jsonl"
step "needle $YARN_SIZES" needle --sizes "$YARN_SIZES" --depths "$DEPTHS" --label yarn --out "$OUT/f9-needle-yarn.jsonl"
accuracy yarn
stop yarn

C=base; SERVE=("${BASE_SERVE[@]}")
start base; startup base
step warm-up python3 bench/cmp_probe.py ttft "$URL" f1 --prompt "$T/warm.txt" --label warm-base --out "$OUT/f9-warm.jsonl"
step "needle $BASE_SIZES" needle --sizes "$BASE_SIZES" --depths "$DEPTHS" --label base --out "$OUT/f9-needle-base.jsonl"
accuracy base
stop base

python3 - "$OUT" "$FACTOR" <<'PY' | tee "$OUT/f9-summary.txt" | tee -a "$TL" || FAIL=1
import json, os, re, sys
out, factor = sys.argv[1], sys.argv[2]
def read(name):
    p = os.path.join(out, name)
    return open(p).read() if os.path.exists(p) else ""
def rows(name):
    return [json.loads(l) for l in read(name).splitlines() if l.strip()]
print(f"F9 YaRN (--rope-yarn {factor}) on one Spark: production flags (int8 KV, --parallel 3, packed cache), text only")
for label in ("yarn", "base"):
    text = read(f"f9-{label}-startup.txt")
    print(f"\n== server {'A: --rope-yarn ' + factor if label == 'yarn' else 'B: plain RoPE (baseline)'} ==")
    print(text.strip() or "(no startup lines: the server did not start; see the server log)")
    est = re.search(r"native (\d+), allocated prompt/reply window (\d+)", text)
    streams = re.search(r"(\d+) streams of (\d+) prompt/reply tokens", text)
    if est and streams:
        native, window, n, per = map(int, (*est.groups(), *streams.groups()))
        print(f"-> native {native}, allocated window {window}; {n} streams of {per}: "
              f"{'FITS' if window >= native else 'SHRUNK to ' + str(window)} ({n} x {native} wanted)")
print("\n== needle in a haystack (found = the reply holds the passcode; TTFT = cold prompt) ==")
print("server | target | depth | prompt tokens | found | TTFT s | total s | note")
for name in ("f9-yarn-cold.jsonl", "f9-needle-yarn.jsonl", "f9-needle-base.jsonl"):
    for r in rows(name):
        note = ("refused: " + str(r["error"])[:80]) if r.get("error") else ("" if r.get("found") else f"reply {r.get('reply', '')[:50]!r}")
        print(f"{r['label']} | {r['target_tokens']} | {r['depth']} | {r.get('prompt_tokens')} | "
              f"{'yes' if r.get('found') else 'NO'} | {r.get('ttft_s')} | {r.get('total_s')} | {note}")
cold = rows("f9-yarn-cold.jsonl")
if cold:
    print(f"-> cold {cold[0]['target_tokens']}-token prompt with YaRN: TTFT {cold[0].get('ttft_s')} s "
          f"({cold[0].get('prompt_tokens')} prompt tokens)")
print("\n== accuracy (greedy, thinking off) ==")
for label in ("yarn", "base"):
    acc = read(f"f9-{label}-acc-summary.json").strip()
    he = read(f"f9-{label}-acc.jsonl.humaneval.txt").strip().splitlines()
    print(f"{label}: gsm8k {acc or '(not run)'}; humaneval {he[-1] if he else '(not run)'}")
PY
log "f9 done: rc $FAIL (summary $OUT/f9-summary.txt)"
exit "$FAIL"
