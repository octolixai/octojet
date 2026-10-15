#!/usr/bin/env bash
# F14 (the GDN prompt chain copies its next stage while it steps the current one; OCTOJET_GDN_CHAIN_AHEAD, default on):
#   tests  (beside production, ~20 min) the F14 CUDA tests (synthetic chains: both forms byte-equal; whole prefills of
#          the 4-layer NVFP4 cut and of the MLX 4-bit checkpoint's first 4 layers with the switch on and off), the GDN
#          and Flash Next suites they touch, then bench/f14_gdn_bench.py (the chain's two forms, front and back, alone)
#   Then production oj-serve is STOPPED (owner's standing approval; PROD_STOP) with the standby watcher paused for the
#   whole window; every exit path removes the pause and the operator's watcher restarts oj-serve.
#   ttft   cold 32k / 128k / 210k (8 output tokens) on main vs f14 with production's flags and checkpoint ($TTFT_CKPT),
#          a throwaway long prompt first after each start; the table goes to f14-ttft.txt
# Usage: bash bench/spark/f14-test.sh [--dry-run] [--steps tests,ttft]   (about 35 min: tests 20 beside production,
#        window ~15)
# Knobs: REPO (f14 checkout), REPO_MAIN (main checkout, without F14), PROD_STOP, PROF_CKPT (host path of the MLX 4-bit
#        checkpoint for its tests; skipped when missing), TTFT_CKPT (container path), SHARED_GPU=1, FORCE=1.
set -euo pipefail
REPO=${REPO:-$HOME/octojet-f14}; REPO_MAIN=${REPO_MAIN:-$HOME/octojet-main}; OUT=${OUT:-$HOME/octojet-runs/f14}
NAME=${NAME:-oj-f14}; PARALLEL=3; PREFIX=f14; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
PROF_CKPT=${PROF_CKPT:-$HOME/tensorfold/flashnext-mlx-4bit-mtp}
TTFT_CKPT=${TTFT_CKPT:-/tf/flashnext-nvfp4-mixed}
STEPS=tests,ttft; DRYRUN=0
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRYRUN=1; shift;;
  --steps) STEPS=${2:-}; shift 2;;
  *) echo "usage: $0 [--dry-run] [--steps tests,ttft]" >&2; exit 2;; esac; done
[[ "$STEPS" =~ ^(tests|ttft)(,(tests|ttft))*$ ]] || { echo "error: --steps takes tests,ttft (got '$STEPS')" >&2; exit 2; }
has(){ [[ ",$STEPS," == *",$1,"* ]]; }
cd "$REPO"
for r in "$REPO" "$REPO_MAIN"; do [ -f "$r/engine/src/tensorfold/cli.py" ] || { echo "error: $r is not an Octojet checkout" >&2; exit 2; }; done
SWITCH=OCTOJET_GDN_CHAIN_AHEAD; GDN_PY=engine/src/tensorfold/cuda/kernels/gdn.py
grep -q "$SWITCH" "$REPO/$GDN_PY" 2>/dev/null || { echo "error: $REPO has no F14 ($SWITCH in $GDN_PY)" >&2; exit 2; }
if grep -q "$SWITCH" "$REPO_MAIN/$GDN_PY" 2>/dev/null; then
  echo "error: $REPO_MAIN already has F14: it must be a checkout of main before F14" >&2; exit 2; fi
if has ttft; then for m in m32k m128k m210k; do [ -f "$M/$m.json" ] || { echo "error: $M/$m.json missing" >&2; exit 2; }; done; fi
T=$OUT/prompts; mkdir -p "$T"

if has ttft; then
  log "prompts: manifests -> text (container, the mixed checkpoint's tokenizer)"
  if [ "$DRYRUN" = 0 ]; then
    docker run --rm -v "$REPO:/octojet:ro" -v "$HOME/tensorfold:/tf" -v "$OUT:/out" -w /octojet --entrypoint python "$IMAGE" \
      bench/cmp_probe.py text --tokenizer /tf/flashnext-nvfp4-mixed/tokenizer.json \
      /tf/octojet-manifests/m32k.json /tf/octojet-manifests/m128k.json /tf/octojet-manifests/m210k.json --out-dir /out/prompts | tee -a "$TL"
    for m in m32k m128k m210k; do [ -s "$T/$m.txt" ] || { echo "error: $T/$m.txt not written" >&2; exit 2; }; done
  fi
  printf "Say hello in five words." > "$T/warm.txt"
fi
variant(){ local f="$T/$2-$1.txt"; { printf "F14 run, %s, %s: a different prompt.\n" "$2" "$1"; cat "$T/$1.txt" 2>/dev/null || true; } > "$f"; echo "$f"; }

FLAGS=(--name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel "$PARALLEL")
serve(){  # $1 label, $2 engine (main|f14): production's flags and checkpoint
  local label=$1 eng=$2 log_="$OUT/f14-$1-server.log" src
  [ -n "$DRY" ] && return 0
  tenants_clear "$label" || exit 1
  drop_cache || { log "$label: page cache drop FAILED; aborting"; exit 1; }; log "$label: page cache dropped"
  free -g | tee -a "$TL"
  src=$REPO; [ "$eng" = main ] && src=$REPO_MAIN
  local args=(serve "$TTFT_CKPT" "${FLAGS[@]}")
  [ "$TTFT_CKPT" = /tf/flashnext-nvfp4-mixed ] && args+=(--packed-cache /tf/octojet-cache/packed)
  printf '%s\n' "$eng" "${args[@]}" > "$OUT/f14-$label-args.txt"
  log "$label: server start ($eng, $TTFT_CKPT)"
  docker run --rm --gpus all --ipc=host --network host --name "$NAME" -v "$src:/octojet:ro" -v "$HOME/tensorfold:/tf" \
    -v /srv/ai:/srv/ai:ro -v "$HOME/tensorfold/torch-ext:/root/.cache/torch_extensions" -v "$HOME/tensorfold/triton-cache:/root/.triton" \
    -e PYTHONPATH=/octojet/engine/src -e PYTHONDONTWRITEBYTECODE=1 -w /octojet/engine --entrypoint python "$IMAGE" \
    -m tensorfold.cli "${args[@]}" > "$log_" 2>&1 &
  PID=$!; local t0=$SECONDS
  until curl -sf --max-time 5 http://127.0.0.1:8080/v1/models | grep -q '"f1"'; do
    kill -0 "$PID" 2>/dev/null || { log "$label: server exited before it was healthy (see $log_)"; exit 1; }
    [ $((SECONDS - t0)) -lt "$READY_DEADLINE" ] || { log "$label: not healthy after ${READY_DEADLINE}s"; exit 1; }; sleep 1
  done
  log "$label: healthy after $((SECONDS - t0))s"
  grep -h -i "startup estimate\|loaded in\|streams of" "$log_" | tee -a "$TL" || true; }
probe(){ [ -n "$DRY" ] && return 0; python3 bench/cmp_probe.py "$@" 2>&1 | tee -a "$TL" || { log "probe $1 ${*: -1} FAILED"; FAIL=1; }; }
window(){
  for eng in main f14; do
    serve "ttft-$eng" "$eng"
    probe ttft http://127.0.0.1:8080 f1 --prompt "$T/warm.txt" --label "$eng-warm" --max-tokens 8 --out "$OUT/f14-warm.jsonl"
    probe ttft http://127.0.0.1:8080 f1 --prompt "$(variant m32k "$eng-throwaway")" --label "$eng-throwaway" --max-tokens 8 --out "$OUT/f14-warm.jsonl"
    for s in m32k m128k m210k; do
      probe ttft http://127.0.0.1:8080 f1 --prompt "$(variant "$s" "ttft-$eng")" --label "$eng-$s" --max-tokens 8 --out "$OUT/f14-ttft.jsonl"
    done
    stop "ttft-$eng"
  done; }

if has ttft; then DRY=--dry-run; window; DRY=; fi
log "dry-run OK: steps $STEPS, main $REPO_MAIN, f14 $REPO"
[ "$DRYRUN" = 1 ] && exit 0
{ git -C "$REPO" rev-parse HEAD; git -C "$REPO_MAIN" rev-parse HEAD; } > "$OUT/commit.txt" 2>/dev/null || true

if has tests; then
  SCRATCH=$HOME/tensorfold/octojet-scratch; mkdir -p "$SCRATCH"
  MLX_ENV=(); [ -e "$PROF_CKPT/config.json" ] && MLX_ENV=(-v "$PROF_CKPT:$PROF_CKPT:ro" -e "OCTOJET_MLX_FLASHNEXT=$PROF_CKPT")
  log "tests: container tests (F14 chain forms, 4-layer NVFP4 cut, MLX 4-bit layers) and the kernel bench, beside production"
  docker run --rm --gpus all --ipc=host "${MOUNTS[@]}" ${MLX_ENV[@]+"${MLX_ENV[@]}"} -v "$SCRATCH:/scratch" "${ENVS[@]}" -e TMPDIR=/scratch \
    -e OCTOJET_CACHE_DIR=/scratch/default -e OCTOJET_NVFP4_FLASHNEXT=/tf/flashnext-nvfp4-mixed -e OCTOJET_NVFP4_FLASHNEXT_LAYERS=4 \
    -w /octojet/engine --entrypoint bash "$IMAGE" -c "
      set -uo pipefail; pip install -q 'pytest>=8,<10'; rc=0
      python -m pytest tests/test_gdn_chain_ahead_switch.py -q -p no:cacheprovider || rc=1
      python -m pytest tests/cuda/test_gdn_chain_ahead.py -q -p no:cacheprovider --basetemp=/scratch/pytest1 || rc=1
      python -m pytest tests/cuda/test_gdn.py tests/cuda/test_flashnext_gdn_io.py tests/cuda/test_flashnext_kernels.py -q -p no:cacheprovider --basetemp=/scratch/pytest2 || rc=1
      python -m pytest tests/cuda/test_flashnext_forward.py tests/cuda/test_prefill_timing_flashnext.py tests/cuda/test_qwen27_prefill.py -q -p no:cacheprovider --basetemp=/scratch/pytest3 || rc=1
      python ../bench/f14_gdn_bench.py --rows 4096 || rc=1
      echo \"F14 TESTS rc=\$rc\"
      exit \$rc" 2>&1 | tee "$OUT/f14-tests.txt" || { log "tests: container tests FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }
  docker run --rm -v "$SCRATCH:/scratch" --entrypoint sh "$IMAGE" -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null; true' || true
  rmdir "$SCRATCH" 2>/dev/null || true
  [ "$FAIL" = 0 ] || [ "${FORCE:-0}" = 1 ] || { log "tests FAILED: no window (FORCE=1 goes on)"; exit 1; }
fi

if has ttft; then
  pause_on
  log "WINDOW: stopping production oj-serve ($PROD_STOP); the standby watcher restarts it when the pause is removed"
  bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
  port_free || { log "WINDOW: port 8080 still listening after stopping production"; exit 1; }
  window
  pause_off
  log "WINDOW: done, standby pause removed (the watcher restarts oj-serve)"
  if [ -s "$OUT/f14-ttft.jsonl" ]; then
    python3 - "$OUT/f14-ttft.jsonl" <<'PY' | tee "$OUT/f14-ttft.txt" | tee -a "$TL"
import json, sys
rows = {r["label"]: r for r in (json.loads(l) for l in open(sys.argv[1]) if l.strip())}
print("size | main TTFT s | f14 TTFT s | f14 / main")
for s in ("m32k", "m128k", "m210k"):
    a, b = (rows.get(f"{e}-{s}", {}).get("ttft_s") for e in ("main", "f14"))
    print(f"{s} | {a} | {b} | {round(b / a, 3) if a and b else None}")
PY
  fi
fi
log "f14 test done: rc $FAIL (tests $OUT/f14-tests.txt, ttft $OUT/f14-ttft.txt)"
exit "$FAIL"
