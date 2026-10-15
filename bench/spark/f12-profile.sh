#!/usr/bin/env bash
# F12 (fused prompt hyper-connection read-out): tests, a black-box kernel profile against upstream, and TTFT.
#   tests    (beside production, ~25 min) the F12 CUDA tests (synthetic shapes; the 4-layer NVFP4 cut; the MLX 4-bit
#            checkpoint's first 4 layers) and the forward / prefill-timing suites they touch, on $REPO (f12)
#   Then production oj-serve is STOPPED (owner's standing approval; PROD_STOP) with the standby watcher paused for the
#   whole window; every exit path removes the pause and the operator's watcher restarts oj-serve.
#   profile  nsys (CUDA trace only, black box: `nsys launch` around the server, `nsys start/stop` around one request)
#            of a cold 32k and a cold 128k prompt (1 output token) on
#              main   Octojet main ($REPO_MAIN)          on $PROF_CKPT (default: upstream's MLX 4-bit checkpoint)
#              f12    Octojet f12 ($REPO)                on $PROF_CKPT
#              up     upstream TensorFold $UP_TAG ($UP_SRC, $UP_IMAGE) on $PROF_CKPT
#            each after a short warm-up and a throwaway long prompt (the first-request cost is not captured)
#   ttft     cold 32k / 128k / 210k (8 output tokens) on main vs f12 with production's flags and checkpoint
#            ($TTFT_CKPT), a throwaway long prompt first after each start
#   After production is restored (CPU only): nsys stats per capture and bench/kern_compare.py: per-kernel GPU time
#   side by side, and the kernels where f12 and upstream spend more / less than main (f12-kernels-32k.txt,
#   f12-kernels-128k.txt), plus f12-ttft.txt.
# Usage: bash bench/spark/f12-profile.sh [--dry-run] [--steps tests,profile,ttft]   (about 75 min with every step:
#        tests 25 beside production; window ~40 (profile ~25, ttft ~15); stats ~10 after the restore)
# Knobs: REPO (f12 checkout), REPO_MAIN (main checkout), PROD_STOP, UP_TAG, UP_IMAGE, UP_SRC, PROF_CKPT (host path),
#        TTFT_CKPT (container path), SHARED_GPU=1, OCTOJET_PREFILL_HC_TILE (f12's up+mix block shape, 0-3).
set -euo pipefail
REPO=${REPO:-$HOME/octojet-f12}; REPO_MAIN=${REPO_MAIN:-$HOME/octojet-main}; OUT=${OUT:-$HOME/octojet-runs/f12}
NAME=${NAME:-oj-f12}; PARALLEL=3; PREFIX=f12; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
UP_TAG=${UP_TAG:-v0.6.2}; UP_IMAGE=${UP_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}; UP_SRC=${UP_SRC:-$HOME/tensorfold/tf-upstream-$UP_TAG}
PROF_CKPT=${PROF_CKPT:-$HOME/tensorfold/flashnext-mlx-4bit-mtp}
TTFT_CKPT=${TTFT_CKPT:-/tf/flashnext-nvfp4-mixed}
P=${P:-$HOME/tensorfold/octojet-profiles/f12}; NSYS_HOST=${NSYS_HOST:-/opt/nvidia/nsight-systems/2025.3.2}
POLL=${POLL:-5}; STEPS=tests,profile,ttft; DRYRUN=0; PROF_ORDER=${PROF_ORDER:-main f12 up}
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRYRUN=1; shift;;
  --steps) STEPS=${2:-}; shift 2;;
  *) echo "usage: $0 [--dry-run] [--steps tests,profile,ttft]" >&2; exit 2;; esac; done
[[ "$STEPS" =~ ^(tests|profile|ttft)(,(tests|profile|ttft))*$ ]] || { echo "error: --steps takes tests,profile,ttft (got '$STEPS')" >&2; exit 2; }
has(){ [[ ",$STEPS," == *",$1,"* ]]; }
cd "$REPO"
for r in "$REPO" "$REPO_MAIN"; do [ -f "$r/engine/src/tensorfold/cli.py" ] || { echo "error: $r is not an Octojet checkout" >&2; exit 2; }; done
[ -f "$REPO/engine/src/tensorfold/families/qwen4_exp/cuda/hc_prefill.py" ] || { echo "error: $REPO has no F12 (hc_prefill.py)" >&2; exit 2; }
if [ -f "$REPO_MAIN/engine/src/tensorfold/families/qwen4_exp/cuda/hc_prefill.py" ]; then
  echo "error: $REPO_MAIN already has F12 (hc_prefill.py): it must be a checkout of main before F12" >&2; exit 2; fi
for m in m32k m128k m210k; do [ -f "$M/$m.json" ] || { echo "error: $M/$m.json missing" >&2; exit 2; }; done
if has profile; then
  [ -e "$PROF_CKPT/config.json" ] || { echo "error: $PROF_CKPT/config.json missing (PROF_CKPT)" >&2; exit 2; }
  [ -d "$UP_SRC/tensorfold" ] || { echo "error: $UP_SRC has no tensorfold (bash bench/spark/compare-upstream.sh --prepare)" >&2; exit 2; }
fi
mkdir -p "$P" "$OUT"
T=$OUT/prompts; mkdir -p "$T"

# prompts: text versions of the manifests (upstream's completions take text only), each with its own first line so no
# run shares a prefix with another (every timed prompt is cold)
log "prompts: manifests -> text (container, the mixed checkpoint's tokenizer)"
docker run --rm -v "$REPO:/octojet:ro" -v "$HOME/tensorfold:/tf" -v "$OUT:/out" -w /octojet --entrypoint python "$IMAGE" \
  bench/cmp_probe.py text --tokenizer /tf/flashnext-nvfp4-mixed/tokenizer.json \
  /tf/octojet-manifests/m32k.json /tf/octojet-manifests/m128k.json /tf/octojet-manifests/m210k.json --out-dir /out/prompts | tee -a "$TL"
for m in m32k m128k m210k; do [ -s "$T/$m.txt" ] || { echo "error: $T/$m.txt not written" >&2; exit 2; }; done
printf "Say hello in five words." > "$T/warm.txt"
variant(){ local f="$T/$2-$1.txt"; { printf "F12 run, %s, %s: a different prompt.\n" "$2" "$1"; cat "$T/$1.txt"; } > "$f"; echo "$f"; }

# nsys: the host's Nsight Systems install ($NSYS_HOST, e.g. /opt/nvidia/nsight-systems/<version>), mounted read-only at
# /opt/nsys in every server container (upstream's image included), so the three captures use the same profiler. When
# it is missing the profile step is skipped (logged) and the TTFT step still runs.
nsys_prepare(){
  if [ -x "$NSYS_HOST/bin/nsys" ]; then
    NSYS=/opt/nsys/bin/nsys; log "nsys: $NSYS_HOST/bin/nsys mounted at $NSYS"
  else
    log "nsys: no Nsight Systems at $NSYS_HOST (set NSYS_HOST); the profile step is skipped"
    STEPS=$(echo ",$STEPS," | sed 's/,profile,/,/; s/^,//; s/,$//')
  fi; }

FLAGS=(--name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel "$PARALLEL")
serve(){  # $1 label, $2 engine (main|f12|up), $3 checkpoint (path inside the container), $4 nsys (0|1)
  local label=$1 eng=$2 ckpt=$3 prof=$4 log_="$OUT/f12-$1-server.log"
  [ -n "$DRY" ] && return 0
  tenants_clear "$label" || exit 1
  drop_cache || { log "$label: page cache drop FAILED; aborting"; exit 1; }; log "$label: page cache dropped"
  free -g | tee -a "$TL"
  local args=(serve "$ckpt" "${FLAGS[@]}")
  [ "$ckpt" = /tf/flashnext-nvfp4-mixed ] && [ "$eng" != up ] && args+=(--packed-cache /tf/octojet-cache/packed)
  local mounts=(-v "$HOME/tensorfold:/tf" -v /srv/ai:/srv/ai:ro -v "$PROF_CKPT:$PROF_CKPT:ro" -v "$P:/prof" -v "$NSYS_HOST:/opt/nsys:ro")
  local envs=(-e PYTHONDONTWRITEBYTECODE=1) image
  case "$eng" in
    main) mounts+=(-v "$REPO_MAIN:/octojet:ro" -v "$HOME/tensorfold/torch-ext:/root/.cache/torch_extensions"
                   -v "$HOME/tensorfold/triton-cache:/root/.triton"); envs+=(-e PYTHONPATH=/octojet/engine/src); image=$IMAGE;;
    f12)  mounts+=(-v "$REPO:/octojet:ro" -v "$HOME/tensorfold/torch-ext:/root/.cache/torch_extensions"
                   -v "$HOME/tensorfold/triton-cache:/root/.triton"); envs+=(-e PYTHONPATH=/octojet/engine/src); image=$IMAGE
          [ -z "${OCTOJET_PREFILL_HC_TILE:-}" ] || envs+=(-e "OCTOJET_PREFILL_HC_TILE=$OCTOJET_PREFILL_HC_TILE");;
    up)   mkdir -p "$HOME/tensorfold/torch-ext-up" "$HOME/tensorfold/triton-cache-up"
          mounts+=(-v "$UP_SRC:/up:ro" -v "$HOME/tensorfold/torch-ext-up:/root/.cache/torch_extensions"
                   -v "$HOME/tensorfold/triton-cache-up:/root/.triton"); envs+=(-e PYTHONPATH=/up); image=$UP_IMAGE;;
  esac
  local cmd=(python -m tensorfold.cli "${args[@]}") wd=/octojet/engine
  [ "$eng" = up ] && wd=/tmp
  [ "$prof" = 1 ] && cmd=("$NSYS" launch --session-new=f12 --trace=cuda "${cmd[@]}")   # launch takes launch options only (2025.x)
  printf '%s\n' "$eng" "$image" "${cmd[@]}" > "$OUT/f12-$label-args.txt"
  log "$label: server start ($eng, $ckpt, nsys $prof)"
  docker run --rm --gpus all --ipc=host --network host --cap-add SYS_ADMIN --name "$NAME" "${mounts[@]}" "${envs[@]}" \
    -w "$wd" --entrypoint "${cmd[0]}" "$image" "${cmd[@]:1}" > "$log_" 2>&1 &
  PID=$!; local t0=$SECONDS
  until curl -sf --max-time 5 http://127.0.0.1:8080/v1/models | grep -q '"f1"'; do
    kill -0 "$PID" 2>/dev/null || { log "$label: server exited before it was healthy (see $log_)"; exit 1; }
    [ $((SECONDS - t0)) -lt "$READY_DEADLINE" ] || { log "$label: not healthy after ${READY_DEADLINE}s"; exit 1; }; sleep 1
  done
  log "$label: healthy after $((SECONDS - t0))s"
  grep -h -i "startup estimate\|loaded in\|streams of" "$log_" | tee -a "$TL" || true; }
probe(){ [ -n "$DRY" ] && return 0; python3 bench/cmp_probe.py "$@" 2>&1 | tee -a "$TL" || { log "probe $1 ${*: -1} FAILED"; FAIL=1; }; }
warm(){  # $1 label: a short request, then a throwaway long prompt (JIT, graphs, first-request allocations)
  probe ttft http://127.0.0.1:8080 f1 --prompt "$T/warm.txt" --label "$1-warm" --max-tokens 8 --out "$OUT/f12-warm.jsonl"
  probe ttft http://127.0.0.1:8080 f1 --prompt "$(variant m32k "$1-throwaway")" --label "$1-throwaway" --max-tokens 8 --out "$OUT/f12-warm.jsonl"; }
capture(){  # $1 label, $2 size: one cold prompt inside nsys start / stop; the report is written by the stop
  local rep="f12-$1-$2" f; f=$(variant "$2" "$1-prof")
  [ -n "$DRY" ] && return 0
  rm -f "$P/$rep.nsys-rep" "$P/$rep.sqlite" "$P/$rep-kern.csv"
  docker exec "$NAME" "$NSYS" start --session=f12 --output="/prof/$rep" --force-overwrite=true --sample=none --cpuctxsw=none 2>&1 | tee -a "$TL" \
    || { log "$1 $2: nsys start FAILED"; FAIL=1; return 0; }
  probe ttft http://127.0.0.1:8080 f1 --prompt "$f" --label "$1-prof-$2" --max-tokens 1 --out "$OUT/f12-prof-ttft.jsonl"
  docker exec "$NAME" "$NSYS" stop --session=f12 2>&1 | tee -a "$TL" || { log "$1 $2: nsys stop FAILED"; FAIL=1; return 0; }
  local i sz prev=-1 stable=0
  for i in $(seq 180); do      # the stop writes the report; wait until its size holds still (bounded, 15 min)
    sz=$(wc -c < "$P/$rep.nsys-rep" 2>/dev/null | tr -d ' ' || true); sz=${sz:-0}
    if [ "$sz" -gt 0 ] && [ "$sz" = "$prev" ]; then stable=$((stable + 1)); else stable=0; fi; prev=$sz
    [ "$stable" -ge 2 ] && { log "$rep: report written ($sz bytes)"; return 0; }
    sleep "$POLL"
  done
  log "$rep: report NOT written after 15 min"; FAIL=1; }

window(){
  if has profile; then
    for eng in $PROF_ORDER; do                         # PROF_ORDER="up f12 main": the reversed-order diagnostic
      serve "prof-$eng" "$eng" "$PROF_CKPT" 1
      warm "prof-$eng"
      for s in m32k m128k; do capture "$eng" "$s"; done
      stop "prof-$eng"
    done
  fi
  if has ttft; then
    for eng in main f12; do
      serve "ttft-$eng" "$eng" "$TTFT_CKPT" 0
      warm "ttft-$eng"
      for s in m32k m128k m210k; do
        probe ttft http://127.0.0.1:8080 f1 --prompt "$(variant "$s" "ttft-$eng")" --label "$eng-$s" --max-tokens 8 --out "$OUT/f12-ttft.jsonl"
      done
      stop "ttft-$eng"
    done
  fi; }
NSYS=/opt/nsys/nsys   # the dry run's placeholder; nsys_prepare sets the real path before the window

DRY=--dry-run; window; DRY=; log "dry-run OK: steps $STEPS, prompts in $T, main $REPO_MAIN, f12 $REPO"
[ "$DRYRUN" = 1 ] && exit 0
{ git -C "$REPO" rev-parse HEAD; git -C "$REPO_MAIN" rev-parse HEAD; echo "$UP_TAG"; } > "$OUT/commit.txt" 2>/dev/null || true

if has tests; then
  SCRATCH=$HOME/tensorfold/octojet-scratch; mkdir -p "$SCRATCH"
  MLX_ENV=(); [ -e "$PROF_CKPT/config.json" ] && MLX_ENV=(-v "$PROF_CKPT:$PROF_CKPT:ro" -e "OCTOJET_MLX_FLASHNEXT=$PROF_CKPT")
  log "tests: container tests (F12 kernels, 4-layer NVFP4 cut, MLX 4-bit layers), beside production"
  docker run --rm --gpus all --ipc=host "${MOUNTS[@]}" ${MLX_ENV[@]+"${MLX_ENV[@]}"} -v "$SCRATCH:/scratch" "${ENVS[@]}" -e TMPDIR=/scratch \
    -e OCTOJET_CACHE_DIR=/scratch/default -e OCTOJET_NVFP4_FLASHNEXT=/tf/flashnext-nvfp4-mixed -e OCTOJET_NVFP4_FLASHNEXT_LAYERS=4 \
    -w /octojet/engine --entrypoint bash "$IMAGE" -c "
      set -uo pipefail; pip install -q 'pytest>=8,<10'; rc=0
      python -m pytest tests/test_flashnext_prefill_hc_switch.py -q -p no:cacheprovider || rc=1
      python -m pytest tests/cuda/test_flashnext_prefill_hc.py -q -p no:cacheprovider --basetemp=/scratch/pytest1 || rc=1
      python -m pytest tests/cuda/test_flashnext_kernels.py tests/cuda/test_flashnext_forward.py -q -p no:cacheprovider --basetemp=/scratch/pytest2 || rc=1
      python -m pytest tests/cuda/test_prefill_timing_flashnext.py tests/cuda/test_qwen4_exp_nvfp4.py -q -p no:cacheprovider --basetemp=/scratch/pytest3 || rc=1
      echo \"F12 TESTS rc=\$rc\"
      exit \$rc" 2>&1 | tee "$OUT/f12-tests.txt" || { log "tests: container tests FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }
  docker run --rm -v "$SCRATCH:/scratch" --entrypoint sh "$IMAGE" -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null; true' || true
  rmdir "$SCRATCH" 2>/dev/null || true
  [ "$FAIL" = 0 ] || [ "${FORCE:-0}" = 1 ] || { log "tests FAILED: no window (FORCE=1 goes on)"; exit 1; }
fi

if has profile; then nsys_prepare; fi
if has profile || has ttft; then
  pause_on
  log "WINDOW: stopping production oj-serve ($PROD_STOP); the standby watcher restarts it when the pause is removed"
  bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
  port_free || { log "WINDOW: port 8080 still listening after stopping production"; exit 1; }
  window
  pause_off
  log "WINDOW: done, standby pause removed (the watcher restarts oj-serve)"
fi

if has profile; then   # CPU only, beside production
  for s in m32k m128k; do
    caps=()
    for eng in main f12 up; do
      rep="f12-$eng-$s"
      [ -s "$P/$rep.nsys-rep" ] || { log "$rep: no report"; FAIL=1; continue; }
      docker run --rm -v "$P:/prof" -v "$NSYS_HOST:/opt/nsys:ro" --entrypoint "$NSYS" "$IMAGE" stats --report cuda_gpu_kern_sum \
        --format csv "/prof/$rep.nsys-rep" > "$P/$rep-kern.csv" 2> "$P/$rep-stats.err" || { log "$rep: nsys stats FAILED"; FAIL=1; continue; }
      cp "$P/$rep-kern.csv" "$OUT/"; caps+=("$eng=$P/$rep-kern.csv")
    done
    [ ${#caps[@]} -ge 2 ] || continue
    python3 bench/kern_compare.py "${caps[@]}" --top 25 --json "$OUT/f12-kernels-$s.json" > "$OUT/f12-kernels-$s.txt" \
      || { log "kern_compare $s FAILED"; FAIL=1; }
    head -40 "$OUT/f12-kernels-$s.txt" | tee -a "$TL"
  done
fi
if has ttft && [ -s "$OUT/f12-ttft.jsonl" ]; then
  python3 - "$OUT/f12-ttft.jsonl" <<'PY' | tee "$OUT/f12-ttft.txt" | tee -a "$TL"
import json, sys
rows = {r["label"]: r for r in (json.loads(l) for l in open(sys.argv[1]) if l.strip())}
print("size | main TTFT s | f12 TTFT s | f12 / main")
for s in ("m32k", "m128k", "m210k"):
    a, b = (rows.get(f"{e}-{s}", {}).get("ttft_s") for e in ("main", "f12"))
    print(f"{s} | {a} | {b} | {round(b / a, 3) if a and b else None}")
PY
fi
log "f12 profile done: rc $FAIL (kernels $OUT/f12-kernels-*.txt, ttft $OUT/f12-ttft.txt, tests $OUT/f12-tests.txt)"
exit "$FAIL"
