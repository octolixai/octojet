#!/usr/bin/env bash
# Multi-stream decode (run 6): aggregate and per-stream decode tok/s with 1-8 requests at once, Octojet against
# upstream TensorFold on Vontra's MLX 4-bit checkpoint, same box, same prompts (engine/tools/bench_concurrent.py:
# code and chat prompts, sampled and greedy, 256 tokens a request, median of 3). Text only, int8 KV, GPU exclusive,
# page cache dropped before each start, one warm-up request per server before timing.
#   ours-p3    Octojet ($REPO), mixed NVFP4 checkpoint, packed cache, --parallel 3 (production's stream count)
#   ours-p8    the same with --parallel 8 (smaller window a stream; what a many-client setup would run)
#   up-mlx-p8  upstream TensorFold $UP_TAG ($UP_SRC) on $MLX, --parallel 8 (its streams grow while memory lasts)
# Levels: LEVELS_P3 (default 1,2,3) for --parallel 3, LEVELS_P8 (default 1,2,3,4,8) for --parallel 8.
# Production oj-serve is STOPPED for the whole run (standing test approval; PROD_STOP), standby pause held and removed
# on every exit path. About 3 cases x ~15 min. Usage: bash bench/spark/multi-stream.sh [--dry-run]; CASES=... picks.
set -euo pipefail
REPO=${REPO:-$HOME/octojet-main}; OUT=${OUT:-$HOME/octojet-runs/run6}; NAME=${NAME:-oj-run6}; PARALLEL=3
PREFIX=run6; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
UP_TAG=${UP_TAG:-v0.6.5}; UP_IMAGE=${UP_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}; UP_SRC=${UP_SRC:-$HOME/tensorfold/tf-upstream-$UP_TAG}
MLX=${MLX:-$HOME/tensorfold/flashnext-mlx-4bit-mtp}
CASES=${CASES:-ours-p3 ours-p8 up-mlx-p8}
LEVELS_P3=${LEVELS_P3:-1,2,3}; LEVELS_P8=${LEVELS_P8:-1,2,3,4,8}
DRYRUN=0; [ "${1:-}" = "--dry-run" ] && DRYRUN=1
cd "$REPO"
[ -e "$MLX/config.json" ] || { echo "error: $MLX missing" >&2; exit 2; }
[ -d "$UP_SRC/tensorfold" ] || { echo "error: $UP_SRC has no tensorfold (run compare-upstream.sh --prepare with UP_TAG=$UP_TAG)" >&2; exit 2; }
[ -f engine/tools/bench_concurrent.py ] || { echo "error: engine/tools/bench_concurrent.py missing" >&2; exit 2; }
for c in $CASES; do case "$c" in ours-p3|ours-p8|up-mlx-p8) ;; *) echo "error: unknown case $c" >&2; exit 2;; esac; done
mkdir -p "$OUT"
if [ "$DRYRUN" = 1 ]; then echo "dry-run OK: cases $CASES (levels p3 $LEVELS_P3, p8 $LEVELS_P8)"; exit 0; fi
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true

serve(){  # $1 case: start its server, wait for /v1/models
  local c=$1 image engine env=() args
  case "$c" in
    ours-p3|ours-p8) image=$IMAGE; engine=ours
      args=(serve /tf/flashnext-nvfp4-mixed --name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel "${c#ours-p}"
            --packed-cache /tf/octojet-cache/packed)
      env=(-v "${DEPS:-$HOME/tensorfold/octojet-pydeps}:/deps:ro" -v /srv/ai:/srv/ai:ro);;   # /srv/ai: the mixed index points there
    up-mlx-p8) image=$UP_IMAGE; engine=up
      args=(serve "$MLX" --name f1 --host 0.0.0.0 --port 8080 --kv-dtype int8 --parallel 8)
      env=(-v "$MLX:$MLX:ro");;
  esac
  tenants_clear "$c" || exit 1
  drop_cache || { log "$c: page cache drop FAILED"; exit 1; }
  local tag=${image//[^a-zA-Z0-9]/_}
  mkdir -p "$HOME/tensorfold/torch-ext-$engine-$tag" "$HOME/tensorfold/triton-cache-$engine-$tag"
  local src=(-v "$REPO:/octojet:ro" -e PYTHONPATH=/octojet/engine/src:/deps)
  [ "$engine" = up ] && src=(-v "$UP_SRC:/up:ro" -e PYTHONPATH=/up)
  grep -E "MemAvailable|MemFree|^Cached|Shmem:|Dirty|Mlocked|Unevictable" /proc/meminfo > "$OUT/$c-meminfo-before.txt" || true
  docker run --rm --gpus all --ipc=host --network host --name "$NAME" "${src[@]}" -v "$HOME/tensorfold:/tf" \
    -v "$HOME/tensorfold/torch-ext-$engine-$tag:/root/.cache/torch_extensions" -v "$HOME/tensorfold/triton-cache-$engine-$tag:/root/.triton" \
    -e PYTHONDONTWRITEBYTECODE=1 ${env[@]+"${env[@]}"} --entrypoint python "$image" -m tensorfold.cli "${args[@]}" \
    > "$OUT/$c-server.log" 2>&1 &
  PID=$!; local t0=$SECONDS
  until curl -sf --max-time 5 http://127.0.0.1:8080/v1/models | grep -q '"f1"'; do
    kill -0 "$PID" 2>/dev/null || { log "$c: server exited before it was healthy (see $OUT/$c-server.log)"; return 1; }
    [ $((SECONDS - t0)) -lt "$READY_DEADLINE" ] || { log "$c: not healthy after ${READY_DEADLINE}s"; return 1; }; sleep 2
  done
  log "$c: healthy after $((SECONDS - t0))s ($image, $engine)"
  grep -h -i "startup estimate\|streams\|loaded in" "$OUT/$c-server.log" | head -4 | tee "$OUT/$c-startup.txt" | tee -a "$TL" || true; }

warm(){ curl -sf --max-time 300 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"f1","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32}' > /dev/null \
  || log "$1: warm-up request failed"; }

pause_on
log "WINDOW: stopping production oj-serve ($PROD_STOP)"
bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
port_free || { log "WINDOW: port 8080 still listening"; exit 1; }
for c in $CASES; do
  levels=$LEVELS_P8; [ "$c" = ours-p3 ] && levels=$LEVELS_P3
  if serve "$c"; then
    warm "$c"
    python3 engine/tools/bench_concurrent.py http://127.0.0.1:8080 f1 --levels "$levels" --label "$c" \
      --output "$OUT/$c-concurrent.json" 2>&1 | tee "$OUT/$c-concurrent.txt" | tee -a "$TL" || { log "$c: bench FAILED"; FAIL=1; }
  else FAIL=1; fi
  stop "$c"
done
log "run6 multi-stream done: rc $FAIL (files in $OUT)"
exit "$FAIL"
